"""
GPU-specific optimizations: Flash Attention via Triton kernel.

Standard attention materializes the full T×T scores matrix — O(T²) memory.
Flash Attention tiles Q×K^T in GPU SRAM — O(T) memory, same result.

    pip install torch triton
    python3 engine_cuda.py

Requires NVIDIA GPU. Triton needs Linux or WSL2 on Windows.
"""

import torch
import torch.nn as nn
import math
import time
import tiktoken
import triton
import triton.language as tl
from engine import FeedForward, load_openai_weights, apply_sampling


# ============================================================
# Flash Attention — Triton Kernel
# ============================================================
#
# Instead of computing the full T×T attention matrix:
#   scores = Q @ K^T          ← T×T matrix, O(T²) memory
#   weights = softmax(scores)  ← another T×T matrix
#   out = weights @ V
#
# We tile the computation: load BLOCK_M queries at a time,
# iterate over BLOCK_N keys at a time, never materializing
# the full T×T matrix. Each tile lives in GPU SRAM (~20MB),
# not HBM (~8GB on your 4060 Ti).
#
# The trick: "online softmax" — compute softmax incrementally
# by maintaining a running max and sum across K tiles.

@triton.jit
def _flash_attn_fwd(
    Q, K, V, Out,
    stride_qz, stride_qm, stride_qd,
    stride_kz, stride_kn, stride_kd,
    stride_vz, stride_vn, stride_vd,
    stride_oz, stride_om, stride_od,
    N_CTX,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, D: tl.constexpr,
):
    pid_m = tl.program_id(0)   # which tile of queries (0..ceil(T/BLOCK_M))
    pid_z = tl.program_id(1)   # which batch×head

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, D)

    # --- Load Q tile: (BLOCK_M, D) --- stays in SRAM for entire loop
    q_ptrs = Q + pid_z * stride_qz + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd
    q = tl.load(q_ptrs, mask=offs_m[:, None] < N_CTX, other=0.0)
    q = q * (1.0 / tl.sqrt(tl.cast(D, tl.float32)))

    # --- Online softmax accumulators ---
    m_i = tl.full([BLOCK_M], value=float('-inf'), dtype=tl.float32)  # running max
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)                      # running sum(exp)
    acc = tl.zeros([BLOCK_M, D], dtype=tl.float32)                    # output accumulator

    # Causal: query at position i only attends to keys at positions <= i
    hi = tl.minimum((pid_m + 1) * BLOCK_M, N_CTX)

    # --- Iterate over K,V tiles ---
    for start_n in range(0, hi, BLOCK_N):
        cur_n = start_n + offs_n

        # Load K tile transposed: (D, BLOCK_N) for Q @ K^T
        kt_ptrs = K + pid_z * stride_kz + offs_d[:, None] * stride_kd + cur_n[None, :] * stride_kn
        kt = tl.load(kt_ptrs, mask=cur_n[None, :] < N_CTX, other=0.0)

        # S = Q @ K^T for this tile: (BLOCK_M, BLOCK_N)
        s = tl.dot(q, kt)

        # Causal mask: zero out future positions
        s = tl.where(offs_m[:, None] >= cur_n[None, :], s, float('-inf'))

        # Online softmax update
        m_new = tl.maximum(m_i, tl.max(s, 1))
        alpha = tl.exp(m_i - m_new)       # rescale factor for old accumulator
        p = tl.exp(s - m_new[:, None])     # softmax numerator for this tile

        # Load V tile: (BLOCK_N, D)
        v_ptrs = V + pid_z * stride_vz + cur_n[:, None] * stride_vn + offs_d[None, :] * stride_vd
        v = tl.load(v_ptrs, mask=cur_n[:, None] < N_CTX, other=0.0)

        # Accumulate: rescale old values + add new contribution
        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
        l_i = l_i * alpha + tl.sum(p, 1)
        m_i = m_new

    # Normalize by softmax denominator
    acc = acc / l_i[:, None]

    # Store output: (BLOCK_M, D)
    o_ptrs = Out + pid_z * stride_oz + offs_m[:, None] * stride_om + offs_d[None, :] * stride_od
    tl.store(o_ptrs, acc.to(Out.dtype.element_ty), mask=offs_m[:, None] < N_CTX)


def flash_attention(q, k, v):
    B, H, T, D = q.shape
    q = q.reshape(B * H, T, D).contiguous()
    k = k.reshape(B * H, T, D).contiguous()
    v = v.reshape(B * H, T, D).contiguous()
    out = torch.empty_like(q)

    BLOCK_M, BLOCK_N = 64, 64
    grid = (triton.cdiv(T, BLOCK_M), B * H)

    _flash_attn_fwd[grid](
        q, k, v, out,
        q.stride(0), q.stride(1), q.stride(2),
        k.stride(0), k.stride(1), k.stride(2),
        v.stride(0), v.stride(1), v.stride(2),
        out.stride(0), out.stride(1), out.stride(2),
        T,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, D=D,
    )
    return out.reshape(B, H, T, D)


# ============================================================
# Standard Attention (for comparison)
# ============================================================

def standard_attention(q, k, v):
    B, H, T, D = q.shape
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(D)
    mask = torch.tril(torch.ones(T, T, device=q.device))
    scores = scores.masked_fill(mask == 0, float('-inf'))
    weights = torch.softmax(scores, dim=-1)
    return torch.matmul(weights, v)


# ============================================================
# GPT-2 with Flash Attention + KV-Cache
# ============================================================
# Prefill (full prompt): Flash Attention — no T×T matrix
# Decode (1 token):      Standard attention with KV-cache

class FlashMultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.c_attn = nn.Linear(d_model, 3 * d_model)
        self.c_proj = nn.Linear(d_model, d_model)

    def forward(self, x, kv_cache=None):
        B, T, C = x.size()
        qkv = self.c_attn(x)
        q, k, v = qkv.split(C, dim=2)
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        if kv_cache is not None:
            prev_k, prev_v = kv_cache
            k = torch.cat([prev_k, k], dim=2)
            v = torch.cat([prev_v, v], dim=2)
            scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
            scores = torch.softmax(scores, dim=-1)
            out = torch.matmul(scores, v)
        else:
            out = flash_attention(q, k, v)

        new_cache = (k, v)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(out), new_cache


class FlashTransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.ln_1 = nn.LayerNorm(d_model)
        self.attn = FlashMultiHeadAttention(d_model, n_heads)
        self.ln_2 = nn.LayerNorm(d_model)
        self.mlp = FeedForward(d_model)

    def forward(self, x, kv_cache=None):
        attn_out, new_cache = self.attn(self.ln_1(x), kv_cache)
        x = x + attn_out
        x = x + self.mlp(self.ln_2(x))
        return x, new_cache


class GPT2Flash(nn.Module):
    def __init__(self, vocab_size=50257, d_model=768, n_heads=12, n_layers=12, max_len=1024):
        super().__init__()
        self.d_model = d_model
        self.wte = nn.Embedding(vocab_size, d_model)
        self.wpe = nn.Embedding(max_len, d_model)
        self.blocks = nn.ModuleList([FlashTransformerBlock(d_model, n_heads) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(d_model)

    def forward(self, input_ids, past_kv=None, start_pos=0):
        B, T = input_ids.size()
        positions = torch.arange(start_pos, start_pos + T, device=input_ids.device).unsqueeze(0)
        x = self.wte(input_ids) + self.wpe(positions)

        new_kv = []
        for i, block in enumerate(self.blocks):
            layer_cache = past_kv[i] if past_kv is not None else None
            x, cache = block(x, layer_cache)
            new_kv.append(cache)

        x = self.ln_f(x)
        logits = x @ self.wte.weight.T
        return logits, new_kv


def generate_flash(model, tokens, max_tokens=50, temperature=0.8, top_k=40, repetition_penalty=1.2):
    model.eval()
    with torch.no_grad():
        device = next(model.parameters()).device
        input_ids = torch.tensor([tokens], device=device)
        logits, past_kv = model(input_ids)
        next_token = apply_sampling(logits[0, -1, :].cpu(), tokens, temperature, top_k, repetition_penalty)
        tokens.append(next_token)
        yield next_token

        for _ in range(max_tokens - 1):
            input_ids = torch.tensor([[tokens[-1]]], device=device)
            logits, past_kv = model(input_ids, past_kv=past_kv, start_pos=len(tokens) - 1)
            next_token = apply_sampling(logits[0, -1, :].cpu(), tokens, temperature, top_k, repetition_penalty)
            tokens.append(next_token)
            yield next_token


# ============================================================
# Benchmarks
# ============================================================

def benchmark_attention():
    """Compare standard vs flash attention at various sequence lengths."""
    print("=" * 60)
    print("Attention Kernel Benchmark (standard vs flash)")
    print("=" * 60)
    print(f"{'T':>6} | {'Standard':>12} | {'Flash':>12} | {'Speedup':>8} | {'Match':>5}")
    print("-" * 60)

    for T in [128, 256, 512, 1024]:
        q = torch.randn(1, 12, T, 64, device='cuda', dtype=torch.float32)
        k = torch.randn(1, 12, T, 64, device='cuda', dtype=torch.float32)
        v = torch.randn(1, 12, T, 64, device='cuda', dtype=torch.float32)

        # Verify correctness
        out_std = standard_attention(q, k, v)
        out_flash = flash_attention(q, k, v)
        match = torch.allclose(out_std, out_flash, atol=1e-2, rtol=1e-2)

        # Warmup
        for _ in range(10):
            standard_attention(q, k, v)
            flash_attention(q, k, v)
        torch.cuda.synchronize()

        # Benchmark standard
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(100):
            standard_attention(q, k, v)
        torch.cuda.synchronize()
        std_ms = (time.time() - start) / 100 * 1000

        # Benchmark flash
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(100):
            flash_attention(q, k, v)
        torch.cuda.synchronize()
        flash_ms = (time.time() - start) / 100 * 1000

        speedup = std_ms / flash_ms
        print(f"{T:>6} | {std_ms:>9.2f} ms | {flash_ms:>9.2f} ms | {speedup:>6.2f}x | {'✓' if match else '✗':>5}")

    print("=" * 60)


def benchmark_generation():
    """Full GPT-2 generation: CPU (KV-cache) vs GPU (Flash + KV-cache)."""
    enc = tiktoken.get_encoding("gpt2")
    prompt = "The meaning of life is"
    prompt_tokens = enc.encode(prompt)
    max_tokens = 100
    runs = 3

    print("\n" + "=" * 60)
    print(f"Generation Benchmark: \"{prompt}\"")
    print(f"Prompt tokens: {len(prompt_tokens)} | Generate: {max_tokens} tokens | Runs: {runs}")
    print("=" * 60)

    # --- GPU with Flash Attention ---
    print("\nLoading model (GPU + Flash Attention)...")
    model_flash = GPT2Flash()
    load_openai_weights(model_flash)
    model_flash = model_flash.cuda().eval()

    times_flash = []
    for r in range(runs):
        tokens = list(prompt_tokens)
        torch.manual_seed(42)
        torch.cuda.synchronize()
        start = time.time()
        for _ in generate_flash(model_flash, tokens, max_tokens=max_tokens):
            pass
        torch.cuda.synchronize()
        times_flash.append(time.time() - start)
        if r == 0:
            text_flash = enc.decode(tokens)

    avg_flash = sum(times_flash) / len(times_flash)
    tps_flash = max_tokens / avg_flash

    print(f"  GPU Flash:  {avg_flash:.2f}s  ({tps_flash:.1f} tokens/sec)")
    print(f"  Output: {text_flash[:100]}...")

    # --- CPU with KV-cache (for comparison) ---
    from engine import GPT2Cached, generate_streaming
    print("\nLoading model (CPU + KV-cache)...")
    model_cpu = GPT2Cached()
    load_openai_weights(model_cpu)
    model_cpu.eval()

    times_cpu = []
    for r in range(runs):
        tokens = list(prompt_tokens)
        torch.manual_seed(42)
        start = time.time()
        for _ in generate_streaming(model_cpu, tokens, max_tokens=max_tokens):
            pass
        times_cpu.append(time.time() - start)

    avg_cpu = sum(times_cpu) / len(times_cpu)
    tps_cpu = max_tokens / avg_cpu

    print(f"  CPU cache:  {avg_cpu:.2f}s  ({tps_cpu:.1f} tokens/sec)")

    speedup = avg_cpu / avg_flash
    print(f"\n  GPU speedup: {speedup:.1f}x faster")
    print("=" * 60)


def benchmark_memory():
    """Show VRAM usage: standard vs flash attention."""
    print("\n" + "=" * 60)
    print("Memory Benchmark: peak VRAM for attention")
    print("=" * 60)

    for T in [256, 512, 1024]:
        # Standard: T×T scores matrix = T² × 12 heads × 4 bytes
        scores_mem = T * T * 12 * 4 / 1024 / 1024
        # Flash: BLOCK_M × BLOCK_N tile = 64×64 × 12 heads × 4 bytes (in SRAM)
        tile_mem = 64 * 64 * 12 * 4 / 1024 / 1024
        ratio = scores_mem / tile_mem

        print(f"  T={T:>4}: standard {scores_mem:>6.1f} MB scores matrix | flash {tile_mem:.3f} MB tile | {ratio:.0f}x less memory")

    print("=" * 60)


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("ERROR: No CUDA GPU detected. This file requires an NVIDIA GPU.")
        print("If on Windows, use WSL2: https://docs.microsoft.com/en-us/windows/wsl/install")
        exit(1)

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB\n")

    benchmark_attention()
    benchmark_memory()
    benchmark_generation()
