import torch
import torch.nn as nn
import math
import time
import tiktoken


# ============================================================
# GPT-2 WITHOUT KV-Cache (baseline — recomputes everything)
# ============================================================

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.c_attn = nn.Linear(d_model, 3 * d_model)
        self.c_proj = nn.Linear(d_model, d_model)

    def forward(self, x):
        B, T, C = x.size()
        qkv = self.c_attn(x)
        q, k, v = qkv.split(C, dim=2)
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        mask = torch.tril(torch.ones(T, T, device=x.device)).unsqueeze(0).unsqueeze(0)
        scores = scores.masked_fill(mask == 0, float('-inf'))
        weights = torch.softmax(scores, dim=-1)
        out = torch.matmul(weights, v)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(out)


class FeedForward(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.c_fc = nn.Linear(d_model, 4 * d_model)
        self.c_proj = nn.Linear(4 * d_model, d_model)

    def forward(self, x):
        x = self.c_fc(x)
        x = x * torch.sigmoid(1.702 * x)
        return self.c_proj(x)


class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.ln_1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, n_heads)
        self.ln_2 = nn.LayerNorm(d_model)
        self.mlp = FeedForward(d_model)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT2(nn.Module):
    def __init__(self, vocab_size=50257, d_model=768, n_heads=12, n_layers=12, max_len=1024):
        super().__init__()
        self.wte = nn.Embedding(vocab_size, d_model)
        self.wpe = nn.Embedding(max_len, d_model)
        self.blocks = nn.ModuleList([TransformerBlock(d_model, n_heads) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(d_model)

    def forward(self, input_ids):
        B, T = input_ids.size()
        positions = torch.arange(T, device=input_ids.device).unsqueeze(0)
        x = self.wte(input_ids) + self.wpe(positions)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        return x @ self.wte.weight.T


# ============================================================
# GPT-2 WITH KV-Cache (optimized — only computes new token)
# ============================================================

class CachedMultiHeadAttention(nn.Module):
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

        new_cache = (k, v)

        total_len = k.size(2)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        mask = torch.tril(torch.ones(total_len, total_len, device=x.device))
        mask = mask[-T:, :]
        scores = scores.masked_fill(mask.unsqueeze(0).unsqueeze(0) == 0, float('-inf'))

        weights = torch.softmax(scores, dim=-1)
        out = torch.matmul(weights, v)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(out), new_cache


class CachedTransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.ln_1 = nn.LayerNorm(d_model)
        self.attn = CachedMultiHeadAttention(d_model, n_heads)
        self.ln_2 = nn.LayerNorm(d_model)
        self.mlp = FeedForward(d_model)

    def forward(self, x, kv_cache=None):
        attn_out, new_cache = self.attn(self.ln_1(x), kv_cache)
        x = x + attn_out
        x = x + self.mlp(self.ln_2(x))
        return x, new_cache


class GPT2Cached(nn.Module):
    def __init__(self, vocab_size=50257, d_model=768, n_heads=12, n_layers=12, max_len=1024):
        super().__init__()
        self.d_model = d_model
        self.wte = nn.Embedding(vocab_size, d_model)
        self.wpe = nn.Embedding(max_len, d_model)
        self.blocks = nn.ModuleList([CachedTransformerBlock(d_model, n_heads) for _ in range(n_layers)])
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


# ============================================================
# Weight Loading
# ============================================================

def load_openai_weights(model):
    from transformers import GPT2LMHeadModel
    print("Downloading GPT-2 weights from OpenAI...")
    hf_model = GPT2LMHeadModel.from_pretrained("gpt2")
    hf_sd = hf_model.state_dict()

    model.wte.weight.data.copy_(hf_sd["transformer.wte.weight"])
    model.wpe.weight.data.copy_(hf_sd["transformer.wpe.weight"])
    model.ln_f.weight.data.copy_(hf_sd["transformer.ln_f.weight"])
    model.ln_f.bias.data.copy_(hf_sd["transformer.ln_f.bias"])

    for i, block in enumerate(model.blocks):
        prefix = f"transformer.h.{i}"
        block.ln_1.weight.data.copy_(hf_sd[f"{prefix}.ln_1.weight"])
        block.ln_1.bias.data.copy_(hf_sd[f"{prefix}.ln_1.bias"])
        block.ln_2.weight.data.copy_(hf_sd[f"{prefix}.ln_2.weight"])
        block.ln_2.bias.data.copy_(hf_sd[f"{prefix}.ln_2.bias"])
        block.attn.c_attn.weight.data.copy_(hf_sd[f"{prefix}.attn.c_attn.weight"].T)
        block.attn.c_attn.bias.data.copy_(hf_sd[f"{prefix}.attn.c_attn.bias"])
        block.attn.c_proj.weight.data.copy_(hf_sd[f"{prefix}.attn.c_proj.weight"].T)
        block.attn.c_proj.bias.data.copy_(hf_sd[f"{prefix}.attn.c_proj.bias"])
        block.mlp.c_fc.weight.data.copy_(hf_sd[f"{prefix}.mlp.c_fc.weight"].T)
        block.mlp.c_fc.bias.data.copy_(hf_sd[f"{prefix}.mlp.c_fc.bias"])
        block.mlp.c_proj.weight.data.copy_(hf_sd[f"{prefix}.mlp.c_proj.weight"].T)
        block.mlp.c_proj.bias.data.copy_(hf_sd[f"{prefix}.mlp.c_proj.bias"])

    print("Weights loaded!\n")


# ============================================================
# Generation: No Cache vs KV-Cache
# ============================================================

def generate_no_cache(model, tokens, max_tokens=50, temperature=0.8, top_k=40):
    model.eval()
    with torch.no_grad():
        for _ in range(max_tokens):
            input_ids = torch.tensor([tokens[-1024:]])
            logits = model(input_ids)
            logits = logits[0, -1, :] / temperature
            if top_k > 0:
                values, _ = torch.topk(logits, top_k)
                logits[logits < values[-1]] = float('-inf')
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, 1).item()
            tokens.append(next_token)
    return tokens


def generate_with_cache(model, tokens, max_tokens=50, temperature=0.8, top_k=40):
    model.eval()
    with torch.no_grad():
        input_ids = torch.tensor([tokens])
        logits, past_kv = model(input_ids)
        logits = logits[0, -1, :] / temperature
        if top_k > 0:
            values, _ = torch.topk(logits, top_k)
            logits[logits < values[-1]] = float('-inf')
        probs = torch.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, 1).item()
        tokens.append(next_token)

        for _ in range(max_tokens - 1):
            input_ids = torch.tensor([[tokens[-1]]])
            logits, past_kv = model(input_ids, past_kv=past_kv, start_pos=len(tokens) - 1)
            logits = logits[0, -1, :] / temperature
            if top_k > 0:
                values, _ = torch.topk(logits, top_k)
                logits[logits < values[-1]] = float('-inf')
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, 1).item()
            tokens.append(next_token)
    return tokens


def apply_sampling(logits, tokens, temperature, top_k, repetition_penalty):
    if repetition_penalty != 1.0:
        for token_id in set(tokens):
            if logits[token_id] > 0:
                logits[token_id] /= repetition_penalty
            else:
                logits[token_id] *= repetition_penalty
    logits = logits / temperature
    if top_k > 0:
        values, _ = torch.topk(logits, top_k)
        logits[logits < values[-1]] = float('-inf')
    probs = torch.softmax(logits, dim=-1)
    return torch.multinomial(probs, 1).item()


def generate_streaming(model, tokens, max_tokens=50, temperature=0.8, top_k=40, repetition_penalty=1.2):
    model.eval()
    with torch.no_grad():
        input_ids = torch.tensor([tokens])
        logits, past_kv = model(input_ids)
        next_token = apply_sampling(logits[0, -1, :], tokens, temperature, top_k, repetition_penalty)
        tokens.append(next_token)
        yield next_token

        for _ in range(max_tokens - 1):
            input_ids = torch.tensor([[tokens[-1]]])
            logits, past_kv = model(input_ids, past_kv=past_kv, start_pos=len(tokens) - 1)
            next_token = apply_sampling(logits[0, -1, :], tokens, temperature, top_k, repetition_penalty)
            tokens.append(next_token)
            yield next_token


# ============================================================
# Benchmarks
# ============================================================

def benchmark(prompt="The meaning of life is", max_tokens=100, runs=3):
    enc = tiktoken.get_encoding("gpt2")
    prompt_tokens = enc.encode(prompt)

    print("=" * 60)
    print(f"Benchmark: \"{prompt}\"")
    print(f"Prompt tokens: {len(prompt_tokens)} | Generate: {max_tokens} tokens | Runs: {runs}")
    print("=" * 60)

    # --- No cache ---
    print("\nLoading model (no cache)...")
    model_no_cache = GPT2()
    load_openai_weights(model_no_cache)

    times_no_cache = []
    for r in range(runs):
        tokens = list(prompt_tokens)
        torch.manual_seed(42)
        start = time.time()
        result = generate_no_cache(model_no_cache, tokens, max_tokens=max_tokens)
        elapsed = time.time() - start
        times_no_cache.append(elapsed)
        if r == 0:
            text_no_cache = enc.decode(result)

    avg_no_cache = sum(times_no_cache) / len(times_no_cache)
    tps_no_cache = max_tokens / avg_no_cache

    print(f"\n  No cache:  {avg_no_cache:.2f}s  ({tps_no_cache:.1f} tokens/sec)")
    print(f"  Output: {text_no_cache[:100]}...")

    # --- With cache ---
    print("\nLoading model (with KV-cache)...")
    model_cached = GPT2Cached()
    load_openai_weights(model_cached)

    times_cached = []
    for r in range(runs):
        tokens = list(prompt_tokens)
        torch.manual_seed(42)
        start = time.time()
        result = generate_with_cache(model_cached, tokens, max_tokens=max_tokens)
        elapsed = time.time() - start
        times_cached.append(elapsed)
        if r == 0:
            text_cached = enc.decode(result)

    avg_cached = sum(times_cached) / len(times_cached)
    tps_cached = max_tokens / avg_cached

    print(f"\n  KV-cache:  {avg_cached:.2f}s  ({tps_cached:.1f} tokens/sec)")
    print(f"  Output: {text_cached[:100]}...")

    # --- Summary ---
    speedup = avg_no_cache / avg_cached
    print("\n" + "=" * 60)
    print(f"  No cache:   {avg_no_cache:.2f}s  |  {tps_no_cache:.1f} tokens/sec")
    print(f"  KV-cache:   {avg_cached:.2f}s  |  {tps_cached:.1f} tokens/sec")
    print(f"  Speedup:    {speedup:.2f}x faster with KV-cache")
    print("=" * 60)

    return {
        "no_cache": {"time": avg_no_cache, "tps": tps_no_cache},
        "kv_cache": {"time": avg_cached, "tps": tps_cached},
        "speedup": speedup,
    }


if __name__ == "__main__":
    benchmark(prompt="The meaning of life is", max_tokens=100, runs=3)
