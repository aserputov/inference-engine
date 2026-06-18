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
# PagedAttention KV-Cache
# ============================================================

class PagedKVCache:
    def __init__(self, n_layers, n_heads, head_dim, page_size=16, max_pages=256):
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.page_size = page_size
        self.max_pages = max_pages

        self.k_pool = torch.zeros(max_pages, n_heads, page_size, head_dim)
        self.v_pool = torch.zeros(max_pages, n_heads, page_size, head_dim)

        self.free_pages = list(range(max_pages))
        self.sequences = {}

    def allocate_sequence(self, seq_id):
        self.sequences[seq_id] = {
            "page_tables": [[] for _ in range(self.n_layers)],
            "length": 0,
        }

    def free_sequence(self, seq_id):
        if seq_id not in self.sequences:
            return
        seq = self.sequences[seq_id]
        for layer_pages in seq["page_tables"]:
            self.free_pages.extend(layer_pages)
        del self.sequences[seq_id]

    def _get_page(self):
        return self.free_pages.pop(0)

    def append(self, seq_id, layer_idx, new_k, new_v):
        seq = self.sequences[seq_id]
        pages = seq["page_tables"][layer_idx]
        pos_in_seq = seq["length"] if layer_idx == 0 else self.sequences[seq_id]["length"]
        n_new = new_k.size(2)

        for i in range(n_new):
            slot = (pos_in_seq + i) % self.page_size
            if slot == 0:
                page_id = self._get_page()
                pages.append(page_id)
            page_id = pages[-1]
            self.k_pool[page_id, :, slot, :] = new_k[0, :, i, :]
            self.v_pool[page_id, :, slot, :] = new_v[0, :, i, :]

        if layer_idx == self.n_layers - 1:
            seq["length"] += n_new

    def get_kv(self, seq_id, layer_idx):
        seq = self.sequences[seq_id]
        pages = seq["page_tables"][layer_idx]
        length = seq["length"]

        if not pages:
            return None, None

        k_parts = []
        v_parts = []
        remaining = length
        for page_id in pages:
            n = min(remaining, self.page_size)
            k_parts.append(self.k_pool[page_id, :, :n, :])
            v_parts.append(self.v_pool[page_id, :, :n, :])
            remaining -= n

        k = torch.cat(k_parts, dim=1).unsqueeze(0)
        v = torch.cat(v_parts, dim=1).unsqueeze(0)
        return k, v

    def stats(self):
        used = self.max_pages - len(self.free_pages)
        return {
            "total_pages": self.max_pages,
            "used_pages": used,
            "free_pages": len(self.free_pages),
            "active_sequences": len(self.sequences),
            "memory_used_mb": (used * self.n_heads * self.page_size * self.head_dim * 4 * 2) / 1024 / 1024,
        }


class PagedMultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.c_attn = nn.Linear(d_model, 3 * d_model)
        self.c_proj = nn.Linear(d_model, d_model)

    def forward(self, x, cached_k=None, cached_v=None):
        B, T, C = x.size()
        qkv = self.c_attn(x)
        q, k, v = qkv.split(C, dim=2)
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        new_k, new_v = k, v

        if cached_k is not None:
            k = torch.cat([cached_k, k], dim=2)
            v = torch.cat([cached_v, v], dim=2)

        total_len = k.size(2)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        mask = torch.tril(torch.ones(total_len, total_len, device=x.device))
        mask = mask[-T:, :]
        scores = scores.masked_fill(mask.unsqueeze(0).unsqueeze(0) == 0, float('-inf'))

        weights = torch.softmax(scores, dim=-1)
        out = torch.matmul(weights, v)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(out), new_k, new_v


class PagedTransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.ln_1 = nn.LayerNorm(d_model)
        self.attn = PagedMultiHeadAttention(d_model, n_heads)
        self.ln_2 = nn.LayerNorm(d_model)
        self.mlp = FeedForward(d_model)

    def forward(self, x, cached_k=None, cached_v=None):
        attn_out, new_k, new_v = self.attn(self.ln_1(x), cached_k, cached_v)
        x = x + attn_out
        x = x + self.mlp(self.ln_2(x))
        return x, new_k, new_v


class GPT2Paged(nn.Module):
    def __init__(self, vocab_size=50257, d_model=768, n_heads=12, n_layers=12, max_len=1024):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.head_dim = d_model // n_heads
        self.wte = nn.Embedding(vocab_size, d_model)
        self.wpe = nn.Embedding(max_len, d_model)
        self.blocks = nn.ModuleList([PagedTransformerBlock(d_model, n_heads) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(d_model)

    def forward(self, input_ids, paged_cache=None, seq_id=None, start_pos=0):
        B, T = input_ids.size()
        positions = torch.arange(start_pos, start_pos + T, device=input_ids.device).unsqueeze(0)
        x = self.wte(input_ids) + self.wpe(positions)

        for i, block in enumerate(self.blocks):
            cached_k, cached_v = None, None
            if paged_cache is not None and seq_id is not None:
                cached_k, cached_v = paged_cache.get_kv(seq_id, i)

            x, new_k, new_v = block(x, cached_k, cached_v)

            if paged_cache is not None and seq_id is not None:
                paged_cache.append(seq_id, i, new_k, new_v)

        x = self.ln_f(x)
        logits = x @ self.wte.weight.T
        return logits


def generate_paged(model, tokens, max_tokens=50, temperature=0.8, top_k=40, repetition_penalty=1.2):
    paged_cache = PagedKVCache(
        n_layers=model.n_layers, n_heads=model.n_heads,
        head_dim=model.head_dim, page_size=16, max_pages=256,
    )
    seq_id = 0
    paged_cache.allocate_sequence(seq_id)
    model.eval()

    with torch.no_grad():
        input_ids = torch.tensor([tokens])
        logits = model(input_ids, paged_cache=paged_cache, seq_id=seq_id)
        next_token = apply_sampling(logits[0, -1, :], tokens, temperature, top_k, repetition_penalty)
        tokens.append(next_token)
        yield next_token

        for _ in range(max_tokens - 1):
            input_ids = torch.tensor([[tokens[-1]]])
            logits = model(input_ids, paged_cache=paged_cache, seq_id=seq_id, start_pos=len(tokens) - 1)
            next_token = apply_sampling(logits[0, -1, :], tokens, temperature, top_k, repetition_penalty)
            tokens.append(next_token)
            yield next_token

    paged_cache.free_sequence(seq_id)


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
# Continuous Batching Scheduler
# ============================================================

import threading
import queue

class Request:
    def __init__(self, prompt_tokens, max_tokens=50, temperature=0.8, top_k=40, repetition_penalty=1.2):
        self.prompt_tokens = prompt_tokens
        self.tokens = list(prompt_tokens)
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_k = top_k
        self.repetition_penalty = repetition_penalty
        self.generated = 0
        self.past_kv = None
        self.started = False
        self.finished = False
        self.output_queue = queue.Queue()


class ContinuousBatchingScheduler:
    def __init__(self, model, max_batch_size=8):
        self.model = model
        self.max_batch_size = max_batch_size
        self.waiting = queue.Queue()
        self.active = []
        self.lock = threading.Lock()
        self.running = True

        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def submit(self, req):
        self.waiting.put(req)

    def _loop(self):
        while self.running:
            with self.lock:
                finished = [r for r in self.active if r.finished]
                for r in finished:
                    self.active.remove(r)

                while len(self.active) < self.max_batch_size and not self.waiting.empty():
                    req = self.waiting.get()
                    self.active.append(req)

            if not self.active:
                time.sleep(0.01)
                continue

            with self.lock:
                batch = list(self.active)

            self._step(batch)

    def _step(self, batch):
        self.model.eval()
        with torch.no_grad():
            for req in batch:
                if not req.started:
                    input_ids = torch.tensor([req.tokens])
                    logits, req.past_kv = self.model(input_ids)
                    req.started = True
                else:
                    input_ids = torch.tensor([[req.tokens[-1]]])
                    logits, req.past_kv = self.model(
                        input_ids, past_kv=req.past_kv, start_pos=len(req.tokens) - 1
                    )

                next_token = apply_sampling(
                    logits[0, -1, :], req.tokens,
                    req.temperature, req.top_k, req.repetition_penalty
                )
                req.tokens.append(next_token)
                req.generated += 1
                req.output_queue.put(next_token)

                if req.generated >= req.max_tokens:
                    req.finished = True
                    req.output_queue.put(None)


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


def benchmark_batching(prompts=None, max_tokens=50):
    if prompts is None:
        prompts = [
            "The meaning of life is",
            "In a shocking finding, scientists discovered",
            "Once upon a time, in a land far away,",
            "The best programming language is",
        ]

    enc = tiktoken.get_encoding("gpt2")
    print("\n" + "=" * 60)
    print(f"Batching Benchmark: {len(prompts)} prompts, {max_tokens} tokens each")
    print("=" * 60)

    model = GPT2Cached()
    load_openai_weights(model)

    # --- Sequential (one by one) ---
    torch.manual_seed(42)
    start = time.time()
    for prompt in prompts:
        tokens = list(enc.encode(prompt))
        for token_id in generate_streaming(model, tokens, max_tokens=max_tokens):
            pass
    sequential_time = time.time() - start
    total_tokens = len(prompts) * max_tokens
    seq_tps = total_tokens / sequential_time

    print(f"\n  Sequential:         {sequential_time:.2f}s  ({seq_tps:.1f} tokens/sec)")

    # --- Continuous batching ---
    scheduler = ContinuousBatchingScheduler(model, max_batch_size=len(prompts))
    torch.manual_seed(42)
    start = time.time()
    requests = []
    for prompt in prompts:
        tokens = list(enc.encode(prompt))
        req = Request(tokens, max_tokens=max_tokens)
        scheduler.submit(req)
        requests.append(req)

    for req in requests:
        while True:
            token = req.output_queue.get()
            if token is None:
                break
    batch_time = time.time() - start
    batch_tps = total_tokens / batch_time
    scheduler.running = False

    print(f"  Continuous batch:   {batch_time:.2f}s  ({batch_tps:.1f} tokens/sec)")
    speedup = sequential_time / batch_time
    print(f"  Speedup:            {speedup:.2f}x faster with batching")
    print("=" * 60)


def benchmark_paged(prompt="The meaning of life is", max_tokens=100, runs=3):
    enc = tiktoken.get_encoding("gpt2")
    prompt_tokens = enc.encode(prompt)

    print("\n" + "=" * 60)
    print(f"PagedAttention Benchmark: \"{prompt}\"")
    print(f"Prompt tokens: {len(prompt_tokens)} | Generate: {max_tokens} tokens | Runs: {runs}")
    print("=" * 60)

    # --- torch.cat cache ---
    print("\nLoading model (torch.cat KV-cache)...")
    model_cat = GPT2Cached()
    load_openai_weights(model_cat)

    times_cat = []
    for r in range(runs):
        tokens = list(prompt_tokens)
        torch.manual_seed(42)
        start = time.time()
        for _ in generate_streaming(model_cat, tokens, max_tokens=max_tokens):
            pass
        times_cat.append(time.time() - start)
        if r == 0:
            text_cat = enc.decode(tokens)

    avg_cat = sum(times_cat) / len(times_cat)
    tps_cat = max_tokens / avg_cat
    print(f"  torch.cat cache: {avg_cat:.2f}s  ({tps_cat:.1f} tokens/sec)")

    # --- Paged cache ---
    print("\nLoading model (PagedAttention)...")
    model_paged = GPT2Paged()
    load_openai_weights(model_paged)

    times_paged = []
    for r in range(runs):
        tokens = list(prompt_tokens)
        torch.manual_seed(42)
        start = time.time()
        for _ in generate_paged(model_paged, tokens, max_tokens=max_tokens):
            pass
        times_paged.append(time.time() - start)
        if r == 0:
            text_paged = enc.decode(tokens)

    avg_paged = sum(times_paged) / len(times_paged)
    tps_paged = max_tokens / avg_paged
    print(f"  PagedAttention:  {avg_paged:.2f}s  ({tps_paged:.1f} tokens/sec)")

    speedup = avg_cat / avg_paged
    print(f"\n  Speedup: {speedup:.2f}x")
    print(f"  Output match: {text_cat == text_paged}")
    print("=" * 60)


# ============================================================
# Prefix Caching
# ============================================================

class PrefixCache:
    def __init__(self, n_layers, n_heads, head_dim, page_size=16, max_pages=512):
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.page_size = page_size
        self.max_pages = max_pages

        self.k_pool = torch.zeros(max_pages, n_heads, page_size, head_dim)
        self.v_pool = torch.zeros(max_pages, n_heads, page_size, head_dim)
        self.free_pages = list(range(max_pages))

        self.cache = {}
        self.next_seq_id = 0
        self.sequences = {}

    def _alloc_page(self):
        return self.free_pages.pop(0)

    def lookup(self, tokens):
        key = tuple(tokens)
        best_len = 0
        best_entry = None
        for cached_key, entry in self.cache.items():
            prefix_len = min(len(cached_key), len(key))
            match = 0
            for i in range(prefix_len):
                if cached_key[i] != key[i]:
                    break
                match += 1
            if match > best_len:
                best_len = match
                best_entry = entry
        return best_len, best_entry

    def allocate_sequence(self, seq_id):
        self.sequences[seq_id] = {
            "page_tables": [[] for _ in range(self.n_layers)],
            "length": 0,
        }

    def free_sequence(self, seq_id):
        if seq_id in self.sequences:
            del self.sequences[seq_id]

    def clone_pages_from_entry(self, seq_id, entry, n_tokens):
        seq = self.sequences[seq_id]
        pages_needed = (n_tokens + self.page_size - 1) // self.page_size

        for layer_idx in range(self.n_layers):
            src_pages = entry["page_tables"][layer_idx][:pages_needed]
            new_pages = []
            for src_page in src_pages:
                dst_page = self._alloc_page()
                self.k_pool[dst_page] = self.k_pool[src_page].clone()
                self.v_pool[dst_page] = self.v_pool[src_page].clone()
                new_pages.append(dst_page)
            seq["page_tables"][layer_idx] = new_pages
        seq["length"] = n_tokens

    def append(self, seq_id, layer_idx, new_k, new_v):
        seq = self.sequences[seq_id]
        pages = seq["page_tables"][layer_idx]
        pos_in_seq = seq["length"]
        n_new = new_k.size(2)

        for i in range(n_new):
            slot = (pos_in_seq + i) % self.page_size
            if slot == 0:
                page_id = self._alloc_page()
                pages.append(page_id)
            page_id = pages[-1]
            self.k_pool[page_id, :, slot, :] = new_k[0, :, i, :]
            self.v_pool[page_id, :, slot, :] = new_v[0, :, i, :]

        if layer_idx == self.n_layers - 1:
            seq["length"] += n_new

    def get_kv(self, seq_id, layer_idx):
        seq = self.sequences[seq_id]
        pages = seq["page_tables"][layer_idx]
        length = seq["length"]

        if not pages:
            return None, None

        k_parts, v_parts = [], []
        remaining = length
        for page_id in pages:
            n = min(remaining, self.page_size)
            k_parts.append(self.k_pool[page_id, :, :n, :])
            v_parts.append(self.v_pool[page_id, :, :n, :])
            remaining -= n

        k = torch.cat(k_parts, dim=1).unsqueeze(0)
        v = torch.cat(v_parts, dim=1).unsqueeze(0)
        return k, v

    def save_prefix(self, tokens, seq_id):
        seq = self.sequences[seq_id]
        key = tuple(tokens)
        self.cache[key] = {
            "page_tables": [list(layer_pages) for layer_pages in seq["page_tables"]],
            "length": seq["length"],
        }

    def stats(self):
        used = self.max_pages - len(self.free_pages)
        return {
            "total_pages": self.max_pages,
            "used_pages": used,
            "free_pages": len(self.free_pages),
            "cached_prefixes": len(self.cache),
            "active_sequences": len(self.sequences),
        }


def generate_prefix_cached(model, tokens, prefix_cache, prefix_tokens=None,
                           max_tokens=50, temperature=0.8, top_k=40, repetition_penalty=1.2):
    seq_id = prefix_cache.next_seq_id
    prefix_cache.next_seq_id += 1
    prefix_cache.allocate_sequence(seq_id)
    model.eval()

    if prefix_tokens is None:
        prefix_tokens = tokens

    hit_len, entry = prefix_cache.lookup(prefix_tokens)

    with torch.no_grad():
        if hit_len > 0 and entry is not None:
            prefix_cache.clone_pages_from_entry(seq_id, entry, hit_len)
            remaining_tokens = tokens[hit_len:]
            start_pos = hit_len

            if remaining_tokens:
                input_ids = torch.tensor([remaining_tokens])
                positions = torch.arange(start_pos, start_pos + len(remaining_tokens)).unsqueeze(0)
                x = model.wte(input_ids) + model.wpe(positions)

                for i, block in enumerate(model.blocks):
                    cached_k, cached_v = prefix_cache.get_kv(seq_id, i)
                    x, new_k, new_v = block(x, cached_k, cached_v)
                    prefix_cache.append(seq_id, i, new_k, new_v)

                x = model.ln_f(x)
                logits = x @ model.wte.weight.T
            else:
                cached_k, _ = prefix_cache.get_kv(seq_id, 0)
                last_pos = prefix_cache.sequences[seq_id]["length"] - 1
                input_ids = torch.tensor([[tokens[-1]]])
                positions = torch.tensor([[last_pos]])
                x = model.wte(input_ids) + model.wpe(positions)

                for i, block in enumerate(model.blocks):
                    cached_k, cached_v = prefix_cache.get_kv(seq_id, i)
                    x, new_k, new_v = block(x, cached_k, cached_v)

                x = model.ln_f(x)
                logits = x @ model.wte.weight.T
        else:
            input_ids = torch.tensor([tokens])
            logits = model(input_ids, paged_cache=prefix_cache, seq_id=seq_id)
            prefix_cache.save_prefix(prefix_tokens, seq_id)

        next_token = apply_sampling(logits[0, -1, :], tokens, temperature, top_k, repetition_penalty)
        tokens.append(next_token)
        yield next_token

        for _ in range(max_tokens - 1):
            input_ids = torch.tensor([[tokens[-1]]])
            start_pos = len(tokens) - 1
            positions = torch.tensor([[start_pos]])
            x = model.wte(input_ids) + model.wpe(positions)

            for i, block in enumerate(model.blocks):
                cached_k, cached_v = prefix_cache.get_kv(seq_id, i)
                x, new_k, new_v = block(x, cached_k, cached_v)
                prefix_cache.append(seq_id, i, new_k, new_v)

            x = model.ln_f(x)
            logits = x @ model.wte.weight.T

            next_token = apply_sampling(logits[0, -1, :], tokens, temperature, top_k, repetition_penalty)
            tokens.append(next_token)
            yield next_token

    prefix_cache.free_sequence(seq_id)


def benchmark_prefix_cache(max_tokens=50, runs=5):
    enc = tiktoken.get_encoding("gpt2")

    system_prompt = "You are a helpful assistant that answers questions concisely and accurately."
    questions = [
        "What is attention in transformers?",
        "Explain gradient descent briefly.",
        "What is a GPU kernel?",
        "How does backpropagation work?",
        "What is the softmax function?",
    ]

    system_tokens = enc.encode(system_prompt)

    print("\n" + "=" * 60)
    print("Prefix Caching Benchmark")
    print(f"System prompt: {len(system_tokens)} tokens")
    print(f"Questions: {len(questions)} | Generate: {max_tokens} tokens each")
    print("=" * 60)

    model = GPT2Paged()
    load_openai_weights(model)

    # --- Without prefix caching (fresh PagedKVCache each time) ---
    print("\n  Without prefix caching:")
    torch.manual_seed(42)
    start = time.time()
    for q in questions:
        full_tokens = system_tokens + enc.encode(" " + q)
        paged = PagedKVCache(
            n_layers=model.n_layers, n_heads=model.n_heads,
            head_dim=model.head_dim, page_size=16, max_pages=256,
        )
        seq_id = 0
        paged.allocate_sequence(seq_id)
        tokens = list(full_tokens)
        for _ in generate_paged(model, tokens, max_tokens=max_tokens):
            pass
    time_no_prefix = time.time() - start
    total_tokens = len(questions) * max_tokens
    tps_no_prefix = total_tokens / time_no_prefix
    print(f"    Time: {time_no_prefix:.2f}s  ({tps_no_prefix:.1f} tokens/sec)")
    print(f"    Prefill per request: {len(system_tokens)} system + question tokens")

    # --- With prefix caching ---
    print("\n  With prefix caching:")
    pcache = PrefixCache(
        n_layers=model.n_layers, n_heads=model.n_heads,
        head_dim=model.head_dim, page_size=16, max_pages=512,
    )

    torch.manual_seed(42)
    start = time.time()
    for i, q in enumerate(questions):
        full_tokens = system_tokens + enc.encode(" " + q)
        tokens = list(full_tokens)
        for _ in generate_prefix_cached(
            model, tokens, pcache,
            prefix_tokens=system_tokens,
            max_tokens=max_tokens,
        ):
            pass
        if i == 0:
            pcache.save_prefix(system_tokens, 0)
    time_prefix = time.time() - start
    tps_prefix = total_tokens / time_prefix
    print(f"    Time: {time_prefix:.2f}s  ({tps_prefix:.1f} tokens/sec)")
    hit_len, _ = pcache.lookup(system_tokens)
    print(f"    Cache hit: {hit_len} tokens reused per request (after first)")
    print(f"    Prefill per request: only question tokens (after first)")

    stats = pcache.stats()
    print(f"    Cached prefixes: {stats['cached_prefixes']}")

    speedup = time_no_prefix / time_prefix
    saved_prefill = (len(questions) - 1) * len(system_tokens)
    print(f"\n  Speedup: {speedup:.2f}x")
    print(f"  Prefill tokens saved: {saved_prefill} (system prompt computed once)")
    print("=" * 60)


if __name__ == "__main__":
    benchmark(prompt="The meaning of life is", max_tokens=100, runs=3)
    benchmark_paged()
    benchmark_batching()
    benchmark_prefix_cache()
