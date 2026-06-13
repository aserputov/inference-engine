# Inference Engine

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> A from-scratch inference engine for GPT-2 with KV-cache, streaming generation, and performance benchmarks. Same model, same weights — but 2.6x faster through caching. No libraries for inference, every optimization written by hand.

**Part of the [Deep Learning from Scratch](https://github.com/aserputov?tab=repositories) series:**
[Word2Vec](https://github.com/aserputov/word2vec-from-scratch) → [RNN / LSTM](https://github.com/aserputov/rnn-from-scratch) → [Transformer](https://github.com/aserputov/attention-from-scratch) → [GPT-2](https://github.com/aserputov/gpt2-from-scratch) → **Inference Engine**

---

## Interactive Demo

Flask-based streaming interface with real-time performance metrics. Tokens appear one by one via Server-Sent Events, powered by KV-cache under the hood.

![Inference Engine Demo](assets/demo.png)

```bash
python3 demo.py    # downloads weights on first run, serves at http://localhost:5004
```

---

## Abstract

This project takes the GPT-2 implementation from the [previous project](https://github.com/aserputov/gpt2-from-scratch) and makes it fast. The naive approach recomputes the entire sequence for every new token — O(n²) total work. KV-cache stores the Key and Value matrices from previous steps, reducing each decode step to O(n) and achieving a **2.6x speedup** on 100-token generation. The engine also implements streaming (token-by-token output via SSE) and repetition penalty.

## The Problem: Redundant Computation

Without cache, generating token 100 means running the full model on all 100 tokens — even though tokens 1-99 haven't changed:

```
Step 1:  process [The]                          → predict "meaning"
Step 2:  process [The, meaning]                 → predict "of"       ← recomputes "The"
Step 3:  process [The, meaning, of]             → predict "life"     ← recomputes "The", "meaning"
...
Step 99: process [The, meaning, of, ..., word₉₈] → predict word₉₉   ← recomputes ALL 98 tokens
```

Total attention computations: 1 + 2 + 3 + ... + n = **O(n²)**

## The Solution: KV-Cache

In attention, we compute Q, K, V for each token. But K and V for past tokens never change (causal mask means they can't see the future). So we cache them:

```
Prefill:  process [The, meaning, of, life, is]  → cache K,V for all 5 tokens → predict "to"
Decode 1: process [to]     + cached K,V          → append to cache → predict "live"
Decode 2: process [live]   + cached K,V          → append to cache → predict "in"
Decode 3: process [in]     + cached K,V          → append to cache → predict "harmony"
```

Each decode step only processes **1 new token** and concatenates its K,V with the cache.

### Two Phases of Generation

| Phase | Input | Work | When |
|-------|-------|------|------|
| **Prefill** | Entire prompt (N tokens) | Process all N at once, build KV-cache | Once, at the start |
| **Decode** | 1 token at a time | Compute Q,K,V for new token only, reuse cached K,V | Repeated for each generated token |

This is why you see two metrics in the demo:
- **TTFT (Time to First Token)** — prefill duration
- **Decode speed** — tokens/sec during generation

## Benchmark Results

```
Prompt: "The meaning of life is" | 100 tokens | 3 runs average

  No cache:   5.32s  |  18.8 tokens/sec
  KV-cache:   2.02s  |  49.4 tokens/sec
  Speedup:    2.63x faster with KV-cache
```

The speedup grows with sequence length — longer generations benefit more from caching.

## Implementation

### Without Cache (Baseline)

```python
# Every step: feed ALL tokens, recompute everything
for _ in range(max_tokens):
    input_ids = torch.tensor([tokens[-1024:]])   # all tokens
    logits = model(input_ids)                     # full forward pass
    next_token = sample(logits[0, -1, :])
    tokens.append(next_token)
```

### With KV-Cache

```python
# Prefill: process entire prompt, get cache
logits, past_kv = model(prompt_ids)
next_token = sample(logits[0, -1, :])

# Decode: only feed NEW token, reuse cache
for _ in range(max_tokens):
    logits, past_kv = model(
        torch.tensor([[next_token]]),   # just 1 token
        past_kv=past_kv,               # reuse cached K,V
        start_pos=current_pos          # correct position embedding
    )
    next_token = sample(logits[0, -1, :])
```

### Key Code: Cached Attention

```python
def forward(self, x, kv_cache=None):
    q, k, v = self.c_attn(x).split(C, dim=2)     # compute Q,K,V for new token(s)

    if kv_cache is not None:
        prev_k, prev_v = kv_cache
        k = torch.cat([prev_k, k], dim=2)          # append new K to cached K
        v = torch.cat([prev_v, v], dim=2)          # append new V to cached V

    new_cache = (k, v)                              # save for next step
    scores = (q @ k.T) / sqrt(d)                   # attend to ALL keys (cached + new)
    return output, new_cache
```

## Additional Optimizations

| Optimization | What it does | Why |
|-------------|-------------|-----|
| **Repetition penalty** | Reduce probability of already-generated tokens | GPT-2 Small tends to loop without it |
| **Streaming (SSE)** | Send each token to client immediately | User sees output as it's generated |
| **Top-K sampling** | Only consider top K most likely tokens | Filters out low-probability noise |
| **Temperature** | Scale logits before softmax | Controls randomness vs determinism |

## Why This Matters for Production

Every LLM serving system (vLLM, TGI, TensorRT-LLM) uses KV-cache. Understanding it from scratch reveals:

- **Memory trade-off**: cache uses O(n × layers × d_model) memory per request
- **Why context length is expensive**: longer context = bigger cache = more memory
- **Why batching is hard**: each request has different cache sizes
- **PagedAttention** (vLLM): manages cache memory like virtual memory pages

## Quick Start

```bash
git clone https://github.com/aserputov/inference-engine.git
cd inference-engine
pip install -r requirements.txt

python3 engine.py    # run benchmark (no cache vs KV-cache)
python3 demo.py      # launch streaming demo at localhost:5004
```

## Project Structure

```
inference-engine/
├── engine.py         # GPT-2 with and without KV-cache, benchmarks
├── demo.py           # Flask streaming demo with SSE
├── assets/
│   └── demo.png      # Demo screenshot
├── requirements.txt
├── LICENSE
└── README.md
```

## References

1. Radford, A., et al. (2019). *Language Models are Unsupervised Multitask Learners.* OpenAI
2. Vaswani, A., et al. (2017). *Attention Is All You Need.* NeurIPS 2017
3. Pope, R., et al. (2022). *Efficiently Scaling Transformer Inference.* MLSys 2023
4. Kwon, W., et al. (2023). *Efficient Memory Management for Large Language Model Serving with PagedAttention.* SOSP 2023

## Series

| Project | Architecture | Status |
|---------|-------------|--------|
| [word2vec-from-scratch](https://github.com/aserputov/word2vec-from-scratch) | Skip-gram embeddings | Done |
| [rnn-from-scratch](https://github.com/aserputov/rnn-from-scratch) | RNN, LSTM, Bahdanau Attention | Done |
| [attention-from-scratch](https://github.com/aserputov/attention-from-scratch) | Transformer encoder-decoder | Done |
| [gpt2-from-scratch](https://github.com/aserputov/gpt2-from-scratch) | GPT-2 decoder-only, real weights | Done |
| **inference-engine** | **KV-cache, streaming, benchmarks** | **Done** |
