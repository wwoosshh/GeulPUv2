# Same bits? — the same request alone and inside a batch

*Korean version: [문서/같은비트.md](../문서/같은비트.md).*

[`도구/same_bits.py`](../도구/same_bits.py) sends one request to an OpenAI-compatible server — temperature 0, a fixed seed,
`cache_prompt: false` — first alone, then concurrently with seven other prompts, five rounds, and compares every answer with the first
one: the tokens and, when the server returns them, the log-probability of every token. It needs only Python's standard library and
works with any server that speaks `/v1/chat/completions`: GeulPU, llama.cpp, vLLM, SGLang, Ollama.

```
python 도구/same_bits.py --url http://127.0.0.1:44544/v1 --label GeulPU
python 도구/same_bits.py --url http://127.0.0.1:8080/v1 --label llama.cpp
```

Options: `--rounds 5` · `--batch 8` (the probe plus seven fillers) · `--max-tokens 160` · `--seed 42` · `--prompt "…"` ·
`--fillers FILE` (one filler prompt per line) · `--strict` (exit 1 when any answer differs). The last line is the verdict:
`<label>: alone N/N identical · in a batch of B M/N identical · first fork at token k`.

## What it showed on one PC (2026-10-30)

Setup: NVIDIA GeForce RTX 4070 Ti (12 GB) · Qwen2.5-7B-Instruct Q4_K_M (bartowski, SHA-256 `65b8fcd9…` — the file of
[prompt-cache-reproducibility.md](prompt-cache-reproducibility.md)) · the tool's default prompt (a word problem, 160 tokens) ·
temperature 0 · seed 42 · `cache_prompt: false` · 5 rounds · a batch of 8. llama.cpp b11721 (official Windows CUDA build,
`-ngl 99 -c 32768 -np 8 -fa on`, f16 KV cache). GeulPU 0.15.0's server with 8 slots and a context of 4,096 per slot, default
settings (fast mode, 8-bit KV cache with fast attention, speculative decoding on).

```
llama.cpp b11721 — http://127.0.0.1:44751/v1
probe: 'A bakery sells muffins for $2.50 and scones for $3.25. On Monday it sold'… · temperature 0 · seed 42 · max_tokens 160 · cache_prompt false
  reference      160 tokens, finish length, 1993 ms — compared with per-token log-probabilities
  answer starts: "Let's break down the problem step by step to find out how much more or less money the bakery made on"
  round 1 alone  identical (160 tokens, 1980 ms)
  round 1 batch8 DIFFERENT — same tokens, log-probabilities differ (max |Δ| 0.0626)
  round 2 alone  identical (160 tokens, 1888 ms)
  round 2 batch8 DIFFERENT — same tokens, log-probabilities differ (max |Δ| 0.0435)
  round 3 alone  identical (160 tokens, 2021 ms)
  round 3 batch8 DIFFERENT — forks at token 66: 'S' vs 'Revenue', max |Δ log-prob| 0.064 before the fork
  round 4 alone  identical (160 tokens, 2493 ms)
  round 4 batch8 DIFFERENT — same tokens, log-probabilities differ (max |Δ| 0.0539)
  round 5 alone  identical (160 tokens, 2021 ms)
  round 5 batch8 DIFFERENT — same tokens, log-probabilities differ (max |Δ| 0.0784)
llama.cpp b11721: alone 5/5 identical · in a batch of 8 0/5 identical · first fork at token 66 — the answer changes
```

```
GeulPU 0.15.0 — http://127.0.0.1:44752/v1
probe: 'A bakery sells muffins for $2.50 and scones for $3.25. On Monday it sold'… · temperature 0 · seed 42 · max_tokens 160 · cache_prompt false
  reference      160 tokens, finish length, 1761 ms — compared with per-token log-probabilities
  answer starts: "Let's break down the problem step by step to find out how much more or less money the bakery made on"
  round 1 alone  identical (160 tokens, 1437 ms)
  round 1 batch8 identical (160 tokens, 5116 ms)
  round 2 alone  identical (160 tokens, 1490 ms)
  round 2 batch8 identical (160 tokens, 5623 ms)
  round 3 alone  identical (160 tokens, 2054 ms)
  round 3 batch8 identical (160 tokens, 5198 ms)
  round 4 alone  identical (160 tokens, 1440 ms)
  round 4 batch8 identical (160 tokens, 10342 ms)
  round 5 alone  identical (160 tokens, 3915 ms)
  round 5 batch8 identical (160 tokens, 10294 ms)
GeulPU 0.15.0: alone 5/5 identical · in a batch of 8 5/5 identical — same bits every time
```

## Reading it

- **Alone, both engines repeat themselves.** With one request in flight the kernels do the same work in the same order every time.
  llama.cpp says it does not aim for batch invariance, and one slot without the prompt cache is deterministic — that is what the
  `alone` rows show.
- **In a batch, llama.cpp's numbers move.** The per-token log-probabilities changed in every batched round (largest difference
  0.04–0.08 nat), and in round 3 the answer itself forked at token 66 — `S` became `Revenue` — where the top two candidates were
  close. Which rounds fork depends on what else is in the batch, so other PCs, builds and prompts will show other counts; the
  log-probabilities moving in every round is the stable part.
- **GeulPU's tokens and log-probabilities were the same bits in all ten answers.** Its precision contract makes the order of
  summation irrelevant — products are summed exactly as integers and every remaining rounding point is fixed
  ([contract](../계약/contract.md)) — so a batch cannot change the result. The same holds for the prompt cache (restored from disk
  or recomputed), speculative decoding, and, in exact mode, the GPU vendor ([user guide — what is different](user-guide.md#what-is-different)).
- **At agent scale** the same effect was measured with the cache-divergence-study harness on 80 tool-use episodes:
  llama.cpp 64 of 80 episodes diverged between the prompt cache on and off, GeulPU 0 of 80 in every comparison
  ([prompt-cache-reproducibility.md](prompt-cache-reproducibility.md), data [doi:10.5281/zenodo.23631969](https://doi.org/10.5281/zenodo.23631969)).

## Limits

- One PC, one model, one prompt, five rounds. The tool compares a server with its own first answer; it does not compare engines with
  each other (GeulPU's and llama.cpp's answers differ from each other because their arithmetic differs).
- llama.cpp b11721 was the build on this PC on 2026-10-30, not the newest release; the tool makes re-running with the latest build a
  two-minute job.
- The tool is a script, not a page in the app: it needs the second server and a terminal.
