# Prompt-cache reproducibility

*Korean version: [문서/프롬프트캐시재현성.md](../문서/프롬프트캐시재현성.md).*

Does reusing a cached prompt prefix change what the server generates? Aditi Patodiya's study *Same Request, Different Answer:
Quantization Amplifies Cache-Induced Divergence in LLM Serving* ([arXiv:2609.04748](https://arxiv.org/abs/2609.04748)) measures it
with agentic tool-use episodes. Reusing the prefix changes the floating-point accumulation order, which can change one token and,
from there, the rest of the episode: in the study's own run of llama.cpp b10434 with Qwen2.5-7B Q4_K_M, 60 of 80 episodes diverge
between the cache on and off. The harness, data and analysis code are in the study's repository,
[cache-divergence-study](https://github.com/aditi-p31/cache-divergence-study).

We ran that harness, unchanged, against GeulPU and — as a positive control — against the study's llama.cpp build, on one PC. In
GeulPU the cache changed nothing: 0 of 80 episodes diverged in every comparison, down to the log-probability of every token. The raw
data is on Zenodo: [doi:10.5281/zenodo.23631969](https://doi.org/10.5281/zenodo.23631969).

## Settings

- **Harness**: commit `bc0551d` (archived as [doi:10.5281/zenodo.23633821](https://doi.org/10.5281/zenodo.23633821)), unchanged, in
  the environment its repository pins (`uv.lock` — bfcl-eval 2026.3.23). BFCL v4
  `multi_turn_base`, episodes 0–79, temperature 0. Each configuration runs in one server process, in two passes (`main`, then
  `repeat`); each pass runs the cache arm on (`cache_prompt: true`), then off. GeulPU is driven through the harness's llama.cpp
  backend: its server takes the same per-request `cache_prompt` field. Scoring and analysis are the study's
  `harness/score_results.py` and `analysis/analyze.py:analyse_config`.
- **Model**: Qwen2.5-7B-Instruct Q4_K_M (bartowski, SHA-256 `65b8fcd9…` — the same file as `7b-q4km` in
  [file-pair.md](file-pair.md)).
- **PC**: NVIDIA GeForce RTX 4070 Ti 12 GB · driver 616.56 · Intel Core i7-14700K · Windows 11. One slot, context 16,384.

| Configuration | Engine and build | Settings |
|---|---|---|
| llama.cpp (positive control) | llama.cpp b10434 (`7e4c0a968`), the official Windows CUDA 12.4 build | the study's `server_llamacpp.sh`: `--parallel 1 -c 16384 -ngl 99 --seed 42 --slots`, f16 KV, flash attention auto |
| GeulPU 0.11.0 defaults | GeulPU 0.11.0 (`5d6d58c`) | fast mode, 8-bit KV with fast attention (F2), speculative decoding on, the SSD prompt cache (`store`) on, starting empty |
| GeulPU matched | GeulPU 0.12.0 + `cached_tokens` (`1c1074d`, #88) | fast mode, 16-bit KV (fast attention exists only on 8-bit KV, so attention follows the exact contract), speculative decoding off, SSD prompt cache off — prefixes are reused from the slot's RAM only, as in llama.cpp |
| GeulPU defaults again | the same build | 0.11.0's defaults, with a new, empty SSD prompt cache |

## Results

Episodes that diverged, out of 80, by the study's `analyse_config` (it compares token ids for llama.cpp and token strings for
GeulPU):

| Configuration | cache on ↔ off | cache on: main ↔ repeat | cache off: main ↔ repeat | `cached_tokens` mean, on / off | BFCL valid, on / off |
|---|---|---|---|---|---|
| llama.cpp b10434 | 64 | 53 | 0 | 7,001 / 0 | 8 / 5 |
| GeulPU 0.11.0 defaults | 0 | 0 | 0 | not reported by 0.11.0 | 6 / 6 |
| GeulPU matched | 0 | 0 | 0 | 6,750 / 0 | 6 / 6 |
| GeulPU defaults again | 0 | 0 | 0 | 7,036 / 0 | 6 / 6 |

Aditi Patodiya, the study's author, ran these files through the study's analysis code and obtained the same numbers
(email, 25 October 2026).

- **Positive control.** For the same llama.cpp build and model on an RTX 4090 the study reports 60 / 62 / 0
  (`analysis/revision_findings.json`, `table1/lcpp-qwen7b-q4km`). On this PC it gives 64 / 53 / 0, with the first divergence at
  request 2 (median): the workload detects the effect here, and `cached_tokens` shows that the cache-on arm reused the prefix (7,001
  tokens per request on average) and the cache-off arm did not.
- **GeulPU, token by token.** In every comparison of the three runs, every completion token has the same token id, log-probability
  and entropy — 22,520 tokens in 863 paired requests (defaults), 22,433 in 866 (matched). The defaults run again with the newer
  build equals the 0.11.0 run bit for bit.
- **The settings change the answers; the cache does not.** Matched and defaults differ in 66 of 80 episodes; GeulPU and llama.cpp,
  both with the cache off, in 68 (defaults) and 66 (matched). Within one configuration the cache path and the recompute path give
  the same bits.
- **`cached_tokens`.** GeulPU reports `usage.prompt_tokens_details.cached_tokens` from #88 on (in the next release). Cache-on requests
  after the first report the reused prefix, cache-off requests report 0. The last prompt token is always computed, so a repeated
  prompt reports `prompt_tokens − 1`.
- **One context overflow (matched).** Request 18 of `multi_turn_base_65` is 16,450 tokens, over the 16,384 context; the server answers
  400 `context_length_exceeded` and the harness counts the episode as an error — at the same request in all four cells.
- **Task success is near the floor** (6–8 of 80): the benchmark is hard for a 7B model, so these runs say nothing about the cache's
  effect on task success. The study draws its outcome conclusions from a math (GSM8K) experiment instead.

## Data

- Zenodo [doi:10.5281/zenodo.23631969](https://doi.org/10.5281/zenodo.23631969) (the DOI of all versions) — data under CC BY 4.0,
  `scripts/` under MIT. The file `geulpu-cache-divergence-runs-2026-10-24.zip`: 14,526,271 bytes, SHA-256
  `c5cfeb1db61aa59c2a6fa9510064217f3d9bd9efec0ed5f9da58ff05533dcd62`.
- One folder per configuration, with the cells `{main,repeat}/arm_{on,off}/` laid out as in the study's `results/`
  (`raw_requests.jsonl`, `model_results.json`, `scores.json`, …), plus `analysis.json` and `finish.log`; the scripts that ran on this
  PC (`scripts/`), a README and `SHA256SUMS.txt`.
- Prompts appear only as SHA-256 hashes; the BFCL questions and answer keys are not included.

## Running it yourself

This measurement does not run in the app — it needs the study's Python harness. Start a GeulPU server (one slot, context 16,384),
run `uv sync` in the harness repository, then:

```
PYTHONPATH=harness uv run python harness/run_episodes.py --model-hf-id Qwen/Qwen2.5-7B-Instruct --served-model geulpu \
  --base-url http://127.0.0.1:<port>/v1 --backend llamacpp --family qwen --arm on --n-episodes 80 --out-dir <out>/main
```

Then `--arm off`, and both again into `<out>/repeat`. Score each cell with
`harness/score_results.py <cell> --model-hf-id Qwen/Qwen2.5-7B-Instruct --category multi_turn_base` and analyse the folder with
`analyse_config` from `analysis/analyze.py`. The matched configuration's app settings (`/api/settings`) were
`{"precision": "fast", "kv8_cache": false, "fast_attention": true, "speculative": false, "slots": 1, "context": 16384, "context_store": "off"}`
(fast attention switches itself off without 8-bit KV); `scripts/run_pat2.sh` in the data shows both GeulPU configurations.

## Limits

- One PC (CUDA), one model, one workload, temperature 0. Whether other GPUs give the same bits is not tested here — see the
  [cross-check](../교차검증/) and [file-pair.md](file-pair.md).
- GeulPU ran in fast mode (its default), not in exact mode.

## Citing

The study: Aditi Patodiya, arXiv:2609.04748, DOI 10.48550/arXiv.2609.04748 (BibTeX in the study repository's README); the harness at
commit `bc0551d`, doi:10.5281/zenodo.23633821. These runs: Sunghyeun Woo, Zenodo, doi:10.5281/zenodo.23631969. GeulPU: see
[CITATION.cff](../CITATION.cff) (Zenodo concept DOI 10.5281/zenodo.23548056).
