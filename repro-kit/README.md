# GeulPU reproducibility kit (GeulPU 0.21.0)

This kit checks whether your machine gives **the same bits** as our reference PC when it runs GeulPU 0.21.0. It loads two
models under two settings, sends 12 fixed chat requests one at a time and then all at once, and compares three things with the
reference values in `expected.json`: the prompt tokens, the generated tokens and a hash of every step's logits (the full logit
row). Any difference, even in the last bit of one logit, shows up as a different hash at that step.

[한국어](README.ko.md)

| | |
|---|---|
| Reference | RTX 4070 Ti (Ada, sm_89), Windows 11, GeulPU 0.21.0 installed from PyPI |
| Models | Qwen3-8B Q8_0 and Qwen3.5-9B Q4_K_M (14.4 GB together) |
| Settings | `default` (the app's default settings — for these two models, fast computation with a 16-bit KV cache; the 8-bit KV cache and fast attention apply only to some k-quant models) and `exact` (exact computation). The two give different logits, so each is checked against its own reference |
| Requests | math (one with thinking on), logic, code, two tool calls, Korean, Spanish (sampled), a 6k-token log, a multi-turn chat (sampled) |
| Time | About 5 minutes on the reference card, including the SHA-256 check of the model files |

The app's own **reproducibility check** (Benchmark page — DeepSeek-R1-Distill-Qwen-7B Q8_0, 32 math problems) is a good first
step: it has already matched on RTX 3080 Ti, 4070 Ti and 5070 cards, an H100 MIG slice, a Radeon RX 9060 XT and Intel integrated
graphics. The kit can run it too (`--builtin-check`).

## 1. Install GeulPU 0.21.0

```bash
pip install geulpu==0.21.0        # or: pipx install geulpu==0.21.0
```

Or download `GeulPU-0.21.0-linux-x64.tar.gz` (or the Windows zip) from the
[release page](https://github.com/wwoosshh/GeulPUv2/releases/tag/v0.21.0).

- Linux x86-64 needs an NVIDIA GPU of the RTX 30 series or newer (compute capability 8.0 or higher — A100, A6000, H100, Ada cards)
  with its driver, and glibc 2.34 or later (Ubuntu 22.04 or newer).
- Windows 10 or 11 (64-bit) also runs on AMD and Intel GPUs through Vulkan, but Qwen3.5 runs only on NVIDIA (CUDA) in 0.21.0.
- A card with 12 GB of graphics memory is enough (the reference card has 12 GB).

## 2. Get the models

Put both files in one folder. The kit checks their SHA-256 through the app before it uses them.

| Model | File | Size (bytes) | SHA-256 |
|---|---|---|---|
| Qwen3-8B Q8_0 | [Qwen/Qwen3-8B-GGUF](https://huggingface.co/Qwen/Qwen3-8B-GGUF) `Qwen3-8B-Q8_0.gguf` | 8,709,518,112 | `408b955510e196121c1c375201744783b5c9a43c7956d73fc78df54c66e883d6` |
| Qwen3.5-9B Q4_K_M | [unsloth/Qwen3.5-9B-GGUF](https://huggingface.co/unsloth/Qwen3.5-9B-GGUF) `Qwen3.5-9B-Q4_K_M.gguf` | 5,680,522,464 | `03b74727a860a56338e042c4420bb3f04b2fec5734175f4cb9fa853daf52b7e8` |

```bash
mkdir -p ~/geulpu-models && cd ~/geulpu-models
curl -L -O https://huggingface.co/Qwen/Qwen3-8B-GGUF/resolve/main/Qwen3-8B-Q8_0.gguf
curl -L -O https://huggingface.co/unsloth/Qwen3.5-9B-GGUF/resolve/main/Qwen3.5-9B-Q4_K_M.gguf
```

For `--builtin-check`, the app also needs DeepSeek-R1-Distill-Qwen-7B Q8_0
([unsloth/DeepSeek-R1-Distill-Qwen-7B-GGUF](https://huggingface.co/unsloth/DeepSeek-R1-Distill-Qwen-7B-GGUF),
`DeepSeek-R1-Distill-Qwen-7B-Q8_0.gguf`, 8,098,524,896 bytes) — put it in the same folder, or download it in the app
(Models → Find models).

## 3. Run the kit

Start GeulPU in one terminal. A separate data folder keeps the kit's models and settings apart from your own use of the app:

```bash
geulpu --data ~/geulpu-kit-data --no-open
```

Then, in a second terminal, from this folder:

```bash
python3 run_kit.py --models-dir ~/geulpu-models
```

Each request prints one line, for example `math-1     alone    256 tokens    4.8 s  same`. The last line gives the total and the
verdict:

```
96/96 requests identical — IDENTICAL to the reference. Result file: result-NVIDIA-H100-...-20261105-101500.json
```

Please send us the result file. It holds the GPU, driver, OS and GeulPU version, the verdict of each request and, only for
requests that differ, their hashes — no answers or other text.

| Option | |
|---|---|
| `--model qwen3-8b-q8_0` · `--config exact` | Run only one model or one settings set (repeatable) |
| `--builtin-check` | Also run the app's reproducibility check (R1, 32 problems, about 5 minutes) |
| `--url http://127.0.0.1:44544` | Address of the running app (default) |
| `--out FILE` | Name of the result file |
| `--concurrency 8` | Requests sent at once in the batch pass |

Exit code: 0 identical, 1 a difference, 2 the kit could not finish (the message says why — a missing model file, for example).

While it runs, the kit loads models and changes the app's settings (computation, KV cache, attention, concurrent chats, context);
it puts the previous settings back at the end and deletes nothing. Do not use the same app for anything else meanwhile.

## What a difference means

- `prompt tokens differ` — the chat template or the tokenizer gave another prompt. Check that the model file's SHA-256 matches.
- `logits differ from step k` — the first `k` steps matched and step `k` did not; the rest of the answer usually differs too.
- `note: the server's computation path differs from the reference's` — the app chose another computation path for this model
  (for example no 8-bit KV cache). That alone can explain a difference; the result file records both.

A difference between the one-at-a-time pass and the batch pass on your machine would mean that batching changed the result, which
GeulPU's precision contract rules out. Please report either kind with the result file.

## For maintainers

- `make_requests.py` writes `requests.json` (the long-context request is a synthetic log made from a fixed seed, 6,076 prompt
  tokens with Qwen3's template).
- `expected.json` comes from `run_kit.py --record expected.json` on the reference PC with a fresh data folder. Recording stops if
  the batch pass differs from the one-at-a-time pass, and the reference is accepted only after a second run from another fresh data
  folder matches it.
- The kit uses the app's HTTP API of 0.21.0 (`/api/state`, `/api/models/add`, `/api/settings`, `/api/server/start`,
  `/api/bench/crosscheck`) and `/v1/chat/completions` with `"geulpu": {"logit_hash": true, "prompt_tokens": true}`. A new GeulPU
  version needs a new kit version and a new reference.
