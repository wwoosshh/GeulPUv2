# Changelog

All notable changes to GeulPU. Version numbers follow the [versioning rules](docs/versioning.md) (from 0.7.0). The Korean changelog,
with more measurement detail, is [문서/바뀐것.md](문서/바뀐것.md).

Versions up to 0.5.0 were released from the research repository [GeulPU](https://github.com/wwoosshh/GeulPU); this repository
(GeulPUv2) continues the same app from 0.5.1 on.

## 0.7.1 — New app icon · project direction in the README (2026-10-20)

### Changed

- **App icon**: the browser-tab icon and the mark at the top of the left rail now show the GeulPU logo — the 글 glyph in the green of the
  글(geul) language site (#10b981) on a dark tile. The glyph is a vector path traced from Malgun Gothic Bold, so it looks the same
  whatever fonts are installed. The rest of the screen keeps its colors.

### Project

- The README explains where the project is going ([Where GeulPU is going](README.md#where-geulpu-is-going)) instead of listing
  planned items, and [CITATION.cff](CITATION.cff) names the author and affiliation. The README pictures show the new icon.
- Release notes: lines wrapped in the changelogs are joined, because GitHub renders every newline in release notes as a line break.

## 0.7.0 — Debugger v1 · long context with YaRN · standard terminology (2026-10-20)

### Added

- **Debugger v1** — for people who build and use local models. It works on **runs** (an agent session, a test set) rather than single
  answers: the **Debugger** on the app's left rail, and the script [`도구/debugger.py`](도구/debugger.py) (Python standard library only).
  Guide with screenshots: [docs/debugger-guide.md](docs/debugger-guide.md).
  - **Record**: while recording is on, every generation request through the app's `/v1` (agents, scripts, chat) is written to a run log
    (JSONL in the data folder) — request and response with the seed, the prompt tokens and every step's logit hash; the header holds the
    model's SHA-256, computation mode, backend, GeulPU version and server build. A test set (JSONL) can be run as one run (`run-set`).
    Recording adds 0.35 % to request time.
  - **Replay**: sends the requests again with the same seeds and compares the prompt tokens, the tokens, every step's logit hash and the
    finish reason. A real Qwen Code session and a test set matched bit for bit (fast mode on CUDA; exact mode recorded on CUDA and
    replayed on Vulkan). Replay warns first when the server, model, computation mode or YaRN factor differ from the recording.
  - **Score and breakpoints**: echo-scores each answer into score lines (and checks that the logit hashes match generation), and finds
    low-margin steps, low-probability samples and the starts of tool calls.
  - **Branch**: generates again from an alternative token at some step — same seed, and `geulpu.step_offset` keeps the original random
    numbers — or from an edited request (tool result, system prompt, settings). It checks that the steps before the branch keep their
    logit hashes and that two runs give the same bits, and saves the branch in the run log.
  - **Compare**: two runs of the same requests (another model file, quantization, settings or version), request by request — same
    answer?, the first diverging step, and the log-probability difference of the same answer scored by both models. The same inputs give
    the same report, byte for byte.
  - **Export and import**: move a run log to another PC and replay it there. [`도구/repro_check.py`](도구/repro_check.py) also checks run
    logs without a GPU, on the precision contract's CPU reference.
  - API: `GET /api/debug`, `GET /api/debug/run`, `POST /api/debug/record/start`, `/api/debug/record/stop`, `/api/debug/run/delete`,
    `/api/debug/run/append`, `/api/debug/run/import`. The screen is in English and Korean.
- **Long context with YaRN**: Qwen2.5 (7B and larger) and Qwen3 take contexts beyond the model file's position count (32,768 · 40,960)
  with YaRN scaling (factor 2 or 4), up to 131,072 tokens. Qwen2.5-7B Q4_K_M on a 12 GB GPU goes from 32,768 to 87,488. The model list
  shows "YaRN" next to the context. Within the native positions the computation is byte-identical to 0.6.0; with a scaled context, exact
  mode is still bit-identical across CUDA ↔ Vulkan, and a needle-in-a-haystack test at 16k–80k found 12 of 12.

### Changed

- **Behavior change — large context settings**: if *context length per chat* is set above a Qwen model's native length (for example
  131072), 0.7.0 turns YaRN on. Static YaRN — as in vLLM and llama.cpp — changes the output for short prompts too, so answers differ from
  0.6.0, which capped the context at the native length. Keep the setting at or below 32,768 (Qwen2.5) or 40,960 (Qwen3) to get 0.6.0's
  outputs.
- **Faster request parsing**: JSON numbers were parsed in time quadratic in their count; now in one pass. Reading a `/v1/completions`
  request with 18,414 prompt tokens given as an array went from 742 ms to 17–30 ms (the path of token-array prompts, echo scoring and
  branches). The prompt-cache index is read the same way, and logit hashes are computed behind the next GPU step. Output unchanged
  (cross-check R1 32/32).
- **Standard terminology**: words coined for the screens, documents and tools were replaced with established terms (echo scoring,
  branch, breakpoint, margin, prompt cache, …); CLI and API names are English. The tool `도구/캡슐확인.py` is now
  [`도구/repro_check.py`](도구/repro_check.py) — scripts calling the old path must be updated; the old option names are still accepted.

### Fixed

- The research panel's *cache break* no longer shows a broken first character (U+FFFD) when the text before the break starts in the
  middle of a character.

### Project

- The quick check (CI) adds GPU-free tests of the debugger tool and the screen's computations (1,255 cases), a screen check (`app.js`
  syntax; every screen string has an English translation), the GPU-free prompt-cache test, and the YaRN RoPE table against the CPU
  reference.
- New: this changelog, [versioning rules](docs/versioning.md), [contributing guide](CONTRIBUTING.md), [code of conduct](CODE_OF_CONDUCT.md),
  [security policy](SECURITY.md), issue and pull request templates, [CITATION.cff](CITATION.cff). The README is now an English
  overview ([Korean](README.ko.md)); the full guides moved to [docs/user-guide.md](docs/user-guide.md) ([사용안내.md](사용안내.md)) and
  [docs/development.md](docs/development.md) ([문서/개발.md](문서/개발.md)).

## 0.6.0 — Prompt cache · slot choice · tools for coding agents · automatic chat summary (2026-10-20)

- **Prompt cache**: the KV cache of long prompts is kept on the SSD in 256-token chunks and restored instead of recomputed whenever a
  request starts the same way — after restarting the app, switching models, or another chat taking the slot. Answers, log-probabilities
  and research-panel values are bit-identical to recomputation (in exact mode, chunks written by CUDA restore on Vulkan too). After a
  restart, the first token of a coding agent's first request (Qwen Code, 15k tokens) went from 3.2 s to 0.3 s. Settings → Prompt cache
  (on for an SSD, off for an HDD by default; location, size limit, write mode, clear). The research panel shows where the prompt was
  reused from and the **cache break** — the first token where an agent's prompt changed.
- **Slot choice**: a new conversation goes to the slot it overwrites least and copies a shared prefix from another slot on the GPU.
  Alternating between two recorded agent tasks: the first tokens of 10 requests took 11.6 s instead of 44.7 s.
- **Tools for coding agents**: the OpenAI-compatible API takes `tools`, `tool_choice` (auto · none), `parallel_tool_calls`, `role: tool`
  messages and assistant `tool_calls`. Prompts with tools match the model file's official chat template token for token (Qwen2.5,
  Qwen3, Llama 3.1 · 3.2 — 62 cases). Qwen3's thinking comes as `reasoning_content`. Context overflow returns 400 with
  `code: context_length_exceeded`; `/v1/models` reports `max_model_len`.
- **Automatic chat summary**: when a chat nears the context length, the same model summarizes the earlier conversation (temperature 0,
  thinking off) and the chat continues; the summary stays as a card you can expand.
- **Fixed**: Qwen models failed to load when the context per chat was set above the model's maximum; the context is now capped to it.

## 0.5.1 — Llama 3.1 · 3.2 · Mistral · fast mode · speculative decoding · long-chat fix (2026-10-19)

- **New model family — architecture `llama`**: Llama-3.1-8B-Instruct, Llama-3.2-3B-Instruct, Mistral-7B-Instruct-v0.3 and fine-tunes with
  the same chat templates. Bit-identical to the CPU reference; mean KL against llama.cpp ≤ 4 × 10⁻⁴ per token. Llama 3 byte BPE and
  Mistral SentencePiece tokenizers, written in 글(geul), match llama.cpp's tokens on 2,000 lines.
- **Fast mode** (default): the activations entering the matrix multiplications are 8-bit (the same class as llama.cpp), under a new
  contract F1 with its own CPU reference — still deterministic. Prompts are 1.3–1.6× faster than exact mode (7B · 8B). Exact mode is the
  0.5.0 contract, byte for byte; the reproducibility check and Vulkan devices always use it.
- **Speculative decoding** (default on, no draft model): drafts from the prompt and the answer so far, verified in one step. Answers,
  log-probabilities and logit hashes are bit-identical to speculation off; editing code or rewriting documents generates 3.5–4.1×
  faster (Qwen2.5-7B Q4_K_M).
- **Faster kernels**: k-quant matrix tiles and SwiGLU (−1.5 to −1.9 % prompt time) and the long-prompt attention (−10.4 % at 8,192
  tokens), with identical bits.
- **Fixed — wrong tokens past 10,240 tokens**: with a context set above 10,240, head-size-128 models produced different tokens without
  an error once a chat grew past that length, and the server could crash near 31,000 tokens. Now bit-identical to the prompt path and the
  CPU reference up to 32,768 tokens. **Output change** — 0.5.0 and earlier have this bug.
- **Fixed**: a rare crash (0.27 %) in the kernel generator's string replacement that could make "preparing kernel modules" fail.
- **Distribution**: the PyPI package [`geulpu`](https://pypi.org/project/geulpu/) (`pipx install geulpu`); the app's new-version notice
  follows GeulPUv2's releases (or PyPI for the package).

## 0.5.0 — Research panel · English screens (2026-10-15)

- **Research panel** beside the chat: per-token probability, rank, top-10 alternatives, entropy and margin; branch from any token with
  the same seed and random numbers; compare answers, prompts or models (e.g. Q8_0 ↔ Q4_K_M); score pasted text; reproduction files
  checked bit for bit; export as JSON, CSV or Python.
- **API**: `logprobs` · `top_logprobs` in chat, `logprobs` · `echo` · `max_tokens: 0` in completions, `/tokenize` · `/detokenize` ·
  `/tokenizer_info` (vLLM's shape) and the `geulpu` extension fields (entropy, top ids, logit hash, `echo_from`, `step_offset`) —
  bit-identical on any PC or GPU (contract §5.2). lm-evaluation-harness connects as `local-completions` with identical per-item results
  across concurrency and CUDA ↔ Vulkan.
- `도구/repro_check.py` checks reproduction files without a GPU. The app's screens are available in English.

## 0.4.0 — Qwen3 · model browser (2026-10-15)

- Qwen3 (1.7B · 8B; the 14B shape checked on a synthetic model) with thinking on or off, bit-identical to the CPU reference.
- Find models: browse every GGUF repository on Hugging Face (trending, popular, likes, updated), filtered to files GeulPU can run and
  that fit this PC.

## 0.3.0 — Redesigned app (2026-10-15)

- A new design, collapsible reasoning, tables, code blocks and math, message versions, download management, model fit per GPU, and
  Hugging Face search with a header check before download.

## 0.2.0 — The app (2026-10-15)

- `GeulPU.exe`, a manager written in 글(geul), with a web UI on `127.0.0.1:44544`: model catalog and downloads with SHA-256 checks,
  chat, benchmarks (diagnosis, speed, cross-check), settings, new-version notice, and release packaging checked against the cross-check
  build hashes.

## 0.1.0 — First chat model (2026-10-10)

- Qwen2.5-0.5B-Instruct Q8_0 chat with the precision contract — bit-identical to the CPU reference.
