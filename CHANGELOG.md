# Changelog

All notable changes to GeulPU. Version numbers follow the [versioning rules](docs/versioning.md) (from 0.7.0). The Korean changelog,
with more measurement detail, is [문서/바뀐것.md](문서/바뀐것.md).

Versions up to 0.5.0 were released from the research repository [GeulPU](https://github.com/wwoosshh/GeulPU); this repository
(GeulPUv2) continues the same app from 0.5.1 on.

## Unreleased

### Fixed

- **The YaRN factor no longer depends on free graphics memory** ([#18](https://github.com/wwoosshh/GeulPUv2/issues/18)). The engine
  chose YaRN's factor from the context left after fitting the graphics memory, so the same settings could compute with no YaRN, factor 2
  or factor 4 depending on what else was using the GPU — for example Qwen3-8B with a context of 131,072 ran without YaRN at 35,328 tokens
  and with factor 2 at 59,392. The factor now follows the context you set (clamped to the model's limit): 4 in both cases. Files with
  fewer than 32,768 positions (for example Qwen2.5-Math-7B, 4,096) are no longer scaled from an assumed 32,768. The server state, debugger
  run logs and reproduction files record the factor the server reports (`yarn`) instead of recomputing it. Outputs without YaRN are
  unchanged.
- **A model file replaced at the same path no longer keeps its old SHA-256** ([#19](https://github.com/wwoosshh/GeulPUv2/issues/19)).
  The app passes the model's SHA-256 to the server as the prompt cache's seed (`--모델해시`) and kept the value in `모델목록.json` even
  after the file at that path was replaced — for example by a requantized or fine-tuned file of the same architecture — so chunks written
  for the old file could be restored as the new file's KV, and answers were wrong without a warning. The list now stores the file's size
  and modification time with the hash (`sha256_size`, `sha256_mtime`); when they differ, the hash and the verified mark are dropped at
  start or at the next load, and the SHA-256 is counted again before the prompt cache uses it. A load waits for a verification in
  progress instead of taking the old value, and the reproducibility check also counts again when the file changed. Hashes in lists
  written by 0.10.1 and earlier have no time stamp: they are shown as before and counted once more before the prompt cache uses them (the
  verified mark stays when the bytes are unchanged). Like other quick checks (rsync, make), a replacement that keeps both the size and
  the modification time is not noticed — use **Verify file integrity again** then.
- **A malformed message after a tool result no longer stops the server** ([#21](https://github.com/wwoosshh/GeulPUv2/issues/21)).
  With Qwen2.5 and Qwen3, the chat template looked at the message after a `tool` message before checking it, so an element that was
  not an object, or whose `role` was not a string, made the server read a null pointer and exit — one local request dropped every
  request in progress. The template now checks every message first, as the Llama 3 and Mistral templates already did, and answers 400
  (`/v1/chat/completions` and `/tokenize`). Valid requests render the same prompt as before.
- **DeepSeek-R1-Distill-Qwen-14B computes with its own RMSNorm eps** ([#16](https://github.com/wwoosshh/GeulPUv2/issues/16)). The
  kernels for the qwen2 architecture had eps 10⁻⁶ built in, and neither the engine nor the app read the file's value. This catalog model
  declares 10⁻⁵ (its 7B sibling and Qwen2.5 use 10⁻⁶), so it loaded without a warning and every logit differed from the model and from
  GeulPU's own CPU reference. qwen2 files with eps 10⁻⁵ now get their own module (`qwen2e5…`, generator `qwen2e5큰`) whose RMSNorm uses
  10⁻⁵, and the engine checks that the module matches the file; other eps values are refused with a message. The first load builds the
  new module once. The 14B's answers change (they now match the CPU reference bit for bit); every other model is unchanged.
- **Qwen2 files with RoPE scaling metadata are refused instead of run unscaled** ([#17](https://github.com/wwoosshh/GeulPUv2/issues/17)).
  The engine and the app dropped `rope.scaling.type` for the qwen2 architecture and ran plain RoPE up to the file's position count —
  community "128K" GGUFs (131,072 positions) then computed long contexts without the scaling the file asks for. qwen2 now follows qwen3,
  llama and the CPU reference: such a file is refused with a message. The same model's file without scaling information still reaches
  131,072 tokens (7B and up) through the engine's own YaRN.
- **Loading a SentencePiece model no longer fails at random** ([#20](https://github.com/wwoosshh/GeulPUv2/issues/20)). While loading, the
  tokenizer worker built the byte-token names (`<0x00>` … `<0xFF>`) with the formatting functions, whose one global buffer the main
  thread was using at the same time to name tensors, so the two could overwrite each other's text and the load stopped with
  "바이트 토큰 … 가 없다" or "텐서 … 없다" (Mistral-7B-Instruct-v0.3). The worker now builds those names without formatting. The console
  runner also waits for the kernel-plan worker before it prints (that worker formats the plan-memory file names), as the server did.
- **`repro_check` and the CPU reference apply the server's YaRN factor** ([#23](https://github.com/wwoosshh/GeulPUv2/issues/23)). Runs
  recorded with a context past the model's position count (YaRN on) were recomputed with unscaled RoPE and reported "different" even
  though they were right. The run log's server `yarn` and the reproduction file's new `made_on.yarn` now go to the CPU reference, score
  lines are matched on the factor too, and the reproduction file check tells you when the factor differs from the current server. With
  no `top_p` · `min_p` in a reproduction file the check now uses the server's defaults (0.95 · 0.05), as for run logs.
- **The hardware diagnosis stays inside the vocabulary** ([#24](https://github.com/wwoosshh/GeulPUv2/issues/24)). The diagnosis made
  its own token ids from 1,000 to 100,999, so with a small vocabulary (Mistral-7B-Instruct-v0.3 has 32,768) most of them were out of
  range and the embedding read past its table — a CUDA error or meaningless timings. The ids are now taken modulo the vocabulary size;
  for vocabularies of 101,000 or more (Qwen, Llama 3) they are unchanged, and so are the measurements.
- **The app and the server no longer leak memory while they run** ([#22](https://github.com/wwoosshh/GeulPUv2/issues/22)). The
  screen asks for `/api/state` every second, and writing the catalog copied twelve strings per entry without freeing them — about
  14 KB per call, over 1 GB a day. The JSON parser shared by the app and the server kept a small header per string, and when a body was
  malformed it kept the whole partly built tree (a truncated 16 MB body cost over 1 GB). Both are fixed; a new GPU-less quick check
  (`시험/앱누수.py`) repeats `/api/state` and valid and truncated bodies and fails if the app's private memory grows by more than 1 MB.
- **Three robustness fixes** ([#26](https://github.com/wwoosshh/GeulPUv2/issues/26)). A request with `cache_prompt: false` no longer
  writes its prompt's KV to the context store on disk (only reading was off; the prompt stayed in the data folder). A GGUF whose
  `rope.freq_base` is not a finite number above 1 is refused instead of hanging the load. A damaged tuning cache (plan memory) for a
  k-quant model is dropped and measured again instead of indexing past its kernel table — a bad entry ended in a divide-by-zero crash.

## 0.10.1 — Security fixes (2026-10-22)

### Security

- **Inference server: requests from web pages and from other host names are refused**
  ([GHSA-8jjh-hfp7-hg9p](https://github.com/wwoosshh/GeulPUv2/security/advisories/GHSA-8jjh-hfp7-hg9p)). The app already did this, but
  the inference server it starts (127.0.0.1, from port 44545) checked neither and sent `Access-Control-Allow-Origin: *`. It now accepts
  only `127.0.0.1` and `localhost` as `Host` (DNS rebinding), refuses every request that carries an `Origin` header, and sends no CORS
  headers. Programs (OpenAI SDK, coding agents, curl) and the app's `/v1` send no `Origin` and are not affected; outputs are unchanged.
  The quick check tests the rule without a GPU (`엔진/시험_요청지킴.gl`).
- **Crafted model files can no longer make the engine read or write out of bounds**
  ([GHSA-4v72-rvjg-8pvw](https://github.com/wwoosshh/GeulPUv2/security/advisories/GHSA-4v72-rvjg-8pvw)). The engine's GGUF reader now
  checks every length, count, nesting depth, dimension and tensor range against the file size and stops with an error instead of crashing
  (an alignment of 0, a negative or oversized length, a tensor past the end of the file …). Before laying out the weights it checks that
  every matrix, norm and bias has the shape the model settings imply, and it refuses vocabularies under 256, as the app's model check now
  does too. Non-finite logits no longer make sampling pick a token outside the vocabulary. Outputs of valid files are unchanged. The quick
  check opens 32 crafted GGUF files and samples rows with NaN and ±Inf without a GPU (`시험/망가진입력.py`).
- **Release and PyPI workflows hardened**
  ([GHSA-4cgh-2775-jg3m](https://github.com/wwoosshh/GeulPUv2/security/advisories/GHSA-4cgh-2775-jg3m)). The release workflow no longer
  writes the tag name into its PowerShell scripts, where a tag containing `$(…)` ran commands with write access to the repository: it reads
  the tag from an environment variable and accepts only `v<MAJOR>.<MINOR>.<PATCH>`, and its checkouts no longer leave the token in
  `.git/config`. The PyPI workflow downloads only this version's wheel, requires exactly one matching line in `SHA256SUMS.txt` and uploads
  only that file — before, another wheel attached to the release was uploaded unchecked. Only someone who can push tags or edit releases
  could use either.

### Project

- [SECURITY.md](SECURITY.md) lists 0.10.x as the supported version (it still said 0.7.x).

## 0.10.0 — Fast attention: faster long prompts (2026-10-22)

### Added

- **Fast attention** (setting — off by default): used only with the 8-bit KV cache on (Fast mode, CUDA, k-quant models). Attention is
  computed in a lighter form (contract F2 — [contract.md](계약/contract.md) section 2.2: one value scale per head instead of one per 32
  values, a base-2 exponential, and probabilities as 16-bit integers with one exponent per 64 keys). On an RTX 4070 Ti, Qwen2.5-7B Q4_K_M
  processes 1,024 new tokens at depth 32k and 64k 22.6 % and 27.6 % faster than with the 8-bit KV cache alone, and empty-cache prompts of
  32k and 64k 17.2 % and 23.0 % faster (Llama-3.1-8B: 24.0, 27.5, 18.6 and 23.7 %). For Qwen2.5-7B all four are faster than llama.cpp
  b11796 with an f16 or a q8_0 KV cache (0.74–0.96× its time). Generation is unchanged and speculative verify steps are 4–6 % faster.
  Quality (criterion B″) passes for Qwen2.5-7B, Llama-3.1-8B and Mistral-7B — KL ratios 0.77–1.55 against llama.cpp's f16 KV, the same
  order as the 8-bit KV cache. Qwen3-8B misses the top-1 bar (−0.51 points against −0.5), so Qwen3 models keep the 8-bit KV cache without
  fast attention. Every path matches the new CPU reference bit for bit. Prompt-cache chunks carry the contract name `|F2` and never mix
  with the other formats, and the context store refuses chunks whose setting differs from the engine's. The server state, debugger run
  logs, benchmarks and reproduction files record the setting, and the debugger's compare and the reproduction check warn when it differs.
  `도구/debugger.py load --fast-attention`; `도구/repro_check.py` checks with the F2 CPU reference. Design, criteria fixed before
  measuring, and results: [문서/빠른어텐션.md](문서/빠른어텐션.md) (Korean). With the setting off (the default), outputs are
  bit-identical to 0.9.0 for the same model file, computation mode, YaRN factor and request.

### Project

- The 글(geul) compiler is pinned to its public release 1.4.0 (tag `2g-1.4.0`), and the PTX and SPIR-V backends now live in this
  repository ([백엔드/](백엔드/)). The engine programs and all 22 kernel modules are byte-identical.
- For contributors: the tests take the tokenizer from the GGUF file they are given, and the measurement scripts read the model folder and
  llama.cpp from environment variables (`GEULPU_MODELS`, `GEULPU_LLAMA_BENCH`). The files GPU tests need are listed in
  [docs/development.md](docs/development.md#checks).
- Published releases are archived on Zenodo with a DOI; Zenodo reads [CITATION.cff](CITATION.cff) (release step 9 in
  [CONTRIBUTING.md](CONTRIBUTING.md)).

## 0.9.0 — Faster generation in long sessions (2026-10-21)

### Changed

- **Faster generation in long sessions** (Fast mode, CUDA — outputs are bit-identical to 0.8.0): the speculative-decoding verify step no
  longer reads the KV cache once per draft row — rows of the same chat now share each KV tile and the attention merges every 1,024-key
  group in registers. On an RTX 4070 Ti the 8-row verify step at 64k tokens takes 23.3 ms instead of 91.1 (Qwen2.5-7B, 16-bit KV),
  19.3 instead of 62.9 (8-bit KV) and 27.1 instead of 114.8 (Llama-3.1-8B, 8-bit KV); at 130k (8-bit KV) 26.0 instead of 111.7. It now costs
  1.2× a one-row step at every depth (it was 2.3–4.8× at 32k–64k), and the server's draft-length model was refitted to it, so drafts stay
  long in long sessions. One-row decode is 5–7 % faster at 64k and 12 % at 130k, the decode kernel no longer launches blocks for the whole
  context setting (8-row steps at short depth 3–12 % faster, most with long context settings), and long prompts are 2–3 % faster. With
  ~30k tokens of context (8-bit KV), code edits with speculation now generate 289–291 tokens/s instead of 173–184 (0.8.0); llama.cpp b11768's
  fastest speculative mode reaches 126–175 on the same requests. Long prompts are still slower than llama.cpp (1.0–1.5×). Details:
  [문서/긴문맥.md](문서/긴문맥.md) ③c (Korean).

## 0.8.0 — 8-bit KV cache: longer contexts, faster long prompts (2026-10-21)

### Added

- **8-bit KV cache** (setting — off by default): with Fast mode, CUDA and k-quant models, the KV cache is kept as 8-bit integers with
  one scale per 32 values and multiplied in that form (contract F1A′ — [contract.md](계약/contract.md) section 2.1). On an RTX 4070 Ti
  (12 GB) 1.68× the context fits on the same GPU — Qwen2.5-7B Q4_K_M goes from 87,488 to 131,072 tokens for one chat (47,808 to 80,320 for
  two), Qwen3-8B from 35,328 to 59,392 and Llama-3.1-8B from 39,936 to 66,688. Nothing gets slower, and at 64k tokens prompts are 22 %,
  generation 19 % and speculative verification 32 % faster (two tensor-core MMAs instead of four, half the KV bytes read). Quality
  (criterion B″) for all four k-quants is a KL ratio of 0.75–1.55 against llama.cpp's f16 KV — the same order as llama.cpp's q8_0 KV
  cache — and Q8_0 and Q4_0 files are blocked because they do not pass. Every path matches the CPU reference bit for bit (long context up
  to 131,072). Prompt-cache chunks are 0.554× the size and never mix with the 16-bit format. Debugger run logs, reproduction files and the
  research panel's provenance record `kv8`, and the debugger and the reproduction check warn when the setting differs.
  `도구/debugger.py load --kv8 on|off`; `도구/repro_check.py` checks with the 8-bit KV CPU reference. With the setting off (the
  default), outputs are bit-identical to 0.7.1 for the same model file, computation mode, YaRN factor and request.

### Changed

- **Guides**: every picture in the user guide, the debugger guide and the research panel guide was retaken with 0.7.1 — the new icon and
  the current screen terms (the research panel pictures still showed 0.5.0's wording). The research panel guide now tells you to switch
  to exact mode: its numbers are exact-mode numbers, and the default fast mode answers its first question differently
  (`Vincent van Gogh` instead of `Leonardo da Vinci`).

### Project

- Changes to `main` now go through pull requests with green CI and a squash merge — see [CONTRIBUTING.md](CONTRIBUTING.md).
- Prompt-cache eviction study ([issue #1](https://github.com/wwoosshh/GeulPUv2/issues/1) —
  [문서/캐시축출.md](문서/캐시축출.md), Korean): a simulator compared six policies on recorded agent workloads, and the current
  leaf-first LRU stays.
- Terminology: coined Korean words in the long-context and cache-eviction documents were replaced with standard terms, and new script
  options are English.

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
