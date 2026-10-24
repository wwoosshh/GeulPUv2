# Changelog

All notable changes to GeulPU. Version numbers follow the [versioning rules](docs/versioning.md) (from 0.7.0). The Korean changelog,
with more measurement detail, is [문서/바뀐것.md](문서/바뀐것.md).

Versions up to 0.5.0 were released from the research repository [GeulPU](https://github.com/wwoosshh/GeulPU); this repository
(GeulPUv2) continues the same app from 0.5.1 on.

## Unreleased

### Added

- **Qwen3.5** (architecture `qwen35` — checked with the 0.8B Q8_0 and 4B Q4_K_M files). Every fourth layer is full attention with head
  size 256 (RoPE on the first 64 dimensions, q · k RMSNorm and a sigmoid output gate); the others are linear attention — a gated delta
  net: a short causal convolution, L2-normalized q and k, and a 128 × 128 state per head updated token by token. Both are written into
  the precision contract (sections 4.9 · 4.10 — the recurrence is a fixed chain of fp32 fused multiply-adds, so a prompt in one piece,
  in chunks or one token at a time gives the same bits). The engine's logits equal the CPU reference bit for bit on both files, and the
  contract's loss against the fp64 model is 1/5,400 of llama.cpp's (0.8B). The KV cache holds only the attention layers (a quarter of
  them); each slot keeps its own linear-attention state. For now: CUDA and Exact mode only, no speculative decoding or context store for
  this model, and a slot reuses its cache only when the new prompt continues everything the slot holds.
- **Qwen3.5 · 3.6 MoE** (architecture `qwen35moe` — checked with Qwen3.6-35B-A3B Q4_K_M, now in the app's model list). The router, the
  top-k choice, the expert FFNs and the shared expert are written into the precision contract (section 4.11); the engine's logits equal
  the CPU reference bit for bit on files cut from the model (4 layers · 32 experts). Its 18.1 GB of experts do not fit in a 12 GB card:
  as many as fit stay in graphics memory and the rest in pinned host memory, which the GPU reads directly over PCIe, and while the server
  is idle between requests it moves the most used experts into graphics memory. Where an expert lives does not change any bit. On an
  RTX 4070 Ti it uses about 10.5 GB of RAM and generates 64 tokens/s in Exact mode (llama.cpp b11846's fastest setting on the same PC:
  80). For now: CUDA and Exact mode only, like Qwen3.5.
- **DeepSeek-R1-Distill-Llama.** Llama-architecture models with DeepSeek-R1's chat template now load and chat. The server renders the
  DeepSeek-R1 template exactly as the GGUF's `chat_template` does under jinja2 (the BOS text, the last system message, `<｜User｜>` and
  `<｜Assistant｜>` turns with earlier answers cut after their last `</think>`, and the `<think>\n` generation prompt of the Llama-based
  GGUF); its tool-call form is not accepted. In the app these models behave like R1-Distill-Qwen — the thinking is folded, and the app
  now reads each model's `</think>` token id from its vocabulary instead of assuming R1-Distill-Qwen's.
- **HyperCLOVAX-SEED and other Llama-architecture fine-tunes with a plain ChatML template.** The server renders the basic ChatML template
  (`<|im_start|>role\ncontent<|im_end|>\n` per message, then `<|im_start|>assistant\n` — no default system message, tools or thinking)
  when the vocabulary has `<|im_start|>` and `<|im_end|>` as special tokens, and the tokenizer accepts the `dbrx` pre-tokenizer (the
  Llama 3 split without the whole-word shortcut, as in llama.cpp): on twelve texts, including the whole user guide (14,424 tokens),
  the token ids equal llama.cpp b11827's.
- **MiniCPM5** (OpenBMB, Llama architecture). Its `minicpm5` pre-tokenizer splits digit runs into groups of three first and then applies
  the Llama 3 split to each piece on its own, as llama.cpp does; its RMSNorm eps of 10⁻⁶ gets a module family of its own (`llamae6…` —
  the qwen2 kernels with a marker kernel the engine checks against the file); and the server renders its chat template (BOS, every answer
  with its thinking block, `enable_thinking` true · false · not given).
- **MiniCPM5 tool calls**, and the XML tool-call form of Qwen3.5 and Qwen3-Coder (`<function=…><parameter=…>` inside `<tool_call>`).
  The server reads MiniCPM's `<function name="…"><param name="…">` calls, and converts XML argument values to the types in the tool's
  JSON schema as vLLM's qwen3coder parser does (string, integer, number, boolean, object, array, null).
- **Nine more models in the app's model list**: Qwen3-0.6B Q8_0, Qwen3-4B Q4_K_M · Q8_0, Qwen2.5-1.5B · 3B Instruct Q8_0,
  Qwen2.5-Coder-14B Instruct Q4_K_M, DeepSeek-R1-Distill-Llama-8B, HyperCLOVAX-SEED 1.5B and MiniCPM5-2B (Q4_K_M). Each was run on this
  project's PC — five execution paths bit-identical, all-row logits equal to the CPU reference, CUDA and Vulkan chat, and tool calls
  where the model writes them correctly. Quality against llama.cpp (f16 KV): the four small Q8_0 files fall short of the fast-mode bar
  (top-1 agreement 0.8–1.1 points below llama.cpp, KL ratio within 2.5) and are listed with that recorded; exact mode is unaffected.
  Qwen2.5-3B uses the Qwen Research License (non-commercial), noted in its description.
- The model finder no longer calls head-size-128 llama models of 1.4–2.5B (HyperCLOVAX-SEED 1.5B, MiniCPM5 2B) unsupported, and the
  suggested model skips the new 0.6B.
- **Unquantized GGUF files: F16, BF16 and F32.** Files written straight from training (`convert_hf_to_gguf.py --outtype f16 | bf16 |
  f32`) now load for the supported architectures and shapes (qwen2 · qwen3 · llama, head size 128). On load each weight matrix is
  converted to the same 16-bit block format as the activations — two 8-bit digits per value and one exponent per 32 values (precision
  contract 1.3) — the only rounding is that conversion (F16 values of at least 1/8 of their block's maximum, BF16 values of at least
  1/64, are kept exactly), and each matrix product is an exact integer sum per block rounded once. The token embeddings are read as the
  file stores them. New kernels for this format (`--W16` and the module type `W16`, about 3 MB); all execution paths are bit-identical
  to the CPU reference. These files load in **Exact mode by default**, with their own setting (Settings → "Computation mode — F16 · BF16
  · F32 files", or the mode chip while such a file is loaded): Fast stays available, but on these files it drifts further from Exact
  than llama.cpp does (Qwen2.5-1.5B F16: next-token KL 5.9× llama.cpp's, top-1 agreement 97.4 %), short of the B′ bar.
- **Head sizes 64 and 96, RoPE on part of each head, narrower widths.** qwen2 and llama models with a head size of 64 or 96 now load —
  Qwen2.5-0.5B (run on this project's PC), and Llama-architecture models of the Llama-3.2-1B shape with its 32-entry `rope_freqs` (tested
  on a synthetic model) — as do
  models that rotate only the first n_rot dimensions of each head (any multiple of 8) and widths that are multiples of 32 (k-quant files
  still need 256). Inside attention each head is padded with zeros to 128, which gives the same bits as computing at the real size
  (contract, "Head size and RoPE dimensions"); the KV cache is kept at the padded size, so a head-64 model uses twice the KV cache per
  token of its real size. The number of heads must be even. On Vulkan these models use the plain module (no cooperative-matrix module),
  and they don't run yet on devices with 32 KB of shared memory.
- **Automatic model verification: one command tests a GGUF file before it joins the model list** (`재기/모델검증.py`,
  [문서/모델검증.md](문서/모델검증.md)). It reads the file and the packaged app's own model check, then runs what the app will run
  for that file: five-path bit tests for every CUDA module the app would build (exact and fast, plus the 8-bit KV and fast-attention
  modules where the app uses them), long-context and speculative-decoding bit tests, the CPU reference, quality against llama.cpp (the
  app's default mode against the B′ bar), the chat template against jinja2 and the tokenizer against llama-tokenize, and chat and tool
  calls in the packaged app on CUDA and Vulkan. It writes a pass / falls-short table and a draft model-list entry.

### Changed

- **The server runs each model file's own chat template.** `/v1/chat/completions` and `/tokenize` render the GGUF's
  `tokenizer.chat_template` with a jinja interpreter written in geul that produces the same text as transformers' jinja2 environment,
  byte for byte — on 10,944 conversations over the 228 distinct templates of the 1,000 most-downloaded GGUF repositories on Hugging
  Face. The hand-written templates (ChatML, Qwen3, Llama 3, Mistral, DeepSeek-R1, basic ChatML, MiniCPM5) are gone; for the models
  already supported the prompts are the same as before (the chat and tool conversations of twelve models all match jinja2). Fine-tunes
  with other chat templates now load — the app refuses only templates the interpreter cannot parse. Files without a chat template get
  the basic ChatML template, as in llama.cpp. Other values in `chat_template_kwargs` are passed to the template, and requests with
  tools now split the thinking of any template that uses `<think>` into `reasoning_content` (before, only Qwen3's).
- **Tokenizers follow llama.cpp's pre-tokenizer table.** Byte-level BPE models load by their `tokenizer.ggml.pre` name from a table
  generated from llama.cpp b11829 — 57 of its 66 name groups, among them Qwen3.5 / 3.8's `qwen35`, GPT-2, Falcon, StarCoder, DeepSeek,
  Command-R, Tekken and GPT-4o / Llama 4 — split by a regex engine that follows llama.cpp's rules (its hand-written fast paths, and
  std::regex's order on the category-collapsed text for the rest). On llama.cpp's own vocabulary tests (13 vocabularies, 602 cases) the
  tokens are identical, and the models already supported give the same tokens as before. Letter and number categories now come from
  llama.cpp's Unicode 15.1 table (the 622 CJK Extension I characters are letters). The app's model check uses the same table.

### Fixed

- GGUF files without tensors (vocabulary-only files) were refused as damaged.
- The app showed the module family "KI" next to the name of files mixing IQ4_XS with k-quants (model picker, chat header, status
  line); it now shows the file's quantization name, as for other k-quant files.

- **R1-Distill-Qwen's `/v1/chat/completions` used the wrong chat template.** The server rendered Qwen's ChatML, whose `<|im_start|>`
  is not in this model's vocabulary, so the markers were split into ordinary text; it now renders the DeepSeek-R1 template (the chat
  in the app was not affected — it builds the R1 prompt itself and uses `/v1/completions`). **Output change** for R1-Distill-Qwen
  requests to `/v1/chat/completions`.

## 0.11.1 — File-pair reproducibility benchmark · k-quants on 32 KB GPUs (2026-10-24)

0.11.1 adds a benchmark item and brings k-quant files to GPUs with 32 KB of shared memory. Under the [versioning rules](docs/versioning.md)
  these would make a MINOR release; they are shipped on the 0.11 line as 0.11.1 so that 0.12.0 stays the model-expansion release
  (Qwen3.5, F16 · BF16 · F32, chat templates — still in progress). Outputs of 0.11.0 are unchanged.

### Added

- **File-pair reproducibility** (Benchmark). Runs GGUF files that quantize the same model in different ways in exact mode and reports
  two things: whether every step of 32 math problems gives the same logit hash (FNV-1a 64) as the reference PC — the same check on any
  GPU — and how the files differ: the first step where their greedy answers diverge, and the per-token log-probability difference when
  the reference file's answers are scored by another file (echo scoring, the same arithmetic as `도구/debugger.py compare`). It comes
  with the files of Liu and Tsai's paper on the verifiability gap in quantized GGUF supply chains (arXiv:2610.19498): two
  Qwen2.5-1.5B-Instruct Q4_K_M files that llama.cpp b10448 made from the same f16 with floating-point contraction on and off (perplexity
  can't tell them apart), bartowski's imatrix quantization, and bartowski's Qwen2.5-7B-Instruct Q4_K_M and Q8_0, which the paper
  re-derived byte for byte. The card downloads each file and checks its SHA-256. The result folder holds every response with its
  per-step hashes, a run log per file that `도구/repro_check.py` can re-check on a CPU without a GPU, and `pair-<A>-<B>.json` with the
  per-token differences; export it as a zip. Reference results and how to reproduce them: [docs/file-pair.md](docs/file-pair.md).
- Result environments (and version.json) record the SHA-256 of the contract document and of its CPU reference
  (`contract_sha256` · `contract_reference_sha256`).

### Changed

- **k-quant files (Q4_K · Q5_K · Q6_K) run on GPUs with 32 KB of shared memory** — AMD's Windows driver, Intel integrated graphics.
  Their kernels prefetch fewer stages to fit in 32 KB; the order of every sum is unchanged, so the output bits are the same as on CUDA.
  Checked with the 32 KB limit emulated on an RTX 4070 Ti's Vulkan driver and on an Intel UHD Graphics 770; not yet on AMD hardware.
  The four 256-column tile kernels don't fit and are skipped there, so long prompts are slower on those GPUs.
  (From the 0.12.0 work, #81.)

### Fixed

- Loading a model on a 32 KB GPU sometimes stopped: the Vulkan layer wrote error text from the worker thread that prepares kernels
  while the main thread used the same formatting buffer. (From #82.)

## 0.11.0 — Faster by default: agent replay 22 % shorter (2026-10-23)

### Changed

- **The 8-bit KV cache and fast attention are on by default.** Both fast-mode settings now default to on. Where they apply is
  unchanged — CUDA and k-quant models (Qwen3 uses F2B); Q8_0 and Q4_0 files, exact mode and Vulkan keep the 16-bit KV cache. Together
  with this release's kernel work, the recorded Qwen Code agent replay (54 requests, Qwen2.5-7B Q4_K_M, RTX 4070 Ti) takes 22 % less
  time than with 0.10.2's defaults — 131.6 → 102.3 s adjusted total (prompt processing 34.0 → 24.0 s, generation 6.06 → 4.87 ms per
  token). On the same replay, llama.cpp b11827's fastest setting (draft model Qwen2.5-0.5B) takes 173.3 s, and its prompt processing
  takes 28.4 s and 28.1 s with f16 and q8_0 KV caches. In the engine benchmarks, empty-cache prompts of 32k and 64k tokens take 0.68×
  and 0.60× of 0.10.2's default, 1,024 new tokens at depths of 32k and 64k 0.60× and 0.55×, and generation at a depth of 64k 0.79×; every
  length measured is now faster than llama.cpp (0.76–0.97× on Qwen2.5-7B, 0.48–0.99× on Llama-3.1-8B). Settings saved by an earlier
  version are moved to the new defaults once, the first time 0.11.0 starts (the settings file gains `settings_version` 2); to go back,
  turn either setting off — the choice is kept from then on. **Default output change**: if you never changed these settings, answers
  differ from 0.10.2 because the computation mode changed. Under the same computation mode the output bits are unchanged — with the
  16-bit KV cache you get 0.10.2's answers, log-probabilities and hashes. Before/after tables and the release criteria:
  [문서/0.11.0.md](문서/0.11.0.md) (Korean), step 1-5.

- **Faster generation and speculative verification in long conversations** (CUDA fast modules and exact `--먼` modules). The
  table-step kernels for the full KV groups in front of the rows load the next 64-key tile while computing the current one (8-bit KV),
  the attention merge folds those groups' states directly and runs as a 128-thread kernel when such groups exist, and each step launches
  only the kernels it needs (a graph per row count and kernel set). On Qwen2.5-7B Q4_K_M an 8-row verification step is 5.8–8.3 % faster
  at depths of 16k–64k with 8-bit KV and fast attention (12.84 → 12.10 ms at 16k, 18.18 → 16.67 ms at 64k) and 4.3–6.4 % faster with
  16-bit KV; a 1-row step is 0.6–2.3 % faster. The draft-length cost model was refitted to the new kernels (it had priced an extra row
  1.7–3.4× too high). In the recorded Qwen Code agent replay, generation is 4.5 % faster per token with 8-bit KV and fast attention
  (5.10 → 4.87 ms) and 2.0 % faster with 16-bit KV. Outputs are bit-identical (every speculative-step and long-context test passes on
  eight module kinds, and the agent replay's prompt and answer tokens are identical); Vulkan modules are unchanged.
- **The first token comes sooner on long agent prompts.** The tokenizer remembers the tokens of each text piece between special tokens
  (identical bytes give identical tokens), so a request that resends the earlier messages tokenizes only the new ones: 31 ms → 4 ms per
  request on average in the agent replay (1.5 s less over its 54 requests). Token ids are identical.
- **Faster prompts in both modes** (CUDA). The prompt matrix-multiply kernels do less per 32-value block: k-quant tiles load through
  running pointers instead of 64-bit index arithmetic and interleave each column tile's MMAs with the previous tile's epilogue; in exact
  mode the row sums that Q4_K · Q5_K need for their min term are written once per block by the kernel that produces the activations
  (RMSNorm, the attention merge, the SwiGLU epilogue) instead of being recomputed with integer MMAs in every tile (11 % of the tensor
  work); exact Q8_0 tiles drop one multiply per value from the epilogue; fast Q8_0 tiles also load through running pointers. On an
  RTX 4070 Ti, prompt processing (512 · 2,048 · 8,192 tokens) is 2.4–3.5 % faster in exact mode and 1.2–1.9 % in fast mode on
  Qwen2.5-7B and Llama-3.1-8B Q4_K_M, and 4.1–5.6 % (exact) and 0.9–2.2 % (fast) faster on Qwen2.5-7B Q8_0. Outputs are bit-identical (every row of logits is byte-identical
  to the previous modules on all paths and models tested, and matches the CPU reference); Vulkan modules are unchanged.
- **Exact mode checks speculated tokens much faster in long conversations** (CUDA). Exact modules now use the same table-step kernels
  for the full KV groups in front of the rows as fast mode does (the rows of one slot read those groups once): on Qwen2.5-7B Q4_K_M a
  verification step of 8 rows takes 18.8 ms instead of 51.9 ms at a depth of 32k (1.24× a 1-row step instead of 3.42×) and 23.6 ms instead
  of 91.3 ms at 64k. The results are bit-identical. Vulkan modules are unchanged.
- **The computation mode is shown where you chat.** A chip next to the model shows **Exact** or **Fast** (with **8-bit KV** and **fast
  attention** when they are on) and switches it — the loaded model reloads in that mode. Every answer carries a tag with the mode it was
  computed in, and the research panel says which mode the answer was made in. The panel computes its values with the server that is
  loaded; when that server's mode differs from the answer's, it now says so and offers to recompute them in the answer's mode (before, it
  silently showed values from whatever mode was loaded — a greedy fast-mode answer could show tokens that were not the top choice).
  The pictures in the user guide and the research panel guide are retaken with the chip, the tags and the panel's mode line.

### Added

- **Fast attention for Qwen3 models.** The fast-attention setting now applies to Qwen3 models through F2B, a variant of the contract
  that keeps one exponent of q per block of 32 values instead of one per head: Qwen3 normalizes q per head before RoPE, which leaves the
  blocks of a head at different sizes, and with one exponent per head Qwen3-8B fell short of the quality bar's top-1 agreement (−0.65
  points against llama.cpp with an f16 KV cache). F2B passes it (KL ratio 1.50, top-1 −0.42 points; the 8-bit KV cache alone is 1.48 ·
  −0.37). On Qwen3-8B Q4_K_M it makes a 32k-token prompt 19.3 % faster and 1,024 new tokens at a depth of 32k 24.9 % faster than the
  8-bit KV cache, with generation unchanged. The modules are new (`_빠른_F2B_KV8`); F2 and every other module are byte-identical, and
  F2B's prompt-cache chunks are kept apart from F2's. Every path matches the CPU reference bit for bit.
- **Models whose query width differs from the hidden size.** The query width (heads × 128 — the width of q, of the attention output
  and of the input to the output projection) is now its own size instead of being taken from the hidden size, so Qwen3-0.6B · 4B · 32B
  load (before, the engine and the app refused them). The kernel generator, the engine (weights, buffers and the output projection in the
  prompt path, the generation graph and the table step), the app's model check and the CPU reference all follow it. On Qwen3-0.6B
  (hidden 1,024, query width 2,048) every path matches the CPU reference bit for bit in the exact and fast modes, with Q8_0 and k-quant
  weights. Modules for models that ran before are byte-identical, so their outputs do not change. Mistral NeMo · Small still need their
  tekken tokenizer.
- **IQ4_XS files** (CUDA for now). IQ4_XS (4.25 bits per weight — smaller than Q4_K_M at close quality, common in community repositories)
  joins the k-quant family: its codebook values are integers within ±127, so the contract computes it like Q8_0 with a per-32 scale
  d · (ls − 32). Files that mix IQ4_XS with Q5_K · Q6_K tensors (as llama.cpp writes them) load with a new module family (KI — the k-quant
  kernels plus IQ4_XS kernels); every IQ4_XS kernel variant matches the CPU reference bit for bit with real weights, extreme values and
  both modes, and so does every path of a whole model. On Vulkan the app says IQ4_XS is not supported yet.

### Project

- **The cross-check's pinned build hashes follow the new generator and servers.** On the reference PC the 32 problems give the same
  output as the reference with the CUDA modules, and the Vulkan quick check (cooperative-matrix 16x16x32 and base modules) matches;
  every pinned module source, PTX and SPIR-V variant is byte-identical.

## 0.10.2 — Fixes: correctness, reproducibility, stability (2026-10-22)

### Fixed

- **The YaRN factor no longer depends on free graphics memory** ([#18](https://github.com/wwoosshh/GeulPUv2/issues/18)). The engine
  chose YaRN's factor from the context left after fitting the graphics memory, so the same settings could compute with no YaRN, factor 2
  or factor 4 depending on what else was using the GPU — for example Qwen3-8B with a context of 131,072 ran without YaRN at 35,328 tokens
  and with factor 2 at 59,392. The factor now follows the context you set (clamped to the model's limit): 4 in both cases. Files with
  fewer than 32,768 positions (for example Qwen2.5-Math-7B, 4,096) are no longer scaled from an assumed 32,768. The server state, debugger
  run logs and reproduction files record the factor the server reports (`yarn`) instead of recomputing it. Outputs without YaRN are
  unchanged. **Output change** — with YaRN on, wherever 0.10.1 picked another factor for the same settings.
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
  **Output change** — DeepSeek-R1-Distill-Qwen-14B only.
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
- **Long contexts load on GPUs with a small single-allocation limit (Vulkan)** ([#25](https://github.com/wwoosshh/GeulPUv2/issues/25)).
  The KV cache was one allocation across all layers, so on AMD (2 GiB per allocation) a long context failed to load even with enough
  memory — with two slots Qwen2.5-7B past about 37k tokens, Llama 3.1 8B past 16k. When the cache would exceed the limit it is now
  allocated per layer, as the weights already were; the kernels see the same bytes. A buffer that cannot be split (one layer of the
  cache, the attention's intermediate buffers) now shrinks the context to fit the limit, with a message, instead of failing to load.

### Project

- **CI checks more of the host-side arithmetic** ([#27](https://github.com/wwoosshh/GeulPUv2/issues/27)): the quick check now compares
  the geul sampler and log-probabilities with the contract's Python reference (12,000 sampling cases including sums of exactly 1 and
  just above 1), checks the fast contract's one-digit matmuls and the RMSNorm eps constants in the contract self-test, compiles and runs
  `--help` for the tools in `도구/`, and treats `엔진/파일.gl` (included by the kernel generator) as a kernel file.
- **The guides agree with the code again** ([#28](https://github.com/wwoosshh/GeulPUv2/issues/28)): the conditions for the same bits (model
  file, computation mode, 8-bit KV cache, fast attention, YaRN factor, and in fast mode the backend) in the debugger guide and the
  versioning rules; the development guide says the server also has a Vulkan build (only the terminal runner is CUDA-only); the glossary
  points at `엔진/`; the `repro_check` instructions install numpy in a virtual environment (`python -I` does not see packages installed
  with `pip install --user`); the bug report template's version example is 0.10.1.

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
