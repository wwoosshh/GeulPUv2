# Changelog

All notable changes to GeulPU. Version numbers follow the [versioning rules](docs/versioning.md) (from 0.7.0). The Korean changelog,
with more measurement detail, is [문서/바뀐것.md](문서/바뀐것.md).

Versions up to 0.5.0 were released from the research repository [GeulPU](https://github.com/wwoosshh/GeulPU); this repository
(GeulPUv2) continues the same app from 0.5.1 on.

## 0.23.0 — Research platform: experiments · CLI for AI agents · run, step and layer screens · reproduction files (2026-11-09)

GeulPU becomes a research platform — experiment records, a CLI for people and AI agents, run, step and layer screens, and reproduction files that another PC checks bit for bit. The inference server is byte-identical to 0.22.0 (and, on CUDA, to 0.21.0), so every output bit is unchanged.

### Added

- **CLI (subcommands).** `GeulPU <command> [--param value …]` (or `geulpu <command> …`) calls any operation of the running GeulPU app.
  Commands come from the operation registry and accept the English name (`run branch`), the Korean name (`실행 분기`) or the id
  (`run.branch`); results and errors are JSON; exit codes are 0 ok · 1 operation failed · 2 usage error · 3 app not reachable.
  `GeulPU ops` lists every command with its parameters, `GeulPU help <command>` explains one, `--wait` waits for a job, and
  `--session <name>` puts the name on the app's events ([명령줄.md](문서/명령줄.md)).
- **A one-page CLI guide for agents.** `GeulPU guide` prints a one-page English guide ([docs/cli.md](docs/cli.md)): command shape
  and discovery, results, errors and jobs, loading models, the research fields of a generation, branching, scoring, the layer lens and
  experiment records. A fresh AI session that read only this guide completed a hesitation · branch · layer-lens study through the CLI.
  Also: `ops` · `help` · `guide` follow `--out`, usage lines show `--experiment`, and `experiment create` records the app's computation
  mode ([명령줄.md](문서/명령줄.md) §3b).
- **Run summary and run comparison operations, and `--select` in the CLI.** The run screen's summary values (mean log-probability,
  perplexity, the step with the smallest margin, steps with margin < δ …), the compare tab's first differing position and step, and
  the logit lens summary are now operations the app computes — `experiment summary` and `experiment compare` (read from the records,
  no model needed). The screens show these operations' values, so the CLI and agents get the same numbers. The CLI and the command
  console take `--select <path>[,<path>…]` to print only the values needed from an answer. `GEULPU_EXPERIMENT` in a shell no longer
  applies to the command that creates an experiment or to an experiment command that names another experiment
  ([사용성시험.md](문서/사용성시험.md) §3).
- **Command console.** Every screen now has a command console on the right. Each operation call — clicked in the app, or made from the
  CLI in a terminal, an AI session or the Python API — appears as one line with the CLI command that does the same thing and its status, and
  each line's command can be copied. Typing a GeulPU command (e.g. `run complete --prompt …`) in its input line sends the same request as
  the CLI. It folds (`]`) and resizes, and becomes a bottom panel on narrow screens ([화면구현.md](문서/화면구현.md) §1).
- **Branch, logit-lens and score a recorded run with one command.** `experiment branch --run r1 --step 3 --token 785`,
  `experiment lens --run r1 --step 3` and `experiment score --run r1 [--step 3]` build the request from the run's record (prompt
  tokens, answer, sampling settings, seed), send it to the engine and record the result as a run of the experiment, so request
  bodies no longer have to be built by hand. The run record still holds the full request sent and the command log holds the short
  command, which makes the same run again when sent again. The branch, logit lens, top candidates and whole-answer scoring on the
  run screen and the step and layer screen use the same operations, so the console shows the short command. The app keeps
  answering other requests while a long branch is generated ([사용성시험.md](문서/사용성시험.md) §5).
- **Experiment records from the CLI.** `GeulPU experiment create --name <name>` makes an experiment folder from the template (in
  `experiments` of the data folder, or `--root`); `experiment list` · `show` find and show experiments; `experiment check --wait` checks the
  format and hashes with the experiment checker (Python 3.10 or later). With `--experiment <folder>` on any command, the app records that
  call (operation, request body, source, status) in the experiment's `records/commands.jsonl` ([명령줄.md](문서/명령줄.md) §3).
- **Experiment records: format and checker.** One experiment is kept as one folder (`experiments/<date>-<name>/`) in the format
  `geulpu-experiment/1` — the research question, criteria written before any run, models by SHA-256 and source, runs, and the SHA-256 of
  every included file. The checker `도구/experiment_check.py` verifies format, hashes and size (at most 5 MB compressed) without a GPU
  ([experiments/README.md](experiments/README.md)).
- **Experiment screens.** The app now opens on the experiment list ("Experiments" in the sidebar). Each experiment shows its status,
  models, number of runs, criteria and number of recorded commands; new experiments can be created on screen and the experiments folder
  can be changed. An experiment opens with Plan (criteria, plan.md), Results (results.md) and Records (commands log, files, commands to
  reproduce, check) tabs; a check run on the screen and CLI calls with `--experiment` appear in the commands log at once. The CLI gains
  `experiment read`, which reads a text file inside an experiment folder ([화면구현.md](문서/화면구현.md) §2).
- **llama.cpp replay check — streamed token-less records.** Streamed responses written by the transparent recorder (the default),
  the usual form for chat UIs and agents, are now verified. The recorder additionally stores only what it gathers from the chunks the
  server sent (the text, whether reasoning or tool-call chunks appeared, finish_reason, usage, log-probabilities the client asked
  for); verification brings the record to the non-streamed form and compares the regenerated tokens' text byte for byte. Token-less
  receipts now compare the response text with the tokens' text even when the tokens are known (temperature 0 only,
  [문서/라마재현.md](문서/라마재현.md), stage 33).
- **llama.cpp replay check — sampled (temperature > 0) token-less records.** When the request carried a seed, token-less records at
  temperature > 0 (the usual setting of chat UIs and agents) — from the transparent recorder, streamed or not, and from client
  conversation logs — are now verified: the replay runs the server's sampler chain (top_k, top_p, min_p, temperature, seed) in the
  order it computes the logits, picks the server's tokens again and compares their text with the response byte for byte. Requests
  without a seed are refused with the reason ([문서/라마재현.md](문서/라마재현.md), stage 34).
- **llama.cpp replay check — responses ended by a stop string.** Responses to requests with stop strings are now verified: the
  server's stop-string handling (earliest match, holding back a partial match, trimming the log-probability entries) is emulated on
  the generated tokens' text, and the stop step and the cut text must match the response. Chats and client conversation logs with stop
  strings are accepted too ([문서/라마재현.md](문서/라마재현.md), stage 35).
- **llama.cpp replay check — repetition penalties and logit_bias.** Requests with repetition penalties (repeat, frequency and
  presence penalty, repeat_last_n), logit_bias given by token id, and ignore_eos are now verified: the emulated sampler chain accepts
  the prompt tokens and the chosen tokens like the server and applies the penalties. DRY, mirostat and the other samplers, and
  logit_bias given as text, are refused with the reason ([문서/라마재현.md](문서/라마재현.md), stage 36).
- **llama.cpp replay check — branching conversations, request/response logs.** Conversation-log import accepts two more forms:
  `requests` (the messages sent and the answer received, in order) and `exchanges` (the request body and the response as sent and
  received, a JSON array or JSONL). Branched conversations (edited messages) and interleaved conversations are emulated in the order
  the server received them; when the responses carry usage or timings, the generated-token count and the cache state are compared
  with the server's too ([문서/라마재현.md](문서/라마재현.md), stage 37).
- **llama.cpp replay kit for other GPUs.** `repro-kit/llama-replay/` holds six receipts recorded from llama.cpp b12008 (Qwen3.5-9B
  and Qwen3-8B), the reference value counts, and a script (Python standard library only) that starts a private GeulPU, verifies every
  receipt on Vulkan (or CUDA) and writes a result file — for checking the replay on GPUs we do not have, such as AMD
  ([문서/라마재현.md](문서/라마재현.md), stage 38).
- **Operation-call events (`/api/events`).** The app records a start and an end event for every call that computes or changes records
  (the registry's POST operations — generation, branch, lens, score, model load, evaluation …). Read them with
  `GET /api/events?after=<seq>&wait=<ms>`, which waits up to `wait` ms when there is no new event. Each event carries the operation id,
  the source (app screen, Python API, other), the HTTP status and the duration; existing API answers are unchanged
  ([기능목록.md](문서/기능목록.md) §7).
- **Every operation on a screen.** Every operation in the registry is now reachable from a screen — "Server info" in the model menu
  on Models › Files (model name, context length and tokenizer as the server reports them), Anthropic Messages and OpenAI Responses in
  the new-run form (with token counting — these two formats take no seed, so only the answer text is recorded), and token ids as score
  input on the step and layer screen (scores for special tokens that cannot be typed as text). The screens call the app only by
  operation id, and CI checks that screen code holds no paths and that every operation is reachable ([화면구현.md](문서/화면구현.md) §6).
- **Make and check reproduction files from the CLI too.** `experiment repro` makes a run's reproduction file (from the experiment's
  whole-answer score run — no model needed) and `check repro --json @<file>` checks one (scoring, regeneration and fingerprint bit for bit
  on this PC with the file's model loaded; when they differ, the first differing step and the settings that differ). The run screen's
  reproduction file export and Reproduce › Checks use the same two operations, so an exported file no longer misses its logit lens
  runs or stops right after scoring, and it records the model's quantization (`model.quant`) ([사용성시험.md](문서/사용성시험.md) §3).
- **Run records and reproduction files carry the device and the sampling settings actually used.** A recorded run now has `device`
  (the name and driver of the device that computed it) and `sampling` (the temperature, top_k, top_p, min_p and seed the engine used —
  the defaults 0.8 · 40 · 0.95 · 0.05 when the request leaves them out), and the experiment's `devices` gets the device. Reproduction
  files record the device they were made on and the sampling settings actually used. The new-run dialog shows the default top-p, top-k
  and min-p as numbers, the run and experiment screens show the device and the sampling values, and `system state`
  (`server.sampling_defaults`) and `help` list the defaults ([사용성시험.md](문서/사용성시험.md) §5).
- **Run screens.** The experiment screen gains Runs and Compare tabs. A run operation called with `--experiment` (or from the
  experiment screen) is recorded by the app: the request and the answer in the experiment's `records/runs/<id>.json`, and `runs` (kind,
  base run, intervention), `models` and `files` in `experiment.json`. The run screen shows the token-by-token output (bars under the
  tokens — margin, probability or entropy) and the top candidates of the selected step, and branches, re-runs or starts a new run from
  there. The Compare tab shows where two runs first differ in tokens and in logit hashes. The plan, the criteria (before the first run),
  verdicts, results.md and status are written on the screen and with `experiment update` in the CLI. The command console gains "Open AI
  runs" ([화면구현.md](문서/화면구현.md) §3).
- **Step and layer screen.** Pick a step on the run screen and press Enter (or "Step and layer screen") to look at that one step: the
  top candidates (1 to 128, each can be branched), the logit lens table (top three per layer and the sampled token's rank, with the
  first layer from which the last layer's top token stays first marked — switch to the Jacobian lens when a J file sits next to the
  model), and the score of a text from that step on. Every computation on the screen is recorded as a run of the experiment, so it is
  there again when the screen opens, and score and logit-lens runs show their tables on the run screen. When a score or lens puts other
  tokens on from a step, the app records the base run and the intervention; intervention steps use the screens' numbering (a branch
  continues its base run's numbers). The registry's "layer lens" wording is now "logit lens" ([화면구현.md](문서/화면구현.md) §4).

### Changed

- **Five sidebar groups.** The sidebar is now Experiments, Models, Reproduce, Jobs and Settings, each group with tabs — Models (files,
  evaluation, performance), Reproduce (llama.cpp, checks, check results), Settings (app, devices, prompt cache). The benchmark cards
  moved to their tabs (speed measurement to Models › Performance, hardware diagnosis to Settings › Devices, the reproducibility and
  file-pair checks to Reproduce › Checks), model evaluation to Models › Evaluation. The new Jobs screen shows the current job's
  progress, log and stop button and the earlier jobs this browser saw; a console line follows the job its call started until it ends.
- **Chat lives in the default experiment "Conversations" (대화).** Open the chat screen from "Conversations" in the experiment list.
  Every answer is recorded as a run of that experiment (the first message creates it), and an answer's "Run rN" opens its
  token-by-token output, branching and the step and layer screen. The debugger opens from the experiment's Records tab, the agents
  screen from "AI sessions" in the console ([화면구현.md](문서/화면구현.md) §5).
- **The research panel moved to the run screen.** The research panel beside the chat is gone and its features are on the new screens:
  the run screen has summary numbers (mean log-probability, perplexity, steps with margin < δ) with previous and next by δ, matching
  this run's model and computation mode, fingerprints, top candidates for every step, a cache-free re-run, scoring with another model
  and exports (token table, reproduction file, Python code). The compare tab shows the first differing step's probabilities and ranks
  and scores one answer under the other prompt, the step and layer screen scores without context and lists alternatives, and
  Reproduce › Checks checks reproduction files and shows engine and kernel hashes. Older answers without a run link become runs of
  "Conversations" through "Record in the experiment" ([화면구현.md](문서/화면구현.md) §6).
- **Experiment template language, authors and check warnings.** Experiments created in the Korean UI start from a Korean template
  (plan.md, results.md, question, criteria), and the new-experiment dialog takes the authors (CLI: `experiment create --lang ko
  --authors`). `experiment update` takes authors, devices, reproduce commands and files you added (`add_files` — the app writes their
  SHA-256 and size), and refuses fields it does not know. The experiment check warns about a plan, question, criteria or authors left
  as the template has them (it still passes) and writes what it checked into the job log. Score and lens runs no longer record a seed
  they do not use ([사용성시험.md](문서/사용성시험.md) §3).
- **Guide and screen text — what the re-check sessions found.** The agent guide now says where lens support is shown, how to see what
  the experiment check looked at (`job log`), the verdict values, what `check_seed` means and where the score request is in the
  reproduction file's 409. The new-experiment dialog says where to write the criteria and the plan, and the run screen's export says
  that the score run stays in the experiment and where to check a reproduction file you received. The Korean plan template no longer
  assumes git or the CLI ([사용성시험.md](문서/사용성시험.md) §3).

### Fixed

- **llama.cpp replay check — settings that change the computation are checked against a closed list.** Verification now checks
  the server's command line and the request fields against llama-server's b12008 option table. Only settings the replay emulates or
  that do not change the computation are accepted; settings that change it without emulation (KV cache types, speculative decoding,
  LoRA, layers or KV on the CPU, RoPE overrides, post-sampling probabilities, several completions, image input and others) and options
  not in the table are refused with the reason. Before, such settings were passed over, so a token-less record could be called "same"
  under different arithmetic. Stop strings now follow the server: empty entries are skipped, and every -r is read and unescaped
  ([문서/라마재현.md](문서/라마재현.md), stage 39).
- **Reproduction files of runs whose request had no temperature.** The file wrote such a run as temperature 0, so the check
  regenerated with greedy decoding an answer the engine had sampled at temperature 0.8 and reported a difference. The file now writes
  the temperature, top_k, top_p, min_p and seed actually used — also for runs recorded earlier ([사용성시험.md](문서/사용성시험.md) §5).
- **Clearer branches, run list, navigation and model adding.** A branch run's description shows the kept and the new tokens and where
  the original random numbers continue ("4 kept + 28 new = 32 tokens · original random numbers from step 4"). Generation runs in the
  run list show the start of their prompt, and the Results tab can check the experiment too. The sidebar's Experiments always opens the
  experiment list. Adding a GGUF file takes a .gguf file path directly and shows My models afterwards. The compare tab pairs the newest
  generation run with its base (or the first generation run), and the reproduction-file button reads "Reproduction file (JSON)"
  ([사용성시험.md](문서/사용성시험.md) §3).
- **Model and computation-mode changes in an experiment's command record.** Loading this run's model, scoring with another model and
  using this run's computation mode on the run and step screens now go into that experiment's command record, so replaying the record
  gives the same scores with the other model. The console's copied command writes text with line breaks (plans, results) as `--json`, so
  it fits on one line in the console and in a shell, and `experiment create` typed in the console no longer lands in the record of the
  experiment on screen ([사용성시험.md](문서/사용성시험.md) §1).
- **Load a model from the models screen; the compare tab's first differing answer step.** Models › Files › My models has "Load this
  model" — before, a model loaded only when a chat message was sent. Comparing two runs with different prompts no longer shows another
  step's probabilities at the prompt position, and shows the first step where the answers differ with both tokens' probabilities and
  ranks. In the CLI, `--wait` waits for the job also with `--out` and writes the finished job to that file, and a `--json` body with a
  syntax error is reported as "--json is not valid JSON" ([사용성시험.md](문서/사용성시험.md) §2).
- **Run screen — token texts of text-generation runs that ended with the end token.** When a text-generation run made with echo —
  every branch made from a chat run — stopped at the end token, the run screen showed each answer token with the text of the token
  before it, so the branch description named a line break instead of the token put in. The end token now shows as its own text
  (`<|im_end|>` …) at the end of the token-by-token output and in Δ lists, and the other-model list and the Δ title name the
  quantization (`Qwen2.5 7B Instruct · Q4_K_M`). Found while retaking the research guide's pictures.

## 0.22.0 — llama.cpp replay check · Python API · Linux Vulkan · reproducibility kit (2026-11-06)

### Added

- **Vulkan on Linux (#135).** The Linux package now carries the Vulkan build too (server-vulkan, device-vulkan, diag-vulkan,
  spirv-backend and the diagnosis kernels) — the way to run GeulPU on GPUs other than NVIDIA (AMD, Intel …) on Linux. The Linux
  Vulkan declarations import `libvulkan.so.1`; the quick check verifies that they declare the same functions as the Windows ones
  and that the Linux Vulkan programs import only `libc.so.6` and `libvulkan.so.1`. Windows binaries, modules and output bits are
  unchanged.
- **llama.cpp replay check — dense architecture (Qwen3).** Plain architectures without the gated delta net can now be checked
  too (attention in every layer, head size 128, Q/K norms, NEOX RoPE). The replay kernels now cover flash attention with head
  size 128 (the decode step and the four prompt configurations) and the fused kernel in which llama.cpp runs the Q/K norm and
  RoPE together, and the check emulates how the server splits the prompt for a dense model (no checkpoints). Ten Qwen3-8B
  Q4_K_M receipts matched bit for bit on CUDA ([문서/라마재현.md](문서/라마재현.md), stage 10).
- **llama.cpp replay check — llama architecture (Llama 3.1).** The llama architecture used by Llama and Mistral can now be
  checked too. The replay kernels cover the normal RoPE (adjacent pairs) and Llama 3.1's long-context scaling (rope_freqs),
  and the replay host follows the graph without Q/K norms. Ten Meta-Llama-3.1-8B-Instruct Q4_K_M receipts matched bit for
  bit on CUDA ([문서/라마재현.md](문서/라마재현.md), stage 11).
- **llama.cpp replay check — GQA ratios other than 4, and Q8_0.** Models whose query/KV head ratio is 2 or 3 (Qwen3-0.6B,
  Llama-3.2-3B) can now be checked too: the replay kernels follow how llama.cpp changes the number of heads it packs into a
  flash-attention tile for each GQA ratio. Ten Q8_0 receipts of each of those models, and ten Q8_0 receipts each of
  Qwen3-8B and Llama-3.1-8B, matched bit for bit on CUDA ([문서/라마재현.md](문서/라마재현.md), stage 12).
- **llama.cpp replay check — qwen2 architecture (Qwen2.5).** The qwen2 architecture used by Qwen2, Qwen2.5 and
  DeepSeek-R1-Distill-Qwen can now be checked too. The replay host follows the Q/K/V matmul biases and NEOX RoPE (no kernel
  changes). Ten Qwen2.5-7B-Instruct Q4_K_M receipts matched bit for bit on CUDA ([문서/라마재현.md](문서/라마재현.md), stage 13).
- **llama.cpp replay check — narrower widths (Qwen2.5-1.5B) and DeepSeek-R1-Distill.** Models whose RMSNorm uses 512-thread
  blocks (width 1536) can now be checked too. Ten Qwen2.5-1.5B-Instruct Q4_K_M receipts, and — with no code change — ten
  DeepSeek-R1-Distill-Qwen-7B Q8_0 receipts, matched bit for bit on CUDA ([문서/라마재현.md](문서/라마재현.md), stage 14).
- **llama.cpp replay check — SPM tokenizer (Mistral).** When the check builds each step's token bytes it now also handles SPM
  vocabularies (`▁` is a space, `<0xAB>` is that byte). Ten Mistral-7B-Instruct-v0.3 Q4_K_M receipts matched bit for bit on
  CUDA, including steps where a character spans several byte tokens, as in Korean ([문서/라마재현.md](문서/라마재현.md),
  stage 15).
- **llama.cpp replay check — conversation requests whose cache ends inside a ubatch (dense models).** When the server had
  computed a whole ubatch of an earlier request and then removed the cache past the reused prefix (the same prompt sent again,
  an edited earlier turn, another conversation on the same slot), the check now recomputes in the same order and verifies the
  request. These requests were refused before ([문서/라마재현.md](문서/라마재현.md), stage 16).
- **llama.cpp replay check — models wider than 4096 (14B class).** The check now computes the RMSNorm that llama.cpp uses
  for width 5120 (block 1024, four float4 per thread), and the flash-attention partial buffers no longer run short on
  prompt ubatches of models with many KV heads. Verified with Qwen2.5-Coder-14B-Instruct Q4_K_M
  ([문서/라마재현.md](문서/라마재현.md), stage 17).
- **llama.cpp replay check — streamed responses.** The recorder now assembles streamed responses (/completion and the chat
  API) into the receipt, and the replay check accepts them. Tokens the server never sends as a chunk (steps that end inside
  a character, and chat steps that produce no text) are chosen by the replay itself at temperature 0, then the server's
  log-probability entries are compared ([문서/라마재현.md](문서/라마재현.md), stage 18).
- **llama.cpp replay check — llama.cpp b11963 (the build Ollama v0.40.14 uses).** The check now also accepts receipts from a
  llama-server built at b11963, besides b12008. The two builds compute the same for the supported models and default settings:
  the same requests gave identical tokens and log probabilities ([문서/라마재현.md](문서/라마재현.md), stage 19).
- **llama.cpp replay check — the llama-server that Ollama launches.** Receipts from the llama-server that Ollama (v0.40.14)
  builds and launches are now accepted (it reports its build as a commit — b11963 — rather than a build number). Nine
  receipts recorded by attaching the recorder to Ollama's llama-server matched bit for bit; on this PC Ollama ran its CUDA
  kernels from PTX compiled by the driver at run time, and the values still matched the official build
  ([문서/라마재현.md](문서/라마재현.md), stage 20).
- **llama.cpp replay check — the latest llama.cpp (b12138).** From b12081 llama.cpp computes the Q8_1 correction sum of its prompt
  matrix multiplications (MMQ) from the quantized integers (sum × d) instead of the float values. The check now does the same
  for those builds, and b12138 receipts verify bit for bit ([문서/라마재현.md](문서/라마재현.md), stage 21).
- **llama.cpp replay check — chat streams above temperature 0.** Receipts of the common chat-UI setting (streaming, temperature
  0.7–1) are now accepted. The end token, which a chat stream never sends, is the last token and is not fed back, so every
  step before it is replayed with known tokens and the last step takes the token the sampler chain picks with the same seed (it
  must be a control token) ([문서/라마재현.md](문서/라마재현.md), stage 22).
- **llama.cpp replay check — default servers whose slots are shared by several conversations.** A default llama-server saves
  idle slots to its RAM cache and clears them whenever it starts a new task, so while one request is computed the KV cache holds
  only that slot's cells. Receipts from servers where other slots were used are now accepted: requests computed without the
  cache, and follow-ups with no other slot's request in between. Follow-ups continued from a state loaded from the RAM cache are
  still refused with a reason ([문서/라마재현.md](문서/라마재현.md), stage 23).
- **llama.cpp replay check — requests loaded from the RAM cache into another slot.** On default servers, requests continued
  from a RAM prompt-cache load after another conversation (refused with a reason up to stage 24) are now replayed. The check
  emulates the server's KV cell allocation (free-cell search, removal, state restore), so the replay uses the server's cell
  layout and attention length ([문서/라마재현.md](문서/라마재현.md), stage 25).
- **llama.cpp replay check — servers that keep idle slots.** Receipts from servers with several slots and a unified KV cache
  that run with `--no-cache-idle-slots` or without a RAM prompt cache are now replayed even when other slots were used (they
  were refused before). The check emulates the server's slot selection (LCP ratio, empty slot, least recently used slot) and
  the KV cells of every slot, so requests computed while other slots' cells stayed in the cache are recomputed with the
  server's cell layout ([문서/라마재현.md](문서/라마재현.md), stage 26).
- **llama.cpp replay check — servers with a KV cache per slot (-np N).** Receipts from servers started with `-np N` (one KV
  stream per slot, unified KV off) are now replayed (they were refused before). The check emulates the KV cells of each
  stream and the server's habit of only saving idle slots to the RAM prompt cache when a task starts
  ([문서/라마재현.md](문서/라마재현.md), stage 27).
- **llama.cpp replay check — overlapping requests (continuous batching) from the server log.** Receipts of requests that the
  server computed together with other requests in one ubatch are now replayed bit for bit by the ubatch batches written in the
  server log, when that server was started with the environment variable `LLAMA_BATCH_DEBUG=1` and `-v --log-file <file>`
  (they were refused with a reason before). Without a server log they are still refused, and the reason says how to record one
  ([문서/라마재현.md](문서/라마재현.md), stage 29).
- **llama.cpp replay check — token-less records (request and response text only).** Records that keep only what the server
  returns by default (token-less receipts, `geulpu-llama-receipt/2`) can now be verified. The prompt tokens are rebuilt at
  verification time by llama.cpp itself (a temporary CPU-only llama-server of the same build), and the replay regenerates the
  generated tokens and checks that their text is byte-identical to the response (temperature 0). The original run is not
  touched. Records of one server are verified in order from the first ([문서/라마재현.md](문서/라마재현.md), stage 30). The docs
  now state correctly that the recorder's added `n_probs` changes the server's sampler chain even at temperature 0.
- **llama.cpp replay check — without the recorder, from conversation logs.** A JSON file with the conversation the client sent
  directly to llama-server (messages with answers, and the sampling values it sent) can be imported ("Import conversation log" ·
  `POST /api/llama/import`); each assistant message becomes a token-less receipt and is verified. This is post-hoc verification
  of a run that nothing touched — the generated token count is unknown, so the replay regenerates tokens matching the answer text
  and checks that it ended with an end token or at max_tokens (temperature 0 only, [문서/라마재현.md](문서/라마재현.md), stage 32).
- **llama.cpp replay check.** Send requests for an unmodified llama-server through the app's recorder (`/llama/completion`)
  and it keeps a receipt; 'llama.cpp replay check' on the Benchmark page recomputes the request from the receipt alone on
  this PC's GPU (CUDA or Vulkan) and checks that the tokens and log probabilities the server returned are bit-identical.
  The recomputed run opens step by step in the debugger (top candidates, log probabilities, entropy) without a GeulPU
  server. Supported runs: llama.cpp b12008, RTX 4070 Ti, Qwen3.5, one slot, prompt cache off, temperature 0
  ([문서/라마재현.md](문서/라마재현.md)).
- **llama.cpp replay check — temperature > 0.** Receipts of requests with temperature > 0 are accepted when the request
  gave a seed. The check computes llama-server's sampling chain (top_k, top_p, min_p, temperature, then sampling from the
  distribution with the same random numbers) and compares the token the server chose at every step. Non-default chains
  (repetition penalties, mirostat, grammar and so on) and requests without a seed are refused with the reason
  ([문서/라마재현.md](문서/라마재현.md), stage 5).
- **llama.cpp replay check — Qwen3.5-4B.** Other sizes of the same architecture (qwen35) can now be checked. The check
  accepts models without a separate output layer (tied embeddings) and Q6_K token embeddings, and follows llama.cpp's work
  split where it depends on the matrix shape (the matrix-vector loop and the RMSNorm block size). Ten Qwen3.5-4B Q4_K_M
  receipts matched bit for bit on CUDA, and three short ones on Vulkan (RTX 4070 Ti)
  ([문서/라마재현.md](문서/라마재현.md), stage 6).
- **llama.cpp replay check — default server settings and the prompt cache.** Receipts from a default llama-server (no
  `-np`: 4 slots, unified KV cache, prompt cache on) are now accepted. When a conversation continues and the server reuses
  its cache, the check rebuilds the slot's state from the earlier receipts of the same slot. It emulates the server's
  cache logic (common prefix, checkpoint restore, prompt batching). The recorder no longer turns the prompt cache off and
  also records the slot state. Servers where another slot was used, and requests whose earlier receipts are missing, are
  refused with the reason ([문서/라마재현.md](문서/라마재현.md), stage 7).
- **llama.cpp replay check — Qwen3.5-0.8B.** Sizes with different head counts can now be checked too (8/2 attention heads,
  16 GDN value heads). The replay kernels for flash attention and the gated delta net take the head counts from the model
  file, and RMSNorm for width 1024 and Q8_0 token embeddings were added. Ten Qwen3.5-0.8B Q8_0 receipts matched bit for bit
  on CUDA, and three short ones on Vulkan (RTX 4070 Ti) ([문서/라마재현.md](문서/라마재현.md), stage 8).
- **llama.cpp replay check — chat API.** The recorder now also forwards llama-server's chat API (`/v1/chat/completions`) and
  writes receipts for it, and the check accepts them. It emulates how the server splits the prompt and creates checkpoints
  at message positions (the start of each user message), and the case where the slot is saved to and looked up in the RAM
  prompt cache. Streamed responses are not accepted yet ([문서/라마재현.md](문서/라마재현.md), stage 9).
- **Python API.** `import geulpu` starts a GeulPU app without a window (`geulpu.start`) or connects to a running one
  (`geulpu.connect`), loads a model, generates with every step's logit hash (`chat`, `complete`), sends a request again (`rerun`),
  scores tokens (`score`), branches at a token (`branch`) and compares two runs bit for bit (`geulpu.compare`); `info()` returns the
  version, GPU, driver, model SHA-256 and computation path. Standard library only, through the app's HTTP API — the engine and its
  output bits are unchanged ([docs/python-api.md](docs/python-api.md)).
- **Reproducibility kit (`repro-kit/`).** One command checks whether another machine running GeulPU 0.21.0 gives the same bits as
  the reference PC: it loads Qwen3-8B Q4_K_M, Qwen3-8B Q8_0 and Qwen3.5-9B Q4_K_M with the default settings (for Q4_K_M including the
  8-bit KV cache and fast attention) and the exact settings, sends 12 fixed requests one at a time and all at once, and compares the
  prompt tokens, the generated tokens and every step's logit hash with the reference values (RTX 4070 Ti). It can also run the app's
  reproducibility check and writes a result file to share. Standard-library Python only ([README.md](repro-kit/README.md)).

### Changed

- **llama.cpp replay check — the recorder is transparent by default.** The recorder now forwards requests to llama-server byte for
  byte, never queries the server during a run (no `/slots` · `/props` · `/tokenize`, and it does not read the model file), and
  writes token-less receipts (`geulpu-llama-receipt/2`). With the llama-server executable in the settings, the build is read with
  `--version` on save (a separate process). The previous recorder (adding probability and token fields to requests and querying
  slot state — receipts with tokens) remains available as the "probing" mode, which states what it changes
  ([문서/라마재현.md](문서/라마재현.md), stage 31).

### Fixed

- **llama.cpp replay check — server-level chains for default servers.** On default servers whose slots are shared by several
  conversations, the chain is now built from all slots' receipts of the same server, emulating the server's slot selection and
  RAM prompt cache (save and load). Requests continued from a RAM-cache load after another conversation still cannot be
  replayed (their KV cell layout differs from the original computation) and are now refused with that reason instead of being
  reported as different ([문서/라마재현.md](문서/라마재현.md), stage 24).
- **llama.cpp replay check — overlapping requests are refused with a reason.** Receipts of requests that the server computed
  together with another request (continuous batching) are now refused with a reason, including the request that finished
  later, which was reported as "different" before. The recorder also stores the slot state before forwarding a request
  (`slots_before`), so overlaps with requests that bypassed the recorder are caught as well
  ([문서/라마재현.md](문서/라마재현.md), stage 28).
- **Vulkan multiply-add rounds once.** The SPIR-V backend now decorates fused multiply-add (GLSL Fma) with NoContraction.
  Without it a driver may round the product and the sum separately, so Vulkan output bits could differ from CUDA on GPUs other
  than NVIDIA (seen on an Intel UHD 770). Output bits on NVIDIA's Vulkan are unchanged. Every module's SPIR-V changes.

## 0.21.0 — Model evaluation: fixes from the fourth researcher review · accuracy interval · original weight type (2026-11-05)

0.21.0 fixes what the fourth outside-researcher review on 0.20.0 (#240 · #241) and an external review found: Make GGUF reads the original weights' type from the stored tensors rather than config.json, Compare workspaces shows every version of a type with its name, and the 95 % accuracy interval no longer collapses to zero width when the number of graded conversations is a power of two. The workspace list (`GET /api/eval`) gains each result's app version (`app`) and, for workspaces whose config.json disagrees with the tensors, `original_config_dtype`; the CSV gains an `app` column at the end. The engine did not change: the build hashes and the output bits of existing requests are the same as 0.20.0.

### Fixed

- **The 95 % accuracy interval no longer collapses when the number of conversations is a power of two.** With 16 · 32 · 64 ·
  128 · 256 · 512 · 1024 graded conversations, the bootstrap 95 % interval was [accuracy, accuracy] — the resampled conversation
  numbers came from the low bits of a linear congruential generator (x % n), whose period equals n, so every resample was a
  permutation of the original conversations. Numbers now come from the high bits of two steps (64 conversations at accuracy 0.5:
  [0.39, 0.63] — matching the width of the normal approximation). Intervals for other numbers of conversations change in the last
  digits because the draw changed (evaluating the same result again gives the same interval). The quick check now checks the
  interval's properties (contains the point estimate · width of the normal approximation) besides its bits against the reference,
  and recomputes exp · log · KL · tail · percentile with standard functions, a different sum and a direct rank (external review —
  a check against a reference alone misses errors both share).
- **Compare workspaces shows every version of a type.** When a workspace had several versions of one type (Q4_K_M made
  without an imatrix and with two texts, say), the comparison cell showed only the last-evaluated one without saying which,
  and the paired test could not pick the others. Each cell now shows the version and its note, every version of the row is in
  the cell, and versions the app made with an imatrix get a row per imatrix text (for example "Q4_K_M · imatrix
  ko_imatrix.txt"). The paired test can pick every version with a result for that data and setting.
- **Setting names, grouping and old BPB for text results.** The name of a setting that groups several results shows only the
  measured number of chunks ("201 chunks" — the requested number can differ per result). Text results from 0.19.0 (whose
  settings recorded the requested number of chunks) are grouped by the measured number too, together with new results of the
  same setting. BPB from text results made before 0.20.0 carries a * and the reason (bytes counted the old way — evaluating
  again corrects it), and the reference-model and before-fine-tuning rows use the latest result. Each result in the workspace
  list (`GET /api/eval`) has the app version that made it (`app`), and the CSV gains an `app` column at the end.
- **The original weights' type now comes from the stored tensors, not config.json.** Make GGUF's default conversion type and
  its note, the workspace's `original_dtype` and the rounding notice of Compare with the original model took the original
  model's type from config.json's `torch_dtype` · `dtype` alone, so a folder whose weights are float32 but whose config says
  bfloat16 (a training loop that saves its float32 master weights, or DeepSpeed `zero_to_fp32.py` weights next to the config
  from training) was converted to BF16 and called "the original values exactly". The type now comes from the safetensors
  header (the weights are not read) — the type holding the most values among tensors of two or more dimensions (like llama.cpp's
  `--outtype auto`, the tensors rather than the config) — and config.json only when the header cannot be read. When the two
  differ, the notes on the card and the comparison also give config.json's value, and the workspace list (`GET /api/eval`) and
  the comparison result keep `original_config_dtype`. Comparisons already made keep the type they recorded (comparing again
  updates it).

## 0.20.0 — Model evaluation: fixes from the third researcher review · text results across workspaces · sign test (2026-11-03)

0.20.0 fixes what the third outside-researcher review on 0.19.0 (#231–#233) and an external review (#236) found: comparing with the original model again keeps the larger result, original models saved in float32 convert to F16, text results' PPL · BPB line up across workspaces, in the result table and in the CSV, and the sign test's p no longer drops to 0 for large samples. The inference server's `/detokenize` gains `n_bytes` (the server build hashes change); existing requests give the same output bits.

### Fixed

- **Original models saved in float32 convert to F16.** Converting a model whose weights are saved in float32 (common for full
  fine-tuning) with Make GGUF's default conversion type BF16 made the reference model differ from the original by KL 3.6 × 10⁻⁴
  from BF16 rounding alone (F16: 2.0 × 10⁻⁶), so Q8_0 · Q6_K comparisons were measured against the rounded model rather than
  the trained one, and the screen did not point it out (#233). The default conversion type now follows the original weights'
  type (torch_dtype · dtype in config.json): bfloat16 → BF16, float32 · float16 → F16 (the same choice as llama.cpp's
  `--outtype auto`). The Make GGUF card shows the original weights' type and why, and the API picks the same way when `outtype`
  is left out. Compare with the original model notes that a float32 · float16 original compared with a BF16 reference can reach
  a KL in the 10⁻⁴ range from rounding alone (the result keeps `original_dtype`).
- **Compare with the original model — comparing again keeps the larger result.** Comparing the same result again with
  "Conversations to compare" overwrote the full comparison, so 24 minutes of GSM8K (1,319 conversations) turned into the values
  of the first 200 (#231). Each number of conversations and compute type (float32 · bfloat16) now keeps its own result; the card
  shows the one that compared the most and lists the others as "Other comparison" lines. `GET /api/eval/original` adds `file`
  and `runs` (a summary of every stored comparison) to the result.
- **The sign test's p no longer drops to 0 for large samples.** With 1,075 or more conversations right on one side only, the
  sign test returned p = 0 (the first term 2⁻ⁿ vanished in double precision), so even a near-even 600 / 599 counted as a clear
  difference (#236 — external review); from 1,023 to 1,074 it lost precision. Terms are now summed in log space and stay within
  10⁻¹⁰ relative error of the exact value up to 40,000. The result's accuracy p and the better / worse verdict of comparing two
  results use it — the verdict is computed when comparing, so it is right at once; a p already written in a result is right
  after evaluating again. The quick check now also compares the sign test with exact values (including n 1,000 – 40,000).
- **Text results' PPL · BPB compared across workspaces and in the result table.** There was no place to see which model —
  continued pretraining, merges, vocabulary extension — has the lowest PPL · BPB, so results had to be opened one by one (#232).
  For text data, each cell of Compare workspaces now shows the evaluated model's PPL · BPB, the reference-model row shows that
  model's PPL · BPB, and a before-fine-tuning row appears when there is a baseline model (accuracy · PPL · BPB); the workspace's
  result table gains PPL · BPB columns and the reference / before-fine-tuning rows. The CSV adds `chunks` · `nll_ref` · `nll` ·
  `ppl_ref` · `ppl` · `bpb_ref` · `bpb` · `baseline_ppl` · `baseline_bpb`. The result screen's before-fine-tuning cell shows BPB
  too.
- **Text results record the chunks actually measured.** Asking for more chunks than the text has (4096 for a 201-chunk text) kept
  the requested number in the result's settings, so the same measurement split into different settings and did not line up in the
  result table or Compare workspaces. `settings.chunks` is now the measured number, `chunks_requested` keeps a larger request,
  and the screen shows "201 chunks (requested 4096)".
- **Exact bytes for BPB.** Characters cut at chunk boundaries were counted as replacement characters (3 bytes), so the BPB
  denominator was 0.06–0.09 % too large. The inference server's `/detokenize` now also returns `n_bytes`, the byte count before
  cleaning, and evaluation divides by it (ko-heldout, 64 chunks: 49,172 → 49,126 bytes — the same as an independent count with the
  tokenizer).

## 0.19.0 — Model evaluation: fixes from the second researcher review · text PPL · paired test (2026-11-03)

0.19.0 fixes what the second outside-researcher review found (#213–#220 — full fine-tuning · DPO · continued pretraining · distillation · merging · structural edits · from-scratch training · tool-call fine-tuning): evaluating a model the engine does not support no longer closes the app, the conversion check stops flagging correct SentencePiece and transformers 5 files, text mode gains absolute PPL and bits per byte, Compare with the original model shows its progress, and Compare workspaces gains a paired test. On PCs without the NVIDIA driver, the app no longer shows nvcuda.dll error boxes at startup. The engine's refusal messages and the length rule for scoring-only requests changed; existing requests give the same output bits.

### Added

- **Compare with the original model — progress and first n.** While comparing, the job step shows "conversation k/n · about
  … left" (before, only "comparing"), and the number of conversations to compare can be set (the first n — 0 means all). A
  GSM8K-size result takes tens of minutes on the CPU in full, so the first few hundred can check the conversion first (#219).
- **Model evaluation — absolute PPL, bits per byte, and a baseline model for text.** Results now carry the mean NLL and PPL on
  the same tokens for both the reference model and the evaluated model (before, only the PPL ratio), and text-data results add
  bits per byte of the text (BPB — comparable across models with different vocabularies). The baseline model (before
  fine-tuning) also works on text data, so the PPL change from continued pretraining shows directly on the same tokens (#217).

### Fixed

- **Conversion check — SentencePiece tokenizers and the transformers 5 BOS.** Whether the original model adds BOS is now also
  read from the post_processor in `tokenizer.json` when `tokenizer_config.json` does not say. Models saved with transformers 5
  were flagged with a BOS problem, and the suggested fixed version turned BOS off (#214). When it cannot be told, the item is
  "check" and no fix is suggested. A SentencePiece GGUF is compared by vocabulary instead of merges, and the ▁ space mark of
  user-defined tokens and the UNKNOWN type of `<unk>` count as the same. Special tokens with IDs outside the model's vocabulary
  (image tokens and the like) are counted apart (false problems with a self-made SentencePiece tokenizer and with Gemma 3).
- **Why an evaluation request failed is kept.** When the inference server refuses a request (for example, conversations with
  `tools` rejected with 400 because GeulPU does not read the chat template's tool-call format), that line of the run log
  (run-*.jsonl) keeps the body the server returned as `error`, and the run log and summary show the first reason — before,
  there was only `"response": null` (#218). A version that gets no answer at all no longer produces a result; it reports why.
- **Evaluating a model the engine does not support no longer crashes the app.** Evaluating a GGUF the engine does not
  support (a size outside the support table, a file with RoPE scaling, …) as the reference model or a version closed the
  whole app with an access violation (#213). The evaluation request is now refused with the reason, such versions cannot be
  picked in Run evaluation, and an unsupported reference model shows why. An unsupported baseline model only skips the
  learned / forgotten counts; the evaluation goes on.
- **Small model-evaluation fixes** (#220):
  - The conversion check's fixed-version file is written to the workspace's made folder. Before, it went to the data
    folder's 모델\ (a non-ASCII path) that llama.cpp's tools could not open.
  - Each row of the improvement report (`GET /api/eval/improve`) now has the KL reduction against the base version, the size
    difference, the reduction per MiB and the accuracy difference.
  - Text data can be used for improvement measurements.
  - Size and quality also points to a version at most 5 % larger with a KL of 0.7× or less (such as a standard Q5_K_M a
    little larger than the picked-groups file).
  - Compare workspaces shows each workspace's reference-model row (accuracy), and adds a paired test of two results
    (across workspaces too, reference models included): only A right · only B right · sign-test p.
  - "Set up tools first" no longer flashes while the tool state is loading.
  - Meaningless model names set by conversion (such as 0.17.0's "Merged") show the file name instead.
- **Why the engine refuses a GGUF is shown right away.** When Make GGUF produces files the engine does not support, the job
  summary now names those types and the reason (before, only a run-log line did). For small models whose hidden size or FFN
  width is not a multiple of 256, Make GGUF warns before quantizing that llama-quantize turns k-quants into fallback types
  (Q5_0 · Q5_1 · IQ4_NL) and suggests Q8_0 · Q4_0. Q4_0 joins the quantization types (token embeddings and output in Q4_0
  too — the engine's Q4_0 path). The engine's messages name formats instead of numbers, and "unsupported model size" now
  states the one condition that failed (SmolLM2-360M: the head count (15) is odd — before, it read "only head size 64 · 96 ·
  128") (#215).
- **Text mode measures models whose maximum position is 512.** Model evaluation's text mode failed on models whose context
  is 512 positions (small from-scratch models like TinyStories ones) with "Could not get the reference model's
  log-probabilities" (#216). The inference server now takes a prompt that fills the whole context when the request only
  scores it (echo with max_tokens 0) — the same as llama-perplexity -c 512. Models with fewer than 512 positions get the
  reason, and a failed scoring step shows why the server refused.
- **No error boxes at startup on PCs without the NVIDIA driver.** On Windows PCs without the NVIDIA driver (Intel or AMD
  GPUs), starting the app showed "The code execution cannot proceed because nvcuda.dll was not found" (server-cuda.exe ·
  device-cuda.exe) two times more than the number of models, and the app came up only after every box was closed (#229). The
  app starts these two programs to try CUDA first; without nvcuda.dll they cannot start, and Windows showed a box each time.
  Now the app skips the CUDA programs when nvcuda.dll is missing and goes straight to Vulkan, and a program the app starts
  that cannot load returns an error code without a box.

## 0.18.0 — Model evaluation: transformers 5 models · learned and forgotten · result table · guide (2026-11-02)

0.18.0 fixes and widens 0.17.0's Model evaluation (#196–#202): conversion and comparison for models saved with transformers 5, what a fine-tune learned and forgot against the model before fine-tuning, a result table and workspace comparison, accuracy in Improve, and a Model evaluation guide. Existing requests give the same output bits.

### Added

- **Model evaluation guide.** One document now covers a full round of Model evaluation (workspace → Set up tools → Make GGUF →
  conversion check → evaluation → Compare with the original model → Improve → before/after → Result table), the data format, the
  grading rules (number · choice · exact · tool_call — with examples), what the metrics mean and their usual sizes, what a fine-tune
  learned and forgot against the model before fine-tuning, the result table and workspace comparison, the files in a result folder and
  the API — [docs/model-evaluation-guide.md](docs/model-evaluation-guide.md) · [모델평가안내.md](모델평가안내.md) (#201).
- **Against the model before fine-tuning — learned and forgotten.** Choose a baseline model (before fine-tuning) for a Model evaluation
  workspace, and every result on graded data shows its accuracy and how many conversations the reference and evaluated models learned
  (wrong before, right now) and forgot (right before, wrong now). The baseline's answers are made once, like reference values, and
  reused for the same model, data and settings — no more adding the original model as a version for each fine-tune and reading the
  reference and evaluated columns backwards (#199).
- **Accuracy and size versus quality in Improve.** With graded data, the improvement report and table give each trial's accuracy (in %p
  against the base version, with the 95 % interval) and the 'reference only correct' count, and the preselection can go by KL or by
  accuracy. Layer thirds are preselected too. The workspace's versions measured with the same data and settings are listed side by side
  by size and quality (KL · accuracy), and a version is flagged when a smaller one is as good or better — for example a 424 MB Q5_K_M
  next to a 447 MB file made with chosen groups (#197).
- **Result table and workspace comparison.** A Model evaluation workspace now has a Result table — choose one data/settings
  combination to see each version's size · KL · top-1 · answers that differ · accuracy (95 % interval) and the reference model's
  accuracy in one table, per-category accuracy (graded data) or KL with 'By category', and a size–quality chart (KL or accuracy) with its
  frontier (versions with no smaller version that is as good or better). Results export to CSV (with per-category rows). With several
  workspaces, 'Compare workspaces' lines up results with the same data and settings type by type — which fine-tune loses less at
  Q4_K_M (#198).

### Fixed

- **max_tokens on data lines.** In Model evaluation, a data line's max_tokens takes precedence over the run setting, but the result row
  and the reproduction info showed only the run setting (#201). The result row now reads like 'answers 512 tokens (600 from the data
  lines)' and the reproduction info shows how many conversations used the data line's value. An issue number left in GeulPU --help is
  gone too.
- **Names and type labels of made GGUFs.** Fine-tune folders are all called 'merged', so every file made with Make GGUF showed up as
  "Merged" on the Models screen and the file names (merged-Q4_K_M-custom.gguf) did not say what changed. Files with per-matrix types
  also had the wrong type label — a Q4_K_M file with only layers 0–8 at Q6_K showed up as "Q6_K" (#200). The type label now comes from
  the type llama-quantize wrote (general.file_type — Q4_K_M, Q5_K_M and so on), and made files take their file name and model name
  (general.name) from the workspace name. With per-matrix types the file name carries the changed matrices
  (…-Q4_K_M+blk0-8.Q6_K.gguf) and the Models screen shows them next to the type. Make GGUF and Add version have a note field, and
  'Make with the chosen groups' puts the chosen groups in the note.
- **Small Model evaluation fixes** (#202). The reference model field of the New workspace dialog no longer preselects the first 16-bit
  file in the list — 'None — convert with Make GGUF' is the default (with another model or another fine-tune as the reference, every KL
  measures the wrong thing). The follow-up evaluation of 'Make with the chosen groups' no longer measures the base version again when
  a result with the same file · data · settings exists (results are bit-identical — evaluations you start yourself still always run).
  The number of imatrix chunks (512 tokens each) is kept in the run log and the new versions. The workspace header says 'Reference
  not set yet' instead of repeating itself, and the quantization type and evaluated version checkboxes have names (accessible name ·
  value). A run handed from Model evaluation to the debugger says why its scores are made separately.
- **Models saved with transformers 5.** Make GGUF could not convert a fine-tune trained and saved with today's transformers 5.x (the
  conversion tools' transformers 4.57.6 cannot read 5.x's tokenizer_config.json), and Compare with the original model silently gave a
  wrong KL because it did not know config.json's rope_parameters (#196). 'Set up tools' now also makes a transformers 5.22.1
  environment (about 30 MB — torch is shared), and the app picks the environment from the transformers version that saved the folder.
  Compare with the original model first checks the hyperparameters (layers · heads · RoPE base · RMS eps …) in the GGUF, in config.json
  and as read by transformers; it stops without a KL when transformers read them differently, and flags a KL above 10⁻³ for a 16-bit
  reference model. The conversion check has a new 'Settings' item (config.json ↔ GGUF hyperparameters), a failed conversion or
  comparison shows the last line of its run log, and a folder without weight files is reported before converting (llama.cpp wrote a
  tokenizer-only GGUF).

## 0.17.0 — Model evaluation: make, check and improve GGUF files (2026-11-02)

A new **Model evaluation** page turns GeulPU into a test bench for GGUF files: make a GGUF from a Hugging Face folder, check the
conversion, compare versions with a reference model and grade their answers, and measure which matrix groups to keep at a higher
precision — with every result reproducible bit for bit. Output bits of existing requests are unchanged.

### Added

- **Server log-probabilities — top 128 and chosen tokens.** A request's `geulpu.top_logprobs` (0–128) returns more than 20 top
  log-probabilities in the `geulpu` object, and `geulpu.token_ids_logprob` (a list of token ids for each scored position) on a
  `/v1/completions` echo returns the log-probabilities of those tokens at each position. They are the same contract 5.2 values,
  bit-identical to the top-k and score values. The model evaluation's distribution comparison (KL divergence) uses them.
- **Model evaluation backend (API).** The app gains workspaces (the GGUF versions of one model plus a reference model), evaluation jobs,
  results and before/after comparison (`/api/eval` …). Both models echo-score the reference model's greedy answers, giving KL divergence
  (the reference's top 64 tokens plus the remaining probability), top-1 agreement, Δp and the PPL ratio; each model also answers with its
  own chat template, giving the conversations whose answers changed, the first divergent step and the reference margin at that step.
  Text files are scored at the same positions as llama-perplexity (the second half of each 512-token chunk). Built-in data: 'basic 60'
  (prompts written for GeulPU, six categories).
- **Model evaluation screen.** A new 'Model evaluation' entry in the left menu. Create a workspace (a reference model plus the GGUF
  versions to compare), run an evaluation from the screen, then read the result (a one-sentence verdict, KL divergence, top-1 agreement,
  the distribution of first divergent steps and the share that diverged at a low-margin step, answers that ended with the end token,
  categories, every conversation), a before/after comparison of two results (conversations that now match the reference and those that
  newly diverged — sign test), and where one conversation's answers differ (both answers side by side, the probabilities both models gave
  the candidate tokens at the first divergent step, and the steps around it). A result's run log opens in the debugger.
- **GGUF metadata fix.** From a version's menu in Model evaluation, create a new GGUF with a changed chat template, special tokens
  (BOS · EOS · EOT · PAD) or add_bos/add_eos and add it as a new version. The tensors are copied byte for byte and only the metadata
  (keys) change — the original file stays as it is. The new version records how it was made, its parent version and the changed keys,
  and evaluating it again with the same data and settings compares it with its parent.
- **GGUF conversion check.** In a Model evaluation workspace, set the original model folder (Hugging Face format — tokenizer_config.json ·
  tokenizer.json · generation_config.json; the weights are not read) and check a version's GGUF. The chat template (five to seven
  conversation cases rendered with the same jinja interpreter as the server and compared byte for byte — differences shown side by side),
  EOS, BOS, special tokens, and the vocabulary and merges are each marked pass, problem or to check, and problems that the metadata can
  fix become a new version with 'Create fixed version'.
- **Compare with the original model.** A Model evaluation result can feed the reference model's (GGUF) answers to the original model
  (Hugging Face format — transformers, CPU) token by token and compare every position (KL divergence · top-1 agreement · answer-token
  log-probability difference). Near 0 means the conversion and the engine reproduce the original model. It uses the Python
  environment made by Make GGUF's tool setup.
- **Accuracy evaluation (grading).** When a data line in Model evaluation has `answer` and `grader` (number · choice · exact · tool_call),
  both the reference and the evaluated model's answers are graded: results show accuracy (95% interval — bootstrap) and the conversations
  where only one model is correct (sign test), and before/after comparisons show the conversations that went wrong → correct and
  correct → wrong. Each conversation shows the expected answer, both verdicts and the tool calls. Built-in graded data 'tasks40' (questions
  written for GeulPU — math · multiple choice · one word · tool calls).
- **Public tasks (accuracy evaluation).** Model evaluation's built-in data now includes the GSM8K test set (1,319 problems — 'gsm8k') and
  500 MMLU-Pro test questions chosen evenly by category ('mmlu-pro-500'), both MIT (notices in THIRD_PARTY_NOTICES.md, which is now in the
  packages). Choosing a data set shows its source, conversation count and license. The conversion rules and tool are in
  문서/모델평가.md and 도구/공개과제.py.
- **Make GGUF.** A Model evaluation workspace can now convert the original model (a Hugging Face folder) to GGUF (BF16 · F16 · Q8_0) and
  quantize it (Q8_0 · Q6_K · Q5_K_M · Q4_K_M · IQ4_XS, an imatrix made from a text file or an existing one, per-matrix types). The
  converted file becomes the reference model (when there is none) and the quantized files become versions that keep their settings
  and tool versions; the conversion check runs by itself. 'Quantize the reference model' remakes files without converting again. The
  tools are downloaded once with 'Set up tools' — llama.cpp b12032 executables and conversion scripts (SHA-256 checked), plus the
  packages (torch CPU · transformers) in an app-only virtual environment made with this PC's Python. A version shows why when the
  engine does not support its mix of types.
- **Improvement measurement.** In a Model evaluation workspace, taking a quantized version as the base, each matrix group (attn_q · attn_k ·
  attn_v · attn_output · ffn_gate+ffn_up · ffn_down · token_embd · output) and each third of the layers gets a trial file with only that
  group at a higher type (Q6_K · Q5_K · BF16), evaluated on the same data; a report shows how much each group lowers the KL divergence and
  how much the file grows (trial files are deleted after measuring). Remaking with the chosen groups evaluates the new version together
  with the base version.

## 0.16.0 — Concurrent Qwen3.5 requests · more GGUF files · one-line install (2026-11-02)

### Added

- **English aliases for command-line options, and `--help`** ([#137](https://github.com/wwoosshh/GeulPUv2/issues/137)). The server
  (`--host` · `--port` · `--slots` · `--ctx` · `--chunk` · `--name` · `--log-dir` · `--no-spec` · `--store` · `--store-size` ·
  `--model-hash` · `--contract exact|fast` · `--store-write all|evict|off` · `--tool-errors` · `--model-check` · `--header-only` ·
  `--archs` and others), the kernel generator (architecture names `qwen2` · `qwen3` · `llama` · `qwen35` · `gemma4` …, `--fast` ·
  `--exact` · `--shared32` · `--coop` · `--head` and others), the terminal runner and GeulPU (`--port` · `--data` · `--no-open`) take
  English names as aliases of the Korean ones. The Korean names stay, and either name gives byte-identical results. `--help` lists
  both. For terminals that cannot show Korean (SSH, JupyterHub), `build-linux.sh` clones the geul repository, builds the compiler as
  `geulc` and builds `ptx-backend` · `kernel-gen` · `server-cuda` · `runner-cuda` under ASCII names.
- **`GeulPU --hf <repo>/<file>.gguf` — download one GGUF from Hugging Face and load it (one-line install).** For example
  `GeulPU --hf Qwen/Qwen3-0.6B-GGUF/Qwen3-0.6B-Q8_0.gguf` (a pasted huggingface.co file URL works too). It takes the same path as the
  app's finder → check → download → load: it asks for the size and SHA-256, downloads the head and checks it with the GGUF header and the
  engine, adds the file to the list, downloads it, verifies the SHA-256 and loads the model. Progress is printed to the console, and the
  app shows the same download and job. If GeulPU is already running, that instance does the work (`POST /api/hf/pull`). This is the
  one-liner for listing GeulPU in Hugging Face's "Use this model" menu (huggingface.js local-apps). After `pipx install geulpu`:
  `geulpu --hf …`.
- **Downloads in the Linux app.** The Linux app now downloads models itself — the screen's download, the Hugging Face finder's check,
  and `GeulPU --hf <repo>/<file>.gguf`. It works as on Windows: the file goes to `.part`, a pause resumes where it stopped (Range), a
  dropped connection is retried up to five times, and the SHA-256 is verified before the model is added. HTTPS is done by the system's
  curl (proxies from environment variables such as `https_proxy`); without curl the app says to install it.
- **The same request alone and inside a batch — `도구/same_bits.py`.** Sends the same request (temperature 0, fixed seed) to any
  OpenAI-compatible server — GeulPU, llama.cpp, vLLM, SGLang, Ollama — first alone, then concurrently with seven other prompts, five
  rounds, and checks whether the tokens and per-token log-probabilities match the first answer. Standard library only. On this PC
  (RTX 4070 Ti, Qwen2.5-7B Q4_K_M) llama.cpp b11721 matched itself 5/5 alone but changed its log-probabilities in 5/5 batched rounds
  and forked the answer once at token 66; GeulPU 0.15.0 gave the same bits in all ten answers
  ([docs/same-bits.md](https://github.com/wwoosshh/GeulPUv2/blob/main/docs/same-bits.md)).
- **Version information in the executable, and a privacy policy.** The Windows `GeulPU.exe` now carries a product name, version,
  description and license (VS_VERSIONINFO), shown in Explorer's Properties (`앱/버전정보.py`, at packaging). The repository gains a privacy
  policy (`PRIVACY.md` — everything the app sends and stores). The engine binaries in `bin/` are unchanged (the cross-check compares their
  build hashes).
- **Files whose q · k · v formats differ.** The attention q, k and v matrices may each use a different k-quant-path format (Q4_K ·
  Q5_K · Q6_K · IQ4_XS · F16 · BF16 · F32), for every architecture. Runs of equal formats form one segment, so output bits are
  unchanged. This runs files such as bartowski's Qwen3.5-9B Q4_K_M, whose attention layers quantize q and k differently. The kernel
  variant memory format changed, so the first load of a model re-measures its kernel variants once.
- **IQ4_XS matrices in Qwen3.5 / Qwen3.6 files (CUDA).** Qwen3.5-family files (qwen35 · qwen35moe) may now hold IQ4_XS matrices,
  per tensor, through the KI module like the other architectures (CUDA only for now). This runs llama.cpp's IQ4_XS files and k-quant
  files with IQ4_XS mixed in, such as unsloth's UD builds (for example Qwen3.5-9B UD-Q4_K_XL). Files whose gate and up (or q · k · v)
  formats differ within one matmul, and files holding IQ4_NL · Q3_K · IQ3 · IQ2, are still refused.
- **16-bit matrices in Qwen3.5 / Qwen3.6 files (F16 · BF16 · F32).** Qwen3.5-family files (qwen35 · qwen35moe) may now hold 16-bit
  matrices, each converted to 16-bit block weights as in contract 1.3. This runs k-quant files whose output weight is BF16 (such as
  DavidAU's NEO · MAX builds), files whose β · α (`ssm_beta` · `ssm_alpha`) are F16 or F32 (such as unsloth's Qwen3.6-35B-A3B UD and
  Qwen3.6-27B), and 16-bit-only files (F16 · BF16). The app's model finder takes the formats each architecture accepts from the
  engine.

### Changed

- **The Qwen3.5 family reuses more of the prompt cache.** The inference engine keeps a snapshot of the GDN state just before the
  last prompt row, so a request that diverges before the end of the slot (a next turn that renders the previous answer differently,
  or the same prompt sent again) resumes there instead of recomputing from token 0. Output bits are unchanged. On an RTX 4070 Ti,
  resending a 3,000-token Qwen3.5-4B prompt cuts the first token from 351 to 9 ms, and a next request that diverges inside the
  previous answer from 360 to 29 ms. Each slot uses one more GDN state of GPU memory (about 53 MB for 4B and 9B, 157 MB for 27B).
- **The Qwen3.5 family computes concurrent requests together.** When several requests are generating, the inference engine runs
  their decode step at once: matrix multiplications gather the rows of all requests and read the weights once, and attention and
  the GDN state also run as one launch over all requests, each row on its own slot's state. Before, requests were computed one
  after another, so total throughput did not grow with concurrency. Output bits are unchanged (the same as a request sent alone).
  On an RTX 4070 Ti, eight concurrent Qwen3.5-4B Q4_K_M requests generate 615 tokens/s in total instead of 117 (two: 220,
  four: 377).
- **Faster k-quant decoding (CUDA).** The one-row GEMV kernels (`정수줄하나_*`) issue fewer instructions per stage (output bits
  unchanged). On an H100 NVL MIG instance the decode step (depth 1,024) is 30% shorter for Llama-3.3-70B Q4_K_M and 27% for
  Llama-3.1-8B Q4_K_M; on an RTX 4070 Ti it is about 2% shorter.
- **Faster Qwen3.5 decoding.** The decode step's k-quant matmuls now run on the one-row GEMV kernels (output bits unchanged). On an
  H100 NVL MIG instance the Qwen3.5-27B Q4_K_M decode step is 33–35% shorter (9B: 30–31%), and on an RTX 4070 Ti the Qwen3.5-9B
  step is 2.5–2.8% shorter.

### Fixed

- **Fixed false "Runs" labels in Find models.** Find models labelled repositories "Runs" from Hugging Face metadata alone, and in the
  trending top 50, 14 of the 23 so labelled could not be downloaded at all (no file in a supported format, the Gemma 4 E2B · E4B ·
  26B-A4B variants, or split files only). It now also reads the repository's file names (supported formats, split parts, unsloth's
  UD files) and the Gemma 4 variant (parameter count): what the rules know is "Should run", what they cannot know is "Needs a check"
  (picking a file fetches its start and asks the engine before downloading), and what cannot be loaded says why. The default list
  ("only those that run") no longer asks Hugging Face for the tags qwen2 · qwen3 · llama · mistral — Hugging Face's tags rarely
  carry the architecture, so Qwen3.5 · Qwen3.6 · Gemma 4 repositories never appeared in it — but filters each repository by its
  architecture and files, and shows "Needs a check" repositories with that mark. The quick check runs the labelling rules on the
  200 trending repositories of the 2026-10-30 survey (시험/찾기표시.py — the old rules gave 31 false "Runs", the new rules 0).
- **The app accepts file formats as the engine does.** The app's model check leaves weight formats (mixes of Q8_0, k-quants and
  16-bit) to the engine's model check, so files the engine loads but the app's stricter rules refused — for example ggml-org's
  Qwen3.8-27B Q4_K_M (Q8_0 matrices inside a k-quant file) — are accepted. The engine's model check now refuses, with the reason,
  files that used to pass the check and stop during loading (layers whose q · k or gate · up formats differ, F32 β · α in Qwen3.5
  MoE files, and so on).
- **The first Qwen3.5 and Gemma 4 requests no longer wait for kernel plan measurement.** The inference engine now picks the
  plans of every matrix multiplication the layers use when it loads the model, and keeps them in the plan memory. Before, the
  plans of the Qwen3.5 GDN projections (Q8_0 files), β · α (k-quant files), Q8_0 matrices inside k-quant files, the MoE router and
  the Gemma 4 full-attention layers (Q8_0 files) were measured on first use of each row bucket (70–100 ms each), again after every
  server start. Output bits are unchanged. The plan memory format changed (고르기 32-3), so plans are measured once more on the
  first start.

## 0.15.0 — Unbroken tool calls · wider tool-error detection (2026-10-29)

### Added

- **Tool-error correction now covers other clients.** Tool results from clients outside the known agents (Qwen Code · Codex · OpenCode ·
  Kilo · Cline) — custom agents, LangChain, the OpenAI Agents SDK, MCP clients — are detected when they start with a standard error text
  (Python exceptions, HTTP status, MCP, API error JSON) and classified as timeouts, wrong arguments, unknown tools or missing permissions.
  The `is_error` field on tool messages and MCP `isError` are read too. Known agents behave as before.

### Fixed

- **Qwen tool calls with doubled braces are parsed.** When a GGUF chat template shows the tool-call example as `{{"name": …}}`
  (the Qwen2.5-7B-Instruct Q8_0 GGUF), the model copies the doubled braces and the reply came back as plain text instead of a tool call.
  The extra braces are now skipped and the call is parsed, as llama.cpp does.
- **Tool calls are no longer cut off midway.** While the model wrote a Qwen-style tool call (`<tool_call>` … `</tool_call>`), picking a
  special token such as `<|im_start|>` cut the call, the reply came back as plain text, and the agent took it as its final answer and
  stopped (Qwen3-32B, at a rare Hangul character in a path). Inside a call such tokens are now skipped and the next candidate is used,
  the same effect as llama.cpp's tool-call grammar. **Output change** — only for requests where a special token was picked inside a call;
  every other reply stays the same bit for bit.

## 0.14.2 — Start coding agents from the app (2026-10-28)

### Added

- **Start coding agents from the app.** On the new Agents page, pick Claude Code, Codex CLI, Qwen Code, OpenCode or Aider and a working
  folder and press Start: the app opens a new terminal window with that agent attached to the loaded model. The settings (environment
  variables and command-line options) are passed to that run only — the agent's configuration files and your environment variables are
  not changed — and the agent still asks before it edits files or runs commands. The page shows whether each agent is installed (PATH)
  with its install command, the commands to run it yourself (copy), the values for the Cline and Kilo Code VS Code extensions, context
  and concurrency warnings ("Reload for agents") and recently detected tool errors. Settings gain "Tool-error form" (off, C1, C2, C3 —
  off by default), and `HEAD /api/hello`, which Claude Code probes, now returns 200. On Linux the app shows the commands but does not open
  a window yet.

## 0.14.1 — Fix: Vulkan memory on integrated GPUs (2026-10-28)

### Fixed

- **Loading a large model on an integrated GPU no longer stops with "Vulkan 오류 -4".** The Vulkan layer counted free graphics memory
  as heap size minus its own allocations, so on integrated GPUs such as a laptop's Intel Arc 130V — whose heap is system memory shared
  with Windows and other apps — it allocated more than was really there and lost the device. It now sizes the context from the driver's
  budget (VK_EXT_memory_budget) minus current usage, and on integrated GPUs it stops an allocation that would still not fit with a
  clear "graphics memory is short" message (needed and free amounts). When the device is lost anyway, it logs the allocated and free
  memory and the last kernels it issued. The device check reports integrated GPUs and the memory available now, and the app uses that
  value as the GPU's memory. With enough memory, replies are bit-identical.
- **The cross-check's pinned Vulkan server hash follows the new Vulkan layer.** On the reference PC the Vulkan quick check
  (cooperative-matrix 16x16x32 and base modules) matches; the CUDA server, the generator, the backends and the modules are unchanged.

## 0.14.0 — Coding agents: Anthropic and Responses APIs · tool_choice · tool-error detection and forms (2026-10-28)

### Added

- **`tool_choice` that forces a tool call.** `/v1/chat/completions` accepts `tool_choice: "required"` (any call) and a named
  function (`{"type": "function", "function": {"name": …}}`). The server enforces it by placing the model template's call opening after
  the generation prompt (after the template's thinking-off form), so the same request gives the same tokens and requests without this
  field are unchanged. With a single tool, `required` also fixes the name; with several tools only the call is enforced and the model
  picks the name. Qwen Code's side requests (session titles, summaries and others), which failed with 400, now work. MiniCPM's XML call
  form is not supported yet (400).
- **Anthropic Messages API (`/v1/messages`, `/v1/messages/count_tokens`).** Agents that speak the Anthropic format, such as Claude
  Code, can now attach (`ANTHROPIC_BASE_URL=http://127.0.0.1:44544`). Requests run through the same chat template and engine (the same
  request gives the same tokens); replies use Anthropic content blocks (text, tool_use), stop_reason and usage, and streaming emits
  the message_start … message_stop events. Thinking is not returned. A prompt over the context limit is reported as
  `prompt is too long`, so the agent compacts the conversation.
- **OpenAI Responses API (`/v1/responses`, stateless).** Codex CLI, which only speaks the Responses API, can now attach (a provider with
  `base_url = "http://127.0.0.1:44544/v1"` and `wire_api = "responses"`). Requests run through the same chat template and engine (the
  same request gives the same tokens); replies carry output items (message, function_call) and usage, and streaming emits the
  response.created … response.completed events. Hosted tools such as web_search are dropped, and `previous_response_id` (server-stored
  conversations) is rejected with 400 — send the whole conversation with `store: false`. Thinking is not returned.
- **Tool-error detection (first piece of agent-compatibility step 4).** When a conversation from an agent contains a failed tool
  result, the server now finds it, classifies it (transient, fixable, unrecoverable or ambiguous) and records it in the server log
  (`[n] 도구 오류 감지 …`) and in an `X-GeulPU-Tool-Errors` response header. Only reliable signals are used: Anthropic `is_error`,
  Cline's `"success": false`, OpenCode's tool-error JSON, and the error text of Qwen Code, Codex, Kilo and OpenCode selected by
  User-Agent. The text sent to the model is unchanged (replies are bit-identical).
- **Tool-error forms (second piece of agent-compatibility step 4 — off by default).** With the request field
  `geulpu_tool_errors` (`"off"`, `"c1"`, `"c2"`, `"c3"`) or the server option `--도구오류 c1|c2|c3`, a failed tool result the agent
  just received (classified as transient, fixable or unrecoverable) is shown to the model in one of the forms measured in the
  research — C1 appends one guidance sentence to the result, C2 removes the failed call and result and adds a runtime notice, C3
  keeps the result and adds the notice in a following user turn. The text is the research's, byte for byte. All three APIs (OpenAI
  chat, Anthropic, Responses) behave the same, and the `X-GeulPU-Tool-Errors` header reports the form and how many failures were
  changed. When off, replies are bit-identical. The app now passes the agent's User-Agent to the inference server, so the
  per-agent error-text rules also apply to agents connected through the app's port.

### Changed

- **The cross-check's pinned server hashes follow the new servers.** On the reference PC the 32 problems give the same output as
  the reference with the CUDA module, and the Vulkan quick check (cooperative-matrix 16x16x32 and base modules) matches; the
  generator, the backends and the pinned modules are byte-identical. On the fresh-install path the same question to Qwen2.5-7B
  Q4_K_M gives the same text and logit hashes as 0.13.0 (default, 16-bit KV and 8-bit KV).

## 0.13.0 — Gemma 4 · faster Qwen3.5 · layer lens and Jacobian lens · Linux (2026-10-28)

### Added

- **Cached prompt tokens in `usage`.** Every `/v1/chat/completions` and `/v1/completions` response — and the usage chunk of a stream
  with `include_usage` — now carries `usage.prompt_tokens_details.cached_tokens`: the prompt tokens taken from a cache instead of being
  computed (this slot's cache, another slot's copy or the context store — the same count as the extension field
  `geulpu.prompt_tokens_cached`). The last prompt token is always computed, so a repeated prompt reports `prompt_tokens − 1`, and a
  request with `"cache_prompt": false` reports 0. Clients that check per request whether the prompt cache was used read it from the
  same field as with llama.cpp and OpenAI.
- **Linux release files.** Releases now carry `GeulPU-<version>-linux-x64.tar.gz` (x86-64 · NVIDIA · glibc 2.34 or later), and the
  `geulpu` package on PyPI installs on Linux too (`pipx install geulpu`, then `geulpu`), so Linux no longer needs a source build.
  In-app downloads and Vulkan are not on Linux yet.
- **A record on Linux and a datacenter GPU.** 0.12.1 was built from source and run on Ubuntu 22.04 with an H100 NVL MIG instance
  (3g.47gb): it builds with the geul compiler and Python 3 alone, its kernel modules are byte-identical to Windows, and the generated
  tokens of 24 conversations were all identical to an RTX 4070 Ti on Windows. The speed against llama.cpp b11900 and vLLM 0.32.0 over
  the same HTTP requests, and scripts to measure again, are in docs/h100.md and 재기/H100/ (the Linux build is now tested under WSL2
  and on native Linux).
- **The release carries a J file (Jacobian lens) for Qwen3-0.6B.** `Qwen3-0.6B-Q8_0.jlens.safetensors` (56,626,912 bytes, SHA-256
  `20f46104ff65671a8c03ee58c384fe2f01cc2a0b367d8e11955a212329ae4d38`). Put it, under this name, next to Qwen's official
  `Qwen3-0.6B-Q8_0.gguf` and load the model: the research panel's layer lens then reads the Jacobian lens as well (for another quantized
  file of the same model, rename it after that GGUF). Fit record (also in the file's `__metadata__`): original Qwen/Qwen3-0.6B (BF16,
  revision c1899de289a04d12100db370d81485cdf75e47ca), jlens (anthropics/jacobian-lens 581d398613e5602a5af361e1c34d3a92ea82ba8e) through
  `도구/jlens_fit.py`, torch 2.14.1+cu130 · transformers 5.20.2, 100 prompts from the first half of the wikitext-2 test split
  (wiki.test.raw, SHA-256 173c87a5…, paragraphs joined to at least 1,200 characters, first 128 tokens), source layers 0–26 → target layer
  27, first 16 positions skipped, 1,019 s on an RTX 4070 Ti. Checked file: Qwen's official Q8_0 (SHA-256 9465e63a…) — in layers 9–26 the
  J-lens top-1 agreement with the original (BF16) is at least 93.8 % and at most 1.7 points below the logit lens's, which carries over
  under the criterion set before measuring (2c in 문서/층렌즈.md). On low-bit files (Q4_K_M and the like) the per-layer top-1 agrees less
  with the original.
- **Qwen3.5 and Gemma 4 in Find models.** The app's model catalog adds Qwen3.5-0.8B (Q8_0, 0.8 GB), Qwen3.5-4B (Q4_K_M, 2.6 GB),
  Qwen3.5-9B (Q4_K_M, 5.3 GB) and Gemma-4-12B-it (Q4_K_M, 6.6 GB) — unsloth's files, checked to run on CUDA (RTX 4070 Ti); not on
  Vulkan yet.
- **Tool calls and thinking for Gemma 4.** Gemma 4's own tool-call format (`<|tool_call>call:name{key:value,…}<tool_call|>`) is now
  read and returned as OpenAI `tool_calls`, with the arguments turned into JSON by the same rules as llama.cpp (strings `<|"|>…<|"|>`,
  numbers, true, false, null, objects and arrays). Its thinking block `<|channel>thought … <channel|>` comes as `reasoning_content` in
  requests that send tools, and otherwise in front of the answer like other models (the channel markers and name are removed, as is the
  empty channel that may follow an answer with thinking off — with `geulpu.special` the markers come as text). The app's 'Thinking'
  switch now works for Gemma 4 and folds the thinking away when it is on.
- **The Jacobian lens without a GPU.** `도구/repro_check.py --layer-lens 5,12 --jlens <model>.jlens.safetensors` moves the same steps'
  rows into the final layer's basis with J_l fitted on the original model and reads them (Anthropic's Jacobian lens — contract 5.4).
  `도구/jlens_fit.py` fits a J file with jlens on the original (Hugging Face) model. A J fitted on Qwen3-0.6B in BF16 was measured to
  carry over to the GGUF Q8_0 file (top-1 agreement with the original of at least 93.9 % in layers 9–26).
- **The Jacobian lens in the research panel.** Put `<name>.jlens.safetensors` (a J file fitted on the original model with
  `도구/jlens_fit.py`) next to the model file and load the model; the step detail's layer lens then reads both the logit lens and the
  Jacobian lens and switches between them (contract 5.4). What the position points at shows earlier in the layers than with the logit
  lens. API: `geulpu.layer_lens: {"jacobian": true}` on a `/v1/completions` echo — response `geulpu.jacobian_lens`. Reproduction files
  carry the computed Jacobian lens and the J file's SHA-256, and `repro_check --jlens` checks them bit for bit without a GPU.
- **The layer lens (logit lens) without a GPU.** `도구/repro_check.py --layer-lens 5,12` reads, for the chosen steps of a capsule, the
  row that scored the token at every layer with the final logit computation (contract 5.3), and shows the top-1 token, the chosen
  token's probability and rank, the entropy and the top k per layer. The last layer is bit-identical to the final logits.
- **Layer lens (logit lens) in the research panel.** In a step's details, 'Layer lens — Read every layer' reads the computation at the
  position that chose the token at every layer with the final output (contract 5.3) and shows, per layer, the top-1 token and its
  probability, the chosen token's probability and rank, and the entropy (click a layer for its top k). It shows at which layer an
  answer forms and where a failure such as switching to another language starts. The API is `geulpu.layer_lens` on an echo request to
  `/v1/completions`. Reproduction files carry the computed lens, and `도구/repro_check.py` checks it bit for bit without a GPU.
  qwen2, qwen3, llama and gemma4 architectures (one GPU — for Gemma 4 the reading ends with the final logits' softcap; the CPU check of
  reproduction files covers qwen2, qwen3 and llama).
- **The app on Linux (built from source).** On Linux x86-64 with an NVIDIA GPU the app runs — the screen, device check, models, the
  inference server and the benchmark's reproducibility check and file-pair reproducibility. `bash 앱/꾸리기.sh <geul repo>` packages it;
  on a remote server open the screen through SSH port forwarding. New `GeulPU --check r1|pair [--quick]` (`--검사 재현성|쌍 [--빠름]`)
  runs one check without the screen, prints the result folder and zip and exits with 0 when every problem matches the reference.
  Downloads inside the app and Vulkan are not on Linux yet (put the model files in the data folder's model folder).
- **The engine runs Gemma 4 12B (architecture gemma4).** Sliding-window layers (head 256, window 1,024) mixed with global-attention
  layers (head 512, v = k, rope_freqs), GeGLU, post-attention/post-FFN norms, per-layer output scale and the final logit soft-cap
  follow contract 4.12 and match the CPU reference bit for bit. k-quant (Q4_K_M etc.) and Q8_0 files, exact mode on CUDA only (no fast
  mode, Vulkan, speculative decoding or context store yet). E2B/E4B (per-layer embeddings) and 26B-A4B (MoE) are rejected with a reason.
- **Qwen3.8-27B and Qwen3.5 GGUFs with mixed formats.** Files that mix Q8_0 and k-quant matrices (the ggml-org Q4_K_M of Qwen3.8-27B keeps the
  attention and GDN projections in Q8_0) and files whose ssm_alpha · ssm_beta are k-quants (lmstudio Q4_K_M) are now accepted. A
  multi-token-prediction layer at the end of the file is skipped like llama.cpp does (the engine used to stop with "tensor not found").
  Qwen3.8-27B does not fit a 12 GB card. Files with F32 β · α (bartowski) or IQ formats (unsloth UD) are not supported yet.
- **Fast mode for Qwen3.5.** Qwen3.5 dense models (0.8B, 4B, 9B, …) can now run in fast mode (compute mode "fast" — the fast contract
  F1 with one-digit activations in the linear layers); the app loads them in fast mode when that is the setting (the default). The 8-bit
  KV cache and fast attention are not available for them yet, and Qwen3.6 MoE stays exact-only until its quality has been measured.

### Changed

- **The engine decides which models are accepted.** The app no longer keeps its own per-architecture rules; it asks the engine
  (`server --모델확인`). Module names and capabilities (exact only, Vulkan, 8-bit KV cache, F2, …) are unchanged, and an unsupported
  architecture now shows the engine's reason with the list of accepted architectures.
- **Gemma 4 generation is 2.1–2.4× faster at long context.** Attention for up to four rows now computes key tiles in parallel and
  merges them in order (same contract — output bits unchanged). On 12B (RTX 4070 Ti, exact mode), generation at about 1,800 and 3,600
  tokens of context went from 54 and 48 ms per token to 22 ms per token; short chats from 22.7 to 21.4 ms per token.
- **Gemma 4 prompt processing is 7–9× faster.** Attention for prompts (five rows or more) now runs on a tensor-core kernel (same
  contract — logits and generated tokens are bit-identical). On 12B (RTX 4070 Ti, exact mode), time to first token for about 1,800 and
  3,600 prompt tokens went from 5.45 and 15.4 s to 0.77 and 1.63 s.
- **Gemma 4 generation is 7–8 % faster.** Small kernels (post-norm with residual and the next RMSNorm, GeGLU, the attention merge) now
  fuse into one kernel that writes the next linear layer's input, and single-row matmuls use the decode-specific kernel (same contract —
  output bits unchanged). On 12B (RTX 4070 Ti, exact mode), short chats went from 21.0 to 19.4 ms per token and about 1,800 tokens of
  context from 22.1 to 20.4 ms per token.
- **Gemma 4 fits a much longer context on the same GPU.** The sliding-window layers keep their KV cache in a ring buffer the size of
  the window, and the shared KV cache holds only the global-attention layers at their own width (output bits unchanged). On 12B
  (RTX 4070 Ti 12 GB, one slot) the context grows from 7,360 to 118,144 tokens — the KV cache per position drops from about 442 KB to
  18 KB (the window layers' ring is a fixed 0.57 GB per slot). The server's slot cache can still roll back and reuse as far as the
  ring holds (513 tokens with 512-row chunks).
- **3× faster GDN recurrence for Qwen3.5.** The linear-attention (GDN) recurrence of Qwen3.5 and Qwen3.6 MoE now stages each 8-token chunk
  in shared memory ahead of use and gives each thread two columns — the recurrence at a 2,048-token prompt drops from 64 to 20 ms, so such
  prompts are 13–16 % faster on the 4B and 7–10 % on the 9B (fast mode: 4B 261 → 219 ms, 9B 386 → 348 ms). The output is bit-identical
  (same contract, same order).
- **Faster long-context generation for Qwen3.5.** From position 16,384 the decode attention of Qwen3.5 and Qwen3.6 MoE launches the
  query heads that share a KV head as neighbouring blocks and prefetches the next chunk into L2 — generation after a 32,768-token prompt
  is 7.5 % faster on the 4B and 4.5 % on the 9B (fast mode: 4B 78.0 → 83.8, 9B 54.3 → 56.8 tokens/s). Shorter contexts are unchanged,
  and the output is bit-identical (same arithmetic, same order).
- **Faster prompts for Qwen3.5.** The prompt (chunk) attention of Qwen3.5 and Qwen3.6 MoE now runs on int8 tensor cores — 28 times
  shorter attention at a 2,048-token prompt, so such prompts are 3.9–4.6× faster on the 4B and 2.8–3.4× on the 9B (fast mode: 4B 1,207 →
  263 ms, 9B 1,332 → 388 ms). The output is bit-identical (same contract, same order); the byte layout of the value cache changed (KV
  hashes differ — the values do not).
- **Faster generation for Qwen3.5.** For Qwen3.5 and Qwen3.6 MoE, the head-256 attention of a generation step is now split into
  64-key tiles spread over the whole GPU — 16 times shorter per layer at depth 2,048, so generation is 32–53 % faster on the 4B (more at
  deeper context) and 15–35 % faster on the 9B. The output is bit-identical (same contract, same order).

### Fixed

- **Speculative decoding state and rotated GGUF files.** Loading Qwen3.5 or Qwen3.6 MoE showed speculative decoding as on in the
  server state (the server turns it off for those architectures); the app now shows the engine's value. GGUF files whose weights were
  folded with a rotation (llama.cpp's `prism.hadamard.*`) give wrong values when run as is, so they are refused with the reason.
- **RoPE table and kernel-choice memory at load time.** The server and the app build the RoPE table on a worker thread while loading,
  but the loader read the model settings a second time meanwhile and briefly cleared rope_freqs, so models with rope_freqs could get a
  table without them (the engine kept that table, and long prompts could give a different answer from one load to the next). The loader
  no longer re-reads the settings while workers run, and a prebuilt table is used only when θ, rope_freqs and YaRN all match. K-quant
  models of architectures with more than five matmul groups (Gemma 4, Qwen3.5, Qwen3.6) always discarded their remembered kernel
  choices and re-tuned on every load; they now reuse them.

## 0.12.1 — Fix: the speed test and `geulpu.speculative` (2026-10-24)

A bug fix on the 0.12 line. Outputs of 0.12.0 are unchanged.

### Fixed

- **`geulpu.speculative` is accepted again, and the speed test runs.** Since 0.12.0 a request with
  `"geulpu": {"speculative": true | false}` (speculative decoding on or off) was rejected with HTTP 400 ("geulpu.speculative must be a
  boolean") on every model except Qwen3.5: a line added for Qwen3.5 had landed between the "is a boolean" check and its "wrong type"
  branch. The app's speed test (Benchmark → Speed test) sends `"speculative": false` with every measurement, so it stopped with
  "HTTP 400" on those models. Qwen3.5 still runs without speculative decoding, and a value of the wrong type is now rejected on every
  model.

## 0.12.0 — More models: Qwen3.5 · Qwen3.6 MoE · F16 / BF16 / F32 files · chat templates from the file (2026-10-24)

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
