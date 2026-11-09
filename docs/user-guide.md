# GeulPU user guide

**[한국어](../사용안내.md)** · [README](../README.md) · [debugger guide](debugger-guide.md) · [research guide](research-guide.md) ·
[model evaluation guide](model-evaluation-guide.md) · [development](development.md) · [glossary](glossary.md)

Everything about installing and using GeulPU — the app, its settings, the OpenAI-compatible API and the supported models — and what
makes it different, with the measurements. The [README](../README.md) has the overview; building from source is in
[development](development.md).

[Install](#install) · [First steps](#first-steps) · [Supported models](#supported-models) · [How it runs](#how-it-runs) ·
[Troubleshooting](#troubleshooting) · [Update and uninstall](#update-and-uninstall) · [What is different](#what-is-different)

## Install

### Requirements

| | |
|---|---|
| OS | Windows 10 or 11 (64-bit), or from 0.13.0 Linux x86-64 with glibc 2.34 or later ([On Linux](#on-linux)) |
| GPU | NVIDIA RTX 30 series or newer runs on CUDA; other GPUs (AMD Radeon, …) run on Vulkan 1.3 (Windows only) |
| Graphics memory | From 6 GB (small models). 7B / 8B need 8 GB (Q4_K_M) or 12 GB (Q8_0); 14B needs 12 GB or more ([table below](#which-model-fits-your-pc)) |
| Disk | About 5 MB for the app plus 2–9 GB per model |
| Internet | Only to find and download models and to check for new versions |

Keep the graphics driver up to date. NVIDIA cards older than the RTX 30 series (RTX 20, GTX 16 and earlier) cannot use the CUDA path
(compute capability below 8.0); the app then checks by itself whether they can run on Vulkan. On AMD, a Radeon RX 9060 XT has been
verified.

### With Python — one command

Instead of downloading and extracting the zip, you can install with a Python installer and run it right away
([geulpu on PyPI](https://pypi.org/project/geulpu/) — the same app as the release zip).

```powershell
pipx install geulpu        # or: uv tool install geulpu · pip install geulpu
geulpu                     # opens http://127.0.0.1:44544 in your browser
```

- To update, quit the app, then run `pipx upgrade geulpu` (`uv tool upgrade geulpu`, `pip install -U geulpu`). The app shows the
  command when a new version is out.
- Models, settings and results live in `%USERPROFILE%\GeulPU` (`~/GeulPU` on Linux) as with the zip, so uninstalling the package
  (`pipx uninstall geulpu`) keeps them.
- If the `geulpu` command is not found, run `python -m geulpu` (pip's per-user install may leave its scripts folder off PATH). pipx or
  uv is recommended.

Steps 1–3 below install from the zip; on Linux, see [On Linux](#on-linux).

### 1. Download

Get **`GeulPU-<version>-windows-x64.zip`** from the [releases page](https://github.com/wwoosshh/GeulPUv2/releases/latest). It is small
because it contains no model — you download models inside the app.

To check that the file arrived intact (optional), compare it with `SHA256SUMS.txt` on the same page. In PowerShell:

```powershell
Get-FileHash .\GeulPU-*-windows-x64.zip
```

### 2. Unzip

Right-click the zip, choose **Extract All…**, and pick any folder (for example `C:\GeulPU`). Do not run `GeulPU.exe` from inside the
opened zip — extract it first.

### 3. Run

Double-click `GeulPU.exe`.

1. If **"Windows protected your PC"** appears, click **More info → Run anyway**. The executable is not code-signed; release zips are
   built from this repository's source by GitHub Actions ([How it runs](#how-it-runs)).
2. A black console window opens. That window *is* GeulPU — **closing it quits the app.**
3. Your browser opens `http://127.0.0.1:44544`. If it does not, type that address into the address bar.

### On Linux

From 0.13.0 the releases page also has **`GeulPU-<version>-linux-x64.tar.gz`** — for x86-64 with an NVIDIA GPU and its driver, and glibc
2.34 or later (Ubuntu 22.04 and newer) — and `pipx install geulpu` installs the same app (run it with `geulpu`).

```bash
sha256sum -c SHA256SUMS.txt --ignore-missing    # optional: check the download
tar xzf GeulPU-<version>-linux-x64.tar.gz
cd GeulPU-<version> && ./GeulPU
```

- The app does not open a browser; it prints its address, `http://127.0.0.1:44544`. On a remote machine, forward the port first
  (`ssh -L 44544:127.0.0.1:44544 <host>`) and open the address on your own computer.
- Downloads inside the app (and `geulpu --hf`) use the system's `curl` — install it if your distribution lacks it. GGUF files you
  already have can be added with **Models › Files → Add a GGUF file**.
- From 0.22.0 the Linux package also carries the Vulkan path (server-vulkan, device-vulkan, `libvulkan.so.1`) — the way to GPUs other
  than NVIDIA (AMD, Intel …). It has only been exercised with a software Vulkan driver; a run on a real Linux GPU is still open.
- Models, settings and results live in `~/GeulPU` (or in the folder named by `GEULPU_HOME`).

## First steps

The app speaks English unless your browser prefers Korean — switch any time in **Settings › App → Language** (Auto · 한국어 · English).
The sidebar on the left has, from the top, **Experiments · Models · Reproduce · Jobs · Settings**. Models, Reproduce and Settings have
tabs at the top: **Models** (Files · Evaluation · Performance), **Reproduce** (llama.cpp · Checks · Check results), **Settings** (App ·
Device · Prompt cache). **Jobs** shows the job that is running (a download, a measurement, a check …) with its progress, log and a stop
button. On the right is the **command console**: every operation you trigger on a screen appears there as the GeulPU command that does the same
thing, and you can type commands in its input line ([6. Console, CLI and Python](#6-console-cli-and-python--the-screens-operations-as-commands)).

The app opens on the experiment list. With no model installed, opening **Experiments › Conversations** shows this screen. **Download the
recommended model** fetches a model chosen for your graphics memory right away; **Browse models** opens Find models, described below.

![First screen — first, you need a model](img/first-run-en.png)

### 1. Get a model — Models › Files

- Hugging Face's GGUF models are listed by **Trending**. Switch to Popular, Likes or Recently updated, or type part of a name (for
  example `qwen3`, `coder`, `14b`) into the box at the top to narrow the list.
- By default only **models GeulPU can run** are shown. **Also show models that don't run** shows everything, with the ones that don't
  run dimmed and the reason given. **Fits this PC** keeps only models that fit your graphics memory.
- New to this? Switch on **Verified by GeulPU**: only repositories with files GeulPU has run itself remain (picture below); see
  [which model fits your PC](#which-model-fits-your-pc).

![Find models — the list on the left, details of the selected model on the right](img/find-models-en.png)

- Click a model to see its description, downloads, license and **file list** on the right. The file that suits this PC best is marked
  **Recommended for this PC**, files GeulPU has run are marked **Verified to run**, and files that are tight or too large for your
  graphics memory say so. Click **Download** on the file you want.

![File list — "Recommended for this PC" and "Verified to run" marks with download buttons](img/model-files-en.png)

- Before downloading, the app reads only the beginning of the file to check that it will run; interrupted downloads resume; finished
  files are verified with SHA-256. Downloads in progress appear under **My models** and in the status bar at the bottom, where you can
  pause or resume them.
- GGUF files you already have can be added with **Add a GGUF file** at the top; they stay where they are.
- **Load this model** under **My models** loads a model into graphics memory (sending a chat message loads it too).

### 2. Chat — Experiments › Conversations

- Open **Conversations** in the experiment list (**Experiments** in the sidebar). Pick a model with the model button at the top and send a message (**Enter** sends,
  **Shift+Enter** adds a line).
- Every answer is recorded as a run of the default experiment "Conversations" (created when you first send). **Run rN** under an answer opens that
  answer's run screen — per-token probabilities, branching, the step and layer screen
  ([4. Run screen](#4-run-screen-and-step-and-layer-screen--per-token-probabilities-branching-logit-lens)).
- **The first time**, GeulPU builds the kernels for your GPU and model (about 10 seconds on a fast CPU, 1–3 minutes on a slow one) and
  loads the model into graphics memory (about 20 seconds for 7B). Later only the loading happens.
- **Reasoning models** (Qwen3, DeepSeek-R1) think before answering. The thinking is collapsed as "Thought for N s"; click it to expand.
  Token count and speed (tokens/s) appear under the answer.

![Chat — Qwen3-8B Q4_K_M with its thinking collapsed and the speed under the answer; the chip above the chat and the answer's tag show the computation mode (Fast)](img/chat-en.png)

- **To turn Qwen3's thinking off**: **Generation settings** (the slider icon at the top right) → switch off **Thinking**. It then
  answers directly.
- Generation settings also hold the temperature (0 gives the same answer to the same question every time), the maximum answer length
  and a system prompt, saved in this browser.

![Generation settings — temperature, max answer length, thinking switch, auto-summarize switch, system prompt](img/generation-settings-en.png)
- You can regenerate an answer or edit and resend your message; earlier answers stay as versions you can flip through (‹ 1 / 2 ›).
  Code, tables and math (LaTeX) are rendered.
- **Long chats**: when a chat nears the context length per chat, the same model summarizes the earlier conversation (the latest turns
  stay word for word) and the chat goes on without an error. The summary stays in the chat as a card you can expand, and the
  summarized messages stay, dimmed. "context N / L" under an answer shows how much is used. Turn it off with **Auto-summarize long
  chats** in Generation settings.
- The chat list stays in this browser; each answer's request and response are kept in the "Conversations" experiment's folder.

### 3. Experiments — question, plan, runs and results in one folder

One study is one **experiment**. An experiment is a folder (`<date>-<name>` in the `experiments` folder) that keeps the question, the
criteria, the plan (plan.md), the run records and the results (results.md). Hand the folder to someone else and they can run the same
runs again and check them.

- **Experiment list** (**Experiments** in the sidebar): each experiment's status, models, number of runs, verdicts and commands. **New
  experiment** creates one with a name, a question and authors; the experiments folder can be changed here too.
- **Plan** tab: the criteria and plan.md. They can be edited only before the first run, so the record shows that the criteria came
  before the results.
- **Runs** tab: the experiment's runs and **New run** (text completion · chat · Anthropic Messages · OpenAI Responses — prompt, maximum
  tokens, temperature, seed, top-p, top-k, min-p, number of top candidates). Left empty, top-p, top-k and min-p sample with the engine's
  defaults (0.95 · 40 · 0.05 — shown in the dialog), and the values actually used go into the run record. Sending a request equal to an
  earlier run's records a **re-run (check)** of that run, to see whether the bits are the same.
- **Compare** tab: the first step where two runs differ, with that step's probabilities and ranks and the logit hashes.
- **Results** tab: the verdict for each criterion, results.md and the experiment's status.
- **Records** tab: the command log (every operation recorded for this experiment — from the screens, the CLI, AI sessions and Python),
  the file list (SHA-256), the reproduce command and **Check** (format, hashes, missing files). Run logs that the debugger recorded
  outside experiments open from here too.

Each run's record keeps the device that computed it (name and driver) and the sampling values actually used, and the app keeps the
experiment's model and device lists. The folder format and the checker are described in [experiments/README.md](../experiments/README.md).

### 4. Run screen and step and layer screen — per-token probabilities, branching, logit lens

GeulPU's arithmetic is fixed by a written contract, so scoring the same text again gives **bit-identical** probabilities to those at
generation time — whatever the number of concurrent requests, the cache, the GPU or the backend (CUDA · Vulkan), within one
computation mode (Vulkan always computes Exact, so set the computation mode to Exact to compare with CUDA). The difference between two
runs therefore comes only from what you changed. Click a run of an experiment (or a chat answer) to open its **run screen**.

**New to it? Start with the [research guide](research-guide.md)** — eight experiments (where the model hesitated, branching,
comparing two answers, the effect of quantization, the effect of one word, scoring text, reproduction files, the logit lens) with
screenshots, how to read the numbers, and API examples.

- **Run screen** — the run's details (model, computation mode, GeulPU version, device, seed, sampling values, source), summary values
  (mean log-probability, perplexity, mean entropy, smallest margin, steps with margin < δ, steps that did not sample the top token) and
  the output token by token. Bars under the tokens show the margin (the gap between the top two log-probabilities), the probability or
  the entropy; pick a step to see its top candidates and **Branch with this token** (← → previous and next step, Enter step and layer
  screen, b branch with the best other candidate). A branch continues with the same settings, the same seed and the same step's random
  numbers, so the only change is the chosen token; it opens in the Compare tab next to the original run.
- The same screen has **Re-run** (a new run with changed settings), **top candidates at every step**, **re-run without the cache**,
  **score with another model** (another installed model with the same vocabulary and chat template — another quantization of the same
  model, say), where the prompt was reused from (this slot, another slot, the prompt cache, and where the cache broke) and **Export**
  (token table as JSON or CSV, reproduction file, Python code that gets the same values).
- **Step and layer screen** — one step in detail: top candidates (1–128, each with a branch button), the **logit lens** table (the top
  three per layer and the sampled token's rank — at which layer the answer appears; with a J file `<name>.jlens.safetensors` next to the
  model, the Jacobian lens too) and the score of a continuation from that step (without context, or against alternatives).
- Every computation on these screens (branch, score, lens) is recorded as a run of the experiment, so it is still there when you open
  the screen again, and the console shows the same CLI command.

Probabilities are the model's distribution (temperature 1), not the distribution filtered by the sampling temperature, top-k or top-p.

![Run screen — run details and summary values, margin bars under the tokens, the chosen step's top candidates and branch buttons](img/run-screen-en.png)

![Step and layer screen — the logit lens table: the top three per layer and the sampled token's rank](img/step-screen-en.png)

### 5. Reproduction files — get a run checked bit for bit on another PC

**Export › Reproduction file (JSON)** on the run screen writes one file with which another PC can check the run — the prompt and answer
tokens, every step's log-probability, entropy, top candidates and logit hash, the model's SHA-256, the device it was made on, the
sampling settings actually used and the seed (and any logit lens computed). The per-step values come from a whole-answer score run, so
without one the screen scores the whole answer first (that run stays in the experiment).

Whoever receives the file loads its model and drops the file into **Reproduce › Check › Check a reproduction file** — scoring,
regeneration and the fingerprint are compared bit for bit, and a difference shows the first differing step and the settings that differ.
From the CLI: `GeulPU experiment repro --path <experiment> --run r1 --out repro.json` and `GeulPU check repro --json @repro.json`.

A reproduction file can be checked without a GPU, too — clone the repository, install numpy in a virtual environment
(`python -m venv .venv`, then `.venv/Scripts/pip install numpy` — `python -I` ignores packages installed with `pip install --user`) and run
`.venv/Scripts/python -I 도구/repro_check.py <repro.json> <model.gguf>`. The contract's CPU reference implementation recomputes the tokens, logit hashes
and log-probabilities (a few minutes for 7B). A reproduction file records its computation mode and is checked in that mode.

### 6. Console, CLI and Python — the screens' operations as commands

Every screen operation is also a `GeulPU <command>` — people use it from a terminal, AI agents from a shell, with the same operations
and the same arguments. Screens, CLI and console are built from one operation registry, so they give the same results and record into
the same experiments.

- **Command console** (right side of every screen): operations triggered from a screen, the CLI, an AI session or Python appear one per line as
  the command that does the same thing, with its status. Each line has **Copy command**; type a command in the input line (for example
  `run complete --prompt "Why is the sky blue?" --max_tokens 64`) and it is sent as the CLI would send it. `]` collapses it, the width
  is adjustable, and on narrow screens it sits at the bottom.
- **CLI**: `GeulPU ops` (every command), `GeulPU help <command>`, `GeulPU guide` (a one-page guide for AI agents —
  [cli.md](cli.md)). `--experiment <experiment>` (or the `GEULPU_EXPERIMENT` environment variable) records into that experiment, and
  `--select` returns only some fields. Branching, logit lens and scoring from a recorded run take one command each, such as
  `GeulPU experiment branch --run r1 --step 3 --token 785`.
- **AI session** (button above the console): starts a coding agent with the app's model — see "Starting an agent from the app" in
  [Use it from other programs](#use-it-from-other-programs-openai-compatible-api).
- **Python**: `import geulpu` — `geulpu.start()` (a windowless app), `geulpu.connect()`, `load`, `complete`, `chat`, `rerun`, `score`,
  `branch`, `compare` ([python-api.md](python-api.md)).

### 7. Speed, device and reproducibility checks (optional)

- **Models › Performance — speed test**: prompt processing (512 and 2048 tokens) and generation speed in tokens per second.
- **Settings › Device — hardware diagnosis**: measures memory bandwidth and tensor-core throughput and shows how close each kernel comes
  to the hardware's limit.
- **Reproduce › Check — reproducibility check**: solves 32 math problems with DeepSeek-R1-Distill-Qwen-7B Q8_0 and compares the output
  with the reference PC bit for bit (the quick check uses 2 problems and 64 tokens).
- **Reproduce › Check — file-pair reproducibility** (0.11.1): runs GGUF files that quantize the same model in different ways — the
  files of Liu and Tsai's paper on GGUF supply chains (arXiv:2610.19498) — in exact mode. It checks that every step of 32 problems gives
  the same logit hash as the reference PC, and measures how the files differ (the first step where the greedy answers diverge, the
  log-probability difference per token). The card downloads the files and checks their SHA-256. Details and reference results:
  [file-pair.md](file-pair.md).
- Results are listed in **Reproduce › Check results** and can be exported as a zip (user paths are removed). The raw tool output and the
  result files themselves stay in Korean.

### 8. Settings — Settings › App

GPU backend (Auto · CUDA · Vulkan), Vulkan device, concurrent chats (how many chats can answer at once, 1–8) and context length per
chat, **computation mode** (one for quantized files, one for F16 · BF16 · F32 files), **8-bit KV cache**, **fast attention**, **speculative decoding**, **prompt cache**, model folder, language,
new-version notice.
Changes are saved immediately.

Since 0.11.0 the **8-bit KV cache** and **fast attention** are on by default. Settings saved by an earlier version are moved to these
defaults once, the first time 0.11.0 starts; to go back, set either one to **Off** here (the choice is kept from then on).

**Computation mode** — **Fast** (default — Exact for F16 · BF16 · F32 files) or **Exact**. Applies the next time a model is loaded.
The computation mode chip above the chat switches it too and reloads the loaded model in that mode. Every answer carries a tag with the mode it was computed in (Fast · Exact,
8-bit KV · fast attention).
- **Fast**: the activations entering the matrix multiplications are 8-bit (the same class as llama.cpp). Prompt processing is 1.3–1.6×
  faster than Exact (CUDA · 7B · 8B), and the change in next-token probabilities is of the same order as llama.cpp's
  ([문서/2단계.md](../문서/2단계.md)).
- **Exact**: the contract as it was up to 0.5.0 (16-bit activations — research precision).
- Both modes match the contract's CPU reference bit for bit, and within one mode every path gives the same bits. The
  reproducibility check and Vulkan devices always use Exact.
- **F16 · BF16 · F32 files** (unquantized files) have their own computation mode setting, **Exact** by default (since 0.12.0). They
  are mostly tested for research right after training, so accuracy comes first, and on these files Fast drifts further from Exact
  than llama.cpp does (Qwen2.5-1.5B F16: next-token KL 5.9× llama.cpp's). Fast can still be chosen — switching the chip while such
  a file is loaded changes the setting for these files.

**8-bit KV cache** — **On** (default since 0.11.0) or Off. When on, the keys and values of the earlier tokens (the KV cache) are kept as 8-bit
integers with one scale per 32 values and multiplied in that form (contract F1A′ — [contract.md](../계약/contract.md) section 2.1). The
KV cache is 0.55× the size, so longer contexts fit on the same GPU, and long contexts get faster — on an RTX 4070 Ti (12 GB),
Qwen2.5-7B Q4_K_M goes from 87,488 to 131,072 tokens for one chat (47,808 to 80,320 for two), Qwen3-8B from 35,328 to 59,392 and
Llama-3.1-8B from 39,936 to 66,688, and at 64k tokens prompts are 22 % and generation 19 % faster (short contexts are not slower). The
change in next-token probabilities is of the same order as llama.cpp's q8_0 KV cache. It is used only with **Fast**, CUDA and k-quant
models (files such as Q4_K_M) and is otherwise the same as Off (Q8_0 and Q4_0 files do not pass the quality bar, so they are blocked).
The answer bits differ from the 16-bit KV cache but are always the same under the same settings — results match the contract's CPU
reference bit for bit, and the debugger and reproduction files record this setting and warn when it differs. Applies the next time a
model is loaded ([문서/긴문맥.md](../문서/긴문맥.md) ③b, Korean).

**Fast attention** — **On** (default since 0.11.0) or Off. Used only with the **8-bit KV cache** on; otherwise the same as Off. When on, attention is
computed in a lighter form (contract F2 — [contract.md](../계약/contract.md) section 2.2: one value scale per head instead of one per 32
values, a base-2 exponential, and probabilities as 16-bit integers with one exponent per 64 keys). Long prompts get faster — on an RTX
4070 Ti, 1,024 new tokens at depth 32k and 64k are 22.6 % and 27.6 % faster for Qwen2.5-7B Q4_K_M and empty-cache prompts of 32k and
64k 17.2 % and 23.0 % faster (Llama-3.1-8B is similar), and all four are faster than the latest llama.cpp (b11796) with an f16 or a q8_0
KV cache. Generation is unchanged and speculative verify steps are 4–6 % faster. The change in next-token probabilities is of the same
order as the 8-bit KV cache (Qwen2.5, Llama-3.1 and Mistral passed the quality bar; Qwen3 models pass it with F2B, a variant that keeps the
exponent of q per block). The answer bits differ from Off but are always the same under the same settings — results match the contract's CPU reference bit
for bit, and the debugger and reproduction files record this setting and warn when it differs. Applies the next time a model is loaded
([문서/빠른어텐션.md](../문서/빠른어텐션.md), Korean).

**Speculative decoding** — **On** (default) or Off. Looks up likely next tokens in the prompt and the answer so far (no draft model)
and verifies several in one step. Because this engine gives the same bits whatever the number of rows in a step, **answers,
log-probabilities and run-screen values are bit-identical to Off**, while answers that repeat pieces of the prompt — editing
code, rewriting a document, quoting summaries — generate 3.5–4× faster (ordinary chat 1.0–1.5× — [문서/3단계.md](../문서/3단계.md), Korean).
Applies the next time a model is loaded.

**Prompt cache** — **Default** (on when the location is an SSD, off for an HDD) · On · Off. Keeps what the model computed for long
prompts (the KV cache) on the SSD in 256-token chunks and restores it instead of recomputing whenever a request starts the same way —
even after restarting the app, switching models and back, or another chat taking the slot. For coding agents, which resend long system
prompts and conversations, the first token drops sharply — after restarting the app, a new task's first request (15k tokens) goes from
3.2 s to 0.3 s, and reopening a 31k-token conversation from 10.0 s to 0.74 s (0.52 s and 1.1 s when the chunks are not in the OS file
cache, as after a reboot — Qwen2.5-7B, RTX 4070 Ti). Because this engine produces
the same KV bytes whichever way it is filled, **answers, log-probabilities and run-screen values are bit-identical to Off.** Chunks
are told apart by the model file's SHA-256; when the file at a path is replaced (its size or modification time changes), the app counts
the SHA-256 again at the next load. The
location (default: data folder\문맥저장소), size limit (default 20 GB — the longest-unused chunks go first), writing (all · only when
evicted · read only) and **Clear the store** are here too. The chunks are derived from your chats, so turn it off or clear it on a shared
PC. Applies the next time a model is loaded ([문서/문맥저장소.md](../문서/문맥저장소.md), Korean).

### 9. Debugger — record · replay · branch · compare

Where the run screen looks into one answer, the **debugger** (Records tab of an experiment › the debugger's run logs) works on **runs** — many requests,
such as an agent session or a test set. It is for people who build or use local models: run a session or a test set again bit for bit
and find where and why it changed (since GeulPU 0.7.0).

**New to it? Start with the [debugger guide](debugger-guide.md)** — it follows one test set through recording, replay, observing,
branching, comparing, export and import with screenshots, and covers the script (`도구/debugger.py`), the API, the run log format and the
costs.

- **Record** — press **Start recording** and every generation request through the app's `/v1` (agents · scripts · chat) goes into a run
  log (JSONL), one line each — request and response with the seed, the prompt tokens and every step's logit hash. A test set (JSONL) can
  also be run as one run. Recording adds 0.35 % to request time.
- **Replay** — sends the recorded requests and branches again with the same seeds and compares the tokens and every step's logit hash.
  With the same version · model file · computation mode it matches bit for bit on another PC too.
- **Observe · breakpoints** — per request, per token: probability · alternatives · margin (echo scoring — the same bits as generation).
  Breakpoints find low-margin steps · low-probability samples · tool-call starts.
- **Branch** — generate again from an alternative token at some step, or from an edited request (tool result · system prompt ·
  settings). Same seed, same random numbers, so any difference comes only from what you changed.
- **Compare** — two runs of the same request sequence with a different model file · quantization · settings · version, request by
  request: same answer? · first diverging step · the log-probability difference of the same answer scored by both models.
- **Export · import** — move a run to another PC as one run log file and replay it there, or check it without a GPU against the precision
  contract's CPU reference (`도구/repro_check.py`).

![The debugger — run list, run header, request table (after a replay all 50 requests and 2 branches are the same)](img/debugger/01-runs-en.png)

### 10. Model evaluation — make · check · compare · improve GGUF files

**Model evaluation** (Models › Evaluation, since GeulPU 0.17.0) is a test bench for GGUF files. A **workspace** holds one
model's work: a reference model (best a BF16 or F16 file) and the GGUF files to compare with it (**versions**).

- **Make GGUF** — from a Hugging Face folder (config.json and .safetensors): convert to BF16 · F16 · Q8_0 and quantize (Q8_0 · Q6_K ·
  Q5_K_M · Q4_K_M · IQ4_XS, an imatrix from a text file, per-matrix types). The first time, **Set up tools** downloads llama.cpp's
  executables and conversion scripts (a fixed release, SHA-256 checked) and installs the conversion packages into an app-only Python
  virtual environment made with your Python (3.10 or later). The converted file becomes the reference model; quantized files become
  versions that keep their settings.
- **GGUF conversion check** — compares the GGUF's chat template · EOS · BOS · special tokens · tokenizer · hyperparameters (settings)
  with the original model's small files, and fixes metadata problems as a new version (tensors are copied byte for byte). Folders saved
  with transformers 5 convert too (from 0.18.0).
- **Run evaluation** — both models score the reference model's answers token by token (KL divergence · top-1 agreement) and answer with
  their own chat templates (answers that differ · the first diverging step · finish reasons). Data with answers is graded too
  (accuracy with a 95 % interval): built-in `tasks40`, `gsm8k` (GSM8K test) and `mmlu-pro-500` (MMLU-Pro test), or your own JSONL.
  Rerunning with the same files and settings gives bit-identical results, so a before/after comparison shows only what you changed.
  With a baseline model (before fine-tuning) it also counts what the fine-tune learned and forgot (from 0.18.0).
- **Compare with the original model** — feeds the reference model's answers to the original (transformers) model and compares every
  position: near 0 means the conversion and the engine reproduce the original model.
- **Improve** — makes trial files with one matrix group (or a third of the layers) at a higher type, measures how much each improves
  the KL divergence (and the accuracy, with graded data) and how much the file grows, and remakes the version with the groups you
  choose.
- **Result table · Compare workspaces** — see the versions in one table and a size and quality chart, export them as CSV, and line up
  two fine-tunes type by type (from 0.18.0).

New to it? Start with the [model evaluation guide](model-evaluation-guide.md) — it follows one fine-tuned model through a full round.
The plan and the measurements behind these screens are in [문서/모델평가.md](../문서/모델평가.md) (Korean).

### 11. llama.cpp replay check — recompute a llama-server run bit for bit

The **llama.cpp replay check** (Reproduce › llama.cpp, from 0.22.0) takes a run that an **unmodified llama-server** produced and
recomputes it with GeulPU's own GPU kernels. Those kernels follow llama.cpp's CUDA arithmetic instruction by instruction, so the check
compares the generated tokens — and, when the client asked for them, the top-20 log-probabilities of every step — **bit for bit**. The
recomputation runs on CUDA or on Vulkan: a run made by llama.cpp on an RTX 4070 Ti was recomputed bit-identically on the Intel integrated
graphics of the same PC. What the replay imitates is **llama.cpp as it runs on an RTX 4070 Ti (sm_89, 60 SMs)**; runs made on other GPUs
are not accepted yet.

- **Supported** — llama.cpp b11963 (the build Ollama v0.40.14 launches), b12008 and b12138 (CUDA, Windows); Qwen3.5 (0.8B · 4B · 9B),
  Qwen3, Qwen2.5 (incl. Coder 14B), Llama 3.1 · 3.2, Mistral 7B and DeepSeek-R1-Distill-Qwen in Q4_K · Q5_K · Q6_K · Q8_0; `/completion`,
  the chat API, streaming, temperature 0 and seeded temperature > 0; the default server (4 slots, prompt cache, RAM cache), `-np N`, and
  overlapping requests when the server wrote a batch log. From 0.23.0 also token-less records of streamed responses and of seeded
  temperature > 0, answers ended by a stop string, repetition penalties and logit_bias, and imported branched conversations and
  request/response logs; server arguments and request fields that change the computation are checked against a fixed list and refused
  with a reason when they are not on it. A kit for recomputing on other GPUs (AMD and others, through Vulkan) is in
  [repro-kit/llama-replay](../repro-kit/llama-replay/).
- **Three ways to record** — ① **Import a conversation log**: a JSON file with the conversation the client sent directly to
  llama-server (messages with answers, and the sampling values) becomes one receipt per assistant message; nothing touches the run
  (temperature 0 only). ② **The transparent recorder** (default): point the client at `http://127.0.0.1:44544/llama/completion` or
  `…/llama/v1/chat/completions` instead of llama-server; GeulPU forwards each request byte for byte, returns the response unchanged, never
  queries the server during the run, and keeps a receipt. ③ **Probing mode** (opt-in): the recorder adds probability and token fields to
  the request and queries the slot state, so the check can compare log-probabilities bit for bit — it changes the run (even at
  temperature 0 the server's sampler ends differently), so choose it knowingly.
- **Verify** — save the server settings (the command line that started llama-server and its executable — the build is read with
  `--version` on save), pick a receipt and run it on CUDA or Vulkan. The job checks the model file, rebuilds missing prompt tokens with a
  temporary CPU-only llama-server of the same build, emulates the slot and prompt-cache state from the earlier receipts of the same
  server, recomputes and compares. The result is same or different (first diverging step, number of differing values), and the recomputed
  run is kept as a debugger run log with every step's probabilities and alternatives. Receipts it cannot verify are refused with a reason.

The record formats and what we need to accept runs from other GPUs are in [llama-replay.md](llama-replay.md); every stage with its
criteria and results is in [문서/라마재현.md](../문서/라마재현.md) (Korean).

### Use it from other programs (OpenAI-compatible API)

While a model is loaded in the app (sending one chat message loads it), use `http://127.0.0.1:44544/v1` as an OpenAI-compatible base
URL. Any `model` name works — the loaded model answers. The response's `model` echoes the name you sent, so check `/v1/models` to see
which model answered.

Paste this into a PowerShell window (it reads the response as UTF-8 so non-English text is not garbled; to save it as a `.ps1` file,
use "UTF-8 with BOM"):

```powershell
$body = @{ messages = @(@{ role = 'user'; content = 'Hello' }) } | ConvertTo-Json -Depth 4
$r = Invoke-WebRequest -Uri http://127.0.0.1:44544/v1/chat/completions -Method Post -ContentType 'application/json; charset=utf-8' -Body ([Text.Encoding]::UTF8.GetBytes($body)) -UseBasicParsing
([Text.Encoding]::UTF8.GetString($r.RawContentStream.ToArray()) | ConvertFrom-Json).choices[0].message.content
```

```python
from openai import OpenAI   # pip install openai
client = OpenAI(base_url="http://127.0.0.1:44544/v1", api_key="geulpu")   # any key
r = client.chat.completions.create(model="geulpu", messages=[{"role": "user", "content": "Hello"}])
print(r.choices[0].message.content)
```

- To turn thinking off on models that switch it (Qwen3, MiniCPM5 and, from 0.13.0, Gemma 4), add `"chat_template_kwargs": {"enable_thinking": false}` to the
  request (`extra_body=` in Python). Other values in `chat_template_kwargs` are passed to the chat template as they are.
- By default the thinking (Qwen3, DeepSeek-R1, MiniCPM5, Gemma 4) comes in front of the answer without tags. Add
  `"geulpu": {"special": true}` to receive `<think>` and `</think>` (for Gemma 4, `<|channel>thought` and `<channel|>`) as text, so you can
  separate the thinking from the answer. In requests that send `tools`, the
  thinking comes separately as `reasoning_content`.
- Only this PC can connect, and there is no authentication. Requests that other websites send through your browser are refused.
- `/v1/chat/completions` runs the model file's own chat template (the GGUF's `tokenizer.chat_template`) with a jinja interpreter that
  produces the same text as transformers, byte for byte. Files without a template get the basic ChatML template, as in llama.cpp.
- If the template uses a date (the Llama 3 system header, for example), it is fixed to `26 Jul 2024` so that the same question does not
  become a different prompt every day; change it with `"chat_template_kwargs": {"date_string": "16 Oct 2026"}` in the request. If the
  template does not render a system role (Mistral v0.3), a system message is put in front of the first question as `{system}\n\n` (the
  same form as llama.cpp).
- For Llama 3 and Mistral a BOS token is prepended to text prompts (the text of `/v1/completions` and the `prompt` of `/tokenize`), as
  llama.cpp and vLLM do — `/tokenize` leaves it out with `"add_special_tokens": false`. Prompts given as token ids are used as they are.

**Connecting a coding agent (tools)** — agents such as Qwen Code, OpenCode, Continue and Cline that call tools (read and write files, run
commands) through an OpenAI-compatible API can connect. From 0.14.0, agents that speak the Anthropic Messages API (`/v1/messages` —
Claude Code) or the OpenAI Responses API (`/v1/responses` — Codex CLI) connect too. All three APIs run through the same chat template
and engine, so the same request gives the same tokens (the two new APIs do not return thinking).

**Starting an agent from the app (from 0.14.2)** — on the **AI session** page (button above the console), pick an agent and a working folder and press
**Start**: the app opens a new terminal window with that agent attached to the loaded model (Claude Code, Codex CLI, Qwen Code, OpenCode,
Aider). The settings are passed to that run only, so the agent's configuration files and your environment variables stay as they are,
and the agent itself asks before it edits files or runs commands. For an agent that is not installed the page shows its install
command; **Show settings** shows the commands to run it yourself (PowerShell). For Cline and Kilo Code (VS Code extensions) it shows the
values to enter in the extension's settings. With a context below 32k or more than one concurrent chat it suggests **Reload for agents**
(one concurrent chat, context 131072 — the engine shrinks it to fit the graphics memory). On Linux the app shows the commands but does
not open a window yet. The connection examples below are for setting an agent up by hand.

![AI session page — the model, the working folder and, for each agent, whether it is installed and a Start button](img/agents-en.png)

- The address is `http://127.0.0.1:44544/v1`; any API key and any model name work (the model loaded in the app answers — load a model
  in the app before using the agent).
- Models that take tools: those whose chat template renders tools and whose call format is one of the five GeulPU reads (Qwen's
  `<tool_call>` JSON, Llama 3's JSON, the XML of Qwen3.5 and Qwen3-Coder, MiniCPM's XML and, from 0.13.0, Gemma 4's
  `<|tool_call>call:name{…}`) — Qwen2.5, Qwen3, Qwen3.5, Llama 3.1 · 3.2, MiniCPM5 and Gemma 4 (from 0.13.0).
  Prompts with tools are also the model file's own chat template. Mistral, DeepSeek-R1 and HyperCLOVAX-SEED have no tools in their
  templates, so they refuse them (400). In tests Qwen2.5 1.5B and 3B often wrote the call's JSON wrongly (`{{…}}`), and
  Qwen2.5-Coder-14B wrapped its calls in `<tools>` instead of `<tool_call>`, so they were not read as calls (the prompts match the
  official template — this is the models' own behaviour).
- In tests Qwen2.5-7B and Qwen3-8B finished the tasks; Llama 3.x sometimes repeated the same call after getting the tool result — Qwen is
  recommended for agents. `tool_choice` accepts `auto` (default) and `none`, and from 0.14.0 `required` (any call) and a named function
  (`{"type": "function", "function": {"name": …}}`) — the server places the model template's call opening ahead, so the same request
  gives the same tokens (MiniCPM's XML call form is not supported yet — 400).
- **The bigger the context, the better.** An agent's base instructions alone are 10,000–15,000 tokens (Qwen Code: 14,501), and they
  grow as it works. Feature-level development (for example planning, reviewing, implementing and reporting a login feature) without
  compaction needs 100k tokens or more. Set concurrent chats to 1 and the context length per chat high (e.g. 131072) in Settings; the
  engine reduces it to the model's maximum and to the graphics memory. For Qwen2.5 7B and larger and Qwen3, a context beyond the
  model file's position count (32,768 · 40,960) turns on **YaRN** (the position scaling Qwen recommends — factor 2 or 4) up to 131,072;
  the model list shows 'YaRN' next to the context. The factor follows the context you set (2 up to 65,536, 4 above), not the context
  the engine ends up with after fitting the graphics memory, so the same settings always give the same bits. Files with fewer than 32,768
  positions are not scaled. With a scaled context, short texts are computed the same way too (static YaRN, as in other engines), so set a
  large context only when you need long texts ([문서/긴문맥.md](../문서/긴문맥.md), Korean). Maximum reached on an RTX 4070 Ti (12 GB):

  | Model | Max context | Limit |
  |---|---|---|
  | Qwen2.5-7B Q4_K_M | 87,488 (YaRN — 47,808 with two concurrent chats) | graphics memory |
  | Qwen2.5-7B · Coder-7B Q8_0 | 38,784 (YaRN) | graphics memory |
  | Qwen3-8B Q4_K_M | 35,328 | graphics memory |
  | Llama-3.1-8B Q4_K_M | 39,936 | graphics memory |
  | Llama-3.2-3B Q8_0 | 57,984 | graphics memory |
  | Qwen2.5-14B Q4_K_M | 8,448 | graphics memory |

- When the context overflows, the server answers with an OpenAI-style error (`code: context_length_exceeded`, with the token counts) so
  the agent can shorten the conversation itself. `max_model_len` in `/v1/models` is the current context.
- Example (Qwen Code): set the environment variables `OPENAI_BASE_URL=http://127.0.0.1:44544/v1`, `OPENAI_API_KEY=geulpu` and
  `OPENAI_MODEL=qwen2.5-7b-instruct-q4_k_m`, then run `qwen` in your project folder.
- Example (Claude Code — from 0.14.0): set `ANTHROPIC_BASE_URL=http://127.0.0.1:44544`, `ANTHROPIC_AUTH_TOKEN=geulpu`,
  `ANTHROPIC_MODEL=geulpu-local` and `ANTHROPIC_DEFAULT_HAIKU_MODEL=geulpu-local`, then run `claude` in your project folder. Any model
  name works (the model loaded in the app answers); Claude Code notes that it does not know the name and carries on.
- Example (Codex CLI — from 0.14.0): put the lines below in `~/.codex/config.toml`, set `GEULPU_API_KEY` to anything and run `codex`.
  Set `model_context_window` to the current context (`max_model_len` in `/v1/models`). The Responses API is served without state
  (`previous_response_id` gets 400), so Codex sends the whole conversation, and hosted tools such as web_search are dropped. In tests a 7B
  model rarely called Codex's tools — a larger model is recommended.

  ```toml
  model = "geulpu-local"
  model_provider = "geulpu"
  model_context_window = 131072

  [model_providers.geulpu]
  name = "GeulPU"
  base_url = "http://127.0.0.1:44544/v1"
  wire_api = "responses"
  env_key = "GEULPU_API_KEY"
  ```

- **Tool errors (from 0.14.0).** When a conversation from an agent contains a failed tool result (a missing file, a failing command and
  the like), the server finds it, classifies it (transient, fixable, unrecoverable or ambiguous) and records it in the server log
  (`[n] 도구 오류 감지 …`) and in the `X-GeulPU-Tool-Errors` response header. The text sent to the model is unchanged. For research, to
  show the failed result the agent just received in one of the forms measured in the error-delivery study (C1, C2, C3 —
  [문서/에이전트.md](../문서/에이전트.md), Korean), choose **Tool-error form** in Settings (off, C1, C2, C3 — from 0.14.2, applied the
  next time a model is loaded) or add `"geulpu_tool_errors": "c3"` to the request (`"off"`, `"c1"`, `"c2"`, `"c3"` — the request's value
  wins; off by default). When off, replies are bit-identical. The AI session page lists recently detected tool errors. From 0.15.0, tool
  results from clients outside the known agents (custom agents, LangChain, the OpenAI Agents SDK, MCP clients) are detected too when they
  start with a standard error text or carry `is_error` / MCP `isError`.
- **Unbroken tool calls (from 0.15.0).** While the model writes a Qwen-style tool call (`<tool_call>` … `</tool_call>`), a special token
  such as `<|im_start|>` is skipped and the next candidate is used, so the call comes out whole (the same effect as llama.cpp's tool-call
  grammar — contract section 5). Only such requests reply differently from 0.14; every other reply is bit-identical. Each skip writes one
  server log line.

**Log-probabilities · scoring · tokenization (for research)** — every value on the run, step and layer screens is also available from the API.
Contract 5.2 ([contract](../계약/contract.md)) defines the values, so they are bit-identical on any PC or GPU.

- `/v1/chat/completions`: `"logprobs": true, "top_logprobs": 5` (0–20), in the OpenAI shape (`token` · `logprob` · `bytes` ·
  `top_logprobs`).
- `/v1/completions`: `"logprobs": 5`, `"echo": true` (log-probabilities of the prompt tokens too), `"max_tokens": 0` (score only, no
  generation). `prompt` is text or an array of token ids.
- The `"geulpu"` extension: `"logit_hash": true` (a 64-bit hash of each step's logit row), `"prompt_tokens": true` (the prompt tokens
  after the chat template), `"echo_from": n` (echo log-probabilities from the token at index n on — 0-based; earlier ones are null), `"step_offset": n` (start the
  step index of the sampling random numbers at n — a branch continues with the same random numbers as the rest of the original answer),
  and `entropy` · `top_ids` · `top_logprobs` (aligned with the ids) · `seed` in the response.
- Layer lens (from 0.13.0 — [precision contract 5.3 · 5.4](../계약/contract.md)): add `"geulpu": {"echo_from": n, "layer_lens": true}` to
  an echo request (`"echo": true` · `"max_tokens": 0` · `"logprobs": k`) to read the row that scored token n at every layer with the final
  logit computation; `geulpu.layer_lens` holds each layer's log-probability, entropy, top k, logit hash and rank (the logit lens — Qwen2.5,
  Qwen3 and Llama architectures). With a J file (`<name>.jlens.safetensors`) next to the model file, `"layer_lens": {"jacobian": true}`
  also returns the same row's Jacobian lens as `geulpu.jacobian_lens` — see
  [experiment 8 of the research guide](research-guide.md#experiment-8-at-which-layer-does-the-answer-appear--the-logit-lens).
- `/tokenize` · `/detokenize` · `/tokenizer_info` — in the same shape as vLLM.
- In `geulpu`, `tokens` and `logit_hashes` include the step that sampled the stop token (EOS), while the log-probability arrays
  (`logprobs`, `entropy`, `top_ids`, `top_logprobs`) cover only the answer's tokens, as in OpenAI — one shorter when the answer ends with
  EOS (score `tokens` with echo to get the EOS step's probability). Despite its name, `first_token_ms` is in µs (kept for compatibility).
- Asking for log-probabilities during generation adds about 2% per token (7B · 8B — the computation hides behind the next step's GPU
  work). Without them, speed is unchanged.

```python
# Log-probability of each token of a text (scoring) — the first token has no context, so it is null
r = client.completions.create(model="geulpu", prompt="The capital of France is Paris.", echo=True, max_tokens=0, logprobs=5)
print(list(zip(r.choices[0].logprobs.tokens, r.choices[0].logprobs.token_logprobs)))
```

**lm-evaluation-harness** — it connects as `local-completions`. It uses the server's tokenizer (`tokenizer_backend=remote`), so no
Hugging Face tokenizer is downloaded. Per-item results are bit-identical whatever `num_concurrent`, the number of concurrent chats or
CUDA ↔ Vulkan ([stage 38](https://github.com/wwoosshh/GeulPU/blob/main/문서/38단계.md)). Raise the number of concurrent chats in **Settings** (up to 8).

```bash
pip install "lm-eval[api]"
lm_eval --model local-completions --tasks arc_easy,gsm8k --model_args model=geulpu,base_url=http://127.0.0.1:44544/v1/completions,tokenizer_backend=remote,tokenized_requests=True,num_concurrent=8
```

## Supported models

GeulPU takes a single GGUF file (merge split files with llama.cpp's `llama-gguf-split --merge`).

| Architecture | Models | Sizes |
|---|---|---|
| qwen2 | Qwen2.5-Instruct · Qwen2.5-Coder · DeepSeek-R1-Distill-Qwen and their fine-tunes | 0.5B and up (head size 64 · 96 · 128) — 0.5B · 1.5B · 3B · 7B · 14B run on this project's PC (head size 96 on a synthetic model) |
| qwen3 | Qwen3 and its fine-tunes | 0.6B · 1.7B · 4B · 8B · 14B · 32B — 0.6B · 1.7B · 4B · 8B run on this project's PC (14B on a synthetic model) |
| llama | Llama 3.1 · 3.2 Instruct, Mistral 7B Instruct v0.3, DeepSeek-R1-Distill-Llama, HyperCLOVAX-SEED, MiniCPM5, and their fine-tunes | Head size 64 · 96 · 128, including RoPE on part of each head — 1.5B · 2B · 3B · 7B · 8B run on this project's PC; head size 96, partial RoPE and 70B checked as shapes with synthetic models |
| qwen35 | Qwen3.5 (full-attention and linear-attention layers mixed) · Qwen3.8-27B | 0.8B (Q8_0) · 4B · 9B (Q4_K_M) run on this project's PC. From 0.13.0: Fast mode, and Qwen3.8-27B (the ggml-org and lmstudio Q4_K_M files, which mix formats from matrix to matrix — too large for this PC, checked bit for bit with synthetic files of the same shape). CUDA only, without speculative decoding or the context store |
| qwen35moe | Qwen3.5 · 3.6 MoE (Qwen3.5 layers with mixture-of-experts FFNs) | Qwen3.6-35B-A3B (Q4_K_M) runs on a 12 GB card — experts that do not fit in graphics memory stay in RAM (about 10.5 GB); CUDA and Exact mode for now |
| gemma4 | Gemma 4 (sliding-window and full-attention layers mixed) — from 0.13.0 | 12B instruct (Q4_K_M) runs on this project's PC — files whose matrices are all k-quants or all Q8_0 (Q8_0 checked bit for bit on a file with fewer layers). CUDA and Exact mode only, without speculative decoding or the context store |

The chat template is run from the model file as it is, so fine-tunes may change it (only templates GeulPU's jinja interpreter cannot
parse are refused — it parsed all 228 templates of the 1,000 most-downloaded GGUF repositories on Hugging Face).

- **Formats**: Q8_0 · Q4_K_M · Q5_K_M · Q6_K · IQ4_XS, Q4_0 files whose tensors are all Q4_0, and unquantized F16 · BF16 · F32 files
  (each matrix is converted on load to the 16-bit block format of the precision contract — F16 values of at least 1/8 of their block's
  maximum stay exact; in fast mode the activations are 8-bit, which loses more against llama.cpp on these files than on quantized ones).
- **Licenses** differ by model. In the app's list, Llama 3.1 · 3.2 use Meta's community licenses (usage conditions and an Acceptable
  Use Policy — services with more than 700 million monthly users need permission from Meta), the Mistral 7B v0.3, Qwen2.5 (except 3B),
  Qwen3 and MiniCPM5 files are Apache 2.0, Qwen2.5-3B has the Qwen Research License (research and evaluation only — commercial use needs a
  license from Alibaba Cloud), DeepSeek-R1-Distill-Qwen is MIT (R1-Distill-Llama also follows Llama 3.1's license), and HyperCLOVAX-SEED has NAVER's own
  HyperCLOVA X SEED license; the Qwen3.5, Qwen3.6 and Gemma 4 files are Apache 2.0 (also shown in the app's model descriptions).
- **Not yet**: other architectures (Phi, …), Gemma 4's E2B · E4B (per-layer embeddings), 26B-A4B (MoE) and 31B and its Q4_0 files, Mistral
  NeMo · Small (not run yet — their tekken tokenizer is accepted from 0.12.0), small models with head size 64 (Qwen2.5-0.5B, Llama 3.2 1B),
  other MoE models (Mixtral, Qwen3-30B-A3B, …), image input, Q2_K · Q3_K · IQ formats other than IQ4_XS · Q5_0. Find models hides such
  files by default.
- **Depending on the GPU**: Q8_0 · Q4_0 and k-quants (Q4_K_M, …) run on CUDA and on every Vulkan GPU, including those with 32 KB of shared memory
  such as AMD's Windows driver (from 0.12.0 — checked with that limit emulated on an RTX 4070 Ti and on an Intel UHD Graphics 770). F16 · BF16 · F32
  run on CUDA and on Vulkan GPUs with at least 48 KB of shared memory. IQ4_XS, Qwen3.5, Qwen3.6 MoE and Gemma 4 run only on CUDA (NVIDIA
  GPUs) for now.

### Which model fits your PC

Files GeulPU has run itself (**Verified by GeulPU** in the app). Sizes use 1024-based units, as Windows Explorer does.

| Graphics memory | Suggested files |
|---|---|
| 4 GB | Qwen3-0.6B Q8_0 (0.6 GB) · Qwen3.5-0.8B Q8_0 (0.8 GB) · HyperCLOVAX-SEED-Text-Instruct-1.5B Q4_K_M (1.1 GB) · MiniCPM5-2B Q4_K_M (1.5 GB) |
| 6 GB | Qwen3-1.7B Q8_0 (1.7 GB) · Qwen2.5-1.5B-Instruct Q8_0 (1.8 GB) · Qwen3-4B Q4_K_M (2.3 GB) · Qwen3.5-4B Q4_K_M (2.6 GB) · Llama-3.2-3B-Instruct Q8_0 (3.2 GB) · Qwen2.5-3B-Instruct Q8_0 (3.4 GB) |
| 8 GB | Qwen3-8B Q4_K_M (4.7 GB) · Qwen3.5-9B Q4_K_M (5.3 GB) · Qwen2.5-7B-Instruct Q4_K_M (4.4 GB) · DeepSeek-R1-Distill-Qwen-7B Q4_K_M (4.4 GB) · DeepSeek-R1-Distill-Llama-8B Q4_K_M (4.6 GB) · Llama-3.1-8B-Instruct Q4_K_M (4.6 GB) · Mistral-7B-Instruct-v0.3 Q4_K_M (4.1 GB) · Qwen3-4B Q8_0 (4.0 GB) |
| 12 GB | Qwen3-8B Q8_0 (8.1 GB) · Qwen2.5-7B-Instruct Q8_0 · Qwen2.5-Coder-7B-Instruct Q8_0 · DeepSeek-R1-Distill-Qwen-7B Q8_0 (7.5 GB) · Llama-3.1-8B-Instruct Q8_0 (8.0 GB) · Mistral-7B-Instruct-v0.3 Q8_0 (7.2 GB) · Gemma-4-12B-it Q4_K_M (6.6 GB — from 0.13.0, context about 7,300) · Qwen3.6-35B-A3B Q4_K_M (19.0 GB — about 10.5 GB of RAM for the experts that do not fit in graphics memory) — 14B Q4_K_M also runs with a reduced context |
| 16 GB or more | Qwen2.5-14B-Instruct Q4_K_M · Qwen2.5-Coder-14B-Instruct Q4_K_M · DeepSeek-R1-Distill-Qwen-14B Q4_K_M (8.4 GB) |

Qwen3-14B Q4_K_M and Qwen3.8-27B have only been checked bit for bit on synthetic models of the same shape and have not been run yet.
Qwen3.5 and Gemma 4 run only on CUDA. The app tells you for every file whether it fits your graphics memory.

## How it runs

```mermaid
flowchart LR
    B["Browser<br/>127.0.0.1:44544"] -- "UI · /api · /v1" --> M["GeulPU.exe<br/>manager"]
    M -- "first use of a model" --> G["kernel-gen.exe<br/>ptx-backend.exe · spirv-backend.exe"]
    G -- "kernels for this GPU" --> K[("module folder")]
    M -- "starts it, forwards /v1" --> S["server-cuda.exe<br/>server-vulkan.exe"]
    K --> S
    S -- "nvcuda.dll · vulkan-1.dll" --> GPU[("GPU")]
```

1. The **manager** (`GeulPU.exe`) serves the UI on `127.0.0.1:44544` and opens your browser. It listens only inside this PC; if it is
   already running on that port, it just opens the page. 44544 is the code point of '글' (U+AE00).
2. It first checks the GPU — CUDA (`nvcuda.dll`) on an NVIDIA RTX 30 or newer, otherwise Vulkan (`vulkan-1.dll`). You can change this
   in Settings.
3. The first time a model is used, it **builds kernels**: the kernel generator (`kernel-gen.exe`) writes GPU kernel source in geul for
   the model's architecture, shape and format, and the geul compiler's backends (`ptx-backend.exe`, `spirv-backend.exe`) turn it into
   PTX (NVIDIA) or SPIR-V (Vulkan) in the data folder — once per app version.
4. The **inference server** (`server-cuda.exe` or `server-vulkan.exe`) loads the GGUF into graphics memory and runs those kernels
   directly through the graphics driver. No CUDA toolkit, runtime or Python is needed.
5. Chat goes browser → manager → inference server (an OpenAI-style API), and the answer streams back token by token.
6. Closing the console window stops the manager together with every process under it (server, tools).

The manager, the inference server, the kernel generator and the backends are all written in geul; they import only Windows DLLs (the
manager: kernel32, ws2_32, shell32, winhttp, bcrypt, advapi32) and the graphics driver's DLL.

**Inside the zip**

| File | What it does |
|---|---|
| `GeulPU.exe` | The manager — UI, model download and verification, starting the server and tools |
| `bin\server-cuda.exe` · `bin\server-vulkan.exe` | Inference server (OpenAI-style HTTP API) |
| `bin\kernel-gen.exe` | Kernel generator — kernel source (geul) for a model's shape |
| `bin\ptx-backend.exe` · `bin\spirv-backend.exe` | The geul compiler's backends — kernels to PTX and SPIR-V |
| `bin\device-*.exe` · `bin\diag-*.exe` · `kernels\` | GPU detection and hardware diagnosis |
| `web\` | The UI (HTML, CSS, JavaScript) |
| `data\` | Catalog of downloadable models · reproducibility-check reference |
| `version.json` | Version, commit and the SHA-256 of every file |

**Data folder** — `%USERPROFILE%\GeulPU` (kept across versions)

| | |
|---|---|
| `모델\` (models) | Downloaded models (GGUF) |
| `모듈\<version>\` (modules) | Kernels built for this GPU |
| `결과\` (results) | Reproducibility check and speed test results |
| `기록\` (logs) | Server and job logs — look here when something goes wrong |
| `설정.json` · `모델목록.json` · `허깅목록.json` | Settings · your model list · files picked on Hugging Face |

To move the data folder, set the environment variable `GEULPU_HOME` or run `GeulPU.exe --자료 <folder>`. Other options:
`--포트 <number>` (another port) and `--열지않음` (do not open the browser).

**Network and privacy**

- The manager and the inference server listen only on `127.0.0.1` — other devices cannot connect.
- The internet is used for three things only: Find models (the page reads the Hugging Face API directly), downloading models
  (huggingface.co) and the new-version check (GitHub — switch off **New-version notice** in Settings).
- No usage statistics are sent. Conversations stay in your browser's storage.
- There is no installer, and nothing is written to the registry.

**Build and trust**: release zips are built from this repository's source by GitHub Actions when a tag is pushed (`앱/꾸리기.ps1` — it
builds the geul compiler to its self-hosting fixed point, then the backends, servers, tools and the manager). The server, generator and
backends are packaged only after their bytes are checked against the build hashes of the reproducibility reference
(`교차검증/expected.json`).

## Troubleshooting

| Symptom | What to do |
|---|---|
| The browser does not open | Type `http://127.0.0.1:44544` into the address bar. |
| "Windows protected your PC" | **More info → Run anyway** (the executable is not code-signed). |
| Antivirus blocks it | Add an exception for the extracted GeulPU folder. The source and the build process are public in this repository. |
| Port 44544 is taken by another program | Start it on another port, e.g. `GeulPU.exe --포트 44600`. |
| "No usable GPU" | Update the graphics driver, then **Hardware → Check again**. |
| "Not supported on this GPU" · "Not enough VRAM" | Choose a smaller file (Q4_K_M) or a smaller model. A shorter context per chat or fewer concurrent chats in Settings also use less memory. |
| The first message takes long | The first time builds kernels (1–3 minutes). After that, only loading the model (tens of seconds). |
| A download stopped | **Resume download** under My models or in the status bar — it continues where it stopped. |
| Part of the screen is in Korean | Messages from the engine are translated when they are shown; raw logs, tool output and result files stay in Korean. Paths such as `%USERPROFILE%\GeulPU\모델\` are real folder names. |
| Anything else | Open an [issue](https://github.com/wwoosshh/GeulPUv2/issues) and attach the logs from `%USERPROFILE%\GeulPU\기록\`. |

## Update and uninstall

- **New version**: the app tells you when GitHub has a new release. Quit the app, unzip the new zip and run its `GeulPU.exe`. Models,
  settings and results stay in the data folder, so nothing is downloaded again (the kernels are rebuilt once for the new version).
  Changes are listed in the [changelog](../CHANGELOG.md).
- **Uninstall**: quit the app and delete the GeulPU folder. To delete the downloaded models as well, also delete
  `%USERPROFILE%\GeulPU`.
- **Installed with a Python installer**: to update, quit the app, then `pipx upgrade geulpu` (`uv tool upgrade geulpu` ·
  `pip install -U geulpu`); to uninstall, `pipx uninstall geulpu` (`uv tool uninstall geulpu` · `pip uninstall geulpu`) — the data
  folder stays.

## What is different

- **Precision is a written contract.** Activations are integers per 32-value block — two balanced base-256 digits (±2¹⁴ of the block
  maximum) in the exact mode; in the fast mode (the default — [stage 2](../문서/2단계.md), Korean) only the activations entering the matrix
  multiplications have one digit (8 bits) —, the KV cache is two-digit block integers, products are summed exactly on integer tensor
  cores, and exp is a written polynomial. Every rounding point is fixed in the source, so in both modes the GPU result is
  **bit-identical** to the CPU reference implementation ([`계약/정수계약.py`](../계약/정수계약.py)). The only losses are the model file's own
  quantization and the roundings the contract writes down — see [docs/contract.md](../계약/contract.md).
- **The same bits whichever way it computes.** Feeding a prompt at once, in pieces, one token at a time, as a continuing conversation,
  or batched with other sentences gives identical results. Common engines change output bits with the amount of work computed
  together: llama.cpp's wikitext-2 perplexity varies from 15.8171 to 15.8200 with batch settings, while GeulPU gives the same value to
  the last bit for every way of feeding the text (stage 20). Across GPUs too: an RTX 3080 Ti (Ampere), an RTX 4070 Ti (Ada) and an RTX 5070
  (Blackwell) gave bit-identical outputs and per-step logits (82,673 steps) on 32 math problems with DeepSeek-R1-Distill-Qwen-7B, while llama.cpp's
  outputs all changed with the number of concurrent requests on one PC ([교차검증](../교차검증/)). Evaluation tools too: lm-evaluation-harness
  per-item results (arc_easy 500, gsm8k 100) were identical across num_concurrent 1 ↔ 8 and CUDA ↔ Vulkan (with the same setup,
  llama.cpp b11643 changed the generated text of 41 of 100 gsm8k items between 1 and 8 concurrent requests — [stage 38](https://github.com/wwoosshh/GeulPU/blob/main/문서/38단계.md)).
- **Speculative decoding gives the same answers.** Thanks to the property above, the logits of a verification step (several rows)
  are bit-identical to computing the tokens one by one, so greedy decoding and sampling (random numbers fixed by the step number)
  produce the same tokens — with speculation on, the tokens, per-step logit hashes and log-probabilities equal those with it off
  (tests: seventeen requests × correct, wrong and half-wrong drafts, four concurrent slots, CUDA and Vulkan — [stage 3](../문서/3단계.md)).
  In the same measurement, llama.cpp b11721's llama-server changed the answers of two of eight greedy requests (summary, math) when
  `--spec-type ngram-simple` was switched on.
- **The prompt cache gives the same answers.** KV restored from the SSD is byte-identical to recomputed KV (the contract — KV is the same
  whichever way it is filled), so with the store on, the tokens and per-step logit hashes equal those with it off. Tests: 54 recorded Qwen
  Code requests sent with recomputation · store off · empty store · after a server restart · one slot alternating conversations · 40
  corrupted chunks — all 1,728 steps identical; a Vulkan server reading chunks written by CUDA matched too. So did a replay that
  generates the recorded answer lengths (19,552 steps), a new task, reopening a 31k-token conversation and rewinding it (from the OS
  cache and from a cold SSD), and switching between more conversations than slots ([문서/문맥저장소.md](../문서/문맥저장소.md), Korean).
- **The exact mode's hidden loss is small — measured on 0.5B.** Qwen2.5-0.5B-Instruct (Q8_0): the hidden loss (KL against an fp64 forward pass of the same
  weights) is 1/615 of llama.cpp's (a 461-token evaluation text) and 1/385 (wikitext-2, 148,920 tokens); its perplexity is
  indistinguishable from fp64 (relative −6.6 × 10⁻⁶), while llama.cpp's is 0.34% higher — the same order as the Q8_0 weight quantization
  itself (0.24%). How this was measured: [docs/hidden-loss.md](https://github.com/wwoosshh/GeulPU/blob/main/docs/hidden-loss.md). 7B and 14B were not measured with the real engine
  (see the simulation below). The fast mode keeps the hidden loss at llama.cpp's level — measured against the exact mode, its
  next-token KL is 0.65–0.98× llama.cpp's on k-quant 7B · 8B models and 2.2–2.4× on Q8_0 ([stage 2](../문서/2단계.md)).
- **Speed — with the default settings faster than llama.cpp at every length measured.** On one RTX 4070 Ti:
  - Default settings since 0.11.0 (fast mode with the 8-bit KV cache and fast attention — [문서/0.11.0.md](../문서/0.11.0.md) step 1-5,
    Korean; latest official llama.cpp b11827, same GGUF, the faster of its f16 and q8_0 KV caches): for Qwen2.5-7B Q4_K_M, empty-cache
    prompts of 512 / 2,048 / 8,192 / 32,768 / 65,536 tokens take 0.83 / 0.87 / 0.90 / 0.95 / 0.92× llama.cpp's time, 1,024 new tokens
    at session depth 8k / 32k / 64k 0.90 / 0.92 / 0.76×, and decoding at those depths 0.97 / 0.89 / 0.82×; for Llama-3.1-8B Q4_K_M the
    prompts take 0.85 / 0.88 / 0.92 / 0.92× (512–32,768), new input at depth 32k / 64k 0.63 / 0.48× and decoding there 0.90 / 0.80×. A
    recorded Qwen Code session (54 requests, Qwen2.5-7B Q4_K_M) takes 102.3 s adjusted total against 131.6 s for 0.10.2's defaults and
    173.3 s for llama.cpp's fastest setting (a Qwen2.5-0.5B draft model); its prompt processing takes 24.0 s (llama.cpp: 28.4 s with
    f16, 28.1 s with q8_0). The exact mode took 1.19–1.40× llama.cpp's time on prompts up to 8k tokens in the same run (1.74× at 32k). The
    items below are earlier measurements, kept for the record.
  - Fast mode (stage 4 — latest official llama.cpp b11721, same GGUF, median of three alternating rounds, llama.cpp at its fastest
    setting per length): prompt time is 0.89 / 0.90 / 0.95 / 1.04× llama.cpp's for Qwen2.5-7B Q4_K_M, 0.74 / 0.76 / 0.81 / 0.92× for Q8_0
    and 0.92 / 0.93 / 0.99 / 1.11× for Llama-3.1-8B Q4_K_M (512 / 2048 / 4096 / 8192 tokens — [stage 4](../문서/4단계.md)). The slowdown at
    long contexts is the attention, which still has two digits. The exact mode took 1.16–1.47× in the same run.
  - Long sessions ([long-context measurement](../문서/긴문맥재기.md), fast mode): filling an empty cache with 16k / 32k tokens takes 1.19 / 1.40×
    llama.cpp's time (Qwen2.5-7B Q4_K_M) and 1.29 / 1.40× (Llama-3.1-8B Q4_K_M); a new 1,024-token input at session depth 8k / 16k / 31k
    takes 1.18 / 1.40 / 1.57× and 1.29 / 1.44 / 1.08× — the attention costs 1.9–2.4× llama.cpp's per token of depth. Decoding at the same
    depths is 1.01–1.06×. Context goes up to 87,488 for Qwen2.5-7B Q4_K_M with YaRN (③a — 50 s to the first token for 80k tokens,
    decoding at 80k depth 38.8 ms/token) and to about 33k for Llama-3.1-8B on 12 GB.
  - Speculative decoding (stage 3 — on by default; Qwen2.5-7B Q4_K_M, fast mode, one slot, median of three rounds, generated tokens
    per second): editing code 329 / 356, rewriting a document 389, a quoting summary 347 — 3.5–4.1× the 95 with speculation off and
    1.2–2.5× llama.cpp b11721's fastest speculation (ngram-simple: 266 / 245 / 188 / 139). Ordinary chat: greedy 112, sampling 97,
    math 144; translation 96 (no speculation: 95 — llama.cpp also 95).
  - Qwen2.5-7B / 14B (Q8_0 · Q4_0 · k-quants) decode at parity with llama.cpp (0.970–1.012× its time), 1.11–1.17× the bandwidth floor
    (llama.cpp: 1.13–1.19×). Decoding is bound by reading the weights, so the extra arithmetic hides behind it. Fast-mode decoding
    takes 0.993–0.998× the exact mode's time (stage 2 — short context, 2048 and 8192).
  - Exact-mode prompts were faster only when short (stage 30): 7B Q4_0 took 0.85× (32 tokens), 1.03× (128), 1.11× (2048) and 1.27× (8192) the time of the latest
    official llama.cpp release (b11567, same GGUF), Q8_0 0.85–1.30× and k-quants 0.95–1.56× (same run, stage 30). At equal optimization the two-digit activations double
    the integer tensor-core work of llama.cpp's 8-bit activations, and the attention's exact sums and written exp add more; the ratio grows
    with length because of the attention.
  - The 0.5B results (prompts of 128–8192 tokens at 0.55–0.82× llama.cpp's time, decoding at 0.63–0.65× with a generation megakernel)
    and GPT-2 XL (prompts at 0.78–0.87×) come from kernel engineering — fusion and a megakernel — not from the contract being free.
  - The old motto's "it is not slower" was withdrawn in stage 30. The price at equal optimization was measured on 2026-10-13 by switching
    constraints off in the same engine (in a scratch folder; the raw data is not in this repository). Two activation digits → one in the
    tile kernels: 7B Q4_0 prompts of 512 / 2048 / 8192 tokens took 1.38 / 1.35 / 1.26× (a lower bound: epilogues and the attention's q
    keep two digits). Written exponent → approximate exponent: almost free. Decode: about 1.0× (weight reads dominate). Determinism alone
    could not be isolated. Stage 33 (a formal re-measurement) was stopped for that reason.
  - Stage 34, Vulkan speed — the emulated integer tensor op became cooperative-matrix tensor-core work in the tile kernels,
    attention and the decode line kernels; the SPIR-V backend's division and subnormal flushing (no re-flush of values already flushed), the
    Vulkan layer's barriers and the tile kernels' loading were fixed. All bit-exact (CPU reference, CUDA records byte for byte). Measured alternately in one session, 7B on Vulkan takes 1.19–1.27×
    the CUDA path's prompt time (stage 31: 4.3–5.5×) and 1.00–1.06× its decode time (stage 31: 1.09–1.27×; short-context decode varies
    about 3% between runs); prompt tile kernels and short prompts (128 rows: 1.39–1.47×) remain ([stage 34](https://github.com/wwoosshh/GeulPU/blob/main/문서/34단계.md), Korean). Stage 31 (Vulkan): the geul host runs on vulkan-1.dll alone and reproduces the CUDA records byte for byte on NVIDIA and on Intel
    integrated graphics; on a friend's AMD Radeon RX 9060 XT (RDNA4) the 7B R1 run matched the NVIDIA CUDA reference in all 82,673 steps
    (stage 34). Stage 32 (self-tuning to the device's floor) waits for runs on
    friends' PCs. See the [research log](https://github.com/wwoosshh/GeulPU/blob/main/docs/research-log.md).
  - Now: stage 37, Qwen3 — a one-run preview of Qwen3-8B decoding took 0.982–1.007× llama.cpp's time (b11567) for Q8_0 and 1.037–1.055×
    for Q4_K_M; the pre-registered speed criterion (37c, ≤ 1.05×) was not measured officially ([stage 37](https://github.com/wwoosshh/GeulPU/blob/main/문서/37단계.md), Korean).
- **Larger models keep the advantage.** A simulation that reproduces only the rounding points (validated against the measured 0.5B value:
  1/637 simulated vs 1/615 measured) gives a hidden loss of 1/3805 (7B Q8_0) and 1/4467 (14B Q4_0) of a llama.cpp-like pipeline.

