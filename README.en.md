# GeulPU — local AI chat on your own graphics card

*한국어: [README.md](README.md). The code and most stage reports are written in Korean — see the [glossary](docs/glossary.md) for the
terms; English documents live in [docs/](docs/).*

> This repository (GeulPUv2) continues the engine and app of the research repository [GeulPU](https://github.com/wwoosshh/GeulPU)
> (research frozen on 2026-10-16, app 0.5.0) for **everyday use** — performance, stability and features. Every feature is kept
> (including the research panel and the cross-check); the research record (stage reports, paper) stays in GeulPU. What was moved:
> [문서/이관범위.md](문서/이관범위.md) (Korean); what changed per version: [문서/바뀐것.md](문서/바뀐것.md) (Korean).

GeulPU is a Windows app that runs language models such as Qwen3, Qwen2.5, DeepSeek-R1, Llama 3 and Mistral **on your own GPU**. There is no installer:
unzip it, run `GeulPU.exe`, and a chat page opens in your browser. Models are downloaded from Hugging Face inside the app, and your
conversations never leave the PC.

The engine is written in **geul (글)**, a Korean programming language — from the GPU kernels to the inference server and the app. It
runs on the graphics driver alone, without the CUDA toolkit, Python or PyTorch, and its precision is a written contract, so its
results are **bit-identical** to a CPU reference implementation ([What is different](#what-is-different)). So scoring the same text
again gives the same probabilities on any PC or GPU, and the **research panel** lets you inspect every token's probability, branch
with one token changed, compare against another model or prompt, and check someone else's result to the bit
([research panel](#5-research-panel--per-token-probabilities--intervene--compare--reproduce)).

> It loses only what is written down — and it shows what losing less costs.

**[⬇ Download the latest version — Windows 64-bit, zip of about 1.4 MB](https://github.com/wwoosshh/GeulPUv2/releases/latest)** · with Python, one line: `pipx install geulpu` ([install](#with-python--one-command))

[Install](#install) · [First steps](#first-steps) · [Supported models](#supported-models) · [How it runs](#how-it-runs) ·
[Troubleshooting](#troubleshooting) · [Update and uninstall](#update-and-uninstall) · [What is different](#what-is-different) ·
[Development and research](#development-and-research)

## Install

### Requirements

| | |
|---|---|
| OS | Windows 10 or 11 (64-bit) |
| GPU | NVIDIA RTX 30 series or newer runs on CUDA; other GPUs (AMD Radeon, …) run on Vulkan 1.3 |
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
- Models, settings and results live in `%USERPROFILE%\GeulPU` as with the zip, so uninstalling the package (`pipx uninstall geulpu`)
  keeps them.
- If the `geulpu` command is not found, run `python -m geulpu` (pip's per-user install may leave its scripts folder off PATH). pipx or
  uv is recommended.

Steps 1–3 below install from the zip.

### 1. Download

Get **`GeulPU-<version>-windows-x64.zip`** from the [releases page](https://github.com/wwoosshh/GeulPUv2/releases/latest). It is small
because it contains no model — you download models inside the app.

To check that the file arrived intact (optional), compare it with `SHA256SUMS.txt` on the same page. In PowerShell:

```powershell
Get-FileHash .\GeulPU-0.6.0-windows-x64.zip
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

## First steps

The app speaks English unless your browser prefers Korean — switch any time in **Settings → Language** (Auto · 한국어 · English).
The icons in the left bar are, from the top, **Chat · Models · Benchmark · Hardware**, with **Settings** at the bottom.

With no model installed, the app first shows this screen. **Download the recommended model** fetches a model chosen for your graphics
memory right away; **Browse models** opens Find models, described below.

![First screen — first, you need a model](docs/img/first-run-en.png)

### 1. Get a model — Models → Find models

- Hugging Face's GGUF models are listed by **Trending**. Switch to Popular, Likes or Recently updated, or type part of a name (for
  example `qwen3`, `coder`, `14b`) into the box at the top to narrow the list.
- By default only **models GeulPU can run** are shown. **Also show models that don't run** shows everything, with the ones that don't
  run dimmed and the reason given. **Fits this PC** keeps only models that fit your graphics memory.
- New to this? Switch on **Verified by GeulPU**: only repositories with files GeulPU has run itself remain (picture below); see
  [which model fits your PC](#which-model-fits-your-pc).

![Find models — the list on the left, details of the selected model on the right](docs/img/find-models-en.png)

- Click a model to see its description, downloads, license and **file list** on the right. The file that suits this PC best is marked
  **Recommended for this PC**, files GeulPU has run are marked **Verified to run**, and files that are tight or too large for your
  graphics memory say so. Click **Download** on the file you want.

![File list — "Recommended for this PC" and "Verified to run" marks with download buttons](docs/img/model-files-en.png)

- Before downloading, the app reads only the beginning of the file to check that it will run; interrupted downloads resume; finished
  files are verified with SHA-256. Downloads in progress appear under **My models** and in the status bar at the bottom, where you can
  pause or resume them.
- GGUF files you already have can be added with **Add a GGUF file** at the top; they stay where they are.

### 2. Chat

- Pick a model with the model button at the top and send a message (**Enter** sends, **Shift+Enter** adds a line).
- **The first time**, GeulPU builds the kernels for your GPU and model (about 10 seconds on a fast CPU, 1–3 minutes on a slow one) and
  loads the model into graphics memory (about 20 seconds for 7B). Later only the loading happens.
- **Reasoning models** (Qwen3, DeepSeek-R1) think before answering. The thinking is folded as "Thought for N s"; click it to expand.
  Token count and speed (tokens/s) appear under the answer.

![Chat — Qwen3-8B Q4_K_M with its thinking folded and the speed under the answer](docs/img/chat-en.png)

- **To turn Qwen3's thinking off**: **Generation settings** (the slider icon at the top right) → switch off **Thinking**. It then
  answers directly.
- Generation settings also hold the temperature (0 gives the same answer to the same question every time), the maximum answer length
  and a system prompt, saved in this browser.

![Generation settings — temperature, max answer length, thinking switch, system prompt](docs/img/generation-settings-en.png)
- You can regenerate an answer or edit and resend your message; earlier answers stay as versions you can flip through (‹ 1 / 2 ›).
  Code, tables and math (LaTeX) are rendered.
- **Long chats**: when a chat nears the context length per chat, the same model summarizes the earlier conversation (the latest turns
  stay word for word) and the chat goes on without an error. The summary stays in the chat as a card you can expand, and the
  summarized messages stay, dimmed. "context N / L" under an answer shows how much is used. Turn it off with **Auto-summarize long
  chats** in Generation settings.
- Conversations are stored only in this browser.

### 3. Benchmark (optional)

- **Hardware diagnosis**: measures memory bandwidth and tensor-core throughput and shows how close each kernel comes to the hardware's
  limit.
- **Speed test**: prompt processing (512 and 2048 tokens) and generation speed in tokens per second.
- **Reproducibility check**: solves 32 math problems with DeepSeek-R1-Distill-Qwen-7B Q8_0 and compares the output with the reference
  PC bit for bit (the quick check uses 2 problems and 64 tokens).
- Results can be exported as a zip (user paths are removed). The raw tool output and the result files themselves stay in Korean.

### 4. Settings

GPU backend (Auto · CUDA · Vulkan), Vulkan device, concurrent chats (how many chats can answer at once, 1–8) and context length per
chat, **computation mode**, **speculative decoding**, **context store**, model folder, language, new-version notice. Changes are saved
immediately.

**Computation mode** — **Fast** (default) or **Exact**. Applies the next time a model is loaded.
- **Fast**: the activations entering the matrix multiplications are 8-bit (the same class as llama.cpp). Prompt processing is 1.3–1.6×
  faster than Exact (CUDA · 7B · 8B), and the change in next-token probabilities is of the same order as llama.cpp's
  ([문서/2단계.md](문서/2단계.md)).
- **Exact**: the contract as it was up to 0.5.0 (16-bit activations — research precision).
- Both modes match the contract's CPU reference bit for bit, and within one mode every path gives the same bits. The
  reproducibility check and Vulkan devices always use Exact.

**Speculative decoding** — **On** (default) or Off. Looks up likely next tokens in the prompt and the answer so far (no draft model)
and verifies several in one step. Because this engine gives the same bits whatever the number of rows in a step, **answers,
log-probabilities and research-panel values are bit-identical to Off**, while answers that repeat pieces of the prompt — editing
code, rewriting a document, quoting summaries — generate 3.5–4× faster (ordinary chat 1.0–1.5× — [문서/3단계.md](문서/3단계.md), Korean).
Applies the next time a model is loaded.

**Context store** — **Default** (on when the location is an SSD, off for an HDD) · On · Off. Keeps what the model computed for long
prompts (the KV cache) on the SSD in 256-token chunks and restores it instead of recomputing whenever a request starts the same way —
even after restarting the app, switching models and back, or another chat taking the slot. For coding agents, which resend long system
prompts and conversations, the first token drops sharply — after restarting the app, a new task's first request (15k tokens) goes from
3.2 s to 0.3 s, and reopening a 31k-token conversation from 10.0 s to 0.74 s (0.52 s and 1.1 s when the chunks are not in the OS file
cache, as after a reboot — Qwen2.5-7B, RTX 4070 Ti). Because this engine produces
the same KV bytes whichever way it is filled, **answers, log-probabilities and research-panel values are bit-identical to Off.** The
location (default: data folder\문맥저장소), size limit (default 20 GB — the longest-unused chunks go first), writing (all · only when
evicted · read only) and **Clear the store** are here too. The chunks are derived from your chats, so turn it off or clear it on a shared
PC. Applies the next time a model is loaded ([문서/문맥저장소.md](문서/문맥저장소.md), Korean).

### 5. Research panel — per-token probabilities · intervene · compare · reproduce

GeulPU's arithmetic is fixed by a written contract, so scoring the same text again gives **bit-identical** probabilities to those at
generation time — whatever the number of concurrent requests, the cache, the GPU or the backend (CUDA · Vulkan), within one
computation mode (Vulkan always computes Exact, so set the computation mode to Exact to compare with CUDA). The difference between two
runs therefore comes only from what you changed. Open the panel with the flask button at the top of the chat, or the
flask under an answer (answers received in this version onward — research records stay in this browser).

**New to it? Start with the [research panel guide](docs/research-guide.md)** — seven experiments (where the model hesitated, branching,
comparing two answers, the effect of quantization, the effect of one word, scoring text, reproduction capsules) with screenshots, how to
read the numbers, and API examples.

- **Observe** — the answer as tokens: each token's probability, rank, top-10 alternatives, entropy and the gap between the top two
  (margin). Close calls (margin < δ — places where a slightly different computation or setting could pick another token) and steps
  where sampling picked a token that wasn't ranked first are marked. The answer header shows where the prompt was reused from (this
  slot · another slot · the context store) and the **cache break** (which token, the text before it, before → now token) — when an agent
  puts something that changes every time (a clock, a working-folder path) near the start of its prompt, you can see the cache break there
  ([guide — where did the prompt cache break](docs/research-guide.md#where-did-the-prompt-cache-break-context-store)).
- **Intervene** — pick an alternative token at any step and **branch here**: generation continues with the same settings, the same
  seed and the same per-step random numbers, so the chosen token is the only change. You can also regenerate with only the
  temperature, seed or max length changed. New answers stay as versions.
- **Compare** — the first diverging step of two answers (versions or branches). For an edited question, score the same answer tokens
  under the other prompt and see the per-token difference; score the same text with another model (e.g. Q8_0 ↔ Q4_K_M of the same
  model) to see the effect of the quantization alone.
- **Score** — paste text and get each token's log-probability and the perplexity.
- **Reproduce** — fingerprints (SHA-256 — the recipe is in [contract](계약/contract.md) §5.2), re-run without the cache and check, export a **reproduction capsule** (request, result,
  log-probabilities and logit hashes in one JSON file) and load one to check it, hashes of the engine binaries and kernels.
- **Export** — every value shown as JSON or CSV, and Python code that fetches the same values.

Probabilities are the model's distribution (temperature 1), not the distribution filtered by the sampling temperature, top-k or
top-p. The panel computes only when you look, so chat speed is unchanged.

![Research panel — Observe: pick a step in the token strip to see its top-10 alternatives and numbers](docs/img/research-observe-en.png)

![Research panel — Compare: the same answer scored again with the same model's Q4_K_M file — the effect of the quantization alone, token by token](docs/img/research-compare-en.png)

A capsule can be checked without a GPU, too — clone the repository (`pip install numpy`) and run
`python -I 도구/캡슐확인.py <capsule.json> <model.gguf>`. The contract's CPU reference implementation recomputes the tokens, logit hashes
and log-probabilities (a few minutes for 7B). A capsule records its computation mode and is checked in that mode.

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

- To turn Qwen3's thinking off, add `"chat_template_kwargs": {"enable_thinking": false}` to the request (`extra_body=` in Python).
- By default the thinking of Qwen3 and DeepSeek-R1 comes in front of the answer without tags. Add `"geulpu": {"special": true}` to
  receive `<think>` and `</think>` as text, so you can separate the thinking from the answer. In requests that send `tools`, Qwen3's
  thinking comes separately as `reasoning_content`.
- Only this PC can connect, and there is no authentication. Requests that other websites send through your browser are refused.
- `/v1/chat/completions` uses the Qwen2.5, Qwen3, Llama 3 (3.1 · 3.2) and Mistral (v0.3) chat templates. For DeepSeek-R1, put R1's
  template into `/v1/completions` yourself for now.
- The Llama 3 template puts a date in the system header. So that the same question does not become a different prompt every day, it is
  fixed to `26 Jul 2024`; change it with `"chat_template_kwargs": {"date_string": "16 Oct 2026"}` in the request. Mistral's template has
  no system role, so a system message is put in front of the first question as `{system}\n\n` (the same form as llama.cpp).
- For Llama 3 and Mistral a BOS token is prepended to text prompts (the text of `/v1/completions` and the `prompt` of `/tokenize`), as
  llama.cpp and vLLM do — `/tokenize` leaves it out with `"add_special_tokens": false`. Prompts given as token ids are used as they are.

**Connecting a coding agent (tools)** — agents such as Qwen Code, OpenCode, Continue and Cline that call tools (read and write files, run
commands) through an OpenAI-compatible API can connect.

- The address is `http://127.0.0.1:44544/v1`; any API key and any model name work (the model loaded in the app answers — load a model
  in the app before using the agent).
- Models that take tools: Qwen2.5 (7B · 14B · Coder), Qwen3, Llama 3.1 · 3.2. Prompts with tools match the official chat template in the
  model file token for token. Mistral and DeepSeek-R1 have no tools in their model files' templates, so they refuse them (400).
- In tests Qwen2.5-7B and Qwen3-8B finished the tasks; Llama 3.x sometimes repeated the same call after getting the tool result — Qwen is
  recommended for agents. `tool_choice` accepts only `auto` (default) and `none`.
- **The bigger the context, the better.** An agent's base instructions alone are 10,000–15,000 tokens (Qwen Code: 14,501), and they
  grow as it works. Feature-level development (for example planning, reviewing, implementing and reporting a login feature) without
  compaction needs 100k tokens or more. Set concurrent chats to 1 and the context length per chat high (e.g. 131072) in Settings; the
  engine reduces it to the model's maximum and to the graphics memory. For Qwen2.5 7B and larger and Qwen3, a context beyond the
  model file's position count (32,768 · 40,960) turns on **YaRN** (the position scaling Qwen recommends — factor 2 or 4) up to 131,072;
  the model list shows 'YaRN' next to the context. With a scaled context, short texts are computed the same way too (static YaRN, as in
  other engines), so set a large context only when you need long texts ([문서/긴문맥.md](문서/긴문맥.md), Korean). Maximum reached on an
  RTX 4070 Ti (12 GB):

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

**Log-probabilities · scoring · tokenization (for research)** — every value in the research panel is also available from the API.
Contract 5.2 ([contract](계약/contract.md)) defines the values, so they are bit-identical on any PC or GPU.

- `/v1/chat/completions`: `"logprobs": true, "top_logprobs": 5` (0–20), in the OpenAI shape (`token` · `logprob` · `bytes` ·
  `top_logprobs`).
- `/v1/completions`: `"logprobs": 5`, `"echo": true` (log-probabilities of the prompt tokens too), `"max_tokens": 0` (score only, no
  generation). `prompt` is text or an array of token ids.
- The `"geulpu"` extension: `"logit_hash": true` (a 64-bit hash of each step's logit row), `"prompt_tokens": true` (the prompt tokens
  after the chat template), `"echo_from": n` (echo log-probabilities from the token at index n on — 0-based; earlier ones are null), and `entropy` · `top_ids` · `top_logprobs`
  (aligned with the ids) · `seed` in the response.
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
| qwen2 | Qwen2.5-Instruct · Qwen2.5-Coder · DeepSeek-R1-Distill-Qwen and their fine-tunes | 1.5B and up (head size 128) — 7B and 14B run on this project's PC |
| qwen3 | Qwen3 and its fine-tunes | 1.7B · 8B · 14B |
| llama | Llama 3.1 · 3.2 Instruct, Mistral 7B Instruct v0.3, and their fine-tunes that use the same chat templates | 3B · 7B · 8B — 70B checked as a shape with a synthetic model |

- **Formats**: Q8_0 · Q4_K_M · Q5_K_M · Q6_K, and Q4_0 files whose tensors are all Q4_0.
- **Licenses** differ by model. In the app's list, Llama 3.1 · 3.2 use Meta's community licenses (usage conditions and an Acceptable
  Use Policy — services with more than 700 million monthly users need permission from Meta), the Mistral 7B v0.3, Qwen2.5 and Qwen3 files
  are Apache 2.0, and DeepSeek-R1-Distill is MIT (also shown in the app's model descriptions).
- **Not yet**: other architectures (Gemma, Phi, …), models whose query width differs from the hidden size (Qwen3 0.6B · 4B · 32B,
  Mistral NeMo), small models with head size 64 (Qwen2.5-0.5B, Llama 3.2 1B), MoE (Mixtral, Qwen3-30B-A3B, …), llama-architecture
  fine-tunes with other chat templates (ChatML, …), image input, Q2_K · Q3_K · IQ · Q5_0 · F16 · BF16. Find models hides such files by
  default.
- **Depending on the GPU**: k-quants (Q4_K_M, …) run on CUDA and on Vulkan GPUs with at least 48 KB of shared memory. With AMD's
  Windows driver (32 KB of shared memory) only Q8_0 (and Q4_0) files run for now.

### Which model fits your PC

Files GeulPU has run itself (**Verified by GeulPU** in the app). Sizes use 1024-based units, as Windows Explorer does.

| Graphics memory | Suggested files |
|---|---|
| 6 GB | Qwen3-1.7B Q8_0 (1.7 GB) · Llama-3.2-3B-Instruct Q8_0 (3.2 GB) |
| 8 GB | Qwen3-8B Q4_K_M (4.7 GB) · Qwen2.5-7B-Instruct Q4_K_M (4.4 GB) · DeepSeek-R1-Distill-Qwen-7B Q4_K_M (4.4 GB) · Llama-3.1-8B-Instruct Q4_K_M (4.6 GB) · Mistral-7B-Instruct-v0.3 Q4_K_M (4.1 GB) |
| 12 GB | Qwen3-8B Q8_0 (8.1 GB) · Qwen2.5-7B-Instruct Q8_0 · Qwen2.5-Coder-7B-Instruct Q8_0 · DeepSeek-R1-Distill-Qwen-7B Q8_0 (7.5 GB) · Llama-3.1-8B-Instruct Q8_0 (8.0 GB) · Mistral-7B-Instruct-v0.3 Q8_0 (7.2 GB) — 14B Q4_K_M also runs with a reduced context |
| 16 GB or more | Qwen2.5-14B-Instruct Q4_K_M · DeepSeek-R1-Distill-Qwen-14B Q4_K_M (8.4 GB) |

Qwen3-14B Q4_K_M has only been checked bit for bit on a synthetic model of the same shape and has not been run yet. The app tells you
for every file whether it fits your graphics memory.

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
| `결과\` (results) | Benchmark results |
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
  Changes are listed in [바뀐것.md](문서/바뀐것.md) (Korean).
- **Uninstall**: quit the app and delete the GeulPU folder. To delete the downloaded models as well, also delete
  `%USERPROFILE%\GeulPU`.
- **Installed with a Python installer**: to update, quit the app, then `pipx upgrade geulpu` (`uv tool upgrade geulpu` ·
  `pip install -U geulpu`); to uninstall, `pipx uninstall geulpu` (`uv tool uninstall geulpu` · `pip uninstall geulpu`) — the data
  folder stays.

## What is different

- **Precision is a written contract.** Activations are integers per 32-value block — two balanced base-256 digits (±2¹⁴ of the block
  maximum) in the exact mode; in the fast mode (the default — [stage 2](문서/2단계.md), Korean) only the activations entering the matrix
  multiplications have one digit (8 bits) —, the KV cache is two-digit block integers, products are summed exactly on integer tensor
  cores, and exp is a written polynomial. Every rounding point is fixed in the source, so in both modes the GPU result is
  **bit-identical** to the CPU reference implementation ([`계약/정수계약.py`](계약/정수계약.py)). The only losses are the model file's own
  quantization and the roundings the contract writes down — see [docs/contract.md](계약/contract.md).
- **The same bits whichever way it computes.** Feeding a prompt at once, in pieces, one token at a time, as a continuing conversation,
  or batched with other sentences gives identical results. Common engines change output bits with the amount of work computed
  together: llama.cpp's wikitext-2 perplexity varies from 15.8171 to 15.8200 with batch settings, while GeulPU gives the same value to
  the last bit for every way of feeding the text (stage 20). Across GPUs too: an RTX 3080 Ti (Ampere), an RTX 4070 Ti (Ada) and an RTX 5070
  (Blackwell) gave bit-identical outputs and per-step logits (82,673 steps) on 32 math problems with DeepSeek-R1-Distill-Qwen-7B, while llama.cpp's
  outputs all changed with the number of concurrent requests on one PC ([교차검증](교차검증/)). Evaluation tools too: lm-evaluation-harness
  per-item results (arc_easy 500, gsm8k 100) were identical across num_concurrent 1 ↔ 8 and CUDA ↔ Vulkan (with the same setup,
  llama.cpp b11643 changed the generated text of 41 of 100 gsm8k items between 1 and 8 concurrent requests — [stage 38](https://github.com/wwoosshh/GeulPU/blob/main/문서/38단계.md)).
- **Speculative decoding gives the same answers.** Thanks to the property above, the logits of a verification step (several rows)
  are bit-identical to computing the tokens one by one, so greedy decoding and sampling (random numbers fixed by the step number)
  produce the same tokens — with speculation on, the tokens, per-step logit hashes and log-probabilities equal those with it off
  (tests: seventeen requests × correct, wrong and half-wrong drafts, four concurrent slots, CUDA and Vulkan — [stage 3](문서/3단계.md)).
  In the same measurement, llama.cpp b11721's llama-server changed the answers of two of eight greedy requests (summary, math) when
  `--spec-type ngram-simple` was switched on.
- **The context store gives the same answers.** KV restored from the SSD is byte-identical to recomputed KV (the contract — KV is the same
  whichever way it is filled), so with the store on, the tokens and per-step logit hashes equal those with it off. Tests: 54 recorded Qwen
  Code requests sent with recomputation · store off · empty store · after a server restart · one slot alternating conversations · 40
  corrupted chunks — all 1,728 steps identical; a Vulkan server reading chunks written by CUDA matched too. So did a replay that
  generates the recorded answer lengths (19,552 steps), a new task, reopening a 31k-token conversation and rewinding it (from the OS
  cache and from a cold SSD), and switching between more conversations than slots ([문서/문맥저장소.md](문서/문맥저장소.md), Korean).
- **The exact mode's hidden loss is small — measured on 0.5B.** Qwen2.5-0.5B-Instruct (Q8_0): the hidden loss (KL against an fp64 forward pass of the same
  weights) is 1/615 of llama.cpp's (a 461-token evaluation text) and 1/385 (wikitext-2, 148,920 tokens); its perplexity is
  indistinguishable from fp64 (relative −6.6 × 10⁻⁶), while llama.cpp's is 0.34% higher — the same order as the Q8_0 weight quantization
  itself (0.24%). How this was measured: [docs/hidden-loss.md](https://github.com/wwoosshh/GeulPU/blob/main/docs/hidden-loss.md). 7B and 14B were not measured with the real engine
  (see the simulation below). The fast mode keeps the hidden loss at llama.cpp's level — measured against the exact mode, its
  next-token KL is 0.65–0.98× llama.cpp's on k-quant 7B · 8B models and 2.2–2.4× on Q8_0 ([stage 2](문서/2단계.md)).
- **Speed — fast-mode prompts beat llama.cpp up to 4096 tokens; decoding at parity.** On one RTX 4070 Ti:
  - Fast mode (stage 4 — latest official llama.cpp b11721, same GGUF, median of three alternating rounds, llama.cpp at its fastest
    setting per length): prompt time is 0.89 / 0.90 / 0.95 / 1.04× llama.cpp's for Qwen2.5-7B Q4_K_M, 0.74 / 0.76 / 0.81 / 0.92× for Q8_0
    and 0.92 / 0.93 / 0.99 / 1.11× for Llama-3.1-8B Q4_K_M (512 / 2048 / 4096 / 8192 tokens — [stage 4](문서/4단계.md)). The slowdown at
    long contexts is the attention, which still has two digits. The exact mode took 1.16–1.47× in the same run.
  - Long sessions ([long-context measurement](문서/긴문맥재기.md), fast mode): filling an empty cache with 16k / 32k tokens takes 1.19 / 1.40×
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

## Development and research

The rest is for building the engine from source. To use the app, [Install](#install) is all you need. The research record (stage
reports, paper) lives in the [GeulPU repository](https://github.com/wwoosshh/GeulPU); the "stage N" links in the table below go there.

### Status (2026-10)

| | |
|---|---|
| Models | **Llama-3.1-8B · Llama-3.2-3B · Mistral-7B v0.3** (stage 39 — architecture `llama`: q · k row restore and RoPE scaling factors as contract §4.7; five files bit-identical to the CPU reference, mean KL ≤ 4 × 10⁻⁴ against llama.cpp on every token of the same texts; the 70B shape tested on synthetic models) · **Qwen3-1.7B / 8B / 14B** (stage 37 — per-head Q/K RMSNorm as contract §4.6 and one appended kernel, bit-identical to the CPU reference; the 14B shape tested on synthetic models) · **Qwen2.5-0.5B / 7B / 14B-Instruct** · DeepSeek-R1-Distill-Qwen (chat; Llama-family blocks — RMSNorm · RoPE · SwiGLU · GQA; head sizes 64 and 128). Head-128 Qwen2 shapes are read from the GGUF config; 32B and 72B shapes were bit-tested on 4-layer synthetic models. GPT-2 small and XL (research engine). |
| App | Stages 36–39 — `GeulPU.exe` (a manager written in geul) and a web UI: find models (browse every Hugging Face GGUF repo, sorted), download, add and verify them; chat (Qwen2.5's ChatML, Qwen3's template with thinking on/off, R1's, Llama 3's and Mistral's templates); self-diagnosis, speed and cross-check runs; result zips; a new-version notice; a **research panel** (stage 38 — per-token probabilities, branching, comparison, scoring and reproduction capsules; log-probabilities as contract §5.2) and an English UI. The release zip (1.4 MB) is built from source by CI when a tag is pushed (as a draft; a person publishes it). Windows x64 only — see [Install](#install). |
| Weight formats | GGUF **Q8_0 · Q4_0 · Q4_K · Q5_K · Q6_K**, including per-tensor mixes as in Q4_K_M · Q5_K_M (stage 26 — tested with files re-quantized on this PC from Q8_0; official Qwen files are not yet tested). |
| Runner | `빌드/글/실행기.exe model.gguf` (`실행기` = runner) — a geul program: chat (Qwen's ChatML and the Qwen3, Llama 3 and Mistral templates), streamed output, KV cache reused across turns, greedy decoding (the same conversation gives the same answer). About 0.3 s until the prompt runs. |
| Server | `빌드/글/서버.exe model.gguf` (`서버` = server) — a geul program (stage 28): an OpenAI-compatible HTTP API (chat · completions · SSE streaming) with slots, continuous batching, prompt caching and a sampling contract (the same seed gives the same tokens). A request returns the same bits whatever else runs with it — zero differences over four models × twelve requests × four schedules, while llama-server's greedy answer changed in 11 of 12 cases depending on the concurrent requests. Head-128 Qwen2 (7B · 14B). 7B Q4_0 aggregate throughput: 302 tokens/s with four streams and 508 with eight (llama-server 360 · 632 — a multi-row decode kernel is next); one stream is 4% faster than llama-server. A cached prompt's first token takes 13.8 ms (llama-server 13.0 ms; stage 30 feeds the last rows through the decode step). See [`문서/28단계.md`](https://github.com/wwoosshh/GeulPU/blob/main/문서/28단계.md) · [`문서/30단계.md`](https://github.com/wwoosshh/GeulPU/blob/main/문서/30단계.md) |
| Environment | Windows or Linux x86-64 (glibc) + NVIDIA Ampere (RTX 30) or newer + driver. Linux was tested under WSL2 Ubuntu 24.04 in stage 29 — the same bits as Windows on the same GPU, speed within ~0.5% ([`문서/29단계.md`](https://github.com/wwoosshh/GeulPU/blob/main/문서/29단계.md)). On Linux the programs import only libc.so.6 and libcuda.so.1. All numbers here are from one RTX 4070 Ti (12 GB). Building (generator → kernel source → PTX → runner) needs only the geul compiler. Tests and benchmarks need Python 3.12 + numpy. **Vulkan (stage 31):** Vulkan builds of the geul host (`시험_엔진V` · `시험_7BV` · `재기_7BV` — they import kernel32, shell32 and vulkan-1.dll; kernels come from the geul compiler's SPIR-V backend) reproduce the CUDA records byte for byte on NVIDIA (0.5B, 7B in five formats, 14B) and Intel integrated graphics (0.5B). Speed is not there yet — stage 34 replaced the emulated integer MMA with cooperative-matrix tensor-core work (`--협력` modules) and fixed the backend and the Vulkan layer: 7B prompts take 1.15–1.41× llama.cpp Vulkan's time (stage 31: 4.4–6.1×), decode 1.00–1.04× (stage 31: 1.09–1.24×) — a record, not yet a verdict. No Vulkan build of the runner or the server yet. |
| Layer split | Stage 25 — split layers across GPUs (`--나누기 P` parts, `--장치 a,b,…` devices, `--예산 GB` budget). Emulated on one GPU with one CUDA context per part: 16 split runs are bit-identical to the single-part run; decoding with two parts takes 1.025–1.029× (mostly the emulation's context switching; 1.002–1.003× with a shared context). 14B in two parts did not fit at context 8192 on 12 GB, so that pre-registered cell was not measured (criterion 25c not passed). |
| Not yet | Real multi-GPU runs; measuring the contract's loss with the real engine on 7B+ (done for 0.5B; simulated for 7B · 14B); Q2_K · Q3_K · IQ formats · Q5_0 (real 72B k-quant files use Q5_0 · Q8_0 for ffn_down); MoE; a GPU run on native Linux (tested under WSL2 only); running on Apple GPUs (three NVIDIA generations and an AMD Radeon RX 9060 XT (RDNA4, Vulkan) match bit for bit — [교차검증](교차검증/); AMD speed (RX 9060 XT generation is about 3× slower than an RTX 4070 Ti, mostly attention); Intel integrated graphics matches through Vulkan for 0.5B — stage 31); k-quants on 32 KB shared-memory devices (AMD on Windows); models whose query width differs from the hidden size (Qwen3-0.6B · 4B · 32B, Mistral NeMo); small head-64 llama models (Llama-3.2-1B); other architectures (Gemma, Phi, …). |

### Repository layout

| Folder | What |
|---|---|
| [`엔진/`](엔진/) (engine) | The engine in geul: `gguf.gl` (GGUF reader), `가중치.gl` · `가중치K.gl` (weight layout, k-quants), `로프.gl` (RoPE tables), `토크나이저.gl` (BPE, SentencePiece, Unicode 15.0 tables), `엔진.gl` · `엔진K.gl` · `엔진표.gl` (kernel selection, CUDA graphs, decoding, slots), `조각.gl` (layer split), `뽑기.gl` · `로그확률.gl` (sampling and log-probabilities — contract §5 · §5.2), `서버.gl` · `서버V.gl` · `서버몸.gl` (the OpenAI-compatible HTTP server — CUDA and Vulkan heads), `장치C.gl` · `장치V.gl` (device check), `진단.gl` · `진단V.gl` · `진단몸.gl` (self-diagnosis), `실행기.gl` (terminal runner), `쿠다*.gl` · `벌칸*.gl` (GPU layers). `리눅스/` holds the Linux-target overlays. |
| [`커널/`](커널/) (kernels) | `커널생성.gl` + `생성*.gl` — the kernel generator: it emits a kernel module (geul source) per model architecture, shape, weight format and device. `진단*.gl` — diagnosis kernels. |
| [`앱/`](앱/) (app) | `관리자.gl` (GeulPU.exe) and its modules, `web/` (the UI — self-contained HTML · JS · CSS, Korean and English), `data/models.json` (the model catalog), `꾸리기.ps1` (release packaging), `읽어보기.txt` (the zip's README), `DESIGN.md` (design system). |
| [`계약/`](계약/) (contract) | [계약.md](계약/계약.md) · [contract.md](계약/contract.md) — the precision contract. `정수계약.py` — its CPU reference implementation (every bit comparison is against it), `로그확률.py` — log-probabilities (§5.2) in Python, `gguf읽기.py`. |
| [`도구/캡슐확인.py`](도구/캡슐확인.py) (tools) | Checks a research-panel reproduction capsule without a GPU, on the CPU reference. |
| [`시험/`](시험/) (tests) | `빠른점검.py` (the quick check run by CI), `계약자체.py` (the contract's CPU self-tests), `참조7B.py` (the CPU reference forward pass for 7B — the end-to-end bit oracle), `판23.py` · `판26.py` · `머리128.py` · `줄128.py` (each kernel against the CPU reference — on the GPU, via `쿠다드라이버.py`), `품질.py` (hidden loss relative to the exact mode, compared with llama.cpp), `뽑기.py` · `로그확률38.py` (sampling and log-probability contracts, Python vs geul), `서버28.py` · `로그확률서버38.py` (server regression tests: bits under concurrency, API, log-probabilities), `모듈해시.txt` · `SASS넘침.txt` · `글저장소커밋.txt`. |
| [`재기/`](재기/) (measurements) | Speed and quality runs — `견주기2.sh` (prompt time of exact · fast · llama.cpp, order alternating by round) · `견주기요약.py`, `생성2.sh` (decode ms/token), `품질2.sh`, `라마.py` (llama.cpp's C API). The geul test and measurement programs are in `엔진/`: `시험_7B.gl` · `재기_7B.gl` · `재기_프롬프트.gl` · `재기_커널.gl` · `퍼플렉시티.gl`. |
| [`교차검증/`](교차검증/) (cross-check) | The reference for the app's reproducibility check (Benchmark → cross-check): `expected.json` (the reference PC's outputs and build hashes) · `requests.json` (the 32 problems). Packaging compares the build hashes with it. |
| [`문서/`](문서/) (documents) | [이관범위.md](문서/이관범위.md) (what was moved) · [바뀐것.md](문서/바뀐것.md) (changelog) · stage records ([2단계.md](문서/2단계.md) — fast contract F1 · [2b단계.md](문서/2b단계.md) — fast attention plan · [2c단계.md](문서/2c단계.md) — matmul kernels · [3단계.md](문서/3단계.md) — speculative decoding · [4단계.md](문서/4단계.md) — long-prompt attention kernel · [4b단계.md](문서/4b단계.md) — long-context decode fold fix), [긴문맥재기.md](문서/긴문맥재기.md) (long-session prompt and decode vs llama.cpp) — Korean. |
| [`docs/`](docs/) | English documents — [research-guide.md](docs/research-guide.md) (research panel guide) · [glossary.md](docs/glossary.md) · `img/` (screenshots). |
| [`연구자안내.md`](연구자안내.md) | The research panel guide in Korean. |

### Build from source

1. **Requirements**: Windows (or Linux x86-64 — see 4) with an NVIDIA driver (Ampere or newer) or a Vulkan driver. The app and the
   server need only this and the geul compiler; the tests need Python 3.12 and `pip install -r requirements.txt` (numpy).
2. **geul compiler and backends**: clone the `research/gpu` branch of the geul repository next to this one as `../geul-lang-v2`, check
   out the pinned commit ([`시험/글저장소커밋.txt`](시험/글저장소커밋.txt) — CI, packaging and the cross-check reference all use it) and
   build the self-hosted compiler and the PTX and SPIR-V backends:
   ```
   git clone -b research/gpu https://github.com/wwoosshh/geul-lang ../geul-lang-v2
   cd ../geul-lang-v2
   git checkout $(tr -d '\r\n ' < ../GeulPUv2/시험/글저장소커밋.txt)
   python build.py selfhost 컴파일러
   GEUL_ROOT=. build/self_컴파일러.exe research/gpu/자체PTX/PTX.gl -o build/자체PTX.exe
   GEUL_ROOT=. build/self_컴파일러.exe research/gpu/자체SPIRV/SPIRV.gl -o build/자체SPIRV.exe
   ```
3. **Package the app** (what a release is): `앱\꾸리기.ps1 -GeulRoot ..\geul-lang-v2 [-GeulC ..\geul-lang-v2\build\self_컴파일러.exe] -NoZip`
   builds `빌드\앱\GeulPU-<version>\` — `GeulPU.exe`, `bin\` (server-cuda, server-vulkan, device-cuda, device-vulkan, diag-cuda,
   diag-vulkan, kernel-gen, ptx-backend, spirv-backend), `kernels\`, `web\`, `data\` — and compares the SHA-256 of the server, the
   generator and the backends with the cross-check reference. A model's kernel module is generated on the user's PC the first time that
   model is loaded (data folder `모듈\<version>\`).

   **Server only**: build the generator and the server, generate the module for the model's shape and format, compile it to PTX next
   to the server and run it (if the module is missing, the server prints the command). The terminal runner is `엔진/실행기.gl`
   (`빌드/글/실행기.exe model.gguf`, same module).
   ```
   mkdir -p 빌드/글
   GEUL_ROOT=../geul-lang-v2 ../geul-lang-v2/build/self_컴파일러.exe 커널/커널생성.gl -o 빌드/글/커널생성.exe
   GEUL_ROOT=../geul-lang-v2 ../geul-lang-v2/build/self_컴파일러.exe 엔진/서버.gl -o 빌드/글/서버.exe
   빌드/글/커널생성.exe qwen2큰 K 3584 28 4 18944 152064 빌드/글/qwen2K_3584x28x4x18944x152064.gl     # Qwen2.5-7B k-quant module
   ../geul-lang-v2/build/자체PTX.exe 빌드/글/qwen2K_3584x28x4x18944x152064.gl -o 빌드/글/qwen2K_3584x28x4x18944x152064.ptx
   빌드/글/서버.exe model.gguf --슬롯 4 --문맥 4096 --포트 8080
   ```
4. **Linux**: the same sources build as ELF with the geul compiler's Linux target (they import only `libc.so.6` and `libcuda.so.1`).
   Tested under WSL2 Ubuntu 24.04; not yet on native Linux with a GPU.

### Checks

[![Quick check](https://github.com/wwoosshh/GeulPUv2/actions/workflows/ci.yml/badge.svg)](https://github.com/wwoosshh/GeulPUv2/actions/workflows/ci.yml)

- **Quick check on every push** — GitHub Actions, a Windows and an Ubuntu runner, no GPU and no model files, about 5 minutes
  ([`시험/빠른점검.py`](시험/빠른점검.py); the same command runs locally: `python -X utf8 시험/빠른점검.py --글 ../geul-lang-v2`). The geul
  compiler is a CPU program, so everything up to PTX and SPIR-V can be produced without a GPU: the check builds the geul compiler and
  both backends at the pinned geul commit, builds the generator and every engine program, regenerates the 18 kernel modules (Qwen2 ·
  Qwen3 · Llama shapes and formats, shared-32 and one-digit variants) from scratch and compares source, PTX and SPIR-V byte for byte
  with [`시험/모듈해시.txt`](시험/모듈해시.txt), assembles every module to SASS for sm_89 with NVIDIA `ptxas` (a checker only) and fails on
  new or larger register spills ([`시험/SASS넘침.txt`](시험/SASS넘침.txt)), runs the contract's CPU self-tests
  ([`시험/계약자체.py`](시험/계약자체.py)), checks that the tile tables agree between the generator and the engine, and checks the DLL
  import tables of the runner, the server and the app manager.
- **GPU tests** (by hand): end-to-end bit comparisons against the CPU reference ([`시험/참조7B.py`](시험/참조7B.py)) and the server
  regression tests ([`시험/서버28.py`](시험/서버28.py) · [`시험/로그확률서버38.py`](시험/로그확률서버38.py)). The scripts that produced each
  research stage's verdicts are in [GeulPU's `공식실행/`](https://github.com/wwoosshh/GeulPU/tree/main/공식실행).

### Research log

The engine came out of a research program (stages 0–40, 2026-10-08 to 10-18) whose record — criteria committed before measuring, every
variant compared bit for bit before timing — lives in the GeulPU repository: [stage record](https://github.com/wwoosshh/GeulPU/blob/main/문서/단계기록.md)
(Korean), [English summary](https://github.com/wwoosshh/GeulPU/blob/main/docs/research-log.md), [paper draft](https://github.com/wwoosshh/GeulPU/tree/main/docs/paper).
The research was frozen on 2026-10-16; this repository carries the engine on for everyday use.

## License

MIT — [LICENSE](LICENSE).
