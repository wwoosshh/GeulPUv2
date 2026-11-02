# GeulPU model evaluation guide — make, check and fix GGUF files

**[한국어](../모델평가안내.md)** · [README](../README.md) · [User guide](user-guide.md) · [Debugger guide](debugger-guide.md) · [Research guide](research-guide.md) · [Plan and measurements (Korean)](../문서/모델평가.md)

**Model evaluation** (the flask button on the left rail) is a test bench for GGUF files. Make a GGUF from a fine-tuned model (a Hugging
Face folder), check that the conversion is right, measure what each quantization changes, count what the fine-tune learned and forgot
against the model before fine-tuning, and remake a file that loses less quality — then compare before and after. With the same files,
data and settings a rerun gives bit-identical results, so a before/after comparison shows only what you changed.

## Contents

- [1. Terms](#1-terms)
- [2. One round — a fine-tuned model to GGUF](#2-one-round--a-fine-tuned-model-to-gguf)
- [3. Data](#3-data)
- [4. Grading rules](#4-grading-rules)
- [5. Reading the metrics](#5-reading-the-metrics)
- [6. Compare with the original model](#6-compare-with-the-original-model)
- [7. Against the model before fine-tuning — learned and forgotten](#7-against-the-model-before-fine-tuning--learned-and-forgotten)
- [8. Improve — restore one matrix group at a time](#8-improve--restore-one-matrix-group-at-a-time)
- [9. Result table · Compare workspaces](#9-result-table--compare-workspaces)
- [10. Files](#10-files)
- [11. From scripts — the API](#11-from-scripts--the-api)
- [12. FAQ](#12-faq)

## 1. Terms

- **Workspace** — the work on one model: a reference model, the GGUF files compared with it (**versions**) and the results. The
  workspace name becomes the name of the files it makes and their model name (`general.name` in the GGUF) — letters and digits in it go
  into the file names as they are.
- **Reference model** — what every version is compared with. A BF16 or F16 file of the same model is best (Q8_0 if it does not fit in
  graphics memory); for a fine-tuned model, the BF16 converted with Make GGUF becomes the reference. With a file of another model or
  another fine-tune as the reference, every KL measures the wrong thing.
- **Baseline model** (optional) — a GGUF of the model before fine-tuning. When one is chosen, every result on graded data counts what
  the fine-tune learned and forgot (section 7).
- **Version** — a GGUF file to compare (v1, v2, …). Its making settings (type · imatrix · per-matrix types · tool versions) and a note
  are kept with it.
- **Original model** — the Hugging Face folder (`config.json` · `.safetensors` weights · tokenizer files). Conversion, the conversion
  check and Compare with the original model read it.

## 2. One round — a fine-tuned model to GGUF

1. **New workspace** — give it a name, leave the reference model at 'None — convert from the original model with Make GGUF', and set
   the original model folder.
2. **Set up tools** (once) — the button on the Make GGUF card. It shows what it downloads, with sizes and sources, and downloads only
   after you press it:
   - the executables and conversion scripts of a fixed llama.cpp tag (the SHA-256 of every file is checked),
   - two app-only virtual environments made with your Python (3.10 or later) — one with transformers 4.57.6, the version llama.cpp
     pins, and one with transformers 5.22.1 for folders saved with transformers 5 (it shares torch with the first). Python itself is
     not downloaded.
3. **Make GGUF** — source 'Original model folder', conversion type BF16, quantization types (choose several — Q8_0 · Q6_K · Q5_K_M ·
   Q4_K_M · IQ4_XS). The converted file becomes the reference model and the quantized files become versions. The app picks the
   environment from the `transformers_version` in the folder's `config.json`. You can also use an imatrix (made from a text file, or a
   file you have) and per-matrix types (one per line, like `attn_v=q6_k` — llama-quantize `--tensor-type`). The number of imatrix
   chunks measured (512 tokens each) is kept in the run log and the versions — a short text gives few chunks.
4. **Conversion check** — runs by itself after making, and again from the Conversion check card. It compares the GGUF with the
   original folder's small files (the weights are not read):
   - chat template — 5 to 7 conversation cases rendered with both templates are byte-identical,
   - EOS · BOS — the GGUF's end token is one of the original model's end tokens, and `add_bos_token` matches,
   - special tokens · tokenizer — IDs, text and merges match,
   - settings — the hyperparameters in `config.json` (layers · hidden · FFN · heads · KV heads · head size · RoPE base · RMS eps · vocab
     size) match the GGUF.

   Problems that metadata can fix — template, EOS, `add_bos_token` — get a new version from 'Create fixed version', which keeps the
   tensors and changes only the keys. Different special tokens, tokenizer or settings need a new conversion.
5. **Run evaluation** — choose data (section 3), the versions, the compute mode (exact or fast — fast on CUDA only) and the answer
   length. The reference model's answers and the top 64 at every position (the reference values) are made once and reused for the same
   reference model, data and settings. With graded data you can also choose a baseline model (before fine-tuning — section 7).
6. **Compare with the original model** — the card under the result (section 6). It shows how faithfully conversion and engine
   reproduce the original model.
7. **Improve** — measure which matrix group of a quantized version improves the KL (and the accuracy, with graded data) most when
   restored to a higher type, and remake the version with the groups you choose (section 8).
8. **Before/after** — choose two results in the list. Results with the same data, settings and reference model are compared, and the
   verdict counts conversations that now match the reference model and ones that newly differ.
9. **Result table** — see the workspace's versions in one table and a size and quality chart, and export them as CSV. Compare
   workspaces lines them up type by type with the workspaces of other fine-tunes (section 9).

## 3. Data

Built-in data — `basic60` (60 conversations written by GeulPU, six categories), `tasks40` (40 graded tasks), `gsm8k` (GSM8K test,
1,319) and `mmlu-pro-500` (500 from MMLU-Pro test, even across categories). Sources and licenses show next to the data picker.

**Your own data (JSONL)** — one conversation per line. Import data copies it into `평가\데이터\` in the data folder.

```json
{"id": "q-001", "category": "facts", "messages": [{"role": "user", "content": "When was the Woulwivceim Orchestra founded?"}], "max_tokens": 16, "answer": "1867", "grader": "number"}
```

| Field | Meaning |
|---|---|
| `messages` | The conversation (required) — OpenAI format, with system messages and several turns if you like |
| `id` · `category` | Name and category — metrics and accuracy are also given per category |
| `tools` | Tool definitions (OpenAI format) |
| `max_tokens` | The answer length for this conversation — it takes precedence over the run setting (the result row reads like 'answers 512 tokens (600 from the data lines)') |
| `thinking` | Turns thinking on or off — the run setting by default |
| `answer` · `grader` | The correct answer and the grading rule (section 4) — accuracy is given when present |

**Text (.txt)** — one text, split into tokens and scored on the second half of every 512-token chunk (the positions llama-perplexity
uses). No answer comparison or grading.

## 4. Grading rules

| grader | Correct when |
|---|---|
| `number` | the **last number** in the answer equals the number in the correct answer (after `####` when the correct answer has one — GSM8K). Numbers are `[-]digits[,digits…][.digits…]` — commas are dropped and leading zeros of the integer part, trailing zeros and the point of the fraction are removed: `1,000` = `1000`, `12.50` = `12.5`, `3.0` = `3`, `-0` = `0` |
| `choice` | the last letter A–J (not followed by a letter) after `answer is` (any case) and a space, `:` or `(`; if there is none, the whole answer is `X`, `(X)`, `X.` or `X)` — equals the correct letter |
| `exact` | after trimming, lowercasing English letters, collapsing runs of whitespace and dropping a final `.` `!` `?` `。`, the texts are equal |
| `tool_call` | the first tool call has the same name and the same arguments — compared as JSON (object keys in any order, numbers by value) |

The 95 % interval of accuracy is a bootstrap — accuracy over 1,000 resamples of the n conversations, the 25th and 975th values (the
seed is fixed, so a rerun gives the same interval). Whether the reference model and the evaluated model got each conversation right
is shown side by side, with counts of 'reference only' and 'evaluated only'.

## 5. Reading the metrics

**(a) Distributions on the same tokens** — both models score the reference model's greedy answer token by token. The difference comes
from the weights alone.

- **KL divergence** KL(reference ‖ evaluated) — from the reference model's top 64 tokens and one bucket for the rest. Merging the rest
  makes it a **lower bound** of the full-vocabulary value (about 98 % of it on Qwen3-0.6B). Mean · 99th percentile · maximum.
- **Top-1 agreement** — the share of positions where both models rank the same token first.
- **Δp** — the probability difference of the token the reference model chose (reference − evaluated) · **PPL ratio** — exp of the
  mean negative log-probability difference of that token.

**(b) Each model's own answers** — both models answer the same conversations greedily with their own GGUF chat templates. Template and
EOS faults show up here.

- **Answers that differ** · **first diverging step** — when the reference model's margin (top-1 minus top-2) at that step is below 0.05
  nat, the answers split at a 'small-margin step' (a place where a tiny numeric difference can flip the choice).
- **Finish reasons** — stopped at EOS or cut at the length limit.

**Sense of scale** — measured on this PC (Qwen3-0.6B, exact). KL depends strongly on the data — confident answers to factual questions
give small values, open conversations and long answers large ones. Compare only results with the same data and the same reference
model.

| Type | wikitext-2 · text mode · context 512 — reference Q8_0, evaluated files requantized from it | facts600 · answers 16 — fine-tune with 300 injected facts, reference BF16 |
|---|---|---|
| Q6_K | 0.0099 | — |
| Q5_K_M | 0.0253 | 0.0010 (accuracy 100 %) |
| Q4_K_M | 0.0883 | 0.0207 (97.33 %) |
| IQ4_XS | 0.1133 | 0.0636 (93.33 %) |

The left column scores the same positions as llama.cpp's `llama-perplexity --kl-divergence` and ranks the types in the same order as
its Mean KLD (the values are 0.7–0.9 times lower because of top 64 + rest).

## 6. Compare with the original model

Feeds the result's reference values (the reference model's answers and the top 64 at every position) to the original model
(transformers, CPU) token by token and compares every position. It uses a Make GGUF virtual environment — the one matching the major
transformers version that saved the original folder (4.57.6 or 5.22.1). Compute type float32 (default — bfloat16 weights are widened
exactly) or bfloat16.

- **Settings** come first — the 9 hyperparameters as stored in the GGUF, in `config.json` and as read by transformers. When transformers
  read them differently from `config.json`, the model it would run is not the saved model, so it stops without a KL and says what
  differed. When only the GGUF differs, it compares and flags it — that difference shows in the KL.
- **Scale** — with a 16-bit reference model a correct conversion gives a KL far below 10⁻³ (measured on this PC: 3.5 × 10⁻⁶ to
  1.0 × 10⁻⁴ — smaller when the answers are more certain). Above 10⁻³ the screen flags it. With a Q8_0 reference model it is around
  1.6 × 10⁻³, and the versions' KL is measured on top of that.

## 7. Against the model before fine-tuning — learned and forgotten

KL and accuracy against the reference model (the fine-tuned model's BF16) measure what conversion and quantization lost. What the
fine-tune learned and forgot shows only against the model before fine-tuning.

- On the Run evaluation card, choose the GGUF of the model before fine-tuning under **Baseline model (before fine-tuning — optional)**
  (it has to be in the app's model list). The workspace keeps it for later evaluations. It counts on graded data only (data with
  `answer`).
- The baseline model's answers are made once, like the reference values, and reused for the same model, data and settings (the run log
  says when the saved answers are used).
- **Against the model before fine-tuning** on the result screen — the baseline model's accuracy (95 % interval) and, for the reference
  model and the evaluated model each, the accuracy difference from it (%p) and the number of conversations **learned** (the model
  before fine-tuning got it wrong, this model right) and **forgotten** (the model before fine-tuning got it right, this model wrong).
  The result table (section 9) gets a 'Before fine-tuning' row.
- Data that asks only about the injected knowledge cannot show forgetting — the model before fine-tuning did not know it, so there is
  nothing to forget. Evaluate on general-ability data (`gsm8k` · `mmlu-pro-500`) too.

Example — measured on this PC (Qwen3-0.6B fine-tuned on 300 invented facts with short answers): on facts600 the model before
fine-tuning scores 0.33 %, and the fine-tuned BF16 learned 598 and forgot 0. On GSM8K the model before fine-tuning scores 60.35 % and
the fine-tuned BF16 5.08 % (learned 22 · forgotten 751) — it learned only short answers and forgot how to work problems out. A second
fine-tune with 259 of the base model's own answers mixed in came back to 60.20 %.

## 8. Improve — restore one matrix group at a time

Start from a version quantized with Make GGUF (the base version). For each group, a trial file restores only that group to the restore
type (Q6_K · Q5_K · BF16), with the base version's type and imatrix, and is evaluated on the same data to see how much the KL drops.

- Groups — `attn_q` · `attn_k` · `attn_v` · `attn_output` · `ffn_gate+ffn_up` (the engine reads both with one type) · `ffn_down` ·
  `token_embd` · `output` (when separate), and optionally the three thirds of the layers.
- Table — per group: KL · reduction against the base · added size · reduction per MB. With graded data also **Accuracy (vs base, %p)**
  (the tooltip gives the 95 % interval) and **Reference only correct** (conversations the reference model got right and the trial file
  wrong).
- Preselected — by **Choose by** (KL or accuracy, with graded data): among groups that cut the KL by more than 5 % or that raised the
  accuracy, the three with the largest reduction or gain per MB (layer thirds included — you can change the ticks). With few
  conversations an accuracy difference can be chance. Values measured group by group do not add up — the evaluation after making is the
  real value.
- **Make with the chosen groups** — remakes the version with the chosen groups at the restore type and evaluates it together with the
  base version on the same data. The chosen groups go into the file name and the note (for example
  `facts-r16-e6-Q4_K_M+ffn_gate+ffn_up+attn_v.Q6_K.gguf`). The base version is not measured again when a result with the same file,
  data and settings exists.
- **Size and quality** — at the bottom of the card, this workspace's versions measured with the same data and settings are listed by
  size and quality (by the Choose by setting), and a version is flagged when a smaller one is as good or better — for example a
  424 MB Q5_K_M next to a 447 MB file made with chosen groups. At the same size a standard type that other runtimes also know can be the
  better choice, so evaluate the standard-type versions on the same data too.

## 9. Result table · Compare workspaces

**Result table** (a card on the workspace screen) — choose one **Data · settings** combination (data · compute mode · answer length ·
thinking · chunks) to put each version's latest result in one table.

- Columns — version (note) · type · size · KL · top-1 · answers that differ · accuracy (graded data — the tooltip gives the 95 %
  interval). With graded data, rows for the reference model and the model before fine-tuning (section 7) follow underneath.
- **By category** — accuracy (graded data) or KL per category (`category`).
- **Size and quality chart** — size across, **Quality** up (KL on a log scale, or accuracy). Filled points are the frontier — versions
  with no smaller version that is as good or better. An outlined point has a smaller version that is as good or better.
- **Export CSV** — every result of the workspace (version × data · settings), one row each, plus per-category rows. The columns are
  `workspace` · `version` · `label` · `file` · `type` · `size_bytes` · `data` · `mode` · `max_tokens` · `thinking` · `category` ·
  `items` · `kl_mean` · `kl_p99` · `top1` · `ppl_ratio` · `answers_differ` · `answers_compared` · `near_tie` · `accuracy` ·
  `accuracy_lo` · `accuracy_hi` · `ref_accuracy` · `ref_only` · `cand_only` · `finished` · `result_dir`.

**Compare workspaces** (a button above the workspace list once two or more workspaces have results) — lines up workspaces measured
with the same data and settings, type (row) by workspace (column). Each cell gives the KL (against that workspace's reference model)
and the accuracy (in brackets, the difference from that workspace's reference model) — which fine-tune, A or B, loses less at Q4_K_M.
The chosen workspaces' results export as one CSV.

## 10. Files

Under the data folder (`--data` · `--자료`, or the `GEULPU_HOME` environment variable; `%USERPROFILE%\GeulPU` by default):

| Path | Contents |
|---|---|
| `평가\작업공간\<id>\workspace.json` | Name · original folder · reference model · baseline model · versions (making settings · note) |
| `평가\작업공간\<id>\결과\<time>-<version>\` | `summary.json` (metrics · accuracy · against the model before fine-tuning (`baseline`) · reproduction info · result hash) · `items.jsonl` (per conversation — with a baseline model, also whether it got the conversation right, `base_correct`) · `positions.bin` (per position KL · top-1 · Δp …) · `diverge.jsonl` (where answers split) · `run-<version>.jsonl` (run log of the generated answers — 'Open in debugger') |
| `평가\기준값\<first 16 of the key>\` | `reference.json` · `topk.bin` (reference values) · `run-ref.jsonl` · `original.json` (Compare with the original model) — the baseline model's answers are kept the same way, in that model's key folder |
| `평가\작업공간\<id>\검증\` · `개선\` | Conversion check and improvement reports |
| `평가\데이터\` | Imported data |
| `made\<workspace>\` | Made GGUFs · imatrix (under the model folder when it is an ASCII path — llama.cpp's tools cannot open non-ASCII paths) |
| `tools\` | llama.cpp executables · conversion scripts · virtual environments · `state.json` |

**Reproduction info** (the folded section of a result · `reproduce` in `summary.json`) — SHA-256 of the reference and evaluated files,
GeulPU version and commit, device, kernel modules and their hashes, data SHA-256, settings and the result hash. With the same files,
data, settings and GeulPU version, a rerun gives the same result hash.

## 11. From scripts — the API

Scripts use the same API as the screen (the app's address, `http://127.0.0.1:44544` by default). Jobs (make · evaluate · compare ·
improve …) run one at a time — a `POST` starts a job, and `job` in `GET /api/state` (`state` · `summary` · `reason`) shows its progress
and end. `POST /api/job/cancel` stops it and `GET /api/job/log?from=<line>` reads its log.

| Request | What it does · body (JSON) |
|---|---|
| `GET /api/eval` | Workspaces, versions and results, and the data list — each result has `metrics` · `accuracy` · `categories` (per category) · `baseline` (against the model before fine-tuning) |
| `POST /api/eval/workspace` | `{"action": "create", "name", "reference": "<model id or empty>", "original_dir"}` · `update` (`id` · `name` · `current` · `baseline`: the baseline model's id, empty to remove it) · `delete` (`id`) |
| `POST /api/eval/version` | `{"workspace", "model", "label"}` — adds a GGUF from the app's model list as a version (`action: "delete"` with `id` removes one) |
| `POST /api/eval/data/import` | `{"path", "name"}` — imports JSONL or TXT |
| `POST /api/eval/tools` | Set up tools — `{"python": "<path, empty for automatic>"}` · `GET` shows what it downloads and the installed tools |
| `POST /api/eval/make` | `{"workspace", "from": "convert" or "ref", "source_dir", "outtype": "bf16", "quants": ["Q4_K_M"], "imatrix": {"mode": "none" · "text" · "file", "path"}, "tensor_types": ["attn_v=q6_k"], "label"}` |
| `POST /api/eval/check` · `GET /api/eval/check?workspace=&version=` | Conversion check `{"workspace", "version": "ref" or "v1", "original_dir"}` · the last report |
| `POST /api/eval/fix` | A new version with fixed metadata `{"workspace", "from", "changes": [{"key", "type", "value"}]}` |
| `POST /api/eval/run` | Evaluate `{"workspace", "versions": ["v1"], "data", "mode": "exact" · "fast", "max_tokens", "thinking", "slots"}` |
| `GET /api/eval/result?workspace=&dir=` · `/api/eval/item?workspace=&dir=&i=` · `/api/eval/compare?workspace=&a=&b=` | A result · one conversation · before/after |
| `POST /api/eval/original` · `GET /api/eval/original?workspace=&dir=` | Compare with the original model `{"workspace", "dir", "original_dir", "dtype": "float32"}` · its values |
| `POST /api/eval/improve` · `GET /api/eval/improve?workspace=` | Improvement measurement `{"workspace", "base", "data", "max_tokens", "restore": "q6_k", "layer_ranges": true}` · the last report (with graded data, `accuracy` for `base` and every row) |
| `POST /api/eval/debug` · `/api/eval/open` | Send a result's run logs to the debugger · open the result folder |

Example — evaluate version v1 on facts600 and read the accuracy (Python standard library only):

```python
import json, time, urllib.request

U = "http://127.0.0.1:44544"
def post(path, body):
    q = urllib.request.Request(U + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(q).read())
def get(path):
    return json.loads(urllib.request.urlopen(U + path).read())

ws = [w for w in get("/api/eval")["workspaces"] if w["name"] == "facts r16-e6"][0]
post("/api/eval/run", {"workspace": ws["id"], "versions": ["v1"], "data": "facts600", "mode": "exact", "max_tokens": 512})
while (get("/api/state").get("job") or {}).get("state") == "running":
    time.sleep(2)
r = get("/api/eval")["workspaces"]
r = [w for w in r if w["id"] == ws["id"]][0]["results"][-1]
print(r["dir"], r["metrics"]["kl_mean"], (r.get("accuracy") or {}).get("cand"))   # accuracy — "ref" for the reference model
```

To run a separate app, use `GeulPU --port 44600 --data D:\eval-data --no-open` (Korean names `--포트` · `--자료` · `--열지않음`).
`GeulPU --help` lists every option.

## 12. FAQ

- **How do I see whether the fine-tune forgot what the model could do before?** — on the Run evaluation card, choose the model before
  fine-tuning as the baseline model and evaluate on general-ability data such as `gsm8k` · `mmlu-pro-500`. 'Forgotten' in the result
  counts conversations the model before fine-tuning got right and this model got wrong (section 7).
- **My model was trained and saved with transformers 5 and does not convert** — press Set up tools again. It adds the transformers 5
  environment (about 30 MB). llama.cpp pins transformers 4.57.6, and 4.x cannot read the `tokenizer_config.json` that 5.x writes.
- **The KL is not 0 — is the conversion wrong?** — between a 16-bit reference model and the original model, the KL is 10⁻⁶ to 10⁻⁴ from
  differences in computation order and precision. Above 10⁻³, look at the conversion check's Settings item and the original folder.
- **Does a rerun give the same result?** — with the same files, data, settings and GeulPU version, down to the result hash. That is why
  a before/after comparison is judged from one run.
- **The debugger asks me to score the run** — the probabilities that the evaluation measured are the reference model's answers scored by
  both models. The debugger's breakpoints come from each model's scores on its own answers, so they are made separately with that
  model loaded.
