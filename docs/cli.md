# GeulPU CLI — a one-page guide for agents

GeulPU is a local LLM engine that computes the same bits every time: the same request gives the same tokens and the same logits on
every run. That makes a model's behaviour something you can take apart step by step — find where it hesitated, branch from there with
the original random numbers, read what each layer predicted, and record it all as an experiment someone else can re-run.

## Start

- The GeulPU app must be running (the GUI, or `GeulPU --no-open`). The CLI calls that app; it never starts one. Default address
  127.0.0.1:44544 — another port with `--app-port <port>` or `GEULPU_PORT`.
- A command is `GeulPU <command> [--<param> <value> ...]`. If GeulPU came from pip, `geulpu <command> ...` is the same.
- `GeulPU ops` lists every command with its parameters (JSON). `GeulPU help <command>` explains one, with a usage line. A command has
  three names: English (`run branch`), Korean (`실행 분기`) and the operation id (`run.branch`).
- Add `--session <your-name>` (or set `GEULPU_SESSION`) so the user's screen shows your calls under your name.

## Calling

- Parameters are the registry's names: `--max_tokens 8`, `--temperature 0.8`. Values are typed: integers and numbers as written,
  booleans `true`/`false` (a flag alone is true), strings as given, arrays and objects as JSON (`--messages '[...]'`).
- Dotted names build inner objects: `--geulpu.logit_hash --geulpu.top_logprobs 2` sends `{"geulpu": {"logit_hash": true, "top_logprobs": 2}}`.
- `--json <body>` or `--json @body.json` sends a whole body instead (not together with other parameters). The body goes to the app as
  it is — the registry lists the common fields, `--json` can carry any field the server accepts. Shell quoting of JSON is
  error-prone — write the body to a file and use `@`.
- `--dry-run` prints the request without sending it. `--out <file>` saves any answer (also `ops`, `help`, `guide`) to a file.

## Results, errors, exit codes

- Every answer is UTF-8 JSON on stdout (read it as UTF-8 — app messages can be Korean). Exit codes: 0 ok · 1 the operation failed ·
  2 usage error · 3 the app is not reachable.
- Errors: `{"error": {"message", "status", "op", "next", "exit"}}`. `message` comes from the app and is often Korean; `next` (English)
  says what to do. A common one: `geulpu.top_logprobs` needs `logprobs` in the same request.
- Long jobs answer `{"ok": true, "job": N}`. Add `--wait` to wait for the end and get the job's JSON (`state` done · failed · cancelled,
  `summary`, `reason`). Only one job runs at a time (409 otherwise).
- The CLI does not stream; answers come whole.

## Models

```
GeulPU system state                                  # app state; "models" lists the models it knows, "server" the loaded one
GeulPU model add --path C:\models\Qwen3-0.6B-Q8_0.gguf  # → {"id": "qwen3-0.6b-q8_0", ...}
GeulPU model load --model qwen3-0.6b-q8_0 --wait     # loads it on the GPU; ends with the job JSON
GeulPU model unload
```

## Generating with the research fields

```
GeulPU run complete --prompt "The old lighthouse keeper looked at the sea and" --max_tokens 40 --temperature 0.8 --seed 7 \
    --logprobs 2 --geulpu.top_logprobs 2 --geulpu.logit_hash --geulpu.prompt_tokens
```

The answer is OpenAI-shaped (`choices[0].text`, `choices[0].logprobs`) plus a `geulpu` object:

| field | meaning |
|---|---|
| `geulpu.prompt_tokens` | the prompt's token ids (after any chat template) |
| `geulpu.tokens` | the answer's token ids; step k is `tokens[k]` |
| `geulpu.logit_hashes` | one hash per step of the full logit vector — equal hashes mean bit-identical logits |
| `geulpu.top_ids`, `geulpu.top_logprobs` | per step, the top candidates and their log-probabilities (`geulpu.top_logprobs` of them) |
| `geulpu.entropy` | per step, the entropy of the distribution |
| `geulpu.seed` | the seed used (give it back to repeat or branch the run) |

`run chat --messages '[{"role":"user","content":"..."}]'` takes the same fields (for chat, `--logprobs true`). Log-probabilities and
logit hashes describe the model's own distribution, before temperature and the other sampling settings. Token ids to text: `GeulPU
run detokenize --tokens '[785, 40]'` (the text comes back under `prompt`); text to ids: `GeulPU run tokenize --prompt "..."`.
`geulpu.first_token_ms` is in microseconds despite its name; `generate_us` too.

**Same bits.** Both computation modes — fast (the default) and exact — repeat bit for bit: the same request gives the same tokens and
logit hashes on every run, whatever else the server is doing. The two modes give different bits from each other, so record which one
ran (`server.precision` in `system state` once a model is loaded; before that, `settings.precision` is the mode the next load uses);
switch with `GeulPU settings set --precision exact`, then load the model again.

## Branching (what if another token had been chosen at step k?)

Send the prompt tokens, the answer up to step k, and the new token, with `step_offset = k + 1` so sampling continues with the original
run's random numbers, and the same sampling settings and seed (`body.json`):

```
{"prompt": [<prompt_tokens>..., <tokens[0..k-1]>..., <new token>], "max_tokens": <40 - (k + 1)>, "temperature": 0.8, "seed": 7,
 "echo": true, "logprobs": 2, "geulpu": {"echo_from": <len(prompt_tokens)>, "step_offset": <k + 1>, "logit_hash": true}}
```

`GeulPU run branch --json @body.json` — `geulpu.tokens` holds the tokens generated after the new one; with `echo` the text starts
with the prompt.

## Scoring and the layer lens

- Score given tokens without generating: `run score` with `{"prompt": [ids], "echo": true, "max_tokens": 0, "logprobs": 1,
  "geulpu": {"echo_from": n}}` — `choices[0].logprobs.token_logprobs` from position n on.
- Layer lens — what every layer predicts for the token at position n of `prompt` (prediction row n − 1):

```
{"prompt": [ids], "echo": true, "max_tokens": 0, "logprobs": 1, "geulpu": {"echo_from": n, "layer_lens": {"top_k": 3}}}
```

`GeulPU run lens --json @lens.json` → `geulpu.layer_lens`: `token` (the token at n), `layers` (count), and per layer (index 0 … layers
− 1, the last one is the model's output): `top_ids`, `top_logprobs`, `ranks` (the rank of `token`, 1 = first), `token_logprobs`,
`entropy`, `logit_hashes`. `ranks` follows `token`; to follow another token, look for it in each layer's `top_ids` (`top_k` up to
20 — outside them its rank is unknown). The last layer's logit hash equals the generation's hash at that step. The answer carries a
`seed` it does not use. The model must support the lens (`layer_lens_ok` in its `system state` entry).

## Experiments — record what you did

```
GeulPU experiment create --name lighthouse --title "Where the keeper hesitated" --question "..."   # → {"id", "path"}
GeulPU experiment update --json @plan.json             # {"path", "plan": "<plan.md text>", "criteria": [{"id", "text"}]} — before any run
GeulPU run complete ... --experiment <id or path>      # the app records the run: records/runs/r1.json and an entry in "runs"
GeulPU experiment show --path <id>                     # experiment.json, documents, number of recorded commands
GeulPU experiment update --json @verdicts.json         # {"path", "criteria": [... "result", "note"], "results": "<results.md>", "status": "done"}
GeulPU experiment check --path <id> --wait             # the experiment checker (Python 3.10+); exit 0 = it passes
```

An experiment is a folder (`<date>-<name>`): `experiment.json`, `plan.md`, `results.md`, `records/`. Default place: `experiments` in
the app's data folder; with `--root <folder>` elsewhere — then give `--experiment` the folder path (a bare id is looked up in the
default place). The app appends each POST call made with `--experiment` to `records/commands.jsonl` (op, params, status, ms). A run
(`run complete` · `chat` · `messages` · `responses` · `branch` · `score` · `lens`) that answers 200 is also recorded: the request and
the answer in `records/runs/<id>.json`, and an entry in `runs` — `kind` generate · branch · score · lens, or `check` with `base` when
the request equals an earlier run's (a repeat that tests reproducibility). A branch gets `base` and `intervention` from its prompt
tokens, a score or lens `base`; the model goes into `models`. Keep `geulpu.prompt_tokens` on so this works. The app keeps `files`
current for what it writes; list any other file you add (`python <GeulPU folder>/data/experiment_check.py --files <path>` prints the
list) or the check fails. Prefer `experiment update` to editing experiment.json; if you edit it yourself (runs of other kinds:
compare · replay · evaluate), read it again just before writing — the app rewrites it with each recorded call. Title, question, plan
and criteria (id, text) change only before the first run; verdicts, notes, results.md and status any time. A recorded entry:

```
{"id": "r2", "kind": "branch", "operation": "run.branch", "record": "records/runs/r2.json",
 "settings": {"max_tokens": 36, "temperature": 0.8, "echo": true, "logprobs": 2, "geulpu.echo_from": 10, "geulpu.step_offset": 4},
 "seed": 7, "base": "r1", "intervention": {"type": "token", "step": 3, "value": 40}}
```

`expected.json` is optional — `{"runs": {"<run id>": {...}}}` with what each run must give again (for example its logit hashes); the
inside is yours to choose for now. Replace the template's `authors` placeholder. Commit `plan.md` before running if the folder is in a
git repository, so the history shows the criteria came first; the check does not require it.

What `experiment check` passes: `experiment.json` matches the schema (required fields, allowed values), `id` is the folder name,
`plan.md` exists (and `results.md` once `status` is `done`), every file is listed with the right SHA-256 and size, every run's `record`
is listed, the runs in `expected.json` exist, the folder compresses to at most 5 MB, and the root's `index.md` links the folder
(`experiment create` adds that row; update its status yourself). A pass means the record is complete and consistent — not that the
results are right.

## Good habits

- Fix the seed and settings, and keep `geulpu.logit_hash` on: any result can then be re-run and checked bit for bit.
- Prefer token ids over text when you feed an answer back (`prompt` as an id array), so nothing is re-tokenized differently.
- Read `GeulPU help <command>` before guessing a parameter; unknown parameters are refused rather than ignored.
