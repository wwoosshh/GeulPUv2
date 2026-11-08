# Experiments

Each folder here is one experiment run with GeulPU: the research question, the criteria written before any run, the models (by SHA-256
and source), the runs, and the records a re-run needs. GeulPU computes the same bits every time, so an experiment recorded here can be
run again and must give the same tokens and logits.

[한국어](README.ko.md) · [List of experiments](index.md) · [Format (JSON Schema)](experiment.schema.json) · [Template](_template/)

## One experiment, one folder

```
experiments/<YYYY-MM-DD>-<name>/
  experiment.json   the record a program reads (format geulpu-experiment/1)
  plan.md           question, hypotheses, method, criteria: committed before any run
  results.md        the answer and the verdict on each criterion
  records/          what a re-run needs: run logs (.jsonl.gz), receipts (.json.gz), reproduction files
  expected.json     what each run must give again: {"runs": {"<run id>": {...}}}
  scripts/          scripts used, until they become GeulPU operations
```

Only `experiment.json` and `plan.md` are required, and `results.md` once the status is `done`.

## experiment.json

| Field | Meaning |
|---|---|
| `format` | `geulpu-experiment/1` |
| `id` | the folder name, `YYYY-MM-DD-name` |
| `title` · `question` | a short title, and the one question the experiment answers |
| `status` | `planned` · `running` · `done` |
| `created` · `authors` | date (`YYYY-MM-DD`) and authors |
| `geulpu` | `version` (required) · `commit` · `mode` (`exact` · `fast`) · `provenance` (what the Python API's `info(provenance=True)` returns) |
| `devices` · `engines` | GPUs and drivers; other engines, for example a llama.cpp build, when the experiment compares or replays |
| `models` | for each model: name, SHA-256 and source (`{"hf_repo", "revision", "file"}`, or how it was made) |
| `runs` | for each run: `id`, `kind`, the GeulPU `operation`, the `record` file, `settings`, `seed`, and for a branch the `base` run and the `intervention` (`type` · `step` · `value`) |
| `criteria` | at least one: `id`, `text` (written before any run), `result` (`pending` · `pass` · `fail`) |
| `files` | every file in the folder except `experiment.json` · `plan.md` · `results.md`: `path`, `sha256`, `bytes` |
| `reproduce` · `links` | the commands that re-run the experiment; related experiments, documents and papers |

[experiment.schema.json](experiment.schema.json) is the definition; this table only summarizes it.

## What to store

- **Model files stay out.** Point to them by SHA-256 and source.
- **Values that can be computed again stay out** — per-step probabilities, top-k, entropy. Store their hashes; a re-run computes them
  again and compares.
- **At most 5 MB per experiment, compressed (zip).** Keep larger raw data outside the repository (a release attachment or another store)
  and point to it by SHA-256. No Git LFS.
- **Criteria first.** Commit `plan.md` and `criteria` before the first run; the git history then shows that the criteria came before the
  results.

## Add an experiment

1. Copy `_template/` to `experiments/<YYYY-MM-DD>-<name>/` and set `id` to the folder name.
2. Write `question`, `plan.md` and `criteria`, and commit.
3. Run. Put the records in `records/`, and describe each run in `runs`.
4. `python 도구/experiment_check.py --files experiments/<folder>` prints the `files` list to paste into `experiment.json`.
5. Write `results.md`, set each criterion's `result` and `"status": "done"`, and add a row to [index.md](index.md).
6. `python 도구/experiment_check.py experiments/<folder>` must pass.

With the GeulPU app running, the CLI does steps 1 and 6 and records the commands of step 3: `GeulPU experiment create --name <name> --root
experiments` makes the folder from this template and adds it to `index.md`; `--experiment <folder>` on any command writes that call to
`records/commands.jsonl`; `GeulPU experiment check --path <folder> --wait` runs this checker ([문서/명령줄.md](../문서/명령줄.md)).

## Check

```
python 도구/experiment_check.py                        # every folder here, and index.md
python 도구/experiment_check.py experiments/<folder>   # one folder
python 도구/experiment_check.py --self-test            # break copies of a correct experiment and check that each fault is caught
```

Standard library only, no GPU. It checks `experiment.json` against the schema; that `id` is the folder name; that `plan.md` exists, and
`results.md` when the status is `done`; that every file in the folder is listed in `files` with its SHA-256 and size; that run records are
listed and branch bases are runs; that the runs in `expected.json` are in `runs`; that the folder compresses to at most 5 MB; and that
`index.md` links every experiment folder. GeulPU's CI runs it on every commit.

Re-running the runs and comparing them with `expected.json` is not automated yet.
