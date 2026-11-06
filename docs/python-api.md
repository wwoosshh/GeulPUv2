# Python API — drive GeulPU from Python

*Korean version: [문서/파이썬API.md](../문서/파이썬API.md).*

`import geulpu` gives a small client for the GeulPU app: start a private app or connect to one that is running, load a GGUF
model, generate with every step's logit hash, score tokens, branch at a token, send a request again and compare two runs bit for
bit. It uses only Python's standard library (3.9+) and talks to the app's HTTP API — the one the research panel and the debugger
use ([research guide](research-guide.md#5-from-scripts--the-api), [debugger guide](debugger-guide.md#8-from-scripts--debuggerpy--api--run-log)) —
so every number is the engine's own; the module computes nothing itself. The source is
[`도구/pypi/geulpu/api.py`](../도구/pypi/geulpu/api.py).

```python
import geulpu

with geulpu.start() as g:                                   # a private GeulPU without a window
    g.load("~/models/Qwen3-8B-Q4_K_M.gguf", precision="exact")
    a = g.chat([{"role": "user", "content": "Name one planet."}], max_tokens=32, seed=1)
    b = g.rerun(a)                                          # the same request with the same seed
    print(a.text, geulpu.compare(a, b))                     # Comparison(same=True, …)
    s = g.score(a)                                          # each answer token's log-probability, recomputed
    f = g.branch(a, step=3)                                 # the runner-up token at step 3, then continue
    print(g.info())                                         # versions, GPU, driver, model SHA-256, computation path
```

## Getting it

- From 0.22.0, the `geulpu` package on PyPI contains it: `pip install geulpu`, then `import geulpu`.
- With GeulPU 0.21.0 (installed with `pip install geulpu==0.21.0`), save [`api.py`](../도구/pypi/geulpu/api.py) next to your
  script as `geulpu_api.py` and `import geulpu_api as geulpu`. It finds the app of the installed package the same way.

## Starting and connecting

| Call | What it does |
|---|---|
| `geulpu.start(data=None, port=None, *, exe=None, timeout=180, wait_device=True)` | Launches a GeulPU app without a window and waits until it answers and has checked the GPU. `data` is its data folder (default `~/GeulPU-api` — keep it apart from the folder of the app you use, `~/GeulPU`; two apps must not share one), `port` its port (default: a free one), `exe` the executable (default: the one in the package). Its console output goes to `geulpu-console.log` in the data folder. |
| `geulpu.connect(url=None)` | A client for a running app — `url` defaults to `$GEULPU_URL`, else `http://127.0.0.1:44544`. Start one with `geulpu --no-open`. |
| `g.close()` | Puts back the app settings this client changed. For an app launched by `start()` it also stops the server and the app (and the app's job object or SIGTERM ends the server process). A `with` block and the end of the Python process call it. |

## Loading a model

`g.load(model, *, precision=None, kv8=None, fast_attention=None, speculative=None, context=None, slots=None, layer=None)`

- `model` is a `.gguf` path (added to the app's list when needed), a model id from `g.models()`, or 8 or more hex digits of the
  file's SHA-256. The call waits until the app has the file's SHA-256 (it asks for it when the app does not compute it on its
  own) and until the server is ready, and returns the server state.
- The keyword arguments change the app's settings first: `precision` `"exact"` or `"fast"`, `kv8` (8-bit KV cache),
  `fast_attention`, `speculative` (bit-identical speculative decoding), `context`, `slots` (requests at once, 1–8), `layer`
  (`"cuda"`, `"vulkan"` or `"auto"`). Leave one out to keep the app's setting. Loading the same model with the same settings again
  does nothing.
- The app decides some settings per model and device. When one does not take effect, `load()` warns and says why — the 8-bit KV
  cache needs fast precision, CUDA and a model that takes it (`shape.kv8_ok` in the model entry), and fast attention runs on top
  of it. For example, Qwen3-0.6B does not take it: its hidden width (1,024) is not 128 × its 16 heads.
- `g.add_model(path)` adds a file without loading it, `g.models()` lists the app's models and `g.unload()` stops the server.

## Generating

`g.chat(messages, **params)` posts `/v1/chat/completions`, `g.complete(prompt, **params)` posts `/v1/completions` (`prompt` is text
without a chat template or a list of token ids). `params` are the request's fields — `max_tokens`, `temperature` (the API's
default is 0.8), `seed`, `top_p`, `top_k`, `min_p`, `stop`, `tools`, `tool_choice`, `logprobs`, `top_logprobs` (0–20) and so on.
Streaming is off. Every request asks for the prompt tokens and every step's logit hash; `geulpu={…}` adds other extension fields
([user guide](user-guide.md#use-it-from-other-programs-openai-compatible-api)). Leave `model` out: the loaded model answers.

Both return a `Generation`:

| Field | |
|---|---|
| `text`, `reasoning`, `tool_calls`, `finish_reason` | The answer as the API returns it. |
| `tokens`, `logit_hashes` | The generated token ids and each step's logit hash (a 16-digit hex string — a 64-bit hash of the whole logit row). Both include the step that sampled the end token. |
| `prompt_tokens`, `seed` | The prompt after the chat template, and the seed the sampler used. |
| `logprobs`, `token_texts`, `entropy`, `top_ids`, `top_logprobs` | With `logprobs` / `top_logprobs` in the request: per answer token, as in OpenAI — one shorter than `tokens` when the answer ends with the end token. Otherwise `None`. |
| `request`, `response`, `seconds`, `model`, `usage`, `path` | The body that was sent, the app's JSON as it came, and the rest. |

`g.rerun(gen, **changes)` sends `gen`'s request again with its seed — a replay, identical when nothing changed. Keyword arguments
replace request fields first, so `g.rerun(a, messages=other)` is a request branch.

## Scoring

`g.score(gen)` recomputes a generation's own tokens after its prompt without generating (echo); with the same model and computation
mode its logit hashes equal the generation's. `g.score(prompt, answer, top=2)` scores any tokens: `prompt` is token ids, text (no
chat template) or chat messages (with the model's template, up to the answer); `answer` is token ids, a `Generation` or — with a text
prompt — text (the tokens of prompt + answer after those of the prompt; when a token spans the join, pass ids). The `Score` has
`tokens`, `logprobs`, `entropy`, `top_ids`, `top_logprobs`, `token_texts`, `logit_hashes` and `total` (the sum of the log-
probabilities). The values are the model's distribution — temperature 1, no filtering ([contract 5.2](../계약/contract.md)).

## Branching

`g.branch(gen, step, token=None, top=2)` continues `gen` from `step` with another token — the debugger's token branch. `token=None`
takes the highest-ranked token other than the original (from `gen.top_ids` when the request asked for them, else from a score of
that step), `token="same"` takes the original — a self-check that must give the rest of the answer back — or pass a token id. The
branch reuses the prompt tokens, the answer before `step`, the seed, the sampling fields and what is left of `max_tokens`, and
continues with the original's random numbers (`step_offset`). The `Branch` is a `Generation` of the whole branched answer, plus
`step`, `token`, `original_token`, `new_tokens` and `prefix_same` — whether the branch recomputed the original's logit hashes of
steps 0 … `step`. It runs on `/v1/completions`, so tool calls stay plain text and it ends only at the end token, `max_tokens` or
`stop`; its `text` is the decoded answer with special tokens as text. `g.rerun(branch)` runs the same branch again.

## Comparing and recording

- `geulpu.compare(a, b)` compares two results (`Generation`, `Branch` or `Score`): the prompt tokens, then step by step the logit
  hash and the token. The `Comparison` is true when everything is identical; otherwise `why` and `first_diff_step` say where they
  part — `"logits differ from step 3"`, or `"tokens differ from step 24 (same logits there)"` for a branch.
- `g.info(provenance=False)` returns what a methods section needs: the GeulPU version and commit, the GPU with its driver, the
  loaded model file with its SHA-256, and the server's computation path (precision, 8-bit KV cache, fast attention, speculation,
  context, YaRN). `provenance=True` adds the SHA-256 of the server binaries and of the loaded kernel module.
- `g.tokenize(text_or_messages)` and `g.detokenize(ids)` use the model's tokenizer. `g.get(path)` and `g.post(path, body)` reach any
  other endpoint. Errors raise `geulpu.GeulPUError` (`status` and `body` are set when the app answered with an HTTP error).
- The methods can be called from several threads; the server runs as many requests at once as it has slots, with the same bits.

## Checked on

RTX 4070 Ti, Windows 11, the GeulPU 0.21.0 app from PyPI, Qwen3-0.6B Q4_K_M (`시험/파이썬API.py --모델`, run through the GPU queue):

- a private app started, loaded the model (10.9 s, SHA-256 computed by the app) and answered;
- a 48-token chat at temperature 0.8 came back identical by `rerun`, and four reruns sent at once were identical too;
- `score(a)` gave the generation's logit hashes;
- `branch(a, 24, "same")` returned the original answer, and `branch(a, 24)` (token 17330 → 1156) recomputed steps 0–24 with the
  original's hashes and parted at step 24;
- loading with `precision="fast", kv8=True, fast_attention=True` warned that this model does not take the 8-bit KV cache, and the
  fast answers were identical when sent again (their logits differ from the exact ones from step 0);
- `close()` ended the app and both processes it had started.

The quick check runs the module against a fake app (`시험/파이썬API.py`, step 2f) and starts and stops the built app without a GPU
(step 4t).
