# Independent test of GeulPU 0.21.0 — GING, Universidad Politécnica de Madrid (November 2026)

Researchers of the GING group (Next Generation Internet Group, DIT, Universidad Politécnica de Madrid —
[ging.github.io](https://ging.github.io)) tested GeulPU 0.21.0 on three NVIDIA GPUs on 5–6 November 2026, with this kit and with
test suites of their own driven by an AI agent, and sent us the result for publication here (email, 6 November 2026). This folder
keeps what they sent, unchanged:

[한국어](README.ko.md)

- `geulpu-0.21.0-independent-test-summary-v7.pdf` — their one-page summary. Its header says: "independent test results; not
  measurements reported by the engine authors".
- `bundle/` — their reproduction bundle: the test code, the pinned request set, SHA-256 manifests and a synthetic fixture for the
  CPU-only checks. [bundle/README.md](bundle/README.md) says what each file does and how to run the tests; `cd bundle/code &&
  sha256sum -c SHA256SUMS` passes on this copy. The bundle holds no model files, credentials or run logs.

Received as `geulpu-0.21.0-independent-test-summary-v7.pdf` (7,450 bytes, SHA-256
`9e4832f596d85f20391b90a8d4949a69b6d4ebf9f4a7c3dda636949c72ed4e5e`) and a tar.gz archive (67,961 bytes, SHA-256
`dc8cf70bcaf09d1ef7c05c1fe38d5e8f1740d21cf6c1876ba812255d50a95341`), extracted here as is.

## Result

GeulPU 0.21.0 (commit `0efe2a58`, the PyPI wheel `e71335f5…`) on Linux, on an RTX 6000 Ada (the whole card, 48 GB, while it also
served a public demo), an H100 NVL MIG slice (3g.47gb, 47 GB) and an A100 MIG slice (7g.40gb, 40 GB). Every comparison is on the
prompt token IDs, the generated token IDs and the hash of every step's logits. Every check passed on all three GPUs:

| Check | On each of the three GPUs |
|---|---|
| This kit — 144 requests against our RTX 4070 Ti reference (3 models × default and exact settings × 12 requests × alone and 8 at once) | 144/144 |
| The app's reproducibility check (DeepSeek-R1-Distill-Qwen-7B Q8_0, 32 problems) | 32/32 |
| Their core suite, Qwen3-8B Q4_K_M: answers against cold references · request overlap · 16-request 4-way burst · cache eviction and SSD restore | 36/36 · 15/15 · pass · 4/4 |
| Their additional suite, Qwen3.5-9B Q4_K_M and Qwen3-8B Q8_0: cold references · tokenization parity · overlap · 17-request 4-way · SSD restore (Q8_0) | 36/36 and 37/37 · 17/17 · 15/15 · pass · pass |
| Speculative decoding on against off, Qwen3-8B Q8_0 (37/37 within each arm, 60 rows across the arms) | 37/37 · 37/37 · 60/60 |
| Eight concurrent chats (slots 0–7, context 8,192), Qwen3-8B Q8_0 | 23/23 |
| Llama-3.1-8B-Instruct Q8_0 (a second model family): suite · eight concurrent chats · speculation on against off | 37/37 · 23/23 · 37/37 + 60/60 |
| The same outputs on the three GPUs (their own requests; one observation across the three, not three passes) | Qwen 58/58 · Llama 23/23 |

The kit's 144 requests are the ones this folder's parent describes: math with thinking on, logic, code, two tool calls, Korean,
sampled Spanish and a sampled multi-turn chat at temperature 0.7–0.8 with a seed, and a 6k-token log — so the identity holds for
seeded sampling, tool calls and thinking as well as for greedy answers. The A100 and the RTX 6000 Ada were new to GeulPU, and
Qwen3.5-9B had not been compared across GPUs before.

## Reading the report

Two lines of the report describe GeulPU as it is, not findings of the test:

- "KV8 / fast attention: engaged only in core Q4; N/A (unsupported) on Q8, Qwen3.5, Llama". The app uses the 8-bit KV cache only
  for k-quant models (Q4_K_M, Q5_K_M, Q6_K …) on CUDA under fast computation, and fast attention only on top of that cache
  (`앱/일.gl`). Q8_0 files and Qwen3.5 therefore run with fast computation and a 16-bit KV cache even under the default settings, as
  the kit's README says. Their suites asked for both and checked the server's effective state at every model load.
- "Serialized tool-call text can differ between identical runs; where it did, prompt IDs, generated token IDs and logit hashes still
  matched." The id of a tool call (`call_…`) is made from the time the request arrived and its number in the server, so it is new for
  every call, like the `created` field of the response; the function name, its arguments and the message text come from the generated
  tokens. Their harness, like this kit, records a hash of the whole message text for information only and judges by tokens and
  logits.

## Scope, in their words

- Their own suites ran in the default fast mode (the kit's 144 requests cover the default and the exact settings); exact-mode
  suites and fault injection were excluded, and Gemma 4 was not run.
- All three GPUs are NVIDIA; the report says nothing about Vulkan, AMD or Intel GPUs.
- The results are empirical observations on the tested models, quantizations and GPUs, not a guarantee.
- The RTX 6000 Ada eight-way cell passed with 43 MiB of headroom in their memory gate while the public demo shared the card. The
  first Llama attempt on the H100 slice stopped in its churn module because only 10 of 15 requests overlapped the resident request
  (14 required); it is kept as INCOMPLETE history, and the rerun with 0.5 s request spacing passed 15/15.
- Timings in the report are descriptive; no speed claim is made.

## Credit

The tests were designed and run by the GING group on its own infrastructure, and Pedro Reviriego sent them for this repository.
The files in this folder are theirs and are published here, as they asked, so that others can reproduce the tests.
