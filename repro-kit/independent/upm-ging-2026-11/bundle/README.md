# GeulPU 0.21.0 - independent determinism test suite (reproduction bundle)

This bundle contains the exact test code, pinned configuration and provenance
behind the one-page report "GeulPU 0.21.0 - independent test summary" (v7,
2026-11-06). It is shared so others can reproduce the tests on their own
hardware. It contains no credentials, no internal hostnames or paths, and no
model weights.

## Contents

- `code/additional_runner.py` - run harness: starts its own GeulPU app with its
  own data directory, verifies the wheel and model hashes, enforces a measured
  GPU-memory guard, and verifies cleanup (child processes dead, port closed).
- `code/additional_pilot.py` - the test suite: cold solo references (B0),
  context-boundary checks at the native 8192-token context (M2), sustained
  cache churn with a staggered three-party barrier (M3), KV-eviction barrier
  test (M4), slots=4 burst (M5) and slots=8 eight-way concurrency (M6), plus
  strict /tokenize-vs-chat prompt-ID parity and a native BOS pin.
- `code/stress_pilot.py`, `code/core_pilot.py`, `code/m4_followup.py` - the
  shared machinery the suite imports (request runner, recorder, earlier
  modules).
- `code/compare_arms.py` - offline cross-arm comparison (speculation OFF
  full.json vs ON full.json). Empirical reporting only; never asserts equality.
- `code/cpu_tests.py` - CPU-only gate: argument contract, fail-closed load
  semantics, memory-gate arithmetic, profile pins, and compare_arms
  self/injected-diff behavior. Run this before any GPU run.
- `code/requests.json` - the fixed request set (prompts, seeds, temperatures,
  max_tokens) shared by all arms.
- `fixtures/synthetic-full.json` - small SYNTHETIC fixture for the CPU gate.
  Generated constants, clearly labeled; contains no model output and no
  research data.
- `HASHES.md` - SHA-256 manifest of the audited code, the shipped code, the
  fixture, and the pinned wheel and model files, plus the exact sanitization
  edits applied.

## Requirements

- CPU gate (fixture): any OS, Python 3.10 or newer (tested on 3.10). No GPU.
- GPU runs: Linux x86_64 with glibc >= 2.34 (the pinned wheel is
  manylinux_2_34_x86_64), NVIDIA GPU with the CUDA driver (`libcuda`), and a
  Python 3.11 venv (the audited runs used 3.11). Free GPU memory is measured
  and gated at run time; the slots-8 gate refuses below 8240 MiB free for the
  Llama profile.

## Getting the pinned artifacts

Engine wheel (the runner verifies its sha256 before installing):

    pip download geulpu==0.21.0 --no-deps  # or fetch the exact file directly:
    # https://files.pythonhosted.org/packages/4d/d8/63064700bee45606d3a3925a23903cf64415653a1620e1115b3468b76e0b/geulpu-0.21.0-py3-none-manylinux_2_34_x86_64.whl
    # sha256 e71335f5328c159d420a90563c5d2a9b566162694044d14f6e9e95f1cd5f64a7 (verified against the PyPI JSON API 2026-11-06)

Engine source (for inspection; the tests run the wheel above):
https://github.com/wwoosshh/GeulPUv2 at commit
`0efe2a5824ea202f5c02f98afb0620647966e884`.

Models (GGUF; URLs are the exact sources used for the audited runs, each
verified live 2026-11-06 with the pinned content length; the runner also
verifies size + sha256 + GGUF metadata before every run):

- `Qwen3-8B-Q8_0.gguf` (8,709,518,112 B):
  https://huggingface.co/Qwen/Qwen3-8B-GGUF/resolve/main/Qwen3-8B-Q8_0.gguf
- `Qwen3.5-9B-Q4_K_M.gguf` (5,680,522,464 B):
  https://huggingface.co/unsloth/Qwen3.5-9B-GGUF/resolve/main/Qwen3.5-9B-Q4_K_M.gguf
- `Meta-Llama-3.1-8B-Instruct-Q8_0.gguf` (8,540,775,840 B):
  https://huggingface.co/bartowski/Meta-Llama-3.1-8B-Instruct-GGUF/resolve/main/Meta-Llama-3.1-8B-Instruct-Q8_0.gguf

## Layout expected by the runner (paths are arguments; nothing is hardcoded)

```
ROOT/
  venv/                      # python3.11 -m venv venv
  wheelcheck/geulpu-0.21.0-py3-none-manylinux_2_34_x86_64.whl
  models/<the pinned .gguf files>
```

## How to run

1. CPU gate (no GPU, ~1 second):
   `cd code && python3 cpu_tests.py ../fixtures/synthetic-full.json`
   Expected tail: "ALL CPU TESTS PASS" / "v9 llama cpu tests OK".
2. Speculation-OFF arm:
   `python3 additional_runner.py --root ROOT --code . --profile llama --gpu-label MyGPU --speculative off`
3. Eight-way concurrency (M6): add `--slots8`
4. Speculation-ON arm: `--speculative on`
5. Cross-arm comparison:
   `python3 compare_arms.py OFF_RUN/full.json ON_RUN/full.json crossarm.json`

Profiles: `llama` (Llama-3.1-8B-Instruct Q8_0), `q8` (Qwen3-8B Q8_0),
`hybrid` (Qwen3.5-9B Q4_K_M).

The upstream project also ships a lighter-weight reproducibility kit
(`repro-kit/` in the engine repository) with its own `expected.json` reference
outputs from the author's reference PC. That reference file is optional and is
not included here; this bundle's checks compare arms and runs against each
other, not against the author's machine.

## What the suite checks

For every request: full prompt token-ID sequence, full generated token-ID
sequence, and a per-step logit hash at every generation step. Any single-token
or single-logit difference fails the cell. Within a run, every module replays
requests against their cold references; across arms, `compare_arms.py` reports
every identity/difference per (phase, request). The suite also verifies at
every model load: requested vs server-effective vs live speculative state,
KV8/fast-attention flags, chat template hash, native context 8192 (no silent
shrink, checked from the server startup header), and exactly one leading BOS
token per prompt.

## Safety design

- The runner owns exactly one app instance, started with a stripped environment
  in its own process group; kills are limited to that group.
- A memory guard watches free GPU memory through the runner's own CUDA context
  (works on MIG slices where nvidia-smi cannot see the slice) and trips at a
  fixed threshold, failing closed. Slots-8 admission uses measured free memory,
  never declared capacity or another GPU's numbers.
- CPU affinity pinning is recorded; an absolute hard stop bounds every run; a
  smoke gate runs before each full run; old run data is never overwritten
  (a failed run stays on disk marked INCOMPLETE).
- Cleanup is verified after every run: child process dead, app dead, port
  closed.

## Results being reproduced (audited summary)

GeulPU 0.21.0 (commit 0efe2a58), fast mode, context 8192, on three NVIDIA
GPUs (H100 MIG slice, A100 MIG slice, full RTX 6000 Ada co-serving a live
public demo):

| Cell | H100 MIG | A100 MIG | RTX 6000 Ada |
|---|---|---|---|
| Llama-3.1-8B-Instruct Q8_0, speculation OFF suite | PASS 37/37 | PASS 37/37 | PASS 37/37 |
| Llama, eight-way concurrency (M6) | PASS 23/23 | PASS 23/23 | PASS 23/23 (tight margins, see caveats) |
| Llama, speculation ON vs OFF cross-arm | PASS 37/37 ON + 60/60 cross | PASS 37/37 ON + 60/60 cross | PASS 37/37 ON + 60/60 cross |

A cross-GPU check also observed 23/23 token-and-logit identity for the same
requests across the three allocations - one empirical multi-GPU observation,
not a guarantee.

Earlier cells for Qwen3-8B Q8_0 and Qwen3.5-9B Q4_K_M (37/37 OFF, eight-way,
and ON/OFF cross-arm on the same three GPUs) are historical results produced
by an earlier revision of this same code lineage; they are NOT reproduced by
the exact v10 files shipped here, which are the final Llama-era revision.

## Caveats (please keep these with any results)

- These are empirical observations on the tested configurations, not a
  universal determinism guarantee for other models, versions, hardware or
  settings. Reported timings are descriptive; no speed claims are made.
- The RTX 6000 Ada eight-way cell passed with tight margins: the admission
  gate had 43 MiB of headroom and the memory guard's minimum free sample was
  2047 MiB, 47 MiB above the trip threshold. That is a legitimate PASS, not an
  operational-margin recommendation. The co-resident public demo stayed live
  and untouched throughout.
- The first Llama speculation-OFF attempt on the H100 slice failed its churn
  module (10/15 with the original 1.0 s stagger); it is preserved as
  INCOMPLETE history. The shipped code uses the profile's 0.5 s stagger with
  the unchanged 14/15 overlap threshold, which passed 15/15 on all three GPUs.
- GPU-memory telemetry on MIG slices is limited to the runner's own CUDA
  context; nvidia-smi cannot see per-slice memory.

## Provenance

All runs on 2026-11-05/06 (UTC). Each cell was reviewed before launch and
independently audited after completion from the run artifacts (runner.json,
smoke.json, full.json, crossarm.json, server logs). Final audited runs:

| GPU | OFF suite | Eight-way (M6) | Speculation ON |
|---|---|---|---|
| H100 MIG | additional-llama-specoff-20261105T215436 | additional-llama-specoff-slots8-20261105T220042 | additional-llama-specon-20261105T222253 |
| A100 MIG | additional-llama-specoff-20261105T222837 | additional-llama-specoff-slots8-20261105T224808 | additional-llama-specon-20261105T230405 |
| RTX 6000 Ada | additional-llama-specoff-20261105T222840 | additional-llama-specoff-slots8-20261105T224811 | additional-llama-specon-20261105T230407 |

Retained failed history: first H100 OFF attempt
`additional-llama-specoff-20261105T213855` (INCOMPLETE, M3 10/15).

## Attribution

Tests were designed and executed independently on infrastructure of the GING
research group (DIT, Universidad Politecnica de Madrid) - https://ging.github.io.
The GeulPU engine is the work of its upstream author
(https://github.com/wwoosshh/GeulPUv2). This bundle is provided to enable
independent reproduction of the tests; it is not promotional material and
makes no guarantees beyond the empirical results stated above.
