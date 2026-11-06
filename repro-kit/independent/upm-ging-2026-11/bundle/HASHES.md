# Hash manifest

## Audited code set ("v10"), SHA-256 of the exact bytes that ran and were independently audited

```
ebda87d4fcb0e2f118117c01fe5465bbd9f6af75caffd67d401fca06184941f8  additional_pilot.py
0bae13413e093fdeb6ef4227a7ccb5c3bc8212a45b8c97e033cde4ebe0bb7b1c  additional_runner.py
00fa4ff18aacf44ff627a9dd13880eab8010bf41a069e1ea6d9edaf22ecf6d87  compare_arms.py
682520111d60893c07c9e17f1278744226c0d06b46262192ee5b3f95a0830e14  core_pilot.py
fc2b9d3a2cbe0281e31d6b9c27494228795ec0ca8f0b288fbc368d8c0678afa6  cpu_tests.py
364a262c56e766bbfa97352bf648e9074ebf5fc29cdce22175ae19068eebb796  m4_followup.py
e500db3ce566ea1b0950b55bea2fdaa22022bc751577811cc2bbfda3af2722b7  requests.json
9603c1dd73cd5b604893287d6a41dcf3e3896f88571e5e77db341ff216240647  stress_pilot.py
```

## Sanitization applied for sharing

Two code comments that named test-machine owners were reworded. Nothing else
changed; no executable line was touched:

- `stress_pilot.py` docstring: a machine-owner nickname was removed from the
  phrase "(first pilot, <nickname>)."
- `core_pilot.py` comment: the same kind of nickname in "speeds (<nickname>
  worst-case ~2x estimate)" was replaced by "slowest test machine".

SHA-256 of the shipped (sanitized) files (also machine-checkable:
`cd code && sha256sum -c SHA256SUMS`):

```
ebda87d4fcb0e2f118117c01fe5465bbd9f6af75caffd67d401fca06184941f8  additional_pilot.py
0bae13413e093fdeb6ef4227a7ccb5c3bc8212a45b8c97e033cde4ebe0bb7b1c  additional_runner.py
00fa4ff18aacf44ff627a9dd13880eab8010bf41a069e1ea6d9edaf22ecf6d87  compare_arms.py
71be020b17a66c0325634c602c45e3398c06ac4958b8176994be786fbd237fd8  core_pilot.py
fc2b9d3a2cbe0281e31d6b9c27494228795ec0ca8f0b288fbc368d8c0678afa6  cpu_tests.py
364a262c56e766bbfa97352bf648e9074ebf5fc29cdce22175ae19068eebb796  m4_followup.py
e500db3ce566ea1b0950b55bea2fdaa22022bc751577811cc2bbfda3af2722b7  requests.json
6ea1e9a6c24b7c55b68a106b67a643cc45301e58004288de08a29444bf90d14f  stress_pilot.py
```

## CPU-test fixture

```
ad005cdeeb893b3d843c6909ad76ec9e7c90704dd98f141f883672d0040ef313  fixtures/synthetic-full.json
```

Synthetic generated constants for `cpu_tests.py`/`compare_arms.py`; contains no
model output and no research data. Machine-checkable:
`cd fixtures && sha256sum -c SHA256SUMS`.

## Pinned binary artifacts (verified by the runner at run time; not included here)

- GeulPU wheel `geulpu-0.21.0-py3-none-manylinux_2_34_x86_64.whl`
  sha256 `e71335f5328c159d420a90563c5d2a9b566162694044d14f6e9e95f1cd5f64a7`
- Models (GGUF, verified size + sha256 + native metadata by the runner):
  - `Qwen3-8B-Q8_0.gguf`, 8709518112 bytes,
    sha256 `408b955510e196121c1c375201744783b5c9a43c7956d73fc78df54c66e883d6`
  - `Qwen3.5-9B-Q4_K_M.gguf`, 5680522464 bytes,
    sha256 `03b74727a860a56338e042c4420bb3f04b2fec5734175f4cb9fa853daf52b7e8`
  - `Meta-Llama-3.1-8B-Instruct-Q8_0.gguf`, 8540775840 bytes,
    sha256 `9da71c45c90a821809821244d4971e5e5dfad7eb091f0b8ff0546392393b6283`
