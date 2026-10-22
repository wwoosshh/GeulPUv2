# Versioning

**[한국어](../문서/버전규칙.md)** · [README](../README.md) · [changelog](../CHANGELOG.md)

A GeulPU version is `MAJOR.MINOR.PATCH`, for example `0.7.0`. It is written in one place, [`앱/판.gl`](../앱/판.gl), and a release is
tagged `v` + that number (the release workflow stops if the two differ). The app, the release zip and the PyPI package `geulpu` always
carry the same number.

| Part | Goes up when | Examples |
|---|---|---|
| **MAJOR** | an update is **not compatible** with the previous version | a documented API field is removed; run logs written by the previous version can no longer be read; the same request gives different output bits |
| **MINOR** | **features** are added and everything that worked before keeps working | the debugger (0.7.0) · the prompt cache (0.6.0) · a new model family · a new API field · a faster kernel that keeps the output bits |
| **PATCH** | **bugs are fixed** or small changes are made | a crash fix · a wrong result corrected · a screen text or documentation fix |

When one part goes up, the parts to its right go back to 0: 0.7.0 → 0.7.1 → 0.8.0 → 1.0.0. These rules apply from 0.7.0. The major
version 0 is not an exception: the first incompatible change after 0.7.0 makes the next version 1.0.0.

## What "compatible" means

GeulPU is used by programs as well as people, and what it promises is reproducibility. So compatibility covers four things.

1. **Interfaces.** The HTTP API — the OpenAI-compatible endpoints (`/v1/chat/completions`, `/v1/completions`, `/v1/models`),
   `/tokenize`, `/detokenize`, `/tokenizer_info`, the debugger API (`/api/debug/…`) and the inference server's prompt-cache API
   (`/geulpu/store`) — with the `geulpu` extension fields in requests and responses; the command-line options of `GeulPU.exe` and
   `geulpu`; and the commands and options of the tools in [`도구/`](../도구/) (`debugger.py`, `repro_check.py`). Removing or renaming any
   of these, or changing what an existing one means, is incompatible. Adding endpoints, fields, options or commands is compatible. The
   app's internal endpoints, used only by its own web page, are not covered.
2. **Files.** Run logs (`"format": "geulpu-run-log/1"`), reproduction files, and the settings and model list in the data folder. A version
   must read every file the previous version wrote. A changed format gets a new format version (`geulpu-run-log/2`), and dropping an old
   one is incompatible. Caches — kernel modules, the prompt cache, the tuning cache — are rebuilt when needed and are not covered.
3. **Output bits.** For the same model file, computation mode (with its 8-bit KV cache and fast attention settings), YaRN factor and
   request (seed included), the tokens, log-probabilities and
   logit hashes stay the same from version to version, so a run log recorded on the previous version replays bit for bit. Changing them
   is incompatible — unless the change fixes a result that did not match the [precision contract](../계약/contract.md)'s CPU reference;
   such a fix is a PATCH and is marked **Output change** in the changelog. A new computation mode can come in a MINOR version.
4. **Support.** The model families, weight formats, GPUs and operating systems that the previous version ran. Dropping one is
   incompatible.

Speed, memory use, the look of the app and the wording of messages are not covered — the changelog still mentions their changes.

## Deprecation

Before a MAJOR version removes something, a MINOR version marks it as deprecated in the changelog and keeps accepting it, as the tools
still accept their old option names.

## Releases

Releases are built from the tagged commit by GitHub Actions and published by a person; the steps are in
[CONTRIBUTING.md](../CONTRIBUTING.md#releasing-maintainers). Every release has a section in the [changelog](../CHANGELOG.md) (Korean:
[문서/바뀐것.md](../문서/바뀐것.md)).
