# Contributing to GeulPU

Thank you for your interest in GeulPU. The project has a single maintainer, so clear reports and small, well-checked changes help
the most. Issues and pull requests are welcome in **English or Korean**.

## Ways to help

- **Report a bug** with the [bug report form](https://github.com/wwoosshh/GeulPUv2/issues/new?template=bug_report.yml). Include the
  GeulPU version, the GPU and driver, the model file and the logs from `%USERPROFILE%\GeulPU\기록\` (the logs folder).
- **Report a reproducibility problem.** This is the most valuable report for this project: a run that does not replay bit for bit, or
  two machines that give different bits for the same model file and computation mode. Attach the debugger's run log or the research
  panel's reproduction file ([debugger guide](docs/debugger-guide.md)).
- **Tell us what runs on your hardware.** Results of **Benchmark → Reproducibility check** on a GPU that has not been tested yet (AMD,
  Intel, older NVIDIA cards), or a GGUF model that should run but does not.
- **Improve the documentation**: unclear steps, outdated screenshots, translations.
- **Write code.** For anything bigger than a small fix, open an issue first so that we can agree on the approach before you spend time
  on it.

## Things to know first

### The code is written in 글(geul)

The engine, the kernel generator, the inference server and the app are written in [글(geul)](https://github.com/wwoosshh/geul-lang), a
programming language with Korean keywords, so identifiers and comments are in Korean. English documents live in [docs/](docs/), and
the [glossary](docs/glossary.md) explains the Korean names used in the code.

### Determinism is the product

GeulPU promises the same output bits for the same model file, computation mode and request, whatever the batching, slot, prompt cache
or speculative decoding — and, in exact mode, whatever the GPU. A change has to keep that promise:

- Kernel changes are compared bit for bit with the CPU reference implementation of the [precision contract](계약/contract.md)
  (`계약/정수계약.py`) before anything is timed.
- A change that alters output bits needs a stated reason. It is a breaking change unless it fixes a result that did not match the
  contract ([versioning](docs/versioning.md)).
- Speed claims are measured against the latest official llama.cpp release with the same GGUF file, as medians of alternating rounds.

### Terminology

Use the established ML and software term for every concept, in Korean text too, and don't invent new words. The Korean ↔ English table
is [문서/용어.md](문서/용어.md).

## Development setup

[docs/development.md](docs/development.md) covers the repository layout, building the 글(geul) compiler at the pinned commit, building the
server and packaging the app.

## Checks

CI runs the quick check on every push and pull request, on Windows and Ubuntu, without a GPU or model files (about 5 minutes; longer when
kernels change). Run it locally before you open a pull request:

```bash
python -X utf8 시험/빠른점검.py --글 ../geul-lang-v2
```

The last line is `빠른 점검: 모두 통과` ("quick check: all passed"). If your change regenerates kernel modules on purpose, update
[`시험/모듈해시.txt`](시험/모듈해시.txt) with `--해시갱신` and the register-spill table [`시험/SASS넘침.txt`](시험/SASS넘침.txt) in the same
pull request.

GPU tests — bit comparisons against the CPU reference and the server regression tests — are run by hand. Say in the pull request which
ones you ran and on which GPU.

## Pull requests

- One topic per pull request, without unrelated formatting changes.
- Describe what changed, why, how you checked it, and whether output bits change.
- Commit messages: a short title line, a blank line, then the details (English or Korean).
- If users will notice the change, add a line to [CHANGELOG.md](CHANGELOG.md). The Korean changelog
  ([문서/바뀐것.md](문서/바뀐것.md)) can be left to the maintainer.
- By contributing, you agree that your contribution is licensed under the [MIT License](LICENSE).

## Versioning

Versions are `MAJOR.MINOR.PATCH`: **MAJOR** goes up when an update is not compatible with the previous version, **MINOR** when features
are added, **PATCH** for bug fixes and small changes. What "compatible" covers — interfaces, files, output bits and support — is defined
in [docs/versioning.md](docs/versioning.md).

## Releasing (maintainers)

1. Choose the number by the [versioning rules](docs/versioning.md) and write it in [`앱/판.gl`](앱/판.gl) (`앱판 = "X.Y.Z"`).
2. Turn the "Unreleased" sections into `## X.Y.Z — <title> (<date>)` in [CHANGELOG.md](CHANGELOG.md) and
   [문서/바뀐것.md](문서/바뀐것.md) — the release workflow puts both into the release notes and stops if either is missing. Update
   `version` and `date-released` in [CITATION.cff](CITATION.cff).
3. Package locally with `앱\꾸리기.ps1 -GeulRoot ..\geul-lang-v2`: the five build hashes must match the cross-check reference
   (`[같음]`, "same").
4. Test the fresh-install path with the packaged app on another port and an empty data folder
   (`GeulPU.exe --포트 44639 --자료 <empty folder> --열지않음`): add a model, load it, chat, check the version in `/api/state`.
5. Run the quick check, commit, push, and wait for CI to be green.
6. Tag and push: `git tag -a vX.Y.Z -m "GeulPU X.Y.Z"` then `git push origin vX.Y.Z`. The release workflow builds the zip, the wheel and
   `SHA256SUMS.txt` from the tagged commit and creates a **draft** release.
7. Download the draft's files, compare their SHA-256, install the wheel in a fresh virtual environment and run it.
8. Publish: `gh release edit vX.Y.Z --draft=false --latest`. Publishing starts the PyPI workflow, which waits for approval in the `pypi`
   environment and uploads the same wheel.

## Code of conduct

Everyone taking part in this project is expected to follow the [code of conduct](CODE_OF_CONDUCT.md).
