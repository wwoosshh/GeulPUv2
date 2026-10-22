# Security policy

## Supported versions

Security fixes go into the latest release. Please check that the problem still happens on it before reporting.

| Version | Supported |
|---|---|
| 0.10.x | Yes |
| older than 0.10 | No — update to the latest release |

## Reporting a vulnerability

Please **do not open a public issue** for a security problem. Report it privately with GitHub's
[private reporting form](https://github.com/wwoosshh/GeulPUv2/security/advisories/new) (Security → Report a vulnerability) and include
the GeulPU version, what an attacker could do, and the steps to reproduce it. The maintainer will reply there, keep you informed until a
fix is released, and credit you in the advisory unless you prefer otherwise.

## Security model

What GeulPU is designed to do, so that you can tell a vulnerability from expected behavior:

- **Local only.** The app (`GeulPU.exe` / `geulpu`) and the inference server it starts listen only on `127.0.0.1`; other devices cannot
  connect. There is **no authentication**: any program running on the same PC can use the API.
- **Browser protection.** The app refuses requests whose `Host` header is not its own address (DNS rebinding) and write requests that
  come from another origin (other websites open in your browser). The inference server it starts accepts only `127.0.0.1` and
  `localhost` as `Host` and refuses every request that carries an `Origin` header — browsers send one, while the app's `/v1` and other
  programs do not. Neither sends CORS headers.
- **Downloads.** Models come from Hugging Face over HTTPS and are checked against the SHA-256 that Hugging Face reports. The new-version
  check contacts GitHub (or PyPI for the Python package) and can be switched off in Settings. No usage data is sent.
- **Model files are untrusted input.** The engine parses GGUF files itself; a crafted GGUF file that makes the engine read or write out of
  bounds is in scope.
- **Builds.** The executables are not code-signed. Release zips and wheels are built from the tagged source by GitHub Actions; every
  release lists the SHA-256 of its files in `SHA256SUMS.txt`, and `version.json` in the zip lists the SHA-256 of every file. The PyPI
  package is uploaded with trusted publishing.
- **Your data stays on disk in plain form.** Conversations are kept in your browser's storage. The prompt cache stores the model's
  KV cache for your prompts in the data folder (`%USERPROFILE%\GeulPU`), and debugger run logs store requests and responses there — on a
  shared PC, turn the prompt cache off or clear it, and delete run logs you no longer need.

In scope: the app, the inference server, the tools in `도구/`, and the release and PyPI packages. Out of scope: running GeulPU on a
machine shared with untrusted users (there is no authentication by design), and problems in the models' own output.
