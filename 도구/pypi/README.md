# GeulPU

**Run language models on your own GPU — and get the same bits every time.**

GeulPU is a local LLM engine and debugger for GGUF models — Qwen3, Qwen2.5 / Coder, DeepSeek-R1-Distill-Qwen, Llama 3.1 / 3.2 and
Mistral 7B — with a chat screen in the browser and an OpenAI-compatible API. Its arithmetic is fixed by a written precision contract,
so the same request gives bit-identical tokens and logits whatever the batch, the slot, the prompt cache or speculative decoding — and,
in exact mode, on NVIDIA (CUDA) and AMD or other GPUs (Vulkan) alike. The debugger records runs (agent sessions, test sets), replays
them bit for bit, branches from any token or edited request, and compares models and quantizations request by request. The engine is
written in the 글(geul) programming language.

This package ships the app from the GitHub release — Windows x64 and Linux x86-64 — so one install command is enough: no archive to
download and extract.

## Install and run (Windows 10 / 11, 64-bit)

```
pipx install geulpu        # or: uv tool install geulpu   ·   or: pip install geulpu
geulpu                     # opens http://127.0.0.1:44544 in your browser
```

- Needs an NVIDIA RTX 30 series or newer (CUDA), or another GPU with Vulkan 1.3 (AMD Radeon, ...), and an up-to-date graphics driver.
- Models, the kernels built for your GPU, results and settings live in `%USERPROFILE%\GeulPU` (set `GEULPU_HOME` to move it).
  Uninstalling the package leaves that folder alone.
- Options: `geulpu --포트 44600` (port), `geulpu --자료 D:\GeulPU` (data folder), `geulpu --열지않음` (don't open the browser).
- Update: `pipx upgrade geulpu` (or `pip install -U geulpu`) after quitting the app. The app tells you when a newer version is out.
- The first chat with a model builds kernels for your GPU (once per version).

## Linux (x86-64, NVIDIA)

```
pipx install geulpu        # or: uv tool install geulpu   ·   or: pip install geulpu
geulpu                     # prints http://127.0.0.1:44544 — open it in a browser
```

- Needs an NVIDIA GPU and its driver (the app uses `libcuda.so.1`; no CUDA toolkit) and glibc 2.34 or later (Ubuntu 22.04,
  Debian 12, RHEL 9 or newer). Vulkan (AMD, Intel) is Windows-only for now.
- On a remote machine, open the screen through SSH port forwarding: `ssh -L 44544:127.0.0.1:44544 <host>`.
- Data lives in `~/GeulPU` (set `GEULPU_HOME` to move it). The app does not download models on Linux yet: download the GGUF files
  yourself and add them with **Add model** (or put them in `~/GeulPU/모델`).
- One check without the screen: `geulpu --check pair --quick` (exit code 0 when every problem matches the reference PC).

## 설치 · 실행 (한국어)

```
pipx install geulpu        # 또는 uv tool install geulpu · pip install geulpu
geulpu                     # 브라우저에 http://127.0.0.1:44544 가 열립니다
```

모델 · 이 GPU 용 커널 · 결과 · 설정은 `%USERPROFILE%\GeulPU` 에 저장됩니다(`GEULPU_HOME` 으로 옮김 — 패키지를 지워도 그대로).
업데이트는 앱을 끈 뒤 `pipx upgrade geulpu`(또는 `pip install -U geulpu`). 한국어 안내: [사용안내.md](https://github.com/wwoosshh/GeulPUv2/blob/main/사용안내.md).

리눅스(x86-64 · NVIDIA): 같은 명령으로 깔고 `geulpu` 가 알리는 주소를 브라우저로 엽니다(원격이면 SSH 포트 포워딩). NVIDIA 드라이버와
glibc 2.34 이상(Ubuntu 22.04 · Debian 12 · RHEL 9 부터)이 필요합니다. 자료 폴더는 `~/GeulPU` 이고, 리눅스에서는 앱 안의 내려받기가 아직
없어 GGUF 파일을 따로 받아 '모델 추가'로 넣습니다. 화면 없이 검사 하나: `geulpu --검사 쌍 --빠름`.

## Links

- [README](https://github.com/wwoosshh/GeulPUv2#readme) · [user guide](https://github.com/wwoosshh/GeulPUv2/blob/main/docs/user-guide.md) ·
  [debugger guide](https://github.com/wwoosshh/GeulPUv2/blob/main/docs/debugger-guide.md)
- [Changelog](https://github.com/wwoosshh/GeulPUv2/blob/main/CHANGELOG.md) · [source](https://github.com/wwoosshh/GeulPUv2) ·
  [issues](https://github.com/wwoosshh/GeulPUv2/issues)

MIT License.
