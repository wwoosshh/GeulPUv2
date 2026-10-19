# GeulPU — local AI chat on your own graphics card

GeulPU runs GGUF language models (Qwen3, Qwen2.5 / Coder, DeepSeek-R1-Distill-Qwen, Llama 3.1 / 3.2, Mistral 7B) on your own GPU, with
a chat screen in the browser and an OpenAI-compatible API. Its engine is written in the 글(geul) language, and the same input gives
**bit-identical output** on any PC and GPU (NVIDIA CUDA, AMD and other GPUs through Vulkan), with or without speculative decoding.

This package ships the Windows x64 app from the GitHub release, so one install command is enough — no zip to download and extract.

## Install and run (Windows 10 / 11, 64-bit)

```
pipx install geulpu        # or: uv tool install geulpu   ·   or: pip install geulpu
geulpu                     # opens http://127.0.0.1:44544 in your browser
```

- Needs an NVIDIA RTX 30 series or newer (CUDA), or another GPU with Vulkan 1.3 (AMD Radeon, ...), and an up-to-date graphics driver.
- Models, the kernels built for your GPU, results and settings live in `%USERPROFILE%\GeulPU` (set `GEULPU_HOME` to move it).
  Uninstalling the package leaves that folder alone.
- Options: `geulpu --포트 44600` (port), `geulpu --자료 D:\GeulPU` (data folder), `geulpu --열지않음` (don't open the browser).
- Update: `pipx upgrade geulpu` (or `pip install -U geulpu`). The app tells you when a newer version is out.
- The first chat with a model builds kernels for your GPU (1–3 minutes, once per version).

## 설치 · 실행 (한국어)

```
pipx install geulpu        # 또는 uv tool install geulpu · pip install geulpu
geulpu                     # 브라우저에 http://127.0.0.1:44544 가 열립니다
```

모델 · 이 GPU 용 커널 · 결과 · 설정은 `%USERPROFILE%\GeulPU` 에 저장됩니다(`GEULPU_HOME` 으로 옮김 — 꾸러미를 지워도 그대로).
업데이트는 `pipx upgrade geulpu`(또는 `pip install -U geulpu`).

Guide, source and changelog: https://github.com/wwoosshh/GeulPUv2 (English: README.en.md). MIT License.
