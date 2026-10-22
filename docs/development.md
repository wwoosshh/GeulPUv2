# Development

**[한국어](../문서/개발.md)** · [README](../README.md) · [contributing](../CONTRIBUTING.md) · [user guide](user-guide.md) · [glossary](glossary.md)

This page is for building GeulPU from source and finding your way around the code — to use the app, the [user guide](user-guide.md) is all you need. How to propose changes is in [CONTRIBUTING.md](../CONTRIBUTING.md). The research record (stage reports, paper) lives in the [GeulPU repository](https://github.com/wwoosshh/GeulPU); the "stage N" links below go there.

## Status (2026-10)

| | |
|---|---|
| Models | **Llama-3.1-8B · Llama-3.2-3B · Mistral-7B v0.3** (stage 39 — architecture `llama`: q · k row restore and RoPE scaling factors as contract §4.7; five files bit-identical to the CPU reference, mean KL ≤ 4 × 10⁻⁴ against llama.cpp on every token of the same texts; the 70B shape tested on synthetic models) · **Qwen3-1.7B / 8B / 14B** (stage 37 — per-head Q/K RMSNorm as contract §4.6 and one appended kernel, bit-identical to the CPU reference; the 14B shape tested on synthetic models) · **Qwen2.5-0.5B / 7B / 14B-Instruct** · DeepSeek-R1-Distill-Qwen (chat; Llama-family blocks — RMSNorm · RoPE · SwiGLU · GQA; head sizes 64 and 128). Head-128 Qwen2 shapes are read from the GGUF config; 32B and 72B shapes were bit-tested on 4-layer synthetic models. GPT-2 small and XL (research engine). |
| App | Stages 36–39 — `GeulPU.exe` (a manager written in geul) and a web UI: find models (browse every Hugging Face GGUF repo, sorted), download, add and verify them; chat (Qwen2.5's ChatML, Qwen3's template with thinking on/off, R1's, Llama 3's and Mistral's templates); self-diagnosis, speed and cross-check runs; result zips; a new-version notice; a **research panel** (stage 38 — per-token probabilities, branching, comparison, scoring and reproduction files; log-probabilities as contract §5.2) and an English UI. The release zip (1.5 MB) is built from source by CI when a tag is pushed (as a draft; a person publishes it). Windows x64 only — see [Install](user-guide.md#install). |
| Weight formats | GGUF **Q8_0 · Q4_0 · Q4_K · Q5_K · Q6_K**, including per-tensor mixes as in Q4_K_M · Q5_K_M (stage 26 — tested with files re-quantized on this PC from Q8_0; official Qwen files are not yet tested). |
| Runner | `빌드/글/실행기.exe model.gguf` (`실행기` = runner) — a geul program: chat (Qwen's ChatML and the Qwen3, Llama 3 and Mistral templates), streamed output, KV cache reused across turns, greedy decoding (the same conversation gives the same answer). About 0.3 s until the prompt runs. |
| Server | `빌드/글/서버.exe model.gguf` (`서버` = server) — a geul program (stage 28): an OpenAI-compatible HTTP API (chat · completions · SSE streaming) with slots, continuous batching, prompt caching and a sampling contract (the same seed gives the same tokens). A request returns the same bits whatever else runs with it — zero differences over four models × twelve requests × four schedules, while llama-server's greedy answer changed in 11 of 12 cases depending on the concurrent requests. Head-128 Qwen2 (7B · 14B). 7B Q4_0 aggregate throughput: 302 tokens/s with four streams and 508 with eight (llama-server 360 · 632 — a multi-row decode kernel is next); one stream is 4% faster than llama-server. A cached prompt's first token takes 13.8 ms (llama-server 13.0 ms; stage 30 feeds the last rows through the decode step). See [`문서/28단계.md`](https://github.com/wwoosshh/GeulPU/blob/main/문서/28단계.md) · [`문서/30단계.md`](https://github.com/wwoosshh/GeulPU/blob/main/문서/30단계.md) |
| Environment | Windows or Linux x86-64 (glibc) + NVIDIA Ampere (RTX 30) or newer + driver. Linux was tested under WSL2 Ubuntu 24.04 in stage 29 — the same bits as Windows on the same GPU, speed within ~0.5% ([`문서/29단계.md`](https://github.com/wwoosshh/GeulPU/blob/main/문서/29단계.md)). On Linux the programs import only libc.so.6 and libcuda.so.1. All numbers here are from one RTX 4070 Ti (12 GB). Building (generator → kernel source → PTX → runner) needs only the geul compiler. Tests and benchmarks need Python 3.12 + numpy. **Vulkan (stage 31):** Vulkan builds of the geul host (`시험_엔진V` · `시험_7BV` · `재기_7BV` — they import kernel32, shell32 and vulkan-1.dll; kernels come from the geul compiler's SPIR-V backend) reproduce the CUDA records byte for byte on NVIDIA (0.5B, 7B in five formats, 14B) and Intel integrated graphics (0.5B). Speed is not there yet — stage 34 replaced the emulated integer MMA with cooperative-matrix tensor-core work (`--협력` modules) and fixed the backend and the Vulkan layer: 7B prompts take 1.15–1.41× llama.cpp Vulkan's time (stage 31: 4.4–6.1×), decode 1.00–1.04× (stage 31: 1.09–1.24×) — a record, not yet a verdict. The server has a Vulkan build too (`server-vulkan` — `엔진/서버V.gl`, used by the app for Vulkan devices); the terminal runner does not. |
| Layer split | Stage 25 — split layers across GPUs (`--나누기 P` parts, `--장치 a,b,…` devices, `--예산 GB` budget). Emulated on one GPU with one CUDA context per part: 16 split runs are bit-identical to the single-part run; decoding with two parts takes 1.025–1.029× (mostly the emulation's context switching; 1.002–1.003× with a shared context). 14B in two parts did not fit at context 8192 on 12 GB, so that pre-registered cell was not measured (criterion 25c not passed). |
| Not yet | Real multi-GPU runs; measuring the contract's loss with the real engine on 7B+ (done for 0.5B; simulated for 7B · 14B); Q2_K · Q3_K · IQ formats · Q5_0 (real 72B k-quant files use Q5_0 · Q8_0 for ffn_down); MoE; a GPU run on native Linux (tested under WSL2 only); running on Apple GPUs (three NVIDIA generations and an AMD Radeon RX 9060 XT (RDNA4, Vulkan) match bit for bit — [교차검증](../교차검증/); AMD speed (RX 9060 XT generation is about 3× slower than an RTX 4070 Ti, mostly attention); Intel integrated graphics matches through Vulkan for 0.5B — stage 31); k-quants on 32 KB shared-memory devices (AMD on Windows); models whose query width differs from the hidden size (Qwen3-0.6B · 4B · 32B, Mistral NeMo); small head-64 llama models (Llama-3.2-1B); other architectures (Gemma, Phi, …). |

## Repository layout

| Folder | What |
|---|---|
| [`엔진/`](../엔진/) (engine) | The engine in geul: `gguf.gl` (GGUF reader), `가중치.gl` · `가중치K.gl` (weight layout, k-quants), `로프.gl` (RoPE tables), `토크나이저.gl` (BPE, SentencePiece, Unicode 15.0 tables), `엔진.gl` · `엔진K.gl` · `엔진표.gl` (kernel selection, CUDA graphs, decoding, slots), `조각.gl` (layer split), `뽑기.gl` · `로그확률.gl` (sampling and log-probabilities — contract §5 · §5.2), `서버.gl` · `서버V.gl` · `서버몸.gl` (the OpenAI-compatible HTTP server — CUDA and Vulkan heads), `장치C.gl` · `장치V.gl` (device check), `진단.gl` · `진단V.gl` · `진단몸.gl` (self-diagnosis), `실행기.gl` (terminal runner), `쿠다*.gl` · `벌칸*.gl` (GPU layers). `리눅스/` holds the Linux-target overlays. |
| [`커널/`](../커널/) (kernels) | `커널생성.gl` + `생성*.gl` — the kernel generator: it emits a kernel module (geul source) per model architecture, shape, weight format and device. `진단*.gl` — diagnosis kernels. |
| [`백엔드/`](../백엔드/) (backends) | The PTX and SPIR-V backends that turn a kernel module (geul source) into PTX or SPIR-V — `PTX/PTX.gl`, `SPIRV/SPIRV.gl` (+ `구조.gl`, `낱말.gl`). They include the geul compiler's front end (`self/하강.gl`, from the geul repository) and lower its IR. Imported from the geul repository's `research/gpu` branch (2026-10-21; that branch stays frozen as the record of the GeulPU research). |
| [`앱/`](../앱/) (app) | `관리자.gl` (GeulPU.exe) and its modules, `web/` (the UI — self-contained HTML · JS · CSS, Korean and English), `data/models.json` (the model catalog), `꾸리기.ps1` (release packaging), `읽어보기.txt` (the zip's README), `DESIGN.md` (design system). |
| [`계약/`](../계약/) (contract) | [계약.md](../계약/계약.md) · [contract.md](../계약/contract.md) — the precision contract. `정수계약.py` — its CPU reference implementation (every bit comparison is against it), `로그확률.py` — log-probabilities (§5.2) in Python, `gguf읽기.py`. |
| [`도구/`](../도구/) (tools) | [`debugger.py`](../도구/debugger.py) — the debugger (record · replay · score · breakpoints · branch · compare run logs) from scripts, Python standard library only. [`repro_check.py`](../도구/repro_check.py) — checks a research-panel reproduction file or a debugger run log without a GPU, on the CPU reference. |
| [`시험/`](../시험/) (tests) | `빠른점검.py` (the quick check run by CI), `계약자체.py` (the contract's CPU self-tests), `참조7B.py` (the CPU reference forward pass for 7B — the end-to-end bit oracle), `판23.py` · `판26.py` · `머리128.py` · `줄128.py` (each kernel against the CPU reference — on the GPU, via `쿠다드라이버.py`), `품질.py` (hidden loss relative to the exact mode, compared with llama.cpp), `뽑기.py` · `로그확률38.py` (sampling and log-probability contracts, Python vs geul), `서버28.py` · `로그확률서버38.py` (server regression tests: bits under concurrency, API, log-probabilities), `모듈해시.txt` · `SASS넘침.txt` · `글저장소커밋.txt`. |
| [`재기/`](../재기/) (measurements) | Speed and quality runs — `견주기2.sh` (prompt time of exact · fast · llama.cpp, order alternating by round) · `견주기요약.py`, `생성2.sh` (decode ms/token), `품질2.sh`, `라마.py` (llama.cpp's C API). The geul test and measurement programs are in `엔진/`: `시험_7B.gl` · `재기_7B.gl` · `재기_프롬프트.gl` · `재기_커널.gl` · `퍼플렉시티.gl`. |
| [`교차검증/`](../교차검증/) (cross-check) | The reference for the app's reproducibility check (Benchmark → cross-check): `expected.json` (the reference PC's outputs and build hashes) · `requests.json` (the 32 problems). Packaging compares the build hashes with it. |
| [`문서/`](../문서/) (documents) | [이관범위.md](../문서/이관범위.md) (what was moved) · [바뀐것.md](../문서/바뀐것.md) (changelog) · stage records ([2단계.md](../문서/2단계.md) — fast contract F1 · [2b단계.md](../문서/2b단계.md) — fast attention plan · [2c단계.md](../문서/2c단계.md) — matmul kernels · [3단계.md](../문서/3단계.md) — speculative decoding · [4단계.md](../문서/4단계.md) — long-prompt attention kernel · [4b단계.md](../문서/4b단계.md) — long-context decode merge fix), [긴문맥재기.md](../문서/긴문맥재기.md) (long-session prompt and decode vs llama.cpp) · [디버거.md](../문서/디버거.md) (debugger v1 — plan, stage results, verdicts) · [버전규칙.md](../문서/버전규칙.md) (versioning) · [용어.md](../문서/용어.md) (terminology) · [개발.md](../문서/개발.md) (this page) — Korean. |
| [`docs/`](./) | English documents — [user-guide.md](user-guide.md) (user guide) · [research-guide.md](research-guide.md) (research panel guide) · [debugger-guide.md](debugger-guide.md) (debugger guide) · [development.md](development.md) (this page) · [versioning.md](versioning.md) · [glossary.md](glossary.md) · `img/` (screenshots and the logo). |
| [`사용안내.md`](../사용안내.md) | The user guide in Korean. |
| [`연구자안내.md`](../연구자안내.md) | The research panel guide in Korean. |
| [`디버거안내.md`](../디버거안내.md) | The debugger guide in Korean. |

## Build from source

1. **Requirements**: Windows (or Linux x86-64 — see 4) with an NVIDIA driver (Ampere or newer) or a Vulkan driver. The app and the
   server need only this and the geul compiler; the tests need Python 3.12 and `pip install -r requirements.txt` (numpy).
2. **geul compiler and backends**: clone the geul repository next to this one as `../geul-lang-v2`, check out the pinned release
   ([`시험/글저장소커밋.txt`](../시험/글저장소커밋.txt) — the commit of geul 1.4.0, tag `2g-1.4.0`; CI, packaging and the cross-check
   reference all use it) and build the self-hosted compiler. The PTX and SPIR-V backends live in this repository under
   [`백엔드/`](../백엔드/) and are built with that compiler (the front end they include, `self/하강.gl`, comes from the geul repository
   in `GEUL_ROOT`):
   ```
   git clone https://github.com/wwoosshh/geul-lang ../geul-lang-v2
   cd ../geul-lang-v2
   git checkout $(tr -d '\r\n ' < ../GeulPUv2/시험/글저장소커밋.txt)
   python build.py selfhost 컴파일러
   cd ../GeulPUv2
   mkdir -p 빌드/글
   GEUL_ROOT=../geul-lang-v2 ../geul-lang-v2/build/self_컴파일러.exe 백엔드/PTX/PTX.gl -o 빌드/글/ptx-backend.exe
   GEUL_ROOT=../geul-lang-v2 ../geul-lang-v2/build/self_컴파일러.exe 백엔드/SPIRV/SPIRV.gl -o 빌드/글/spirv-backend.exe
   ```
3. **Package the app** (what a release is): `앱\꾸리기.ps1 -GeulRoot ..\geul-lang-v2 [-GeulC ..\geul-lang-v2\build\self_컴파일러.exe] -NoZip`
   builds `빌드\앱\GeulPU-<version>\` — `GeulPU.exe`, `bin\` (server-cuda, server-vulkan, device-cuda, device-vulkan, diag-cuda,
   diag-vulkan, kernel-gen, ptx-backend, spirv-backend), `kernels\`, `web\`, `data\` — and compares the SHA-256 of the server, the
   generator and the backends with the cross-check reference. A model's kernel module is generated on the user's PC the first time that
   model is loaded (data folder `모듈\<version>\`).

   **Server only**: build the generator and the server, generate the module for the model's shape and format, compile it to PTX next
   to the server and run it (if the module is missing, the server prints the command). The terminal runner is `엔진/실행기.gl`
   (`빌드/글/실행기.exe model.gguf`, same module).
   ```
   mkdir -p 빌드/글
   GEUL_ROOT=../geul-lang-v2 ../geul-lang-v2/build/self_컴파일러.exe 커널/커널생성.gl -o 빌드/글/커널생성.exe
   GEUL_ROOT=../geul-lang-v2 ../geul-lang-v2/build/self_컴파일러.exe 엔진/서버.gl -o 빌드/글/서버.exe
   빌드/글/커널생성.exe qwen2큰 K 3584 28 4 18944 152064 빌드/글/qwen2K_3584x28x4x18944x152064.gl     # Qwen2.5-7B k-quant module
   빌드/글/ptx-backend.exe 빌드/글/qwen2K_3584x28x4x18944x152064.gl -o 빌드/글/qwen2K_3584x28x4x18944x152064.ptx
   빌드/글/서버.exe model.gguf --슬롯 4 --문맥 4096 --포트 8080
   ```
4. **Linux**: the same sources build as ELF with the geul compiler's Linux target (they import only `libc.so.6` and `libcuda.so.1`).
   Tested under WSL2 Ubuntu 24.04; not yet on native Linux with a GPU.

## Checks

[![Quick check](https://github.com/wwoosshh/GeulPUv2/actions/workflows/ci.yml/badge.svg)](https://github.com/wwoosshh/GeulPUv2/actions/workflows/ci.yml)

- **Quick check on every push** — GitHub Actions, a Windows and an Ubuntu runner, no GPU and no model files, about 5 minutes
  ([`시험/빠른점검.py`](../시험/빠른점검.py); the same command runs locally: `python -X utf8 시험/빠른점검.py --글 ../geul-lang-v2`). The geul
  compiler is a CPU program, so everything up to PTX and SPIR-V can be produced without a GPU: the check builds the geul compiler at
  the pinned geul release and, with it, both backends ([`백엔드/`](../백엔드/)), the generator and every engine program, regenerates the 22 kernel modules (Qwen2 ·
  Qwen3 · Llama shapes and formats, shared-32, one-digit and fast-contract variants) from scratch and compares source, PTX and SPIR-V byte for byte
  with [`시험/모듈해시.txt`](../시험/모듈해시.txt), assembles every module to SASS for sm_89 with NVIDIA `ptxas` (a checker only) and fails on
  new or larger register spills ([`시험/SASS넘침.txt`](../시험/SASS넘침.txt)), runs the contract's CPU self-tests
  ([`시험/계약자체.py`](../시험/계약자체.py)), checks that the tile tables agree between the generator and the engine, and checks the DLL
  import tables of the runner, the server and the app manager.
- **GPU tests** (by hand): end-to-end bit comparisons against the CPU reference ([`시험/참조7B.py`](../시험/참조7B.py)) and the server
  regression tests ([`시험/서버28.py`](../시험/서버28.py) · [`시험/로그확률서버38.py`](../시험/로그확률서버38.py)). The scripts that produced each
  research stage's verdicts are in [GeulPU's `공식실행/`](https://github.com/wwoosshh/GeulPU/tree/main/공식실행).
- **Files the GPU tests and measurements need** (not in the repository — `모델/` and `빌드/` are git-ignored):
  - Models (GGUF): the tests and measurement scripts take GGUF paths as arguments. The scripts in `재기/` look for models in the folder
    named by the environment variable `GEULPU_MODELS` (default `F:/geulpu-models`, the maintainer's PC) and for llama.cpp in
    `GEULPU_LLAMA_BENCH` (default an `llama-b<build>` folder under it). Both must be ASCII paths — llama.cpp's tools cannot open Korean
    paths. `재기/MoE/전문가기록.py` takes the llama.cpp DLL folder from `GEULPU_LLAMA`.
  - Tokenizer: the layer-0 chat cases of [`시험/머리128.py`](../시험/머리128.py) use the tokenizer inside the GGUF given to the test, so
    there is nothing else to download.
  - wikitext-2: the quality and long-context runs (`시험/품질.py`, `재기/품질2b.sh`, `재기/추측3.py`, …) read
    `빌드/wikitext-2/wiki.test.raw`, the same file llama.cpp's perplexity tool uses: put `wikitext-2-raw/wiki.test.raw` from the
    `wikitext-2-raw-v1.zip` that llama.cpp's `scripts/get-wikitext-2.sh` downloads there (sha256 `173c87a53759e020…`).

## Research log

The engine came out of a research program (stages 0–40, 2026-10-08 to 10-18) whose record — criteria committed before measuring, every
variant compared bit for bit before timing — lives in the GeulPU repository: [stage record](https://github.com/wwoosshh/GeulPU/blob/main/문서/단계기록.md)
(Korean), [English summary](https://github.com/wwoosshh/GeulPU/blob/main/docs/research-log.md), [paper draft](https://github.com/wwoosshh/GeulPU/tree/main/docs/paper).
The research was frozen on 2026-10-16; this repository carries the engine on for everyday use.

