# Development

**[한국어](../문서/개발.md)** · [README](../README.md) · [contributing](../CONTRIBUTING.md) · [user guide](user-guide.md) · [glossary](glossary.md)

This page is for building GeulPU from source and finding your way around the code — to use the app, the [user guide](user-guide.md) is all you need. How to propose changes is in [CONTRIBUTING.md](../CONTRIBUTING.md). The research record (stage reports, paper) lives in the [GeulPU repository](https://github.com/wwoosshh/GeulPU); the "stage N" links below go there.

## Status (2026-10)

| | |
|---|---|
| Models | Six architectures — qwen2 · qwen3 · llama (Qwen2.5, Qwen3, Llama 3, Mistral, R1-Distill, HyperCLOVAX-SEED, MiniCPM5), qwen35 · qwen35moe (Qwen3.5, Qwen3.8, Qwen3.6 MoE — 0.12.0), gemma4 (Gemma 4 12B — 0.13.0). One architecture = a `구조/<name>/` folder and two lines in [구조/등록.gl](../구조/등록.gl) ([adding an architecture](#adding-an-architecture)). What runs per model, format, GPU and computation mode is in [the user guide's supported models](user-guide.md#supported-models); each architecture's stage record is in `구조/<name>/기록.md` |
| App | Stages 36–39 — `GeulPU.exe` (a manager written in geul) and a web UI: find models (browse every Hugging Face GGUF repo, sorted), download, add and verify them; chat (Qwen2.5's ChatML, Qwen3's template with thinking on/off, R1's, Llama 3's and Mistral's templates); self-diagnosis, speed and cross-check runs; result zips; a new-version notice; a **research panel** (stage 38 — per-token probabilities, branching, comparison, scoring and reproduction files; log-probabilities as contract §5.2) and an English UI. The release zip (1.5 MB) is built from source by CI when a tag is pushed (as a draft; a person publishes it). Windows x64 only — see [Install](user-guide.md#install). |
| Weight formats | GGUF **Q8_0 · Q4_0 · Q4_K · Q5_K · Q6_K**, including per-tensor mixes as in Q4_K_M · Q5_K_M (stage 26 — tested with files re-quantized on this PC from Q8_0; official Qwen files are not yet tested). |
| Runner | `빌드/글/실행기.exe model.gguf` (`실행기` = runner) — a geul program: chat (Qwen's ChatML and the Qwen3, Llama 3 and Mistral templates), streamed output, KV cache reused across turns, greedy decoding (the same conversation gives the same answer). About 0.3 s until the prompt runs. |
| Server | `빌드/글/서버.exe model.gguf` (`서버` = server) — a geul program (stage 28): an OpenAI-compatible HTTP API (chat · completions · SSE streaming) with slots, continuous batching, prompt caching and a sampling contract (the same seed gives the same tokens). A request returns the same bits whatever else runs with it — zero differences over four models × twelve requests × four schedules, while llama-server's greedy answer changed in 11 of 12 cases depending on the concurrent requests. Head-128 Qwen2 (7B · 14B). 7B Q4_0 aggregate throughput: 302 tokens/s with four streams and 508 with eight (llama-server 360 · 632 — a multi-row decode kernel is next); one stream is 4% faster than llama-server. A cached prompt's first token takes 13.8 ms (llama-server 13.0 ms; stage 30 feeds the last rows through the decode step). See [`문서/28단계.md`](https://github.com/wwoosshh/GeulPU/blob/main/문서/28단계.md) · [`문서/30단계.md`](https://github.com/wwoosshh/GeulPU/blob/main/문서/30단계.md) |
| Environment | Windows or Linux x86-64 (glibc) + NVIDIA Ampere (RTX 30) or newer + driver. Linux was tested under WSL2 Ubuntu 24.04 in stage 29 — the same bits as Windows on the same GPU, speed within ~0.5% ([`문서/29단계.md`](https://github.com/wwoosshh/GeulPU/blob/main/문서/29단계.md)) — and natively on Ubuntu 22.04 with an H100 NVL MIG instance (3g.47gb) in 0.12.1 (2026-10): built from source with the geul compiler and Python 3 alone, kernel modules byte-identical to Windows, the generated tokens of 24 conversations identical to an RTX 4070 Ti on Windows ([h100.md](h100.md), with speed against llama.cpp and vLLM). On Linux the programs import only libc.so.6 and libcuda.so.1. All numbers here are from one RTX 4070 Ti (12 GB). Building (generator → kernel source → PTX → runner) needs only the geul compiler. Tests and benchmarks need Python 3.12 + numpy. **Vulkan (stage 31):** Vulkan builds of the geul host (`시험_엔진V` · `시험_7BV` · `재기_7BV` — they import kernel32, shell32 and vulkan-1.dll; kernels come from the geul compiler's SPIR-V backend) reproduce the CUDA records byte for byte on NVIDIA (0.5B, 7B in five formats, 14B) and Intel integrated graphics (0.5B). Speed is not there yet — stage 34 replaced the emulated integer MMA with cooperative-matrix tensor-core work (`--협력` modules) and fixed the backend and the Vulkan layer: 7B prompts take 1.15–1.41× llama.cpp Vulkan's time (stage 31: 4.4–6.1×), decode 1.00–1.04× (stage 31: 1.09–1.24×) — a record, not yet a verdict. The server has a Vulkan build too (`server-vulkan` — `엔진/서버V.gl`, used by the app for Vulkan devices); the terminal runner does not. |
| Layer split | Stage 25 — split layers across GPUs (`--나누기 P` parts, `--장치 a,b,…` devices, `--예산 GB` budget). Emulated on one GPU with one CUDA context per part: 16 split runs are bit-identical to the single-part run; decoding with two parts takes 1.025–1.029× (mostly the emulation's context switching; 1.002–1.003× with a shared context). 14B in two parts did not fit at context 8192 on 12 GB, so that pre-registered cell was not measured (criterion 25c not passed). |
| Not yet | Real multi-GPU runs; measuring the contract's loss with the real engine on 7B+ (done for 0.5B; simulated for 7B · 14B); Q2_K · Q3_K · IQ formats other than IQ4_XS · Q5_0 (real 72B k-quant files use Q5_0 · Q8_0 for ffn_down); other MoE models (Mixtral, Qwen3-30B-A3B); running on Apple GPUs (three NVIDIA generations and an AMD Radeon RX 9060 XT (RDNA4, Vulkan) match bit for bit — [교차검증](../교차검증/); AMD speed (RX 9060 XT generation is about 3× slower than an RTX 4070 Ti, mostly attention); Intel integrated graphics matches through Vulkan for 0.5B — stage 31); Vulkan for Qwen3.5, Qwen3.6 MoE and Gemma 4; small head-64 llama models (Llama-3.2-1B); other architectures (Phi, …) and Gemma 4's E2B · E4B · 26B-A4B · 31B; server penalties (presence and frequency accept only 0) and authentication. |

## Repository layout

| Folder | What |
|---|---|
| [`엔진/`](../엔진/) (engine) | The engine in geul: `gguf.gl` (GGUF reader), `가중치.gl` · `가중치K.gl` (weight layout, k-quants), `로프.gl` (RoPE tables), `토크나이저.gl` (BPE, SentencePiece, Unicode 15.0 tables), `엔진.gl` · `엔진K.gl` · `엔진표.gl` (kernel selection, CUDA graphs, decoding, slots), `조각.gl` (layer split), `뽑기.gl` · `로그확률.gl` (sampling and log-probabilities — contract §5 · §5.2), `서버.gl` · `서버V.gl` · `서버몸.gl` (the OpenAI-compatible HTTP server — CUDA and Vulkan heads), `장치C.gl` · `장치V.gl` (device check), `진단.gl` · `진단V.gl` · `진단몸.gl` (self-diagnosis), `실행기.gl` (terminal runner), `쿠다*.gl` · `벌칸*.gl` (GPU layers). `리눅스/` holds the Linux-target overlays. |
| [`커널/`](../커널/) (kernels) | `커널생성.gl` + `생성*.gl` — the kernel generator: it emits a kernel module (geul source) per model architecture, shape, weight format and device. `진단*.gl` — diagnosis kernels. |
| [`백엔드/`](../백엔드/) (backends) | The PTX and SPIR-V backends that turn a kernel module (geul source) into PTX or SPIR-V — `PTX/PTX.gl`, `SPIRV/SPIRV.gl` (+ `구조.gl`, `낱말.gl`). They include the geul compiler's front end (`self/하강.gl`, from the geul repository) and lower its IR. Imported from the geul repository's `research/gpu` branch (2026-10-21; that branch stays frozen as the record of the GeulPU research). |
| [`앱/`](../앱/) (app) | `관리자.gl` (GeulPU.exe) and its modules, `web/` (the UI — self-contained HTML · JS · CSS, Korean and English), `data/models.json` (the model catalog), `꾸리기.ps1` (release packaging), `읽어보기.txt` (the zip's README), `DESIGN.md` (design system). |
| [`계약/`](../계약/) (contract) | [계약.md](../계약/계약.md) · [contract.md](../계약/contract.md) — the precision contract. `정수계약.py` — its CPU reference implementation (every bit comparison is against it), `로그확률.py` — log-probabilities (§5.2) in Python, `gguf읽기.py`. |
| [`도구/`](../도구/) (tools) | [`debugger.py`](../도구/debugger.py) — the debugger (record · replay · score · breakpoints · branch · compare run logs) from scripts, Python standard library only. [`repro_check.py`](../도구/repro_check.py) — checks a research-panel reproduction file or a debugger run log without a GPU, on the CPU reference. |
| [`시험/`](../시험/) (tests) | `빠른점검.py` (the quick check run by CI; its module and SASS steps are in `모듈점검.py`), `계약자체.py` (the contract's CPU self-tests), `참조7B.py` (the CPU reference forward pass for 7B — the end-to-end bit oracle), `판23.py` · `판26.py` · `머리128.py` · `줄128.py` (each kernel against the CPU reference — on the GPU, via `쿠다드라이버.py`), `품질.py` (hidden loss relative to the exact mode, compared with llama.cpp), `뽑기.py` · `로그확률38.py` (sampling and log-probability contracts, Python vs geul), `서버28.py` · `로그확률서버38.py` (server regression tests: bits under concurrency, API, log-probabilities), `모듈해시.txt` · `SASS넘침.txt` · `글저장소커밋.txt`. |
| [`재기/`](../재기/) (measurements) | Speed and quality runs — `견주기2.sh` (prompt time of exact · fast · llama.cpp, order alternating by round) · `견주기요약.py`, `생성2.sh` (decode ms/token), `품질2.sh`, `라마.py` (llama.cpp's C API), `H100/` (a Linux server GPU: GeulPU, llama.cpp and vLLM over the same HTTP requests — [h100.md](h100.md)). The geul test and measurement programs are in `엔진/`: `시험_7B.gl` · `재기_7B.gl` · `재기_프롬프트.gl` · `재기_커널.gl` · `퍼플렉시티.gl`. |
| [`교차검증/`](../교차검증/) (cross-check) | The reference for the app's reproducibility check (Benchmark → cross-check): `expected.json` (the reference PC's outputs and build hashes) · `requests.json` (the 32 problems). Packaging compares the build hashes with it. |
| [`문서/`](../문서/) (documents) | [이관범위.md](../문서/이관범위.md) (what was moved) · [바뀐것.md](../문서/바뀐것.md) (changelog) · stage records ([2단계.md](../문서/2단계.md) — fast contract F1 · [2b단계.md](../문서/2b단계.md) — fast attention plan · [2c단계.md](../문서/2c단계.md) — matmul kernels · [3단계.md](../문서/3단계.md) — speculative decoding · [4단계.md](../문서/4단계.md) — long-prompt attention kernel · [4b단계.md](../문서/4b단계.md) — long-context decode merge fix), [긴문맥재기.md](../문서/긴문맥재기.md) (long-session prompt and decode vs llama.cpp) · [H100.md](../문서/H100.md) (Linux and an H100 NVL MIG instance — build, reproducibility, speed vs llama.cpp and vLLM) · [디버거.md](../문서/디버거.md) (debugger v1 — plan, stage results, verdicts) · [버전규칙.md](../문서/버전규칙.md) (versioning) · [용어.md](../문서/용어.md) (terminology) · [개발.md](../문서/개발.md) (this page) — Korean. |
| [`docs/`](./) | English documents — [user-guide.md](user-guide.md) (user guide) · [research-guide.md](research-guide.md) (research panel guide) · [debugger-guide.md](debugger-guide.md) (debugger guide) · [development.md](development.md) (this page) · [h100.md](h100.md) (Linux and an H100 NVL MIG instance) · [versioning.md](versioning.md) · [glossary.md](glossary.md) · `img/` (screenshots and the logo). |
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
   Tested under WSL2 Ubuntu 24.04 and natively on Ubuntu 22.04 (glibc 2.35) with an H100 NVL MIG instance — the commands and what
   came up (git on an NFS home) are in [h100.md](h100.md).
   - The app: `bash 앱/꾸리기.sh <geul repo> [out dir]` builds the same folder as `앱\꾸리기.ps1` on Windows (`GeulPU`, `bin/` with
     server-cuda · device-cuda · diag-cuda · kernel-gen · ptx-backend, `kernels/`, `web/`, `data/`, `version.json`) and
     `GeulPU-<version>-linux-x64.tar.gz`. Start it with `./GeulPU --열지않음`; on a remote machine open the screen through SSH port
     forwarding (`ssh -L 44544:127.0.0.1:44544 …`). The data folder is `~/GeulPU` (`--자료` · `GEULPU_HOME`). One check without the
     screen: `./GeulPU --check pair [--quick]` (`r1` or `pair`; Korean `--검사 재현성|쌍 [--빠름]`) prints the result folder and zip and
     exits with 0 when every problem matches the reference (1 different · 2 did not finish · 3 no GPU). Put the model files in the data
     folder's `모델/` (symbolic links work).
   - Everything the app needs from the OS sits in [앱/윈도.gl](../앱/윈도.gl); the Linux version, with the same function names, is
     [앱/리눅스/윈도.gl](../앱/리눅스/윈도.gl) (libc — fork · execv · pthread · mmap; SHA-256 in geul). The app joins paths with `\` — the
     Linux version turns it into `/` wherever it takes a path and in the paths it prints to the console (the screen still shows some
     paths with `\`). Downloading (WinHTTP) is [앱/받기망.gl](../앱/받기망.gl); the Linux version
     ([앱/리눅스/받기망.gl](../앱/리눅스/받기망.gl)) does not download yet — fetch the file yourself and add it with **Add model**. The device
     check's CUDA declarations are in [엔진/장치C선언.gl](../엔진/장치C선언.gl) (Linux: libcuda.so.1).
   - Not on Linux yet: downloads inside the app, Vulkan (AMD · Intel — #135), asking whether the store's drive is an HDD (the context
     store's `auto` turns it on), flushing the store when the window closes. On SIGTERM · SIGHUP · SIGINT the manager sends SIGTERM to
     the servers and tools it started and exits.

## Checks

[![Quick check](https://github.com/wwoosshh/GeulPUv2/actions/workflows/ci.yml/badge.svg)](https://github.com/wwoosshh/GeulPUv2/actions/workflows/ci.yml)

- **Quick check on every push** — GitHub Actions, a Windows and an Ubuntu runner, no GPU and no model files, 2–3 minutes (about 20 when it
  regenerates the modules) ([`시험/빠른점검.py`](../시험/빠른점검.py); the same command runs locally: `python -X utf8 시험/빠른점검.py --글 ../geul-lang-v2`). The geul
  compiler is a CPU program, so everything up to PTX and SPIR-V can be produced without a GPU: the check builds the geul compiler at
  the pinned geul release and, with it, both backends ([`백엔드/`](../백엔드/)), the generator and every engine program, regenerates every kernel module listed in
  [`시험/모듈해시.txt`](../시험/모듈해시.txt) (per architecture, shape, format and mode) from scratch and compares source, PTX and SPIR-V byte
  for byte with that table, assembles every module to SASS for sm_89 with NVIDIA `ptxas` (a checker only) and fails on
  new or larger register spills ([`시험/SASS넘침.txt`](../시험/SASS넘침.txt)), runs the contract's CPU self-tests
  ([`시험/계약자체.py`](../시험/계약자체.py)), checks that the tile tables agree between the generator and the engine, and checks the DLL
  import tables of the runner, the server and the app manager. The module and SASS steps ([`시험/모듈점검.py`](../시험/모듈점검.py)) run
  only on commits that change what defines the modules (`커널/`, `백엔드/`, the hash and spill tables, the pinned geul commit, that script
  or `ci.yml`); other commits skip them. When they run, they compare against the commit CI compared with (the PR's base or the previous
  push — `--기준커밋`): if nothing that defines the backends changed (`백엔드/`, `엔진/파일.gl`, the pinned geul commit, that script and
  `ci.yml`), every module's source is still regenerated and compared, but PTX, SPIR-V and SASS are produced only for the modules whose
  source, hash row or spill lines changed.
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

## Working in parallel

Rules for several pieces of work (people or agent sessions) that share the repository and one GPU. The engine code that belongs to one
model architecture lives in one folder per architecture (`구조/<name>/`), and the engine and the app reach it through a registry
([`엔진/구조표.gl`](../엔진/구조표.gl), [`구조/등록.gl`](../구조/등록.gl)) — see "Adding an architecture" below.

| Role | Does | Edits |
|---|---|---|
| Integration | merges into `main`, releases, the shared core (engine, kernel generator, the app's common parts, the registry, CI, shared test tables), user documentation | everything |
| Model | adds or changes one architecture (contract section, CPU reference, kernels, engine wiring, tests) — one worktree, one branch | that architecture's files and its one registry line. A change to the shared core goes to the integration work as a small pull request (what, why, which test) |
| Q&A and outreach | explanations, reading measurements, drafts of mails and posts | no code |

- **The GPU has a queue.** Every command that uses the GPU (시험_7B, the servers, a test instance of the app, llama.cpp, kernel
  variant tests, measurements) runs through `python -I -X utf8 시험/GPU줄.py 실행 --이름 "<branch> — <task>" -- <command> [args …]`:
  first come, first served, one at a time, and the processes the command starts end with it. `시험/GPU줄.py 상태` shows who holds the
  GPU, who is waiting, and nvidia-smi's memory and processes (including ones outside the queue). Wait with work that needs no GPU
  (contract, CPU reference, documentation).
- **Three levels of checks.** (1) Every commit, minutes, no GPU: the quick check (CI — module hashes, contract self-tests, builds).
  (2) Every model pull request, short GPU time in the queue: one representative file through [`재기/모델검증.py`](../재기/모델검증.py);
  a model that does not fit this PC is checked on a file with fewer layers (like [`시험/작은MoE.py`](../시험/작은MoE.py)). (3) Releases or
  overnight, long GPU time: Vulkan, long context, quality, 32 KB shared memory, cross-check.
- **One pull request per step of a model**: (1) contract and CPU reference (no GPU) → (2) kernels and unit tests (short GPU) → (3) engine
  and app wiring → (4) verification → (5) user documentation and release (integration). Models at different steps only have to queue for
  step 4, the long GPU step.
- **User documentation changes only at release.** Pull requests leave README, 사용안내, user-guide and the support tables alone (the
  integration work updates them for a release). A change users will notice adds one fragment file, `문서/바뀐것.d/<branch>.md`
  (format in [`_꼴.md`](../문서/바뀐것.d/_꼴.md): kind, Korean, `---`, English). On the release branch,
  `python -I -X utf8 시험/바뀐것합치기.py --쓰기` moves the fragments into [문서/바뀐것.md](../문서/바뀐것.md) ("다음 판") and
  [CHANGELOG.md](../CHANGELOG.md) ("Unreleased") and deletes them. The quick check validates the fragments.
- **Merging**: branch → pull request → green CI → squash merge (as before). When `main` is merged into a branch, the tables that grow by
  lines ([`시험/모듈해시.txt`](../시험/모듈해시.txt), [`시험/SASS넘침.txt`](../시험/SASS넘침.txt)) keep the lines of both sides
  (`merge=union` in `.gitattributes`); if both sides changed the same line, the quick check reports the hash mismatch.

### Adding an architecture

One GGUF `general.architecture` is one architecture. The engine's shared code (엔진/엔진.gl, 엔진K.gl, 가중치.gl, 서버몸.gl) never looks at
the architecture name; it calls the current architecture's table (`현재구조` — function slots, values, capabilities). A new architecture
comes in as one folder and two registry lines.

1. **Contract and CPU reference** (no GPU): the contract section ([계약/계약.md](../계약/계약.md) §4, [contract.md](../계약/contract.md)), the
   CPU reference ([계약/정수계약.py](../계약/정수계약.py) functions and `시험/참조<name>.py`), an fp64 math reference, tests in
   [시험/계약자체.py](../시험/계약자체.py).
2. **Kernels**: the architecture's kernels in one generator file (`커널/생성<name>.gl`); [커널/커널생성.gl](../커널/커널생성.gl) gets a mode, an
   include and the rejections of unsupported combinations. Bit tests per kernel against the CPU reference (`시험/<name>판.py`), the module
   hash row (`--해시갱신`) and the SASS spill lines (`시험/SASS_WSL.py`).
3. **Engine**: `구조/<name>/엔진.gl` (plus `가중치.gl` for large weight code) defines one table — start from `기본구조표("<name>")`, the
   shared path, and change only the slots that differ. [엔진/구조표.gl](../엔진/구조표.gl) is the reference for the slots: configuration and
   module naming (`설정읽기`, `모듈구조`, `모듈꼬리`, `생성옵션`, `생성모드`, `표시판`), weights (`층깔기` … `올리기끝`), kernels, memory and
   buffers (`모듈준비`, `장치바이트`, `큰할당`, `버퍼잡기`), the layer recipe (`층띄움들`), logits post-processing (`로짓뒤`, and `로짓뒤칸`
   for one buffer — the layer lens rows), matmul kinds (shared numbers 0 … 5; the architecture defines the rest), the KV layout (`KV폭` —
   elements per position of one cache layer), per-slot state, and capabilities (`생성그래프`, `표판`, `추측`, `슬롯복사`, `저장소`, `나누기`,
   `캐시통째`, `되감기한도` — how many tokens a slot cache may be rewound and reused, which `버퍼잡기` may set, `공통어텐션` — whether the shared
   attention buffers are allocated, `렌즈됨`, `J렌즈됨`, plus what the app chooses: `정밀만`, `벌칸까닭`, `KV8됨`, `F2B`, `공유32됨`, `협력됨`, `먼됨`). Start narrow
   (no CUDA graph, table step, speculative decoding, slot copy, context store or layer split; exact mode only; a Vulkan reason): the engine
   then runs its generic fallbacks. Register it in [구조/등록.gl](../구조/등록.gl) with `포함 "<name>/엔진.gl"` and `<name>구조들등록().`
   (the file merges line by line). Examples: [구조/qwen35/엔진.gl](../구조/qwen35/엔진.gl) (Qwen3.5 and MoE — almost every slot) and
   [구조/표준.gl](../구조/표준.gl) (qwen2, qwen3, llama — the shared path).
4. **Verification**: bit-compare 시험_7B's five paths with the CPU reference on a file with fewer layers (like
   [시험/작은MoE.py](../시험/작은MoE.py)), in the GPU queue; check the real file with [재기/모델검증.py](../재기/모델검증.py); add a case to
   [시험/구조회귀.py](../시험/구조회귀.py), which compares every architecture's bits with a baseline after shared-core changes.
5. **App**: the app asks the engine whether it accepts a file, which module and generator arguments it needs and what it can do
   (`server --모델확인` — [엔진/모델확인.gl](../엔진/모델확인.gl)), so no app code changes. A catalog entry (`앱/data/models.json`), and changes to
   the server's shared parts such as a new tool-call format, go to the integration work as small pull requests.
6. **Documentation**: a changelog fragment (`문서/바뀐것.d/<branch>.md`); stage notes in `구조/<name>/기록.md`. User documentation and the
   support tables change at release time (integration).

## Research log

The engine came out of a research program (stages 0–40, 2026-10-08 to 10-18) whose record — criteria committed before measuring, every
variant compared bit for bit before timing — lives in the GeulPU repository: [stage record](https://github.com/wwoosshh/GeulPU/blob/main/문서/단계기록.md)
(Korean), [English summary](https://github.com/wwoosshh/GeulPU/blob/main/docs/research-log.md), [paper draft](https://github.com/wwoosshh/GeulPU/tree/main/docs/paper).
The research was frozen on 2026-10-16; this repository carries the engine on for everyday use.

