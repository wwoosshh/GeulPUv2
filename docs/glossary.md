# Glossary — reading GeulPU's Korean code and documents

GeulPU is written in **geul (글)**, a Korean programming language whose keywords, types and identifiers are Korean words, and its
documents are written in Korean. This page gives the English meaning of the terms you will meet, so that the code, the commit
history and the Korean documents (`문서/`) can be followed without reading Korean fluently. The English documents in `docs/` use
exactly these translations.

## 1. Reading geul source

| geul | English | Notes |
|---|---|---|
| `[정수 a를 정수 b를 더하기]는 -> 정수 { … }` | `int64 add(int64 a, int64 b) { … }` | A function signature lists parameters first, then the name. Korean particles (조사) mark each parameter's role. |
| `를 / 을`, `에`, `에서`, `로 / 으로`, `와 / 과` | object, target ("into"), source ("from"), means/target ("to", "as"), companion ("with") | Particles on parameters. A call may use them as keywords, e.g. `메모리_복사(목적 에, 원본 에서, 길이 를)` = `memcpy(dst, src, len)`. |
| `x 으로 짧은실수` | `(float)x` | Cast: "x as float32". |
| `조건이면 { … } 아니면 { … }` | `if (cond) { … } else { … }` | `아니면 조건이면` = `else if`. `아닌 조건이면` = `if (!cond)`. |
| `반복 (정수 i는 0부터 n 전까지) { … }` | `for (int64 i = 0; i < n; i++)` | `(조건)동안 { … }` = `while (cond)`. `탈출` = `break`, `계속` = `continue`. |
| `반환 x.` | `return x;` | Statements end with `.` |
| `포함 "파일.gl"` | `#include` | |
| `외부 [ … 이름]는 -> 형.` | `extern` declaration | In kernel modules: a GPU intrinsic. |
| `(* … *)` | comment | |
| `참` / `거짓` | `true` / `false` | `그리고` = `&&`, `또는` = `\|\|`, `아닌` = `!` |
| `~하다` call form, e.g. `x를 정렬하다.` | `정렬(x);` | Verb-style call of a function whose name is the verb stem. |

### Types

| geul | English |
|---|---|
| `정수` | int64 |
| `중간정수` | int32 |
| `짧은정수` | int16 |
| `작은정수` / `부호없는 작은정수` | int8 / uint8 |
| `실수` | float64 |
| `짧은실수` | float32 |
| `반실수` | float16 (half) |
| `참거짓` | bool |
| `문자열` | string (UTF-8) |
| `T 참조` | pointer to T |
| `묶음` (as a type declaration) | struct |

### GPU intrinsics used by the kernel generator

| geul | PTX / CUDA |
|---|---|
| `곱해더하기(a, b, c)` | `fma.rn.f32` — the only place a multiply-add is fused (see the contract) |
| `정수텐서곱(j, a0 … a3, b0, b1, c0 … c3)` | `mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32` (four calls j = 0 … 3 read the four results of one instruction) |
| `정수텐서곱_나부호없음` | same with unsigned B (`.u8`) |
| `정수텐서곱16(j, a0, a1, b0, c0 … c3)` | `mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32` (stage 27) |
| `행렬조각읽기(j, 판, i)` | `ldmatrix` (×4) |
| `바이트넷곱더하기(a, b, c)` | `dp4a` (signed 8-bit dot product of four) |
| `바이트고르기(a, b, s)` | `prmt` (byte permute) |
| `비동기복사16(판, 자리, 원본, 색인)` / `비동기묶기()` / `비동기기다리기(n)` | `cp.async.cg … 16` / `cp.async.commit_group` / `cp.async.wait_group n` |
| `동기화()` / `워프동기화()` | `bar.sync 0` / `bar.warp.sync` |
| `레인에서받기(v, k)` / `아래에서받기(v, s)` | `shfl.sync.idx` / `shfl.sync.down` (float32; `정수레인에서받기` / `정수아래에서받기` for int32) |
| `비트실수(i)` / `실수비트(x)` | reinterpret int32 bits as float32 / float32 as int32 |
| `가까운정수(x)` | round-to-nearest-even float → int (`cvt.rni`), range-checked |
| `큰쪽(x, y)` | `max` |
| `블록안가로()` / `블록가로()` / `블록세로()` | `threadIdx.x` / `blockIdx.x` / `blockIdx.y` |
| `공유판N()` | a shared-memory array of N KiB |

## 2. Domain terms

| Korean | English used in `docs/` | Meaning |
|---|---|---|
| 계약 | **contract** | The written precision specification (`docs/contract.md`). Every rounding point is fixed in the source, so GPU results equal the CPU reference bit for bit. |
| 정수 블록 계약 | integer block contract | Linear layers: activations as per-32-block integers, exact integer sums, one fixed accumulation chain. |
| 정수 KV 계약 | integer KV contract | Attention: keys/values cached as block integers, scores and outputs from exact integer sums. |
| 자리 / 자리 둘 / 자리 넷 | digit / two digits / four digits | An activation integer n (|n| ≤ 2¹⁴) is stored as two balanced base-256 digits D₁, D₀ (n = 256·D₁ + D₀) so that int8 tensor cores can multiply it exactly. |
| 정보 | block info | Per row and per 32-block of activations: the "magic number" word that encodes the block exponent E. |
| 마법수 | magic number | The float 1.5 · 2²³ · 2^(E−13), whose bit pattern turns an integer add into an exact float conversion. |
| 적힌지수 | written exp | The exp function defined by a written polynomial (Cephes coefficients, IEEE float32 ops only) instead of a hardware approximation. |
| 비정규수버림 · 규칙 R | subnormal flush, rule R | The float32 rule of the contract since stage 31: subnormal inputs read as signed zero; a result whose 24-bit rounding (unbounded exponent) is below 2⁻¹²⁶ becomes signed zero; fma does not round the product. Declared in a module's source as `(* 비정규수버림 *)`, emitted as PTX `.ftz`; NVIDIA Vulkan, Intel Vulkan and PTX `.ftz` all follow it bit for bit. |
| 척 / 척도 | scale | Quantization scale of a weight block (Q8_0 · Q4_0 `d`; k-quant `d`, `m`, sub-block scales). |
| 슈퍼블록 / 아래블록 | super-block / sub-block | k-quant structure: 256 values = 8 sub-blocks of 32 (Q4_K · Q5_K) or 16 of 16 (Q6_K). |
| 높은 비트 · 니블 | high bits · nibble | Bits above the low 4 (Q5_K: 1 bit, Q6_K: 2 bits) · a 4-bit value. |
| 깔기 / 깐 / 배치 | lay out / laid-out / layout | Re-arranging GGUF weights into the engine's GPU layout (same number of bytes read). |
| 판 | **kernel** (literally "plate") | One generated GPU kernel variant, e.g. a tile shape. "판 11 개" = the 11 tile shapes. |
| 타일 판 | tile kernel | Integer-tensor-core GEMM used for prompts (prefill). Names: `정수타일선형` (integer tile linear), `정수넓은…` (wide), `정수긴…` (long), `정수깊은…` (deep), `정수세판…` (three-stage) … |
| 줄 판 / 정수줄하나 | row kernel / single-row integer kernel | GEMV-style kernels used for decoding (one or a few rows). |
| 끝손질 | epilogue | What a matmul kernel does after the sums: none (`""`), `_잔차` (add residual), `_SwiGLU정수` (SwiGLU then quantize for the next layer). |
| 판 고르기 | kernel selection | The engine times the candidate kernels once per shape and keeps the fastest (all candidates give identical bits). |
| 메가커널 | megakernel | One cooperative launch that generates a whole token (stage 15). |
| 걸음띄우기 | step-by-step decode | Decoding without the megakernel: one prompt-path computation per token, waiting for each token (the Vulkan path — SPIR-V has no cooperative launch; identical bits, stage 31). |
| 벌칸 · 벌칸 층 | Vulkan layer | `호스트/벌칸.gl`: the CUDA driver API's names and signatures implemented over vulkan-1.dll, so the same engine source runs on CUDA or Vulkan (stage 31). |
| 머리 · 몸 (파일) | head · body (file) | A test split in two: the head includes a GPU layer (`쿠다.gl` or `벌칸.gl`), the body holds the program (e.g. `시험_엔진.gl` / `시험_엔진V.gl` + `시험_엔진몸.gl`). |
| 묶음 (.spvs) | SPIR-V bundle | The SPIR-V backend's output: one SPIR-V module per kernel with its shared-memory size and parameter kinds ("GEULSPV1"). |
| 장치 한도 | device limit | A kernel the device cannot run (its shared memory exceeds the limit — e.g. 32 KB on Intel UHD 770); tests log it instead of skipping it silently. |
| 공유32 | 32 KB variant | `커널생성 qwen2Q8 … --공유32`: the 0.5B module for 32 KB devices — prompt attention with one buffer (36 → 20 KB, same arithmetic order), no branch kernels. |
| 생성 그래프 | decode graph | A CUDA graph replayed per generated token. |
| 묶음 | group | Attention: a group of 16 chunks = 1024 positions (two-level fold). Elsewhere: a batch of sentences. |
| 조각 | chunk / part | Attention: 64 keys. Layer splitting (stage 25): a part = an engine owning a range of layers. |
| 접기 / 두 층 접기 | fold / two-level fold | Online-softmax merge of chunk states (M, L, O), first within a group, then across groups. |
| 덩이 | prompt chunk | Prompts are processed in chunks of up to 1024 rows. |
| 걸음 | step | One decoding step. |
| 칸 | cell / element | "다른 칸 0" = zero differing elements (bit comparison). |
| 길 / 다섯 길 | path / five paths | Ways of computing the same thing: decode graph, all-at-once, in four pieces, one row at a time, again. They must agree bit for bit. |
| 끝에서 끝 | end-to-end | Full forward pass compared with the CPU reference (all rows of logits). |
| 그대로 | unchanged (regression) | Earlier stages' outputs must stay byte-identical. |
| 바닥 | floor | Lower bound: bytes read per token / memory bandwidth (decode), or integer tensor-core peak (prefill). |
| 참조 / 참값 | (CPU) reference / ground truth | `계약/정수계약.py` implements the contract on the CPU; 참값 is an fp64 forward pass of the same weights. |
| 숨은 손실 | hidden loss | Loss an engine adds beyond the model file's own quantization (KL against the fp64 ground truth). |
| 판정 / 미리 적다 | criterion / pre-register | Each stage writes its pass/fail criteria and commits them before measuring. |
| 통과 / 통과하지 못함 / 기록 | pass / not passed / record (not judged) | Verdict words in stage documents. |
| 같다 / 느리다 / 빠르다 | same / slower / faster | Speed classes vs llama.cpp: r = time(geul) / time(llama.cpp); "같다" = 0.95 ≤ r ≤ 1.05. |
| 공식 실행 | official run | A stage's measurement script run on the committed code (`공식실행/`). |
| 빠른 점검 | quick check (CI) | `시험/빠른점검.py`, run on every push without a GPU. |
| 시험 / 재기 | test / benchmark | Folders `시험/` (bit tests) and `재기/` (timing and comparisons). |
| 엔진 / 호스트 / 실행기 / 생성기 / 모듈 | engine / host / runner / kernel generator / PTX module | |
| 올림 / 올리기 | load time / upload, prefetch | Model loading time; in kernels, the asynchronous prefetch of the next block. |
| 층 / 머리 / KV 머리 / 너비 / 확장폭 | layer / attention head / KV head / hidden width / FFN width | |
| 낱말 / 낱말표 / 로짓 / 치우침 / 잔차 | token / embedding table / logits / bias / residual | |
| 층 나누기 / 넘김 / 예산 | layer split / hand-off / memory budget | Stage 25 (pipeline across GPUs, emulated on one GPU). |
| 문맥 | context | Both "context length" and "CUDA context", depending on the sentence. |
| 오류 칸 | error cell | A per-module flag set when a float → int conversion overflows; tests require it to stay 0. |
| k-양자화 | k-quants | GGUF Q4_K · Q5_K · Q6_K (and mixes such as Q4_K_M). |

## 3. Folder names

| Folder | Contents |
|---|---|
| `계약/` | contract — CPU reference implementation (`정수계약.py`) |
| `커널/` | kernels — the kernel generator (`커널생성.gl` and `생성*.gl`, geul), legacy Python generator |
| `호스트/` | host — engine, runner, GGUF loader, tokenizer, tests and benchmarks written in geul |
| `엔진/` | Python engine (legacy, used for comparison and GPT-2 research) |
| `시험/` | tests |
| `재기/` | benchmarks and result collectors |
| `문서/` | Korean documents (stage reports, contract) — English versions in `docs/` |
| `공식실행/` | official run scripts |
| `준비/` | model preparation |
| `컴파일러/` | Python reference front end and the Python PTX backend (kept for comparison) |
| `빌드/`, `모델/`, `도구/` | build output, model files, external tools (not in git) |
