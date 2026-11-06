# GeulPU 재현 키트 (GeulPU 0.21.0)

이 키트는 GeulPU 0.21.0 을 돌리는 기계가 우리 기준 PC 와 **같은 비트**를 내는지 본다. 모델 셋을 설정 둘로 불러, 정해 둔 채팅 요청
12 개를 하나씩, 이어 한꺼번에 보내고, 세 가지를 `expected.json` 의 기준값과 견준다: 프롬프트 토큰, 생성한 토큰, 걸음마다 로짓(로짓 행
전체)의 해시. 로짓 하나의 마지막 비트만 달라도 그 걸음의 해시가 달라진다.

[English](README.md)

| | |
|---|---|
| 기준 | RTX 4070 Ti (Ada, sm_89), Windows 11, PyPI 에서 설치한 GeulPU 0.21.0 |
| 모델 | Qwen3-8B Q4_K_M · Qwen3-8B Q8_0 · Qwen3.5-9B Q4_K_M (모두 19.4 GB) |
| 설정 | `default`(앱의 기본 설정 — CUDA 의 Qwen3-8B Q4_K_M 에서는 빠른 계산 · 8비트 KV 캐시 · 빠른 어텐션. 나머지 두 모델은 둘 다 쓰지 않아 빠른 계산 · 16비트 KV 캐시) · `exact`(정밀 계산). 둘은 로짓이 서로 달라 저마다의 기준과 견준다 |
| 요청 | 수학(하나는 생각 켬) · 논리 · 코드 · 도구 호출 둘 · 한국어 · 스페인어(표집) · 6천 토큰 기록 · 여러 턴 대화(표집) |
| 시간 | 기준 카드에서 7 분쯤(모델 파일의 SHA-256 확인 포함) |

앱에 든 **재현성 검증**(벤치마크 화면 — DeepSeek-R1-Distill-Qwen-7B Q8_0, 수학 32 문제)을 먼저 돌려 보면 좋다. 이 검증은 RTX 3080 Ti ·
4070 Ti · 5070, H100 MIG 구획, Radeon RX 9060 XT, Intel 내장 그래픽에서 이미 같았다. 키트도 `--builtin-check` 로 함께 돌린다.

**독립 검증 결과.** 2026년 11월에 마드리드 공과대학교 GING 그룹이 이 키트와 그들 자신의 시험 모음을 RTX 6000 Ada · H100 MIG 구획 ·
A100 MIG 구획에서 돌렸다: 카드마다 144/144 이고, 그들 모음의 검사(동시 대화 8 개 · 추측 디코딩 켬과 끔 · 캐시 축출 · Llama 3.1)도 모두
통과했다. 보고서와 재현 묶음은 [independent/upm-ging-2026-11](independent/upm-ging-2026-11/README.ko.md) 에 있다.

## 1. GeulPU 0.21.0 설치

```bash
pip install geulpu==0.21.0        # 또는: pipx install geulpu==0.21.0
```

또는 [릴리스 페이지](https://github.com/wwoosshh/GeulPUv2/releases/tag/v0.21.0)에서 `GeulPU-0.21.0-linux-x64.tar.gz`(또는 Windows zip)를
받는다.

- 리눅스 x86-64 는 RTX 30 시리즈 이상의 NVIDIA GPU(계산 능력 8.0 이상 — A100 · A6000 · H100 · Ada 카드)와 드라이버, glibc 2.34
  이상(Ubuntu 22.04 이상)이 필요하다.
- Windows 10 · 11(64비트)은 Vulkan 으로 AMD · Intel GPU 에서도 돈다. 다만 0.21.0 에서 Qwen3.5 는 NVIDIA(CUDA)에서만 돈다.
- 그래픽 메모리는 12 GB 면 된다(기준 카드가 12 GB).

## 2. 모델 받기

세 파일을 한 폴더에 둔다. 키트는 쓰기 전에 앱을 거쳐 SHA-256 을 확인한다.

| 모델 | 파일 | 크기(바이트) | SHA-256 |
|---|---|---|---|
| Qwen3-8B Q4_K_M | [Qwen/Qwen3-8B-GGUF](https://huggingface.co/Qwen/Qwen3-8B-GGUF) `Qwen3-8B-Q4_K_M.gguf` | 5,027,783,488 | `d98cdcbd03e17ce47681435b5150e34c1417f50b5c0019dd560e4882c5745785` |
| Qwen3-8B Q8_0 | [Qwen/Qwen3-8B-GGUF](https://huggingface.co/Qwen/Qwen3-8B-GGUF) `Qwen3-8B-Q8_0.gguf` | 8,709,518,112 | `408b955510e196121c1c375201744783b5c9a43c7956d73fc78df54c66e883d6` |
| Qwen3.5-9B Q4_K_M | [unsloth/Qwen3.5-9B-GGUF](https://huggingface.co/unsloth/Qwen3.5-9B-GGUF) `Qwen3.5-9B-Q4_K_M.gguf` | 5,680,522,464 | `03b74727a860a56338e042c4420bb3f04b2fec5734175f4cb9fa853daf52b7e8` |

```bash
mkdir -p ~/geulpu-models && cd ~/geulpu-models
curl -L -O https://huggingface.co/Qwen/Qwen3-8B-GGUF/resolve/main/Qwen3-8B-Q4_K_M.gguf
curl -L -O https://huggingface.co/Qwen/Qwen3-8B-GGUF/resolve/main/Qwen3-8B-Q8_0.gguf
curl -L -O https://huggingface.co/unsloth/Qwen3.5-9B-GGUF/resolve/main/Qwen3.5-9B-Q4_K_M.gguf
```

`--builtin-check` 에는 DeepSeek-R1-Distill-Qwen-7B Q8_0 도 있어야 한다
([unsloth/DeepSeek-R1-Distill-Qwen-7B-GGUF](https://huggingface.co/unsloth/DeepSeek-R1-Distill-Qwen-7B-GGUF),
`DeepSeek-R1-Distill-Qwen-7B-Q8_0.gguf`, 8,098,524,896 바이트) — 같은 폴더에 두거나 앱에서 받는다(모델 → 모델 찾기).

## 3. 키트 실행

터미널 하나에서 GeulPU 를 띄운다. 자료 폴더를 따로 주면 키트의 모델과 설정이 평소 쓰는 앱과 섞이지 않는다:

```bash
geulpu --data ~/geulpu-kit-data --no-open
```

다른 터미널에서, 이 폴더에서:

```bash
python3 run_kit.py --models-dir ~/geulpu-models
```

요청마다 한 줄씩 나온다(예: `math-1     alone    256 tokens    4.8 s  same`). 마지막 줄에 합계와 판정이 나온다:

```
144/144 requests identical — IDENTICAL to the reference. Result file: result-NVIDIA-H100-...-20261105-101500.json
```

결과 파일을 보내 주면 된다. GPU · 드라이버 · OS · GeulPU 판, 요청마다의 판정, 그리고 다른 요청에 한해 그 해시만 들어 있다 — 답이나
다른 글은 없다.

| 옵션 | |
|---|---|
| `--model qwen3-8b-q8_0` · `--config exact` | 모델 하나 · 설정 하나만(여러 번 줄 수 있음) |
| `--builtin-check` | 앱의 재현성 검증도 함께(R1, 32 문제, 5 분쯤) |
| `--url http://127.0.0.1:44544` | 떠 있는 앱의 주소(기본) |
| `--out FILE` | 결과 파일 이름 |
| `--concurrency 8` | 한꺼번에 보내는 요청 수 |

끝 코드: 0 같음, 1 다름, 2 끝내지 못함(까닭이 함께 나온다 — 이를테면 모델 파일이 없음).

키트는 도는 동안 모델을 불러오고 앱의 설정(계산 방식 · KV 캐시 · 어텐션 · 동시 대화 수 · 문맥)을 바꾼다. 끝에 원래 설정으로
되돌리며 아무것도 지우지 않는다. 도는 동안에는 같은 앱을 다른 일에 쓰지 않는다.

## 다르게 나오면

- `prompt tokens differ` — 채팅 템플릿이나 토크나이저가 다른 프롬프트를 만들었다. 모델 파일의 SHA-256 을 확인한다.
- `logits differ from step k` — 처음 `k` 걸음은 같고 `k` 걸음에서 달라졌다. 보통 그 뒤의 답도 달라진다.
- `note: the server's computation path differs from the reference's` — 앱이 이 모델에 다른 계산 길을 골랐다(이를테면 8비트 KV
  캐시를 쓰지 않음). 그것만으로도 다를 수 있다. 결과 파일에 둘 다 남는다.

키트의 판정은 토큰과 로짓뿐이고 응답의 글은 견주지 않는다. 도구 호출의 id(`call_…`)와 응답의 `created` 시각은 호출마다 새 값이
되게 만든 것이라, 같은 요청은 같은 토큰 · 같은 로짓을 내지만 응답 글까지 같지는 않다.

한 기계에서 하나씩 보낸 답과 한꺼번에 보낸 답이 다르면 묶음이 결과를 바꾼 것이고, GeulPU 의 정밀도 계약에서는 일어나지 않아야 한다.
어느 쪽이든 결과 파일과 함께 알려 주면 된다.

## 관리자용

- `make_requests.py` 가 `requests.json` 을 쓴다(긴 문맥 요청은 정해 둔 씨앗으로 만든 가짜 기록 — Qwen3 템플릿으로 프롬프트 6,076 토큰).
- `expected.json` 은 기준 PC 에서 새 자료 폴더로 `run_kit.py --record expected.json` 을 돌려 만든다. 한꺼번에 보낸 답이 하나씩 보낸
  답과 다르면 기록을 멈추고, 또 다른 새 자료 폴더로 다시 돌려 같을 때만 기준으로 쓴다(기준 PC 에서 기록 144/144, 다시 돌려 144/144).
- 키트는 0.21.0 앱의 HTTP API(`/api/state` · `/api/models/add` · `/api/settings` · `/api/server/start` · `/api/bench/crosscheck`)와
  `"geulpu": {"logit_hash": true, "prompt_tokens": true}` 를 붙인 `/v1/chat/completions` 를 쓴다. GeulPU 의 새 판에는 새 키트 판과 새
  기준이 필요하다.
