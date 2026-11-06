# llama.cpp 재현 키트 — llama.cpp 의 실행을 내 GPU 에서 다시 계산하기

[English](README.md)

GeulPU 는 **고치지 않은 llama-server** 가 만든 실행을 자기 GPU 커널로 다시 계산해, 생성한 토큰마다와 스텝마다의 상위 20 로그확률을
**비트까지** 견줍니다([docs/llama-replay.md](../../docs/llama-replay.md)). 이 키트의 실행은 llama.cpp b12008(공식 Windows CUDA 빌드)이
NVIDIA RTX 4070 Ti 에서 만든 것입니다. 다시 계산은 그 GPU 의 산술을 에뮬레이션하므로 **어느** GPU 에서든 — Vulkan 으로 AMD · Intel GPU
에서도 — 같은 비트가 나와야 합니다. 확인한 것은 RTX 4070 Ti(CUDA · Vulkan)와 Intel UHD 770 입니다. 이 키트는 다른 GPU, 특히 **AMD**
GPU 를 확인하려는 것입니다.

| | |
|---|---|
| 돌리는 것 | 영수증 6 개: Qwen3.5-9B Q4_K_M(하이브리드) · Qwen3-8B Q4_K_M(밀집) 요청 셋씩(프롬프트 86 · 468/401 · 1,181/1,136 토큰, 생성 최대 256 토큰) |
| 필요한 것 | Vulkan 1.3 GPU 가 있는 Windows 또는 리눅스, GeulPU 0.22.0, Python 3.8 이상(표준 라이브러리만), 두 모델의 디스크 11 GB |
| 시간 | 요즘 외장 GPU 에서 영수증마다 몇 분(처음 것은 재현 커널을 짓는 시간까지), 내장 GPU 는 훨씬 오래 |
| 건드리는 것 | 없음: 다른 포트 · 임시 자료 폴더로 자기 GeulPU 를 띄우고, 끝나면 지운다 |

## 순서

1. **GeulPU 0.22.0 설치** — `pip install geulpu==0.22.0`, 또는 [릴리스 페이지](https://github.com/wwoosshh/GeulPUv2/releases/tag/v0.22.0)의
   Windows zip 을 풉니다.
2. **모델 파일 둘**을 한 폴더에 둡니다(또는 3 에서 `--download`):

   | 파일 | 출처 | 크기(바이트) | SHA-256 |
   |---|---|---|---|
   | `Qwen3.5-9B-Q4_K_M.gguf` | [unsloth/Qwen3.5-9B-GGUF](https://huggingface.co/unsloth/Qwen3.5-9B-GGUF) | 5,680,522,464 | `03b74727a860a56338e042c4420bb3f04b2fec5734175f4cb9fa853daf52b7e8` |
   | `Qwen3-8B-Q4_K_M.gguf` | [Qwen/Qwen3-8B-GGUF](https://huggingface.co/Qwen/Qwen3-8B-GGUF) | 5,027,783,488 | `d98cdcbd03e17ce47681435b5150e34c1417f50b5c0019dd560e4882c5745785` |

3. 이 폴더에서 **키트를 돌립니다**:

   ```
   python run_replay_kit.py --models-dir <모델이 있는 폴더>
   ```

   pip 대신 Windows zip 이면 `--app <경로>\GeulPU.exe` 를 더합니다. GPU 가 여럿이면 `--device <Vulkan 장치 이름의 일부>`(예:
   `--device Radeon`). NVIDIA GPU 에서는 `--layer cuda` 도 됩니다.
4. 키트가 쓴 `result-<gpu>-<시각>.json` 파일을 **보내 주세요**(GPU 이름 · 드라이버 · 영수증마다의 결과 — 개인 정보는 없습니다).

## 결과 읽기

영수증마다 토큰과 로그확률이 모두 llama.cpp 의 실행과 비트까지 같고, 견준 값의 수가 기준 PC(`expected.json`)와 같으면 `SAME` 입니다.
`DIFFERENT` 는 비트가 처음 갈린 스텝을 알립니다. 거절(예: 드라이버가 커널을 짓지 못함)은 까닭과 함께 적힙니다. 종료 코드 0 은 여섯
모두 같다는 뜻입니다.

알고 있는 드라이버의 사실 둘은 이미 맞춰 두었습니다: AMD 의 Windows 드라이버는 워크그룹마다 공유 메모리 32 KB(재현 커널은 그 안에
든다), 할당마다 2 GiB(재현은 가중치 텐서마다 따로 할당한다)까지입니다. AMD 에서는 처음 돌리는 키트입니다 — 그 밖에 드러나는 것이
바로 알고 싶은 것입니다.
