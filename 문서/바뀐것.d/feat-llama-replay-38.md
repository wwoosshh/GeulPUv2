갈래: 더함

- **llama.cpp 재현 키트 — 다른 GPU 에서 확인하기**: `repro-kit/llama-replay/` 에 llama.cpp b12008 이 만든 영수증 여섯(Qwen3.5-9B ·
  Qwen3-8B)과 기준 값의 수, 그리고 사설 GeulPU 를 띄워 모든 영수증을 Vulkan(또는 CUDA)에서 검증하고 결과 파일을 쓰는 스크립트(파이썬
  표준 라이브러리만)를 둡니다. AMD GPU 처럼 이 PC 에 없는 GPU 에서 재현을 확인하는 데 씁니다([라마재현.md](../라마재현.md) 38단계).

---

- **llama.cpp replay kit for other GPUs.** `repro-kit/llama-replay/` holds six receipts recorded from llama.cpp b12008 (Qwen3.5-9B
  and Qwen3-8B), the reference value counts, and a script (Python standard library only) that starts a private GeulPU, verifies every
  receipt on Vulkan (or CUDA) and writes a result file — for checking the replay on GPUs we do not have, such as AMD
  ([문서/라마재현.md](../라마재현.md), stage 38).
