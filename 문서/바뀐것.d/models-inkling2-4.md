갈래: 더함

- **Inkling 2(GGUF 구조 `inkling2_loop` — Inkling-2-Mini)의 Q8_0 파일을 받습니다.** 큰 행렬이 모두 Q8_0 인 파일(ggml-org 의
  Inkling-2-Mini-Q8_0)을 CUDA 의 정밀 모드로 계산합니다 — 층 반복(앞 6 층 → 가운데 12 층 × 3 → 뒤 6 층), 블록 어텐션 잔차, 상대 위치 편향과
  어텐션 싱크, 짧은 합성곱, 전문가 128 개 중 8 개의 MoE(전문가 캐시 포함). 같은 입력의 출력은 한 번에 넣는 행의 수 · 덩이 나눔 · 전문가를
  두는 곳과 상관없이 비트까지 같습니다([계약.md](../../계약/계약.md) 의 4.15). 빠른 모드 · Vulkan · 추측 디코딩 · 프롬프트 캐시는 아직
  받지 않습니다.

---

- **Inkling 2 (GGUF architecture `inkling2_loop` — Inkling-2-Mini) Q8_0 files are supported.** Files whose large matrices are all Q8_0
  (ggml-org's Inkling-2-Mini-Q8_0) run in exact mode on CUDA — layer looping (6 prelude layers → 12 core layers × 3 → 6 coda layers),
  block attention residuals, relative position bias and attention sinks, short convolutions, and the 8-of-128-expert MoE (expert cache
  included). The same input gives bit-identical outputs whatever the number of rows per call, the chunking or where the experts are
  placed ([contract.md](../../계약/contract.md), item 4.15). Fast mode, Vulkan, speculative decoding and the prompt cache are not
  supported yet.
