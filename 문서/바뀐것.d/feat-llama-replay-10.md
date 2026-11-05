갈래: 더함

- **llama.cpp 재현 검증 — dense 구조(Qwen3)**: GDN 이 없는 보통의 구조도 검증합니다(모든 층 어텐션 · 머리 차원 128 · Q · K 정규화 ·
  NEOX RoPE). 머리 차원 128 의 flash attention(생성 걸음 · 프롬프트의 네 설정)과, llama.cpp 가 Q · K 의 정규화와 RoPE 를 한 커널로
  엮어 도는 셈을 재현 커널로 옮겼고, dense 모델의 서버가 프롬프트를 나누는 셈(체크포인트 없음)을 흉내 냅니다. Qwen3-8B Q4_K_M 의
  영수증 10개가 CUDA 에서 비트까지 같았습니다([라마재현.md](../라마재현.md) 10단계).

---

- **llama.cpp replay check — dense architecture (Qwen3).** Plain architectures without the gated delta net can now be checked
  too (attention in every layer, head size 128, Q/K norms, NEOX RoPE). The replay kernels now cover flash attention with head
  size 128 (the decode step and the four prompt configurations) and the fused kernel in which llama.cpp runs the Q/K norm and
  RoPE together, and the check emulates how the server splits the prompt for a dense model (no checkpoints). Ten Qwen3-8B
  Q4_K_M receipts matched bit for bit on CUDA ([문서/라마재현.md](../라마재현.md), stage 10).
