갈래: 더함

- **llama.cpp 재현 검증 — 작은 폭(Qwen2.5-1.5B) · DeepSeek-R1-Distill**: 폭 1536 처럼 RMSNorm 이 블록 512 를 쓰는 모델도 검증합니다.
  Qwen2.5-1.5B-Instruct Q4_K_M 의 영수증 10개와, 코드 변경 없이 DeepSeek-R1-Distill-Qwen-7B Q8_0 의 영수증 10개가 CUDA 에서 비트까지
  같았습니다([라마재현.md](../라마재현.md) 14단계).

---

- **llama.cpp replay check — narrower widths (Qwen2.5-1.5B) and DeepSeek-R1-Distill.** Models whose RMSNorm uses 512-thread
  blocks (width 1536) can now be checked too. Ten Qwen2.5-1.5B-Instruct Q4_K_M receipts, and — with no code change — ten
  DeepSeek-R1-Distill-Qwen-7B Q8_0 receipts, matched bit for bit on CUDA ([문서/라마재현.md](../라마재현.md), stage 14).
