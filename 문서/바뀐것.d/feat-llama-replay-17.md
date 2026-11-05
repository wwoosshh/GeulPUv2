갈래: 더함

- **llama.cpp 재현 검증 — 폭이 4096 을 넘는 모델(14B 급)**: 폭 5120 의 RMSNorm(llama.cpp 의 블록 1024 · 스레드마다 float4 넷 판)을
  셈하고, KV 머리가 많은 모델의 프롬프트 ubatch 에서 flash attention 부분 결과의 칸이 모자라던 것을 고쳤습니다.
  Qwen2.5-Coder-14B-Instruct Q4_K_M 으로 확인했습니다([라마재현.md](../라마재현.md) 17단계).

---

- **llama.cpp replay check — models wider than 4096 (14B class).** The check now computes the RMSNorm that llama.cpp uses
  for width 5120 (block 1024, four float4 per thread), and the flash-attention partial buffers no longer run short on
  prompt ubatches of models with many KV heads. Verified with Qwen2.5-Coder-14B-Instruct Q4_K_M
  ([문서/라마재현.md](../라마재현.md), stage 17).
