갈래: 더함

- **llama.cpp 재현 검증 — Qwen3.5-4B**: 같은 구조(qwen35)의 다른 크기도 검증합니다. 출력층을 따로 두지 않는 모델(tied)과 Q6_K
  토큰 임베딩을 받고, 행렬의 모양에 따라 달라지는 llama.cpp 의 일 나누기(행렬-벡터 곱의 반복 · RMSNorm 의 블록)를 그대로 따릅니다.
  Qwen3.5-4B Q4_K_M 의 영수증 10개가 CUDA 에서, 짧은 셋이 Vulkan(RTX 4070 Ti)에서 비트까지 같았습니다([라마재현.md](../라마재현.md)
  6단계).

---

- **llama.cpp replay check — Qwen3.5-4B.** Other sizes of the same architecture (qwen35) can now be checked. The check
  accepts models without a separate output layer (tied embeddings) and Q6_K token embeddings, and follows llama.cpp's work
  split where it depends on the matrix shape (the matrix-vector loop and the RMSNorm block size). Ten Qwen3.5-4B Q4_K_M
  receipts matched bit for bit on CUDA, and three short ones on Vulkan (RTX 4070 Ti)
  ([문서/라마재현.md](../라마재현.md), stage 6).
