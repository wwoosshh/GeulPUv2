갈래: 더함

- **llama.cpp 재현 검증 — Qwen3.5-0.8B**: 머리 수가 다른 크기도 검증합니다(주의 머리 8/2, GDN 의 v 머리 16). flash attention ·
  GDN 의 재현 커널이 머리 수를 모델 파일에서 받고, 폭 1024 의 RMSNorm 과 Q8_0 토큰 임베딩을 더했습니다. Qwen3.5-0.8B Q8_0 의 영수증
  10개가 CUDA 에서, 짧은 셋이 Vulkan(RTX 4070 Ti)에서 비트까지 같았습니다([라마재현.md](../라마재현.md) 8단계).

---

- **llama.cpp replay check — Qwen3.5-0.8B.** Sizes with different head counts can now be checked too (8/2 attention heads,
  16 GDN value heads). The replay kernels for flash attention and the gated delta net take the head counts from the model
  file, and RMSNorm for width 1024 and Q8_0 token embeddings were added. Ten Qwen3.5-0.8B Q8_0 receipts matched bit for bit
  on CUDA, and three short ones on Vulkan (RTX 4070 Ti) ([문서/라마재현.md](../라마재현.md), stage 8).
