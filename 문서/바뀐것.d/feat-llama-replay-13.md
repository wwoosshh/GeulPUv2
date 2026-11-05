갈래: 더함

- **llama.cpp 재현 검증 — qwen2 구조(Qwen2.5)**: Qwen2 · Qwen2.5 · DeepSeek-R1-Distill-Qwen 이 쓰는 qwen2 구조도 검증합니다. Q · K · V
  행렬곱의 치우침(bias)과 NEOX RoPE 를 재현 호스트가 따릅니다(커널은 그대로). Qwen2.5-7B-Instruct Q4_K_M 의 영수증 10개가 CUDA 에서
  비트까지 같았습니다([라마재현.md](../라마재현.md) 13단계).

---

- **llama.cpp replay check — qwen2 architecture (Qwen2.5).** The qwen2 architecture used by Qwen2, Qwen2.5 and
  DeepSeek-R1-Distill-Qwen can now be checked too. The replay host follows the Q/K/V matmul biases and NEOX RoPE (no kernel
  changes). Ten Qwen2.5-7B-Instruct Q4_K_M receipts matched bit for bit on CUDA ([문서/라마재현.md](../라마재현.md), stage 13).
