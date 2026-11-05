갈래: 더함

- **llama.cpp 재현 검증 — llama 구조(Llama 3.1)**: Llama · Mistral 이 쓰는 llama 구조도 검증합니다. 보통 RoPE(이웃한 두 칸이 짝)와
  Llama 3.1 의 긴 문맥 늘림(rope_freqs)을 재현 커널로 옮겼고, Q · K 정규화가 없는 그래프를 재현 호스트가 따릅니다.
  Meta-Llama-3.1-8B-Instruct Q4_K_M 의 영수증 10개가 CUDA 에서 비트까지 같았습니다([라마재현.md](../라마재현.md) 11단계).

---

- **llama.cpp replay check — llama architecture (Llama 3.1).** The llama architecture used by Llama and Mistral can now be
  checked too. The replay kernels cover the normal RoPE (adjacent pairs) and Llama 3.1's long-context scaling (rope_freqs),
  and the replay host follows the graph without Q/K norms. Ten Meta-Llama-3.1-8B-Instruct Q4_K_M receipts matched bit for
  bit on CUDA ([문서/라마재현.md](../라마재현.md), stage 11).
