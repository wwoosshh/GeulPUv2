갈래: 더함

- **llama.cpp 재현 검증 — GQA 비가 4 가 아닌 모델 · Q8_0**: q 머리와 KV 머리의 비가 2 · 3 인 모델(Qwen3-0.6B · Llama-3.2-3B)도
  검증합니다. llama.cpp 가 GQA 비에 따라 flash attention 의 타일에 묶는 머리 수를 바꾸는 셈을 재현 커널이 따릅니다. 두 모델의
  Q8_0 영수증 10개씩과, Qwen3-8B · Llama-3.1-8B 의 Q8_0 영수증 10개씩이 CUDA 에서 비트까지 같았습니다
  ([라마재현.md](../라마재현.md) 12단계).

---

- **llama.cpp replay check — GQA ratios other than 4, and Q8_0.** Models whose query/KV head ratio is 2 or 3 (Qwen3-0.6B,
  Llama-3.2-3B) can now be checked too: the replay kernels follow how llama.cpp changes the number of heads it packs into a
  flash-attention tile for each GQA ratio. Ten Q8_0 receipts of each of those models, and ten Q8_0 receipts each of
  Qwen3-8B and Llama-3.1-8B, matched bit for bit on CUDA ([문서/라마재현.md](../라마재현.md), stage 12).
