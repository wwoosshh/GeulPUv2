갈래: 더함

- **llama.cpp 재현 검증 — 최신 llama.cpp(b12138)**: b12081 부터 llama.cpp 가 프롬프트 셈(MMQ)의 Q8_1 합을 바꿨습니다(값의 실수 합 →
  양자화한 정수의 합 × d). 그 셈을 더해 b12138 의 영수증도 비트까지 같게 검증합니다([라마재현.md](../라마재현.md) 21단계).

---

- **llama.cpp replay check — the latest llama.cpp (b12138).** From b12081 llama.cpp computes the Q8_1 correction sum of its prompt
  matrix multiplications (MMQ) from the quantized integers (sum × d) instead of the float values. The check now does the same
  for those builds, and b12138 receipts verify bit for bit ([문서/라마재현.md](../라마재현.md), stage 21).
