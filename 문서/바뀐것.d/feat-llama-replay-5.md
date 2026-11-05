갈래: 더함

- **llama.cpp 재현 검증 — 온도 > 0**: 시드를 준 요청이면 온도 > 0 의 영수증도 받습니다. llama-server 의 고르기 사슬(top_k · top_p ·
  min_p · 온도 → 분포에서 고르기, 같은 난수)을 그대로 셈해 걸음마다 서버가 고른 토큰과 견줍니다. 기본이 아닌 사슬(반복 벌점 ·
  mirostat · grammar 등)과 시드 없는 요청은 까닭과 함께 받지 않습니다([라마재현.md](../라마재현.md) 5단계).

---

- **llama.cpp replay check — temperature > 0.** Receipts of requests with temperature > 0 are accepted when the request
  gave a seed. The check computes llama-server's sampling chain (top_k, top_p, min_p, temperature, then sampling from the
  distribution with the same random numbers) and compares the token the server chose at every step. Non-default chains
  (repetition penalties, mirostat, grammar and so on) and requests without a seed are refused with the reason
  ([문서/라마재현.md](../라마재현.md), stage 5).
