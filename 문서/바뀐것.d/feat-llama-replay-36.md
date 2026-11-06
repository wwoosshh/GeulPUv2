갈래: 더함

- **llama.cpp 재현 검증 — 반복 벌점 · logit_bias**: 반복 벌점(repeat · frequency · presence penalty, repeat_last_n)과 토큰 번호로 준
  logit_bias · ignore_eos 를 준 요청도 검증합니다. 고르기 사슬의 흉내가 서버처럼 프롬프트 토큰과 고른 토큰을 받아들여 벌점을 셈합니다.
  DRY · mirostat 같은 다른 고르기와 글로 준 logit_bias 는 까닭과 함께 거절합니다([라마재현.md](../라마재현.md) 36단계).

---

- **llama.cpp replay check — repetition penalties and logit_bias.** Requests with repetition penalties (repeat, frequency and
  presence penalty, repeat_last_n), logit_bias given by token id, and ignore_eos are now verified: the emulated sampler chain accepts
  the prompt tokens and the chosen tokens like the server and applies the penalties. DRY, mirostat and the other samplers, and
  logit_bias given as text, are refused with the reason ([문서/라마재현.md](../라마재현.md), stage 36).
