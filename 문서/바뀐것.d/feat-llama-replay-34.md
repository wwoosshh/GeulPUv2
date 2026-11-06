갈래: 더함

- **llama.cpp 재현 검증 — 온도 > 0 의 토큰 없는 기록**: 요청에 시드가 있으면 온도 > 0(채팅 화면 · 에이전트의 흔한 설정)의 토큰 없는
  기록 — 투명한 기록기(스트리밍 포함) · 대화 기록 — 도 검증합니다. 재현이 로짓을 셈하는 차례대로 서버의 고르기 사슬(top_k · top_p ·
  min_p · 온도 · 시드)을 돌려 서버가 고른 토큰을 다시 고르고, 그 글을 응답과 바이트까지 견줍니다. 시드가 없는 요청은 까닭과 함께
  거절합니다([라마재현.md](../라마재현.md) 34단계).

---

- **llama.cpp replay check — sampled (temperature > 0) token-less records.** When the request carried a seed, token-less records at
  temperature > 0 (the usual setting of chat UIs and agents) — from the transparent recorder, streamed or not, and from client
  conversation logs — are now verified: the replay runs the server's sampler chain (top_k, top_p, min_p, temperature, seed) in the
  order it computes the logits, picks the server's tokens again and compares their text with the response byte for byte. Requests
  without a seed are refused with the reason ([문서/라마재현.md](../라마재현.md), stage 34).
