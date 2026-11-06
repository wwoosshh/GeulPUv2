갈래: 더함

- **llama.cpp 재현 검증 — 스트리밍 응답의 토큰 없는 기록**: 투명한 기록기(기본)가 적은 스트리밍 응답(채팅 화면 · 에이전트의 흔한
  꼴)을 검증합니다. 기록기는 서버가 보낸 조각에서 모은 것(글 · 생각과 도구 호출 조각의 유무 · finish_reason · usage · 클라이언트가 요청한
  로그 확률)만 더 적고, 검증은 비스트리밍과 같은 꼴로 맞춰 다시 만든 토큰의 글을 바이트까지 견줍니다. 토큰 없는 영수증은 토큰을 알아도
  응답의 글을 토큰의 글과 견줍니다(온도 0 만, [라마재현.md](../라마재현.md) 33단계).

---

- **llama.cpp replay check — streamed token-less records.** Streamed responses written by the transparent recorder (the default),
  the usual form for chat UIs and agents, are now verified. The recorder additionally stores only what it gathers from the chunks the
  server sent (the text, whether reasoning or tool-call chunks appeared, finish_reason, usage, log-probabilities the client asked
  for); verification brings the record to the non-streamed form and compares the regenerated tokens' text byte for byte. Token-less
  receipts now compare the response text with the tokens' text even when the tokens are known (temperature 0 only,
  [문서/라마재현.md](../라마재현.md), stage 33).
