갈래: 더함

- **llama.cpp 재현 검증 — 기록기 없이, 대화 기록에서**: 클라이언트가 llama-server 에 직접 보낸 대화(messages 와 답, 보낸 고르기 값)를
  JSON 으로 두면("대화 기록 가져오기" · `POST /api/llama/import`) 도우미 메시지마다 토큰 없는 영수증을 만들어 검증합니다. 추론할 때
  아무것도 하지 않은 실행의 사후 검증입니다 — 생성 토큰 수를 몰라도 답의 글에 맞는 토큰을 다시 만들어 끝 토큰 · max_tokens 로 끝났는지
  봅니다(온도 0 만, [라마재현.md](../라마재현.md) 32단계).

---

- **llama.cpp replay check — without the recorder, from conversation logs.** A JSON file with the conversation the client sent
  directly to llama-server (messages with answers, and the sampling values it sent) can be imported ("Import conversation log" ·
  `POST /api/llama/import`); each assistant message becomes a token-less receipt and is verified. This is post-hoc verification
  of a run that nothing touched — the generated token count is unknown, so the replay regenerates tokens matching the answer text
  and checks that it ended with an end token or at max_tokens (temperature 0 only, [문서/라마재현.md](../라마재현.md), stage 32).
