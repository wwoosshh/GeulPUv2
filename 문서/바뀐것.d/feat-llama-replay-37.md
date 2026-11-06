갈래: 더함

- **llama.cpp 재현 검증 — 갈라진 대화 · 요청과 응답 기록 가져오기**: 대화 기록 가져오기가 꼴 둘을 더 받습니다 — 보낸 messages 와 받은
  답을 차례대로 적은 `requests`, 보낸 몸과 받은 응답을 그대로 적은 `exchanges`(JSON 배열 또는 JSONL). 메시지를 고쳐 갈라진 대화 · 여러
  대화가 섞인 기록을 서버가 받은 차례대로 흉내 내어 검증하고, 응답에 usage · timings 가 있으면 생성 토큰 수 · 캐시까지 서버와
  견줍니다([라마재현.md](../라마재현.md) 37단계).

---

- **llama.cpp replay check — branching conversations, request/response logs.** Conversation-log import accepts two more forms:
  `requests` (the messages sent and the answer received, in order) and `exchanges` (the request body and the response as sent and
  received, a JSON array or JSONL). Branched conversations (edited messages) and interleaved conversations are emulated in the order
  the server received them; when the responses carry usage or timings, the generated-token count and the cache state are compared
  with the server's too ([문서/라마재현.md](../라마재현.md), stage 37).
