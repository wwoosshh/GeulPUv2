갈래: 더함

- **llama.cpp 재현 검증 — 스트리밍 응답(stream)**: 기록기가 스트리밍 응답(/completion · 채팅 API)의 조각을 모아 영수증에 적고, 재현
  검증이 그 영수증을 받습니다. 서버가 조각으로 보내지 않는 토큰(글자가 덜 끝난 걸음, 채팅에서 글 조각이 없는 걸음)이 있으면 온도 0 에서
  재현이 같은 셈으로 그 토큰을 고른 뒤 서버의 로그 확률 칸과 견줍니다([라마재현.md](../라마재현.md) 18단계).

---

- **llama.cpp replay check — streamed responses.** The recorder now assembles streamed responses (/completion and the chat
  API) into the receipt, and the replay check accepts them. Tokens the server never sends as a chunk (steps that end inside
  a character, and chat steps that produce no text) are chosen by the replay itself at temperature 0, then the server's
  log-probability entries are compared ([문서/라마재현.md](../라마재현.md), stage 18).
