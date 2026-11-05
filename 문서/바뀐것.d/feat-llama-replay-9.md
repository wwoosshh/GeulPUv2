갈래: 더함

- **llama.cpp 재현 검증 — 채팅 API**: 기록기가 llama-server 의 채팅 API(/v1/chat/completions)도 넘기며 영수증을 남기고, 검증이 그
  영수증을 받습니다. 서버가 메시지 자리(사용자 메시지의 시작)에서 프롬프트를 끊고 체크포인트를 만드는 셈과, 슬롯 캐시를 RAM 프롬프트
  캐시에 저장했다가 불러오는 갈래까지 흉내 냅니다. 스트리밍 응답은 아직 받지 않습니다([라마재현.md](../라마재현.md) 9단계).

---

- **llama.cpp replay check — chat API.** The recorder now also forwards llama-server's chat API (`/v1/chat/completions`) and
  writes receipts for it, and the check accepts them. It emulates how the server splits the prompt and creates checkpoints
  at message positions (the start of each user message), and the case where the slot is saved to and looked up in the RAM
  prompt cache. Streamed responses are not accepted yet ([문서/라마재현.md](../라마재현.md), stage 9).
