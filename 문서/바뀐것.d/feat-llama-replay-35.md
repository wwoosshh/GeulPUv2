갈래: 더함

- **llama.cpp 재현 검증 — 멈춤 글(stop)로 끝난 응답**: 클라이언트가 멈춤 글을 준 요청의 응답도 검증합니다. 서버가 멈춤 글을 찾아 글을
  자르는 셈(가장 앞 자리, 멈춤 글의 앞부분에서 기다림, 로그 확률 칸 자르기)을 생성 토큰의 글에 흉내 내어, 멈춘 걸음과 자른 글이 응답과
  맞아야 "같음"입니다. 멈춤 글을 준 채팅 · 대화 기록도 받습니다([라마재현.md](../라마재현.md) 35단계).

---

- **llama.cpp replay check — responses ended by a stop string.** Responses to requests with stop strings are now verified: the
  server's stop-string handling (earliest match, holding back a partial match, trimming the log-probability entries) is emulated on
  the generated tokens' text, and the stop step and the cut text must match the response. Chats and client conversation logs with stop
  strings are accepted too ([문서/라마재현.md](../라마재현.md), stage 35).
