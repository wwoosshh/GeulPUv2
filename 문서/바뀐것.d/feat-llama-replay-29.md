갈래: 더함

- **llama.cpp 재현 검증 — 겹친 요청(연속 배치)을 서버 기록으로**: 같은 서버에 요청이 겹쳐 여러 대화를 한 ubatch 로 셈한 영수증을,
  그 서버를 환경 변수 `LLAMA_BATCH_DEBUG=1` 과 `-v --log-file <파일>` 로 띄웠으면 그 기록에 적힌 ubatch 묶음대로 다시 셈해 비트까지
  견줍니다(지금까지는 까닭과 함께 거절). 서버 기록이 없으면 지금처럼 거절하고 까닭에 쓰는 법을 적습니다([라마재현.md](../라마재현.md)
  29단계).

---

- **llama.cpp replay check — overlapping requests (continuous batching) from the server log.** Receipts of requests that the
  server computed together with other requests in one ubatch are now replayed bit for bit by the ubatch batches written in the
  server log, when that server was started with the environment variable `LLAMA_BATCH_DEBUG=1` and `-v --log-file <file>`
  (they were refused with a reason before). Without a server log they are still refused, and the reason says how to record one
  ([문서/라마재현.md](../라마재현.md), stage 29).
