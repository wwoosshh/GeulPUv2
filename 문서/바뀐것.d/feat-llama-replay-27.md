갈래: 더함

- **llama.cpp 재현 검증 — 슬롯마다 KV 를 둔 서버(-np N)**: `-np N` 으로 띄운 서버(슬롯마다 KV 흐름 — 통합 KV 꺼짐)의 영수증을
  셈합니다(지금까지는 거절). 흐름마다의 KV 칸과, 작업을 띄울 때 쉬는 슬롯을 RAM 프롬프트 캐시에 저장만 하는 서버의 동작을 흉내 냅니다
  ([라마재현.md](../라마재현.md) 27단계).

---

- **llama.cpp replay check — servers with a KV cache per slot (-np N).** Receipts from servers started with `-np N` (one KV
  stream per slot, unified KV off) are now replayed (they were refused before). The check emulates the KV cells of each
  stream and the server's habit of only saving idle slots to the RAM prompt cache when a task starts
  ([문서/라마재현.md](../라마재현.md), stage 27).
