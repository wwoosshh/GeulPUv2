갈래: 더함

- **llama.cpp 재현 검증 — 기본 설정의 서버 · 프롬프트 캐시**: -np 를 주지 않은 기본 llama-server(슬롯 4 · 통합 KV · 프롬프트 캐시
  켬)의 영수증도 받습니다. 에이전트처럼 한 대화를 이어 보내 서버가 캐시에서 이어 셈한 요청은, 같은 슬롯의 앞 영수증들로 서버의 캐시 셈
  (겹치는 앞 · 체크포인트 되살리기 · 프롬프트 나눔)을 흉내 내 그 상태를 다시 쌓아 검증합니다. 기록기는 프롬프트 캐시를 더 이상 끄지 않고
  슬롯 상태를 함께 적습니다. 다른 슬롯도 쓰인 서버 · 앞 영수증이 모자란 요청은 까닭과 함께 받지 않습니다([라마재현.md](../라마재현.md)
  7단계).

---

- **llama.cpp replay check — default server settings and the prompt cache.** Receipts from a default llama-server (no
  `-np`: 4 slots, unified KV cache, prompt cache on) are now accepted. When a conversation continues and the server reuses
  its cache, the check rebuilds the slot's state from the earlier receipts of the same slot. It emulates the server's
  cache logic (common prefix, checkpoint restore, prompt batching). The recorder no longer turns the prompt cache off and
  also records the slot state. Servers where another slot was used, and requests whose earlier receipts are missing, are
  refused with the reason ([문서/라마재현.md](../라마재현.md), stage 7).
