갈래: 더함

- **llama.cpp 재현 검증 — 여러 대화가 슬롯을 나눠 쓴 기본 설정 서버**: 기본 설정의 llama-server 는 새 작업을 띄울 때 쉬는 슬롯을 RAM
  캐시에 저장하고 비우므로, 한 요청을 셈하는 동안 KV 에는 그 슬롯의 칸만 있습니다. 그래서 다른 슬롯이 쓰인 서버의 영수증도 받습니다 —
  캐시를 쓰지 않은 요청, 그리고 그 사이에 다른 슬롯의 요청이 없었던 이어지는 요청. RAM 캐시에서 불러온 상태로 이은 요청은 아직 까닭과 함께
  거절합니다([라마재현.md](../라마재현.md) 23단계).

---

- **llama.cpp replay check — default servers whose slots are shared by several conversations.** A default llama-server saves
  idle slots to its RAM cache and clears them whenever it starts a new task, so while one request is computed the KV cache holds
  only that slot's cells. Receipts from servers where other slots were used are now accepted: requests computed without the
  cache, and follow-ups with no other slot's request in between. Follow-ups continued from a state loaded from the RAM cache are
  still refused with a reason ([문서/라마재현.md](../라마재현.md), stage 23).
