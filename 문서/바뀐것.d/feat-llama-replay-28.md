갈래: 고침

- **llama.cpp 재현 검증 — 겹친 요청을 까닭과 함께**: 같은 서버에 요청이 겹쳐 한 묶음으로 셈한 영수증(연속 배치)을, 늦게 끝난 요청까지
  "다름" 대신 까닭과 함께 셈하지 않습니다. 기록기가 요청을 넘기기 전의 슬롯 상태(slots_before)도 적어, 기록기를 거치지 않은 요청과 겹친
  것도 알아봅니다([라마재현.md](../라마재현.md) 28단계).

---

- **llama.cpp replay check — overlapping requests are refused with a reason.** Receipts of requests that the server computed
  together with another request (continuous batching) are now refused with a reason, including the request that finished
  later, which was reported as "different" before. The recorder also stores the slot state before forwarding a request
  (`slots_before`), so overlaps with requests that bypassed the recorder are caught as well
  ([문서/라마재현.md](../라마재현.md), stage 28).
