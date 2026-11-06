갈래: 더함

- **llama.cpp 재현 검증 — 쉬는 슬롯을 비우지 않는 서버**: `--no-cache-idle-slots` 이거나 RAM 프롬프트 캐시가 0 인 서버(슬롯 여럿 ·
  통합 KV)에서 다른 슬롯도 쓰인 영수증을 셈합니다(지금까지는 거절). 서버의 슬롯 고르기(LCP 비율 · 빈 슬롯 · 가장 오래 쓰지 않은 슬롯)와
  모든 슬롯의 KV 칸을 흉내 내, 다른 슬롯의 칸이 남은 채로 셈한 요청도 서버와 같은 칸 배치로 다시 셉니다([라마재현.md](../라마재현.md)
  26단계).

---

- **llama.cpp replay check — servers that keep idle slots.** Receipts from servers with several slots and a unified KV cache
  that run with `--no-cache-idle-slots` or without a RAM prompt cache are now replayed even when other slots were used (they
  were refused before). The check emulates the server's slot selection (LCP ratio, empty slot, least recently used slot) and
  the KV cells of every slot, so requests computed while other slots' cells stayed in the cache are recomputed with the
  server's cell layout ([문서/라마재현.md](../라마재현.md), stage 26).
