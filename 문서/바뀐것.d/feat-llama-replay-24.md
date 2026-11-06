갈래: 고침

- **llama.cpp 재현 검증 — 기본 설정 서버의 사슬을 서버 단위로**: 여러 대화가 슬롯을 나눠 쓴 기본 설정 서버에서, 같은 서버의 모든 슬롯
  영수증으로 서버의 슬롯 고르기와 RAM 프롬프트 캐시(저장 · 불러오기)를 흉내 내 사슬을 짓습니다. 다른 대화 뒤 RAM 캐시에서 불러와 이은
  요청은 아직 재현하지 못하며(KV 의 칸 배치가 처음 셈과 다르다), 이제 "다름" 대신 그 까닭으로 알립니다([라마재현.md](../라마재현.md)
  24단계).

---

- **llama.cpp replay check — server-level chains for default servers.** On default servers whose slots are shared by several
  conversations, the chain is now built from all slots' receipts of the same server, emulating the server's slot selection and
  RAM prompt cache (save and load). Requests continued from a RAM-cache load after another conversation still cannot be
  replayed (their KV cell layout differs from the original computation) and are now refused with that reason instead of being
  reported as different ([문서/라마재현.md](../라마재현.md), stage 24).
