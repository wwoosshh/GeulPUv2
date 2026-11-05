갈래: 더함

- **llama.cpp 재현 검증 — 캐시의 끝이 ubatch 가운데인 대화 요청(dense 모델)**: 서버가 앞 요청의 ubatch 를 다 셈한 뒤 캐시 끝 뒤를
  지우고 이어 쓴 요청(같은 프롬프트를 다시 보냄 · 앞을 고침 · 다른 대화가 같은 슬롯을 씀)도 같은 차례로 다시 셈해 검증합니다. 지금까지는
  까닭과 함께 거절했습니다([라마재현.md](../라마재현.md) 16단계).

---

- **llama.cpp replay check — conversation requests whose cache ends inside a ubatch (dense models).** When the server had
  computed a whole ubatch of an earlier request and then removed the cache past the reused prefix (the same prompt sent again,
  an edited earlier turn, another conversation on the same slot), the check now recomputes in the same order and verifies the
  request. These requests were refused before ([문서/라마재현.md](../라마재현.md), stage 16).
