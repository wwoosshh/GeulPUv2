갈래: 더함

- **llama.cpp 재현 검증 — RAM 캐시에서 다른 슬롯에 불러온 요청**: 기본 설정 서버에서 다른 대화 뒤 RAM 프롬프트 캐시에서 불러와 이은
  요청(24단계까지는 까닭과 함께 거절)을 셈합니다. 서버의 KV 칸 고르기(빈 칸 찾기 · 지우기 · 상태 되살리기)를 흉내 내, 재현이 서버와 같은
  칸 배치 · 어텐션 길이로 셉니다([라마재현.md](../라마재현.md) 25단계).

---

- **llama.cpp replay check — requests loaded from the RAM cache into another slot.** On default servers, requests continued
  from a RAM prompt-cache load after another conversation (refused with a reason up to stage 24) are now replayed. The check
  emulates the server's KV cell allocation (free-cell search, removal, state restore), so the replay uses the server's cell
  layout and attention length ([문서/라마재현.md](../라마재현.md), stage 25).
