갈래: 바꿈

- **Gemma 4 의 생성이 긴 문맥에서 2.1 ~ 2.4 배 빨라졌다**: 행 넷까지의 어텐션을 키 타일마다 나란히 셈하고 차례로 접는 커널로 바꿨다(같은
  계약 — 출력 비트는 그대로). 12B(RTX 4070 Ti, 정밀 모드)에서 문맥 약 1,800 · 3,600 토큰의 생성이 54 · 48 ms/토큰 → 22 ms/토큰, 짧은 채팅은
  22.7 → 21.4 ms/토큰.

---

- **Gemma 4 generation is 2.1–2.4× faster at long context.** Attention for up to four rows now computes key tiles in parallel and
  merges them in order (same contract — output bits unchanged). On 12B (RTX 4070 Ti, exact mode), generation at about 1,800 and 3,600
  tokens of context went from 54 and 48 ms per token to 22 ms per token; short chats from 22.7 to 21.4 ms per token.
