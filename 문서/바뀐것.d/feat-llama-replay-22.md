갈래: 더함

- **llama.cpp 재현 검증 — 온도 > 0 의 채팅 스트리밍**: 채팅 화면의 흔한 설정(스트리밍 · 온도 0.7 … 1)의 영수증을 받습니다. 서버가
  보내지 않는 끝 토큰(EOS)은 먹이지 않는 마지막 토큰이라, 그 앞은 알려진 토큰으로 셈하고 마지막 걸음은 같은 시드의 고르기 사슬이 고른
  토큰(제어 토큰이어야 함)으로 봅니다([라마재현.md](../라마재현.md) 22단계).

---

- **llama.cpp replay check — chat streams above temperature 0.** Receipts of the common chat-UI setting (streaming, temperature
  0.7–1) are now accepted. The end token, which a chat stream never sends, is the last token and is not fed back, so every
  step before it is replayed with known tokens and the last step takes the token the sampler chain picks with the same seed (it
  must be a control token) ([문서/라마재현.md](../라마재현.md), stage 22).
