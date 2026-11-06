갈래: 고침

- **llama.cpp 재현 검증 — 셈을 바꾸는 설정은 닫힌 목록으로**: 검증이 서버 명령줄의 인자와 요청 칸을 b12008 의 인자 표에 대어 봅니다.
  재현이 흉내 내는 것과 셈을 바꾸지 않는 것만 받고, 셈을 바꾸는데 흉내 내지 않는 것(KV 캐시 형 · 추측 디코딩 · LoRA · 층이나 KV 를 CPU 에
  둔 서버 · RoPE 덮어쓰기 · 사슬 뒤의 확률 · 답을 여럿 · 그림 입력 등)과 표에 없는 인자는 까닭과 함께 거절합니다. 전에는 이런 설정을
  지나쳐, 토큰 없는 기록이면 다른 셈으로도 "같음"이 나올 수 있었습니다. 멈춤 글은 서버처럼 빈 글을 건너뛰고, 서버의 -r 을 여럿 ·
  이스케이프를 풀어 읽습니다([라마재현.md](../라마재현.md) 39단계).

---

- **llama.cpp replay check — settings that change the computation are checked against a closed list.** Verification now checks
  the server's command line and the request fields against llama-server's b12008 option table. Only settings the replay emulates or
  that do not change the computation are accepted; settings that change it without emulation (KV cache types, speculative decoding,
  LoRA, layers or KV on the CPU, RoPE overrides, post-sampling probabilities, several completions, image input and others) and options
  not in the table are refused with the reason. Before, such settings were passed over, so a token-less record could be called "same"
  under different arithmetic. Stop strings now follow the server: empty entries are skipped, and every -r is read and unescaped
  ([문서/라마재현.md](../라마재현.md), stage 39).
