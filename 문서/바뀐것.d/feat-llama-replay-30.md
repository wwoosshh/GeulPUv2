갈래: 더함

- **llama.cpp 재현 검증 — 토큰 없는 기록(요청 · 응답의 글만으로)**: 서버가 기본으로 돌려주는 응답만 남은 기록(토큰 없는 영수증 —
  `geulpu-llama-receipt/2`)을 검증합니다. 프롬프트 토큰은 검증할 때 같은 빌드의 llama-server 를 CPU 로 잠깐 띄워 llama.cpp 자신의 셈으로
  다시 만들고, 생성 토큰은 재현이 다시 만들어 그 글이 응답과 바이트까지 같은지 봅니다(온도 0). 원래 실행에는 손대지 않습니다. 같은 서버의
  기록은 앞에서부터 차례로 검증합니다([라마재현.md](../라마재현.md) 30단계). 기록기가 요청에 `n_probs` 를 더하면 온도 0 에서도 서버의
  고르기 사슬이 바뀐다는 점을 문서에 바로잡았습니다.

---

- **llama.cpp replay check — token-less records (request and response text only).** Records that keep only what the server
  returns by default (token-less receipts, `geulpu-llama-receipt/2`) can now be verified. The prompt tokens are rebuilt at
  verification time by llama.cpp itself (a temporary CPU-only llama-server of the same build), and the replay regenerates the
  generated tokens and checks that their text is byte-identical to the response (temperature 0). The original run is not
  touched. Records of one server are verified in order from the first ([문서/라마재현.md](../라마재현.md), stage 30). The docs
  now state correctly that the recorder's added `n_probs` changes the server's sampler chain even at temperature 0.
