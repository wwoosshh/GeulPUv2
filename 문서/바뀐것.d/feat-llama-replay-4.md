갈래: 더함

- **llama.cpp 재현 검증**: 고치지 않은 llama-server 로 돌린 요청을 앱의 기록기(`/llama/completion`)로 보내면 영수증이 남고, 벤치마크
  화면의 'llama.cpp 재현 검증'이 그 영수증만으로 이 PC 의 GPU(CUDA · Vulkan)에서 다시 계산해 서버가 낸 토큰과 로그 확률이 비트까지 같은지
  판정합니다. 다시 계산한 실행은 디버거에서 GeulPU 서버 없이 스텝마다(상위 후보 · 로그 확률 · 엔트로피) 봅니다. 받는 실행은 llama.cpp b12008
  · RTX 4070 Ti · Qwen3.5 · 한 슬롯 · 프롬프트 캐시 끔 · 온도 0([라마재현.md](../라마재현.md)).

---

- **llama.cpp replay check.** Send requests for an unmodified llama-server through the app's recorder (`/llama/completion`)
  and it keeps a receipt; 'llama.cpp replay check' on the Benchmark page recomputes the request from the receipt alone on
  this PC's GPU (CUDA or Vulkan) and checks that the tokens and log probabilities the server returned are bit-identical.
  The recomputed run opens step by step in the debugger (top candidates, log probabilities, entropy) without a GeulPU
  server. Supported runs: llama.cpp b12008, RTX 4070 Ti, Qwen3.5, one slot, prompt cache off, temperature 0
  ([문서/라마재현.md](../라마재현.md)).
