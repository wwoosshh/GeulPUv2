갈래: 더함

- **같은 요청을 혼자, 그리고 동시 요청 안에서 — `도구/same_bits.py`**: OpenAI 호환 서버 어디에든(GeulPU · llama.cpp · vLLM · SGLang ·
  Ollama) 같은 요청을 temperature 0 · 고정 시드로 먼저 혼자, 그다음 다른 프롬프트 일곱 개와 동시에, 다섯 번 보내고 토큰과 토큰마다의
  로그 확률이 첫 답과 같은지 본다. 표준 라이브러리만. 이 PC(RTX 4070 Ti, Qwen2.5-7B Q4_K_M)에서 llama.cpp b11721 은 혼자서는 5/5 같았지만
  동시 8 개 안에서는 5/5 로그 확률이 바뀌고 한 번은 66 번째 토큰에서 답이 갈렸고, GeulPU 0.15.0 은 열 번 모두 같은 비트였다
  ([문서/같은비트.md](../같은비트.md)).

---

- **The same request alone and inside a batch — `도구/same_bits.py`.** Sends the same request (temperature 0, fixed seed) to any
  OpenAI-compatible server — GeulPU, llama.cpp, vLLM, SGLang, Ollama — first alone, then concurrently with seven other prompts, five
  rounds, and checks whether the tokens and per-token log-probabilities match the first answer. Standard library only. On this PC
  (RTX 4070 Ti, Qwen2.5-7B Q4_K_M) llama.cpp b11721 matched itself 5/5 alone but changed its log-probabilities in 5/5 batched rounds
  and forked the answer once at token 66; GeulPU 0.15.0 gave the same bits in all ten answers
  ([docs/same-bits.md](https://github.com/wwoosshh/GeulPUv2/blob/main/docs/same-bits.md)).
