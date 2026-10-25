갈래: 더함

- **연구 패널의 층 렌즈(logit lens)**: 걸음 상세의 '층 렌즈 — 층마다 읽기' 가 그 토큰을 고른 자리의 계산을 층마다 끝의 출력으로 읽어(계약
  5.3) 층마다 1위 토큰과 확률, 고른 토큰의 확률과 순위, 엔트로피를 보인다(층을 누르면 그 층의 상위 k). 답이 몇 번째 층에서 생기는지,
  다른 언어로 넘어가기 같은 실패가 어느 층에서 시작되는지 본다. API 는 `/v1/completions` 의 echo 에 `geulpu.layer_lens`. 재현 파일에 셈한
  렌즈가 들어가고 `도구/repro_check.py` 가 GPU 없이 비트까지 견준다. qwen2 · qwen3 · llama · gemma4 구조(한 GPU — Gemma 4 는 끝 로짓의
  소프트캡까지 읽는다 · 재현 파일의 CPU 확인은 qwen2 · qwen3 · llama).

---

- **Layer lens (logit lens) in the research panel.** In a step's details, 'Layer lens — Read every layer' reads the computation at the
  position that chose the token at every layer with the final output (contract 5.3) and shows, per layer, the top-1 token and its
  probability, the chosen token's probability and rank, and the entropy (click a layer for its top k). It shows at which layer an
  answer forms and where a failure such as switching to another language starts. The API is `geulpu.layer_lens` on an echo request to
  `/v1/completions`. Reproduction files carry the computed lens, and `도구/repro_check.py` checks it bit for bit without a GPU.
  qwen2, qwen3, llama and gemma4 architectures (one GPU — for Gemma 4 the reading ends with the final logits' softcap; the CPU check of
  reproduction files covers qwen2, qwen3 and llama).
