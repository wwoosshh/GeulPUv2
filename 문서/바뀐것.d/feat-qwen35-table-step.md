갈래: 바꿈

- **Qwen3.5 계열이 동시 요청을 함께 셈함**: 여러 요청이 함께 생성 중이면 추론 엔진이 그 생성 걸음을 한 번에 셈한다 — 행렬곱은 요청들의 행을
  모아 가중치를 한 번 읽고, 어텐션 · GDN 상태도 요청들의 행을 판 하나로(행마다 그 슬롯의 상태로) 셈한다. 전에는 요청마다 차례로 셈해 동시
  요청이 늘어도 모두의 처리량이 그대로였다. 출력 비트는 그대로다(혼자 보낸 요청과 같다). RTX 4070 Ti 의 Qwen3.5-4B Q4_K_M 에서 동시 요청
  8 개의 모두의 생성 속도가 117 → 615 토큰/초다(2 개 117 → 220, 4 개 117 → 377).

---

- **The Qwen3.5 family computes concurrent requests together.** When several requests are generating, the inference engine runs
  their decode step at once: matrix multiplications gather the rows of all requests and read the weights once, and attention and
  the GDN state also run as one launch over all requests, each row on its own slot's state. Before, requests were computed one
  after another, so total throughput did not grow with concurrency. Output bits are unchanged (the same as a request sent alone).
  On an RTX 4070 Ti, eight concurrent Qwen3.5-4B Q4_K_M requests generate 615 tokens/s in total instead of 117 (two: 220,
  four: 377).
