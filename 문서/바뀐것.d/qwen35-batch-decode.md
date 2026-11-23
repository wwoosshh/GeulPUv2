갈래: 바꿈

- **Qwen3.5 계열에서 여러 요청을 함께 생성할 때 빨라집니다.** 요청이 2 ~ 8 개인 생성 걸음의 행렬곱에, 요청 하나일 때처럼 가중치를 한 번
  읽고 요청마다 셈하는 판이 후보로 더해집니다 — 엔진이 카드마다 재서 고릅니다(V100 처럼 정수 텐서곱 명령이 없는 카드에서 크게 빨라집니다).
  출력 비트는 그대로입니다 — 자세한 것은 [구조/qwen35/기록.md](../../구조/qwen35/기록.md) 의 '속도 S10'.

---

- **Faster Qwen3.5-family generation with several concurrent requests.** Decode steps with 2–8 requests get a new matrix-multiply kernel
  candidate that reads the weights once and computes every request from them, like the one-request kernel; the engine measures and picks per
  card (large gains on cards without integer tensor-core instructions, such as the V100). The output bits are unchanged — details in section
  'S10' of [구조/qwen35/기록.md](../../구조/qwen35/기록.md).
