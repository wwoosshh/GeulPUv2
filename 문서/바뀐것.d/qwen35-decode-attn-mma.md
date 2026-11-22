갈래: 바꿈

- **긴 대화에서 Qwen3.5 계열의 생성이 빨라집니다(정수 텐서곱이 있는 카드 — 계산 능력 8.0 이상).** 토큰 하나를 생성할 때 어텐션은 키 · 값을
  공유하는 질의 헤드들을 정수 텐서곱으로 함께 셈해, 키 · 값을 한 번만 읽습니다. 출력 비트는 그대로입니다. V100 처럼 계산 능력 7.x 인 카드는
  지금의 판을 그대로 씁니다 — 자세한 것은 [구조/qwen35/기록.md](../../구조/qwen35/기록.md) 의 '속도 S8'.

---

- **Faster Qwen3.5-family generation in long conversations (cards with integer tensor-product instructions — compute capability 8.0 and up).**
  When one token is generated, attention now computes the query heads that share a key/value head together with integer tensor products, reading
  the keys and values once. The output bits are unchanged. Cards with compute capability 7.x, such as the V100, keep the current kernel — details
  in section 'S8' of [구조/qwen35/기록.md](../../구조/qwen35/기록.md).
