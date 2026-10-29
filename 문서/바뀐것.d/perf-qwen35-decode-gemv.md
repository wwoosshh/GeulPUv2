갈래: 바꿈

- **Qwen3.5 의 토큰 생성이 빨라짐**: 생성 걸음의 k-양자화 행렬곱을 1행 GEMV 판으로 띄운다(출력 비트는 그대로). RTX 4070 Ti 에서 Qwen3.5-9B
  Q4_K_M 의 생성 걸음이 2.5 ~ 2.8 % 짧다.

---

- **Faster Qwen3.5 decoding.** The decode step's k-quant matmuls now run on the one-row GEMV kernels (output bits unchanged). On an
  RTX 4070 Ti, the Qwen3.5-9B Q4_K_M decode step is 2.5–2.8% shorter.
