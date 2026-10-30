갈래: 바꿈

- **정밀 모드의 Q8_0 프롬프트 처리가 조금 빨라짐(CUDA)**: 정밀 모듈의 Q8_0 타일 행렬곱이 척도를 (열, 블록)마다 워프에서 한 번만 실수로
  바꾼다(출력 비트는 그대로). RTX 4070 Ti 에서 Qwen3-8B · Qwen2.5-7B Q8_0 정밀 모드의 프롬프트 행렬곱이 1.2 ~ 2.1 % 짧다.

---

- **Slightly faster exact-mode Q8_0 prompt processing (CUDA).** The exact-mode Q8_0 tile matmuls now convert each column's scale to
  fp32 once per block per warp instead of once per tensor-core instruction (output bits unchanged). On an RTX 4070 Ti the exact-mode
  prompt matmuls of Qwen3-8B and Qwen2.5-7B Q8_0 are 1.2–2.1% shorter.
