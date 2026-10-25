갈래: 바꿈

- **Gemma 4 의 생성이 7 ~ 8 % 더 빨라졌다**: 작은 판(뒤 정규화 · 잔차 · 다음 RMSNorm, GeGLU, 어텐션의 접기)을 다음 선형층의 입력까지 한 판으로
  합치고, 행 하나의 행렬곱은 생성 전용 판으로(같은 계약 — 출력 비트는 그대로). 12B(RTX 4070 Ti, 정밀 모드)에서 짧은 채팅 21.0 → 19.4 ms/토큰,
  문맥 약 1,800 토큰에서 22.1 → 20.4 ms/토큰.

---

- **Gemma 4 generation is 7–8 % faster.** Small kernels (post-norm with residual and the next RMSNorm, GeGLU, the attention merge) now
  fuse into one kernel that writes the next linear layer's input, and single-row matmuls use the decode-specific kernel (same contract —
  output bits unchanged). On 12B (RTX 4070 Ti, exact mode), short chats went from 21.0 to 19.4 ms per token and about 1,800 tokens of
  context from 22.1 to 20.4 ms per token.
