갈래: 바꿈

- **Gemma 4 의 프롬프트 처리가 7 ~ 9 배 빨라졌다**: 프롬프트(행 다섯 이상)의 어텐션을 텐서 코어 행렬곱 커널로 바꿨다(같은 계약 — 로짓 · 생성
  토큰은 같은 비트). 12B(RTX 4070 Ti, 정밀 모드)에서 프롬프트 약 1,800 · 3,600 토큰의 첫 토큰이 5.45 · 15.4 초 → 0.77 · 1.63 초.

---

- **Gemma 4 prompt processing is 7–9× faster.** Attention for prompts (five rows or more) now runs on a tensor-core kernel (same
  contract — logits and generated tokens are bit-identical). On 12B (RTX 4070 Ti, exact mode), time to first token for about 1,800 and
  3,600 prompt tokens went from 5.45 and 15.4 s to 0.77 and 1.63 s.
