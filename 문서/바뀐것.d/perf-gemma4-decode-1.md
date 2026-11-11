갈래: 바꿈

- **Gemma 4 26B-A4B 의 생성이 빨라졌다**(0.24.0): 생성 스텝에서 전문가의 GeGLU 를 실제로 쓰는 행만 계산하고, Gemma 4 · Gemma 3 · Qwen3.5 ·
  Qwen3.5 · 3.6 MoE · GLM-4.7-Flash · Kanana-2 의 생성 스텝을 CUDA 그래프 하나로 실행한다(전문가 캐시를 쓰지 않을 때). H100 NVL 의 MIG
  인스턴스에서 26B-A4B 의 생성이 초당 58 → 70 토큰이 되었고, 다른 모델은 스텝마다 약 1 ms 빨라졌다. 출력은 비트까지 그대로다.
  기록: [구조/gemma4/기록.md](../../구조/gemma4/기록.md).

---

- **Gemma 4 26B-A4B decodes faster.** Decode steps compute the expert GeGLU only for the rows in use, and Gemma 4, Gemma 3, Qwen3.5,
  Qwen3.5 · 3.6 MoE, GLM-4.7-Flash and Kanana-2 run each decode step as one CUDA graph when no expert cache is in use. On an H100 NVL
  MIG instance the 26B-A4B decodes 70 tokens per second instead of 58, and the other models save about 1 ms per step. Outputs are
  bit-identical. Record: [구조/gemma4/기록.md](../../구조/gemma4/기록.md).
