갈래: 바꿈

- **MoE 모델의 생성이 빨라졌다 — 생성 스텝의 전문가 행렬곱**(0.24.0): 생성 스텝(행 하나)의 전문가 행렬곱을 한 행 커널로 계산한다(Gemma 4
  26B-A4B 는 gate · up · down, Qwen3.5 · 3.6 MoE · GLM-4.7-Flash · Kanana-2 는 down). 같은 계약이라 출력은 비트까지 그대로다. H100 NVL 의 MIG
  인스턴스에서 26B-A4B 의 생성이 초당 70 → 100 토큰이 되었고, Qwen3.6-35B-A3B · GLM-4.7-Flash 는 스텝마다 0.8 · 2.9 ms 빨라졌다.
  기록: [구조/gemma4/기록.md](../../구조/gemma4/기록.md).

---

- **MoE models decode faster — expert matrix products in decode steps.** A decode step (one row) computes the expert matrix
  products with one-row kernels (gate, up and down for Gemma 4 26B-A4B; down for Qwen3.5 · 3.6 MoE, GLM-4.7-Flash and Kanana-2).
  The contract is the same, so outputs are bit-identical. On an H100 NVL MIG instance the 26B-A4B decodes 100 tokens per second
  instead of 70, and Qwen3.6-35B-A3B and GLM-4.7-Flash save 0.8 and 2.9 ms per step. Record: [구조/gemma4/기록.md](../../구조/gemma4/기록.md).
