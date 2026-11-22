갈래: 바꿈

- **Qwen3.5 의 생성이 빨라졌습니다 — 정밀 모드에서 특히.** 층마다의 작은 행렬 둘(`ssm_beta` · `ssm_alpha`)을 토큰 하나를 생성할 때
  한 줄 곱 커널로 계산하고, Qwen3.5 의 정밀 모드 모듈에도 다른 구조의 정밀 모듈이 쓰던 행렬곱 개선을 켰습니다. 출력 비트는 그대로입니다.
  Tesla V100 에서 Qwen3.5-9B Q4_K_M 의 생성이 정밀 40.8 → 76.6, 빠른 63.4 → 81.9 tok/s 가 됐습니다. Qwen3.5 의 MoE · IQ4_XS 파일과
  16비트 블록이 섞인 파일의 모듈은 그대로입니다 — 자세한 것은 [구조/qwen35/기록.md](../../구조/qwen35/기록.md) 의 '속도 S6'.

---

- **Faster Qwen3.5 generation — most of all in exact mode.** The two small per-layer matrices (`ssm_beta` · `ssm_alpha`) now use a
  one-row kernel when one token is generated, and Qwen3.5's exact-mode modules now use the matrix-multiplication improvements the
  exact modules of other architectures already had. The output bits are unchanged. On a Tesla V100, Qwen3.5-9B Q4_K_M generates
  76.6 instead of 40.8 (exact) and 81.9 instead of 63.4 (fast) tok/s. Modules for Qwen3.5 MoE, IQ4_XS and mixed 16-bit-block files are unchanged —
  details in section 'S6' of [구조/qwen35/기록.md](../../구조/qwen35/기록.md).
