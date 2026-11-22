갈래: 바꿈

- **Qwen3.5 계열 파일 가운데 투영 행렬을 Q8_0 으로 담은 파일의 생성이 빨라졌습니다.** 토큰 하나를 생성할 때 그 Q8_0 행렬(어텐션 qkv ·
  GDN qkv · z · ssm_out)도 한 줄 곱 커널로 계산합니다. 출력 비트는 그대로입니다. 그런 파일(예: Qwen3.8-27B Q4_K_M)에서만 달라지고, 그 행렬이
  k-양자화인 파일은 그대로입니다 — 자세한 것은 [구조/qwen35/기록.md](../../구조/qwen35/기록.md) 의 '속도 S7'.

---

- **Faster generation for Qwen3.5-family files that store their projection matrices as Q8_0.** When one token is generated, those Q8_0
  matrices (attention qkv · GDN qkv · z · ssm_out) now also use a one-row kernel. The output bits are unchanged. Only such files change (for
  example Qwen3.8-27B Q4_K_M); files whose projections are k-quants are unchanged — details in section 'S7' of
  [구조/qwen35/기록.md](../../구조/qwen35/기록.md).
