갈래: 더함

- **q · k · v 의 형식이 서로 다른 파일**: 어텐션의 q · k · v 행렬이 서로 다른 k-양자화 길의 형식(Q4_K · Q5_K · Q6_K · IQ4_XS · F16 · BF16 ·
  F32)이어도 받는다 — 모든 구조. 같은 형식이 이어지는 묶음을 구간 하나로 셈하므로 출력 비트는 그대로다. bartowski 의 Qwen3.5-9B Q4_K_M
  같은 파일(어텐션 층의 q 와 k 가 다른 형식)이 실행된다. 판 기억의 꼴이 바뀌어 모델을 처음 올릴 때 커널 변형을 한 번 다시 잰다.

---

- **Files whose q · k · v formats differ.** The attention q, k and v matrices may each use a different k-quant-path format (Q4_K ·
  Q5_K · Q6_K · IQ4_XS · F16 · BF16 · F32), for every architecture. Runs of equal formats form one segment, so output bits are
  unchanged. This runs files such as bartowski's Qwen3.5-9B Q4_K_M, whose attention layers quantize q and k differently. The kernel
  variant memory format changed, so the first load of a model re-measures its kernel variants once.
