갈래: 더함

- **Inkling 2 의 Q4_K_M 파일(bartowski 의 Inkling-2-Mini-Q4_K_M)도 받습니다.** 전문가 행렬이 k-양자화(Q4_K · Q5_K · Q6_K)와 Q8_0 ·
  Q5_0 으로 섞인 파일을 Gemma 4 26B-A4B 와 같은 길로 계산하고(Q5_0 은 읽을 때 Q8_0 으로 — 값이 같다), 이 파일이 F32 로 둔 어텐션
  싱크 · 동적 탭의 투영은 16비트 블록 행렬곱으로 계산합니다([계약.md](../../계약/계약.md) 의 1.3). 어텐션 · 임베딩 · 출력 행렬은 Q8_0
  이어야 합니다.

---

- **Inkling 2 Q4_K_M files (bartowski's Inkling-2-Mini-Q4_K_M) are supported.** Expert matrices that mix k-quants (Q4_K · Q5_K · Q6_K)
  with Q8_0 and Q5_0 take the same path as Gemma 4 26B-A4B (Q5_0 becomes Q8_0 at load — the values are equal), and the attention-sink
  and dynamic-tap projections, which this file keeps in F32, run as 16-bit block matmuls ([contract.md](../../계약/contract.md),
  item 1.3). The attention, embedding and output matrices must be Q8_0.
