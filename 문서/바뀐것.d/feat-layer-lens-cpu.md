갈래: 더함

- **층 렌즈(logit lens)를 GPU 없이**: `도구/repro_check.py --layer-lens 5,12` 가 재현 파일의 고른 스텝에서 그 토큰을 매긴 행을 층마다 끝의
  로짓 계산으로 읽어(계약 5.3), 층마다 1위 토큰 · 고른 토큰의 확률과 순위 · 엔트로피 · 상위 k 를 보인다. 마지막 층은 끝 로짓과 비트까지
  같다.

---

- **The layer lens (logit lens) without a GPU.** `도구/repro_check.py --layer-lens 5,12` reads, for the chosen steps of a capsule, the
  row that scored the token at every layer with the final logit computation (contract 5.3), and shows the top-1 token, the chosen
  token's probability and rank, the entropy and the top k per layer. The last layer is bit-identical to the final logits.
