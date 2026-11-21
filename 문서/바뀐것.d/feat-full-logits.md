갈래: 더함

- **서버가 로짓 행 전체를 돌려준다(`geulpu.logits`)**: 요청에 `geulpu.logits: true` 를 주면 응답의 `geulpu.logits` 에 스텝마다의 로짓 행이,
  echo 로 점수를 매기면 `geulpu.prompt_logits` 에 그 위치들의 로짓 행이 온다 — 로짓 해시를 계산한 바로 그 값(fp32 · 리틀 엔디언)을
  행마다 base64 로. 한 요청에 64 행까지이고, 이 필드를 주어도 토큰 · 로그확률 · 로짓 해시는 그대로다. 토큰마다의 분포 전체가 필요한 외부
  도구(분포 비교 · 시각화)가 상위 128 개 대신 쓸 수 있다. 형식은 [계약 5.2](../../계약/계약.md).

---

- **The server returns full logit rows (`geulpu.logits`).** A request with `geulpu.logits: true` gets the logit row of every step in
  `geulpu.logits`, and an echo that scores tokens gets the rows of those positions in `geulpu.prompt_logits` — the very values the logit
  hashes cover (float32, little-endian), base64 per row. At most 64 rows per request; tokens, log-probabilities and logit hashes are
  unchanged by the field. External tools that need the whole distribution per token (divergence, visualization) can use it instead of the
  top 128. Format: [contract 5.2](../../계약/contract.md).
