갈래: 더함

- **서버의 로그확률 — 상위 128 개와 지정한 토큰**: 요청의 `geulpu.top_logprobs`(0 ~ 128)로 `geulpu` 의 상위 로그확률을 20 개보다 많이 받고,
  `/v1/completions` 의 echo 에 `geulpu.token_ids_logprob`(점수를 매기는 위치마다 토큰 번호 목록)를 주면 위치마다 그 토큰들의 로그확률을 받는다.
  같은 계약 5.2 의 값이라 상위 k · 점수의 값과 비트까지 같다. 모델 평가의 분포 비교(KL divergence)에 쓴다.

---

- **Server log-probabilities — top 128 and chosen tokens.** A request's `geulpu.top_logprobs` (0–128) returns more than 20 top
  log-probabilities in the `geulpu` object, and `geulpu.token_ids_logprob` (a list of token ids for each scored position) on a
  `/v1/completions` echo returns the log-probabilities of those tokens at each position. They are the same contract 5.2 values,
  bit-identical to the top-k and score values. The model evaluation's distribution comparison (KL divergence) uses them.
