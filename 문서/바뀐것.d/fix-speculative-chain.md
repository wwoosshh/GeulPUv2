갈래: 고침

- **`geulpu.speculative` 가 다시 받아진다**: 요청의 `"geulpu": {"speculative": true | false}`(추측 디코딩 켬 · 끔)를 Qwen3.5 가 아닌 모든 모델에서
  400(`geulpu.speculative 는 참거짓이어야 한다`)으로 거절하던 것을 고쳤다 — 0.12.0 의 Qwen3.5 줄이 '참거짓이면 … 아니면 틀린 형' 사이에 들어가
  올바른 값도 틀린 형으로 보았다. Qwen3.5 는 전처럼 추측 디코딩 없이 돌고, 이제 틀린 형(숫자 등)은 모든 모델에서 400 이다.

---

- **`geulpu.speculative` is accepted again.** Since 0.12.0 a request with `"geulpu": {"speculative": true | false}` (speculative
  decoding on or off) was rejected with HTTP 400 ("geulpu.speculative must be a boolean") on every model except Qwen3.5: a line added for
  Qwen3.5 had landed between the "is a boolean" check and its "wrong type" branch. Qwen3.5 still runs without speculative decoding, and a
  value of the wrong type is now rejected on every model.
