갈래: 고침

- **최대 위치 512 인 모델도 텍스트 모드로 잼**: 모델 평가의 텍스트 모드가 최대 위치(문맥)가 512 인 모델(TinyStories 류의 처음부터 학습한
  작은 모델)에서 "기준 모델의 로그확률을 받지 못했습니다"로 실패하던 것을 고쳤다(#216). 추론 서버가 점수만 매기는 요청(echo · max_tokens
  0)은 문맥을 꽉 채운 프롬프트도 받는다 — llama-perplexity -c 512 와 같다. 최대 위치가 512 보다 작은 모델은 그 까닭을 알리고, 로그확률을
  받지 못하면 서버가 거절한 까닭을 함께 보인다.

---

- **Text mode measures models whose maximum position is 512.** Model evaluation's text mode failed on models whose context
  is 512 positions (small from-scratch models like TinyStories ones) with "Could not get the reference model's
  log-probabilities" (#216). The inference server now takes a prompt that fills the whole context when the request only
  scores it (echo with max_tokens 0) — the same as llama-perplexity -c 512. Models with fewer than 512 positions get the
  reason, and a failed scoring step shows why the server refused.
