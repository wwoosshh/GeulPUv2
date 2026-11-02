갈래: 더함

- **원래 모델과 견주기**: 모델 평가의 결과 화면에서 기준 모델(GGUF)의 답을 원래 모델(Hugging Face 형식 — transformers, CPU)에 같은 토큰으로
  넣어 위치마다 견준다(KL divergence · top-1 일치율 · 답 토큰의 로그확률 차). 0 에 가까우면 변환과 엔진이 원래 모델을 그대로 옮긴 것이다.
  'GGUF 만들기'의 도구 준비로 만든 파이썬 환경을 쓴다.

---

- **Compare with the original model.** A Model evaluation result can feed the reference model's (GGUF) answers to the original model
  (Hugging Face format — transformers, CPU) token by token and compare every position (KL divergence · top-1 agreement · answer-token
  log-probability difference). Near 0 means the conversion and the engine reproduce the original model. It uses the Python
  environment made by Make GGUF's tool setup.
