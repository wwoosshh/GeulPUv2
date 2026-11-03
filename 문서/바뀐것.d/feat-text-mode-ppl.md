갈래: 더함

- **모델 평가 — 절대 PPL · bits per byte · 텍스트의 비교용 모델**: 결과에 같은 토큰의 평균 NLL 과 PPL 을 기준 모델 · 평가할 모델 둘 다
  싣고(전에는 PPL 비뿐), 텍스트 데이터의 결과에는 글의 바이트당 비트(BPB — 어휘가 다른 모델끼리도 견줄 수 있다)를 더한다. 비교용
  모델(미세조정 전)을 텍스트 데이터에서도 쓸 수 있어, 이어서 사전학습한 모델의 PPL 이 얼마나 바뀌었는지 같은 토큰으로 바로 본다(#217).

---

- **Model evaluation — absolute PPL, bits per byte, and a baseline model for text.** Results now carry the mean NLL and PPL on
  the same tokens for both the reference model and the evaluated model (before, only the PPL ratio), and text-data results add
  bits per byte of the text (BPB — comparable across models with different vocabularies). The baseline model (before
  fine-tuning) also works on text data, so the PPL change from continued pretraining shows directly on the same tokens (#217).
