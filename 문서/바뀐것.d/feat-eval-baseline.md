갈래: 더함

- **미세조정 전 모델 대비 — 배운 것 · 잊은 것**: 모델 평가의 작업 공간에 '비교용 모델(미세조정 전)'을 고르면, 채점 데이터의 결과마다 그 모델의
  정답률과, 기준 모델 · 평가할 모델이 그 모델 대비 배운 것(전에 틀리고 지금 맞음) · 잊은 것(전에 맞고 지금 틀림)의 수를 보인다. 비교용 모델의 답은
  기준값처럼 한 번 만들고, 같은 모델 · 데이터 · 설정이면 다시 쓴다 — 미세조정을 바꿀 때마다 원래 모델을 버전으로 넣고 다시 평가하거나 기준과
  평가할 모델을 뒤집어 읽지 않아도 된다(#199).

---

- **Against the model before fine-tuning — learned and forgotten.** Choose a baseline model (before fine-tuning) for a Model evaluation
  workspace, and every result on graded data shows its accuracy and how many conversations the reference and evaluated models learned
  (wrong before, right now) and forgot (right before, wrong now). The baseline's answers are made once, like reference values, and
  reused for the same model, data and settings — no more adding the original model as a version for each fine-tune and reading the
  reference and evaluated columns backwards (#199).
