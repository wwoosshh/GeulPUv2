갈래: 더함

- **정확도 평가(채점)**: 모델 평가의 데이터 줄에 `answer` 와 `grader`(number · choice · exact · tool_call)가 있으면 기준 모델과 평가할 모델의
  답을 채점해 결과에 정답률(95 % 구간 — 부트스트랩)과 한쪽만 맞은 대화 수(부호 검정)를, 전후 비교에 틀림 → 맞음 · 맞음 → 틀림 대화를 보인다.
  대화마다 정답 · 두 모델의 판정 · 도구 호출을 볼 수 있다. 내장 채점 데이터 'tasks40'(GeulPU 가 쓴 문항 — 수학 · 객관식 · 한 단어 · 도구 호출).

---

- **Accuracy evaluation (grading).** When a data line in Model evaluation has `answer` and `grader` (number · choice · exact · tool_call),
  both the reference and the evaluated model's answers are graded: results show accuracy (95% interval — bootstrap) and the conversations
  where only one model is correct (sign test), and before/after comparisons show the conversations that went wrong → correct and
  correct → wrong. Each conversation shows the expected answer, both verdicts and the tool calls. Built-in graded data 'tasks40' (questions
  written for GeulPU — math · multiple choice · one word · tool calls).
