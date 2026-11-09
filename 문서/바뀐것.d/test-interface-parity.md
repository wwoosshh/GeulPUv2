갈래: 고침

- **실험의 명령 기록에 모델 · 계산 방식 바꾸기도**: 실행 화면 · 스텝 · 층 화면에서 '이 실행의 모델 불러오기' · '다른 모델로 점수' · '이 실행의
  계산 방식으로'로 바꾼 모델과 계산 방식이 그 실험의 명령 기록에 남습니다 — 명령 기록만으로 다시 실행해도 다른 모델의 점수가 같게 나옵니다.
  명령 창의 명령 복사는 줄바꿈이 든 글(계획 · 결과)을 `--json` 으로 적어 명령 창과 셸의 한 줄에 그대로 들어가고, 명령 창에서 친
  `experiment create` 는 보던 실험의 기록에 남지 않습니다([사용성시험.md](../사용성시험.md) §1).

---

- **Model and computation-mode changes in an experiment's command record.** Loading this run's model, scoring with another model and
  using this run's computation mode on the run and step screens now go into that experiment's command record, so replaying the record
  gives the same scores with the other model. The console's copied command writes text with line breaks (plans, results) as `--json`, so
  it fits on one line in the console and in a shell, and `experiment create` typed in the console no longer lands in the record of the
  experiment on screen ([사용성시험.md](../사용성시험.md) §1).
