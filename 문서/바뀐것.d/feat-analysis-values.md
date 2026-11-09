갈래: 더함

- **실행 요약 · 실행 비교 기능과 CLI 의 `--select`**: 실행 화면의 요약 수치(평균 로그확률 · 퍼플렉서티 · 마진이 가장 작은 스텝 · 마진 < δ
  인 스텝 등)와 비교 탭의 처음 다른 위치 · 스텝, 로짓 렌즈의 요약을 앱이 계산하는 기능이 됐습니다 — `experiment summary` ·
  `experiment compare`(모델 없이 기록에서 읽음). 화면도 이 두 기능의 값을 보이므로 CLI · AI 와 같은 값입니다. CLI 와 명령 창은
  `--select <경로>[,<경로>…]` 로 답에서 필요한 값만 냅니다. 셸의 `GEULPU_EXPERIMENT` 는 새 실험을 만드는 명령과 다른 실험을 가리키는 실험
  명령에는 붙지 않습니다([사용성시험.md](../사용성시험.md) §3).

---

- **Run summary and run comparison operations, and `--select` in the CLI.** The run screen's summary values (mean log-probability,
  perplexity, the step with the smallest margin, steps with margin < δ …), the compare tab's first differing position and step, and
  the logit lens summary are now operations the app computes — `experiment summary` and `experiment compare` (read from the records,
  no model needed). The screens show these operations' values, so the CLI and agents get the same numbers. The CLI and the command
  console take `--select <path>[,<path>…]` to print only the values needed from an answer. `GEULPU_EXPERIMENT` in a shell no longer
  applies to the command that creates an experiment or to an experiment command that names another experiment
  ([사용성시험.md](../사용성시험.md) §3).
