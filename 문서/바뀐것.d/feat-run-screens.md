갈래: 더함

- **실행 화면**: 실험 화면에 실행 · 비교 탭이 생겼습니다. 실행 기능을 `--experiment` 로(화면에서는 실험 화면에서) 부르면 앱이 요청과 답을 실험의
  `records/runs/<id>.json` 에 남기고 `experiment.json` 의 `runs`(종류 · 원래 실행 · 개입) · `models` · `files` 를 맞춥니다. 실행 화면은 토큰별
  출력(토큰 아래 막대 — 마진 · 확률 · 엔트로피)과 고른 스텝의 상위 후보를 보이고, 그 화면에서 분기 · 재실행 · 새 실행을 합니다. 비교 탭은 두 실행이
  처음 다른 스텝과 로짓 해시를 보입니다. 계획 · 판정 기준(첫 실행 전에만) · 판정 · results.md · 상태는 화면과 CLI 의 `experiment update` 로 씁니다.
  명령 창에는 'AI 실행 자동 열기'가 생겼습니다([화면구현.md](../화면구현.md) §3).

---

- **Run screens.** The experiment screen gains Runs and Compare tabs. A run operation called with `--experiment` (or from the
  experiment screen) is recorded by the app: the request and the answer in the experiment's `records/runs/<id>.json`, and `runs` (kind,
  base run, intervention), `models` and `files` in `experiment.json`. The run screen shows the token-by-token output (bars under the
  tokens — margin, probability or entropy) and the top candidates of the selected step, and branches, re-runs or starts a new run from
  there. The Compare tab shows where two runs first differ in tokens and in logit hashes. The plan, the criteria (before the first run),
  verdicts, results.md and status are written on the screen and with `experiment update` in the CLI. The command console gains "Open AI
  runs" ([화면구현.md](../화면구현.md) §3).
