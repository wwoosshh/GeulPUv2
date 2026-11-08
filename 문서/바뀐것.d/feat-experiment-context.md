갈래: 더함

- **실험 기록을 CLI 로**: `GeulPU experiment create --name <이름>` 이 템플릿에서 실험 폴더를 만들고(데이터 폴더의 `experiments`, 또는
  `--root`), `experiment list` · `show` 가 실험을 찾아 보여 주고, `experiment check --wait` 가 점검 도구로 형식 · 해시를 확인합니다(파이썬
  3.10 이상). 어느 명령에든 `--experiment <폴더>` 를 붙이면 앱이 그 호출(기능 · 요청 본문 · 출처 · 상태)을 실험의 `records/commands.jsonl` 에
  남깁니다([명령줄.md](../명령줄.md) §3).

---

- **Experiment records from the CLI.** `GeulPU experiment create --name <name>` makes an experiment folder from the template (in
  `experiments` of the data folder, or `--root`); `experiment list` · `show` find and show experiments; `experiment check --wait` checks the
  format and hashes with the experiment checker (Python 3.10 or later). With `--experiment <folder>` on any command, the app records that
  call (operation, request body, source, status) in the experiment's `records/commands.jsonl` ([명령줄.md](../명령줄.md) §3).
