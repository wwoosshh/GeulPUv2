갈래: 더함

- **CLI(하위 명령)**: `GeulPU <명령> [--인자 값 …]`(또는 `geulpu <명령> …`)으로 떠 있는 GeulPU 의 기능을 모두 부를 수 있습니다. 명령은 기능
  목록에서 만들어져 영어 이름(`run branch`) · 한국어 이름(`실행 분기`) · id(`run.branch`)를 모두 받고, 결과와 오류는 JSON, 종료 코드는 0 성공 ·
  1 기능 실패 · 2 사용법 오류 · 3 앱에 닿지 못함입니다. `GeulPU ops` 가 모든 명령과 인자를, `GeulPU help <명령>` 이 명령 하나를 설명합니다. 오래
  걸리는 작업은 `--wait` 로 끝을 기다리고, `--session <이름>` 을 주면 앱의 이벤트에 그 이름이 적힙니다([명령줄.md](../명령줄.md)).

---

- **CLI (subcommands).** `GeulPU <command> [--param value …]` (or `geulpu <command> …`) calls any operation of the running GeulPU app.
  Commands come from the operation registry and accept the English name (`run branch`), the Korean name (`실행 분기`) or the id
  (`run.branch`); results and errors are JSON; exit codes are 0 ok · 1 operation failed · 2 usage error · 3 app not reachable.
  `GeulPU ops` lists every command with its parameters, `GeulPU help <command>` explains one, `--wait` waits for a job, and
  `--session <name>` puts the name on the app's events ([명령줄.md](../명령줄.md)).
