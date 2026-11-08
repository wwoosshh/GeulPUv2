갈래: 더함

- **실험 화면**: 앱을 열면 실험 목록이 먼저 보입니다(사이드바의 '실험'). 실험마다 상태 · 모델 · 실행 수 · 판정 · 명령 수가 보이고, 화면에서 새
  실험을 만들고 실험 폴더 위치를 바꿀 수 있습니다. 실험을 열면 계획(판정 기준 · plan.md) · 결과(results.md) · 기록(명령 기록 · 파일 · 재현 명령 ·
  점검) 탭이 있고, 실험 화면에서 부른 점검과 CLI 의 `--experiment` 호출이 명령 기록에 바로 나옵니다. CLI 에는 실험 폴더 안의 텍스트 파일을 읽는
  `experiment read` 가 생겼습니다([화면구현.md](../화면구현.md) §2).

---

- **Experiment screens.** The app now opens on the experiment list ("Experiments" in the sidebar). Each experiment shows its status,
  models, number of runs, criteria and number of recorded commands; new experiments can be created on screen and the experiments folder
  can be changed. An experiment opens with Plan (criteria, plan.md), Results (results.md) and Records (commands log, files, commands to
  reproduce, check) tabs; a check run on the screen and CLI calls with `--experiment` appear in the commands log at once. The CLI gains
  `experiment read`, which reads a text file inside an experiment folder ([화면구현.md](../화면구현.md) §2).
