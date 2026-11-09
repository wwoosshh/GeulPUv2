갈래: 더함

- **기록된 실행에서 분기 · 로짓 렌즈 · 점수를 명령 하나로**: `experiment branch --run r1 --step 3 --token 785` · `experiment lens --run r1
  --step 3` · `experiment score --run r1 [--step 3]` 이 실행의 기록(프롬프트 토큰 · 답 · 샘플링 설정 · 시드)에서 요청을 지어 엔진에 보내고 그
  실험의 실행으로 남깁니다 — 요청 본문을 손으로 짓지 않아도 됩니다. 실행 기록에는 지금처럼 보낸 요청 전체가, 명령 기록에는 짧은 명령이 남고, 그
  명령을 다시 보내면 같은 실행이 하나 더 생깁니다. 실행 화면과 스텝 · 층 화면의 분기 · 로짓 렌즈 · 상위 후보 · 답 전체 점수도 이 기능을 써서
  명령 창에 짧은 명령이 보입니다. 긴 분기를 만드는 동안에도 앱이 다른 요청에 바로 답합니다([사용성시험.md](../사용성시험.md) §5).

---

- **Branch, logit-lens and score a recorded run with one command.** `experiment branch --run r1 --step 3 --token 785`,
  `experiment lens --run r1 --step 3` and `experiment score --run r1 [--step 3]` build the request from the run's record (prompt
  tokens, answer, sampling settings, seed), send it to the engine and record the result as a run of the experiment, so request
  bodies no longer have to be built by hand. The run record still holds the full request sent and the command log holds the short
  command, which makes the same run again when sent again. The branch, logit lens, top candidates and whole-answer scoring on the
  run screen and the step and layer screen use the same operations, so the console shows the short command. The app keeps
  answering other requests while a long branch is generated ([사용성시험.md](../사용성시험.md) §5).
