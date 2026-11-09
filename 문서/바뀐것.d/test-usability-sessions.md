갈래: 고침

- **모델 화면에서 바로 불러오기 · 비교 탭의 답의 첫 차이**: 모델 › 파일의 내 모델에 '이 모델 불러오기'가 있습니다 — 전에는 채팅을 보내야
  불러왔습니다. 프롬프트가 다른 두 실행의 비교는 프롬프트 위치에 다른 스텝의 확률을 보이던 것을 고치고, 답이 처음 다른 스텝과 그 스텝의 두
  토큰 · 확률 · 순위를 보입니다. CLI 의 `--wait` 는 `--out` 과 함께 줘도 작업이 끝날 때까지 기다려 끝난 작업을 그 파일에 쓰고, 문법이 틀린
  `--json` 은 "--json is not valid JSON" 으로 알립니다([사용성시험.md](../사용성시험.md) §2).

---

- **Load a model from the models screen; the compare tab's first differing answer step.** Models › Files › My models has "Load this
  model" — before, a model loaded only when a chat message was sent. Comparing two runs with different prompts no longer shows another
  step's probabilities at the prompt position, and shows the first step where the answers differ with both tokens' probabilities and
  ranks. In the CLI, `--wait` waits for the job also with `--out` and writes the finished job to that file, and a `--json` body with a
  syntax error is reported as "--json is not valid JSON" ([사용성시험.md](../사용성시험.md) §2).
