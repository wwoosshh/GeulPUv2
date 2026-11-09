갈래: 바꿈

- **사이드바 다섯**: 사이드바가 실험 · 모델 · 재현 · 작업 · 설정 다섯이 되고, 그룹마다 위에 탭이 있습니다 — 모델(파일 · 평가 · 성능), 재현
  (llama.cpp · 검사 · 검사 결과), 설정(앱 · 장치 · 프롬프트 캐시). 벤치마크의 카드는 그 기능의 탭으로(속도 측정은 모델 › 성능, 하드웨어 진단은
  설정 › 장치, 재현성 검증 · 파일 쌍은 재현 › 검사), 모델 평가는 모델 › 평가로 옮겼습니다. 새 작업 화면은 지금 작업의 진행 · 로그 · 중지와 이
  브라우저가 본 지난 작업을 보이고, 명령 창의 줄은 그 호출이 시작한 작업이 끝날 때까지 진행을 보입니다.
- **채팅은 기본 실험 "대화"로**: 실험 목록의 "대화"에서 채팅 화면을 엽니다. 답마다 그 실험의 실행으로 남고(처음 보내면 실험이 만들어집니다),
  답의 '실행 rN'에서 토큰별 출력 · 분기 · 스텝 · 층 화면을 봅니다. 디버거는 실험 화면의 기록 탭에서, 에이전트는 명령 창의 'AI 세션'에서 엽니다
  ([화면구현.md](../화면구현.md) §5).

---

- **Five sidebar groups.** The sidebar is now Experiments, Models, Reproduce, Jobs and Settings, each group with tabs — Models (files,
  evaluation, performance), Reproduce (llama.cpp, checks, check results), Settings (app, devices, prompt cache). The benchmark cards
  moved to their tabs (speed measurement to Models › Performance, hardware diagnosis to Settings › Devices, the reproducibility and
  file-pair checks to Reproduce › Checks), model evaluation to Models › Evaluation. The new Jobs screen shows the current job's
  progress, log and stop button and the earlier jobs this browser saw; a console line follows the job its call started until it ends.
- **Chat lives in the default experiment "Conversations" (대화).** Open the chat screen from "Conversations" in the experiment list.
  Every answer is recorded as a run of that experiment (the first message creates it), and an answer's "Run rN" opens its
  token-by-token output, branching and the step and layer screen. The debugger opens from the experiment's Records tab, the agents
  screen from "AI sessions" in the console ([화면구현.md](../화면구현.md) §5).
