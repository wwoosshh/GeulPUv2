갈래: 고침

- **안내 한 쪽만 받은 바깥 에이전트가 막힌 곳을 고쳤다**(바깥 연구실의 시험, 2026-11-09 · 11-10).
  안내([docs/cli.md](../../docs/cli.md) · `GeulPU guide`)에 설치 방법이 생겼고, `--wait` 로 기다린 작업의 답이 `{"job": {...}}` 라는
  것(`job.state` · `job.summary` · `job.reason`)을 바르게 적었다 — 전에는 작업의 JSON 을 그대로 준다고 적어, 그대로 읽으면 성공한 모델
  불러오기를 실패로 보았다. `experiment update` 의 판정 기준은 `text` 가 없는 항목(`{id, result, note}`)을 받아 그 id 의 판정만
  바꾼다 — 전에는 판정을 적을 때도 기준마다 id 와 글 전체를 다시 보내야 했고 안내의 예는 그것을 줄여 적었다. 전에 받던 요청은 그대로
  받는다. 판정 기준이 틀린 요청과 첫 실행 뒤에 고정된 것을 바꾸는 요청에는 CLI 가 영어로 할 일(`next`)을 붙인다. 앱이 시작 직후 하드웨어를
  확인하는 동안에 온 모델 불러오기 · 진단 · 속도 측정은 400 이 아니라 503 으로 답하고 CLI 가 잠시 뒤 다시 하라고 알린다. 안내에는 생각부터
  하는 모델(Qwen3 등)이 `max_tokens` 를 생각에 먼저 쓴다는 것과 끄는 방법도 적었다.

---

- **Fixed the places where an outside agent, given only the one-page guide, got stuck** (tests by an outside lab, 2026-11-09 and
  11-10). The guide ([docs/cli.md](../../docs/cli.md), `GeulPU guide`) now says how to install GeulPU and describes the answer
  of a waited job correctly: `{"job": {...}}` with `job.state`, `job.summary` and `job.reason` — it used to promise the job's
  JSON itself, so reading it literally made a successful model load look failed. `experiment update` takes
  criteria entries without `text` (`{id, result, note}`) and sets only that id's verdict; before, recording a verdict meant
  sending every criterion's id and full text again, which the guide's example left out. Requests accepted before are accepted
  as before. For a wrong criteria request, or one that changes what is fixed after the first run, the CLI adds what to do in
  English (`next`). A model load, diagnosis or speed measurement sent while the app is still checking the hardware right after it
  starts is answered with 503 instead of 400, and the CLI says to run it again shortly. The guide also notes that a model that
  thinks first (Qwen3 and others) spends `max_tokens` on thinking, and how to turn that off.
