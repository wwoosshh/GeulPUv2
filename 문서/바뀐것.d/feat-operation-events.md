갈래: 더함

- **기능 호출 이벤트(`/api/events`)**: 앱이 계산하거나 기록을 바꾸는 기능(기능 목록의 POST 기능 — 생성 · 분기 · 렌즈 · 점수 · 모델 불러오기 ·
  평가 등)을 받을 때마다 시작 · 끝 이벤트를 남깁니다. `GET /api/events?after=<seq>&wait=<ms>` 로 받습니다(새 이벤트가 없으면 `wait` 동안
  기다립니다). 이벤트마다 기능 id · 출처(앱 화면 · 파이썬 API · 그 밖) · HTTP 상태 · 걸린 시간이 적히고, 기존 API 의 응답은 그대로입니다
  ([기능목록.md](../기능목록.md) §7).

---

- **Operation-call events (`/api/events`).** The app records a start and an end event for every call that computes or changes records
  (the registry's POST operations — generation, branch, lens, score, model load, evaluation …). Read them with
  `GET /api/events?after=<seq>&wait=<ms>`, which waits up to `wait` ms when there is no new event. Each event carries the operation id,
  the source (app screen, Python API, other), the HTTP status and the duration; existing API answers are unchanged
  ([기능목록.md](../기능목록.md) §7).
