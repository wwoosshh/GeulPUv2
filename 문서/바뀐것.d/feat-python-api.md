갈래: 더함

- **파이썬 API**: `import geulpu` 로 앱을 창 없이 띄우거나(`geulpu.start`) 떠 있는 앱에 붙어(`geulpu.connect`) 모델을 불러오고, 스텝마다
  로짓 해시와 함께 생성하고(`chat` · `complete`), 같은 요청을 다시 보내고(`rerun`), 토큰에 점수를 매기고(`score`), 한 토큰에서 분기하고
  (`branch`), 두 실행을 비트까지 비교한다(`geulpu.compare`). `info()` 는 판 · GPU · 드라이버 · 모델 SHA-256 · 계산 길을 돌려준다.
  표준 라이브러리만 쓰고 앱의 HTTP API 로 묻는다 — 엔진 · 출력 비트는 그대로다([파이썬API.md](../파이썬API.md)).

---

- **Python API.** `import geulpu` starts a GeulPU app without a window (`geulpu.start`) or connects to a running one
  (`geulpu.connect`), loads a model, generates with every step's logit hash (`chat`, `complete`), sends a request again (`rerun`),
  scores tokens (`score`), branches at a token (`branch`) and compares two runs bit for bit (`geulpu.compare`); `info()` returns the
  version, GPU, driver, model SHA-256 and computation path. Standard library only, through the app's HTTP API — the engine and its
  output bits are unchanged ([docs/python-api.md](../../docs/python-api.md)).
