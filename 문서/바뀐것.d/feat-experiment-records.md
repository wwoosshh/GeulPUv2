갈래: 더함

- **실험 기록의 형식과 점검 도구**: 실험 하나를 폴더 하나(`experiments/<날짜>-<이름>/`)로 남기는 형식 `geulpu-experiment/1` — 연구 질문 ·
  실행 전에 적은 판정 기준 · 모델(SHA-256 과 출처) · 실행 · 포함 파일마다 SHA-256 — 을 정했습니다. 점검 도구 `도구/experiment_check.py`
  는 GPU 없이 형식 · 해시 · 크기(압축 5 MB 이하)를 확인합니다([experiments/README.ko.md](../../experiments/README.ko.md)).

---

- **Experiment records: format and checker.** One experiment is kept as one folder (`experiments/<date>-<name>/`) in the format
  `geulpu-experiment/1` — the research question, criteria written before any run, models by SHA-256 and source, runs, and the SHA-256 of
  every included file. The checker `도구/experiment_check.py` verifies format, hashes and size (at most 5 MB compressed) without a GPU
  ([experiments/README.md](../../experiments/README.md)).
