갈래: 더함

- **실행 기록과 재현 파일에 장치와 실제로 쓴 샘플링 값**: 실험에 남는 실행의 기록에 그 실행을 계산한 장치(이름 · 드라이버 — `device`)와
  엔진이 쓴 온도 · top_k · top_p · min_p · 시드(`sampling` — 요청에 없으면 엔진의 기본값 0.8 · 40 · 0.95 · 0.05)가 적히고, 실험의
  `devices` 에 장치가 더해집니다. 재현 파일도 만든 장치와 실제로 쓴 샘플링 값을 적습니다. 새 실행 대화상자는 비운 top-p · top-k · min-p 의
  기본값을 수로 보이고, 실행 화면과 실험 화면에 장치와 샘플링 값이 보이며, `system state` 의 `server.sampling_defaults` 와 `help` 에도
  기본값이 있습니다([사용성시험.md](../사용성시험.md) §5).

---

- **Run records and reproduction files carry the device and the sampling settings actually used.** A recorded run now has `device`
  (the name and driver of the device that computed it) and `sampling` (the temperature, top_k, top_p, min_p and seed the engine used —
  the defaults 0.8 · 40 · 0.95 · 0.05 when the request leaves them out), and the experiment's `devices` gets the device. Reproduction
  files record the device they were made on and the sampling settings actually used. The new-run dialog shows the default top-p, top-k
  and min-p as numbers, the run and experiment screens show the device and the sampling values, and `system state`
  (`server.sampling_defaults`) and `help` list the defaults ([사용성시험.md](../사용성시험.md) §5).
