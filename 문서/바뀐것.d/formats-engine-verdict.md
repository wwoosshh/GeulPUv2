갈래: 고침

- **앱이 파일 형식을 엔진의 판정대로 받음**: 앱의 모델 확인이 형식(Q8_0 · k-양자화 · 16비트의 섞임)을 엔진의 모델 확인에 맡긴다.
  엔진이 받는데 앱의 더 엄한 규칙이 막던 파일 — 예: ggml-org 의 Qwen3.8-27B Q4_K_M(k-양자화 파일 안의 Q8_0 행렬) — 을 받는다. 엔진의
  모델 확인은 불러오기에서야 멈추던 파일(층 안의 q · k · gate · up 형식이 서로 다른 파일, Qwen3.5 MoE 의 F32 β · α 등)을 확인에서 그
  까닭과 함께 받지 않는다.

---

- **The app accepts file formats as the engine does.** The app's model check leaves weight formats (mixes of Q8_0, k-quants and
  16-bit) to the engine's model check, so files the engine loads but the app's stricter rules refused — for example ggml-org's
  Qwen3.8-27B Q4_K_M (Q8_0 matrices inside a k-quant file) — are accepted. The engine's model check now refuses, with the reason,
  files that used to pass the check and stop during loading (layers whose q · k or gate · up formats differ, F32 β · α in Qwen3.5
  MoE files, and so on).
