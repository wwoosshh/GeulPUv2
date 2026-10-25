갈래: 고침

- **추측 디코딩의 상태 · 회전으로 접은 GGUF**: 앱이 Qwen3.5 · Qwen3.6 MoE 를 불러오면 서버 상태의 추측 디코딩을 켜짐으로 보이던 것을
  고쳤다(서버는 그 구조에서 끈다 — 이제 엔진의 값을 보인다). llama.cpp 의 `prism.hadamard.*` 로 가중치를 회전으로 접은 GGUF 는 그대로
  돌리면 값이 틀리므로 까닭을 알리고 받지 않는다.

---

- **Speculative decoding state and rotated GGUF files.** Loading Qwen3.5 or Qwen3.6 MoE showed speculative decoding as on in the
  server state (the server turns it off for those architectures); the app now shows the engine's value. GGUF files whose weights were
  folded with a rotation (llama.cpp's `prism.hadamard.*`) give wrong values when run as is, so they are refused with the reason.
