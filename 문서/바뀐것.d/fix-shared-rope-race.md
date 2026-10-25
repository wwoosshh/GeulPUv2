갈래: 고침

- **불러올 때의 RoPE 표 · 판 기억**: 서버 · 앱은 모델을 불러올 때 RoPE 표를 일꾼 스레드로 미리 짓는데, 그사이 설정을 한 번 더 읽으며
  rope_freqs 를 잠깐 비워, rope_freqs 를 쓰는 모델의 표가 rope_freqs 없이 지어질 수 있었다(엔진은 그 표를 그대로 써서 긴 프롬프트의 답이
  불러올 때마다 확률로 달랐다). 이제 일꾼이 도는 동안 설정을 다시 읽지 않고, 미리 지은 표는 θ · rope_freqs · YaRN 까지 같을 때만 쓴다.
  행렬곱 묶음이 다섯을 넘는 구조(Gemma 4 · Qwen3.5 · Qwen3.6)의 k-양자화 모델은 판 기억을 늘 버려 불러올 때마다 판을 다시 재던 것도 고쳤다.

---

- **RoPE table and kernel-choice memory at load time.** The server and the app build the RoPE table on a worker thread while loading,
  but the loader read the model settings a second time meanwhile and briefly cleared rope_freqs, so models with rope_freqs could get a
  table without them (the engine kept that table, and long prompts could give a different answer from one load to the next). The loader
  no longer re-reads the settings while workers run, and a prebuilt table is used only when θ, rope_freqs and YaRN all match. K-quant
  models of architectures with more than five matmul groups (Gemma 4, Qwen3.5, Qwen3.6) always discarded their remembered kernel
  choices and re-tuned on every load; they now reuse them.
