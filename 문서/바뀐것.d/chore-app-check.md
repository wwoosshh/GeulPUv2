갈래: 바뀜

- **받는 모델을 엔진이 정한다**: 앱이 모델 구조마다의 규칙을 따로 갖지 않고 엔진(`server --모델확인`)에 묻는다 — 모듈 이름과 할 수 있는
  것(정밀만 · Vulkan · 8비트 KV · F2 등)은 전과 같고, 받지 않는 구조의 알림은 엔진의 까닭(받는 구조의 목록)으로 나온다.

---

- **The engine decides which models are accepted.** The app no longer keeps its own per-architecture rules; it asks the engine
  (`server --모델확인`). Module names and capabilities (exact only, Vulkan, 8-bit KV cache, F2, …) are unchanged, and an unsupported
  architecture now shows the engine's reason with the list of accepted architectures.
