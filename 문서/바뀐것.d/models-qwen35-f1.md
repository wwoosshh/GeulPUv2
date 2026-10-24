갈래: 더함

- **Qwen3.5 의 빠른 모드**: Qwen3.5(dense — 0.8B · 4B · 9B 등)를 빠른 모드(계산 방식 '빠름' — 빠른 계약 F1, 선형층의 활성값 한 자리)로
  쓸 수 있다. 앱의 계산 방식이 빠름(기본)이면 빠른 모드로 불러온다. 8비트 KV 캐시 · 빠른 어텐션은 아직 없고, Qwen3.6 MoE 는 아직 정밀
  모드만이다(품질을 아직 재지 못했다).

---

- **Fast mode for Qwen3.5.** Qwen3.5 dense models (0.8B, 4B, 9B, …) can now run in fast mode (compute mode "fast" — the fast contract
  F1 with one-digit activations in the linear layers); the app loads them in fast mode when that is the setting (the default). The 8-bit
  KV cache and fast attention are not available for them yet, and Qwen3.6 MoE stays exact-only until its quality has been measured.
