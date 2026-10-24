갈래: 더함

- **Gemma 4 12B(구조 gemma4)를 엔진이 돌린다**: 슬라이딩 윈도 층(헤드 256 · 창 1,024)과 전체 어텐션 층(헤드 512 · v = k · rope_freqs)이 섞인
  층, GeGLU · 어텐션과 FFN 뒤의 정규화 · 층 배율 · 로짓 소프트캡을 계약 4.12 대로 — CPU 참조와 비트까지 같다. k-양자화(Q4_K_M 등)와 Q8_0
  파일, 정밀 모드 · CUDA 만(빠른 모드 · Vulkan · 추측 디코딩 · 문맥 저장소는 아직). E2B · E4B(PLE) · 26B-A4B(MoE)는 까닭을 알리고 받지 않는다.

---

- **The engine runs Gemma 4 12B (architecture gemma4).** Sliding-window layers (head 256, window 1,024) mixed with global-attention
  layers (head 512, v = k, rope_freqs), GeGLU, post-attention/post-FFN norms, per-layer output scale and the final logit soft-cap
  follow contract 4.12 and match the CPU reference bit for bit. k-quant (Q4_K_M etc.) and Q8_0 files, exact mode on CUDA only (no fast
  mode, Vulkan, speculative decoding or context store yet). E2B/E4B (per-layer embeddings) and 26B-A4B (MoE) are rejected with a reason.
