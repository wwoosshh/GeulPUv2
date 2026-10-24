갈래: 더함

- **Qwen3.8-27B 와 형식이 섞인 Qwen3.5 GGUF**: 한 파일 안에서 행렬마다 Q8_0 과 k-양자화가 섞인 파일(Qwen3.8-27B 의 ggml-org Q4_K_M — 어텐션 ·
  GDN 투영이 Q8_0)과 ssm_alpha · ssm_beta 가 k-양자화인 파일(lmstudio Q4_K_M)을 받는다. 파일 끝의 MTP(다음 토큰 예측) 층은 llama.cpp 처럼
  건너뛴다(전에는 "텐서가 없다"로 멈췄다). Qwen3.8-27B 는 12 GB 장치에는 들지 않는다. F32 β · α(bartowski) · IQ 형식이 섞인 파일(unsloth UD)은 아직.

---

- **Qwen3.8-27B and Qwen3.5 GGUFs with mixed formats.** Files that mix Q8_0 and k-quant matrices (the ggml-org Q4_K_M of Qwen3.8-27B keeps the
  attention and GDN projections in Q8_0) and files whose ssm_alpha · ssm_beta are k-quants (lmstudio Q4_K_M) are now accepted. A
  multi-token-prediction layer at the end of the file is skipped like llama.cpp does (the engine used to stop with "tensor not found").
  Qwen3.8-27B does not fit a 12 GB card. Files with F32 β · α (bartowski) or IQ formats (unsloth UD) are not supported yet.
