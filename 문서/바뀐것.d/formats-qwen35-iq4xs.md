갈래: 더함

- **Qwen3.5 · Qwen3.6 의 IQ4_XS 행렬(CUDA)**: Qwen3.5 계열(qwen35 · qwen35moe) 파일 안의 IQ4_XS 행렬을 받는다(행렬마다 — 다른 구조와 같이
  모듈 KI, 아직 CUDA 만). llama.cpp 의 IQ4_XS 파일과, unsloth 의 UD 판처럼 IQ4_XS 가 섞인 k-양자화 파일(예: Qwen3.5-9B UD-Q4_K_XL)이
  실행된다. 한 행렬곱 안에서 형식이 다른 파일(gate 와 up 이 다름 등)과 IQ4_NL · Q3_K · IQ3 · IQ2 가 든 파일은 아직 받지 않는다.

---

- **IQ4_XS matrices in Qwen3.5 / Qwen3.6 files (CUDA).** Qwen3.5-family files (qwen35 · qwen35moe) may now hold IQ4_XS matrices,
  per tensor, through the KI module like the other architectures (CUDA only for now). This runs llama.cpp's IQ4_XS files and k-quant
  files with IQ4_XS mixed in, such as unsloth's UD builds (for example Qwen3.5-9B UD-Q4_K_XL). Files whose gate and up (or q · k · v)
  formats differ within one matmul, and files holding IQ4_NL · Q3_K · IQ3 · IQ2, are still refused.
