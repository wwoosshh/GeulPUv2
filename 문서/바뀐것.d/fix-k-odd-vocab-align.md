갈래: 고침

- **어휘 수가 홀수인 k-양자화 파일이 CUDA 에서 올라간다**: 출력 행렬(또는 토큰 임베딩)이 Q6_K 이고 어휘 수가 홀수이거나 Q5_K 이고 4 의
  배수가 아니면, 엔진의 가중치 배치에서 상위 비트 행의 주소가 16 바이트에 맞지 않아 커널의 비동기 복사가 CUDA 오류 716(misaligned
  address)으로 멈췄다 — 어휘에 토큰을 더한 Llama 3 미세조정(카카오 Kanana 1.5 — 어휘 128,259) 등. 상위 비트의 열 간격을 4 의 배수로
  올려 배치하고, 커널 생성기는 어휘 수가 4 의 배수가 아닌 모듈에서만 그 간격으로 주소를 계산한다. 지금 받는 모델의 모듈과 출력 비트는
  그대로다.

---

- **k-quant files with an odd vocabulary size load on CUDA.** When the output matrix (or the token embedding) is Q6_K with an odd
  vocabulary size, or Q5_K with one that is not a multiple of 4, the engine's weight layout left the high-bit rows off 16-byte
  alignment and the kernels' asynchronous copies stopped with CUDA error 716 (misaligned address) — for example Llama 3 fine-tunes
  with added tokens such as Kakao's Kanana 1.5 (vocabulary 128,259). The high-bit column stride is now padded to a multiple of 4,
  and the kernel generator uses that stride only in modules whose vocabulary size is not a multiple of 4, so the modules and output
  bits of the models GeulPU already accepts are unchanged.
