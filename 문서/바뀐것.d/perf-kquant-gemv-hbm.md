갈래: 바꿈

- **k-양자화 모델의 토큰 생성이 빨라짐(CUDA)**: 1행 GEMV 커널(`정수줄하나_*`)이 스테이지마다 내는 명령을 덜었다(출력 비트는 그대로).
  H100 NVL 의 MIG 인스턴스에서 생성 스텝(깊이 1,024)이 Llama-3.3-70B Q4_K_M 30 %, Llama-3.1-8B Q4_K_M 27 %, RTX 4070 Ti 에서 2 % 남짓
  짧다.

---

- **Faster k-quant decoding (CUDA).** The one-row GEMV kernels (`정수줄하나_*`) issue fewer instructions per stage (output bits
  unchanged). On an H100 NVL MIG instance the decode step (depth 1,024) is 30% shorter for Llama-3.3-70B Q4_K_M and 27% for
  Llama-3.1-8B Q4_K_M; on an RTX 4070 Ti it is about 2% shorter.
