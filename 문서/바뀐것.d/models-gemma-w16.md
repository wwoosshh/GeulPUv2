갈래: 더함

- **Gemma 4 · Gemma 3 가 양자화하지 않은 파일(F16 · BF16 · F32)을 받는다**(0.24.0): 행렬이 모두 F16 · BF16 · F32 인 Gemma 파일을 계약 1.3 의
  16비트 블록 가중치로 불러온다 — 다른 구조가 이미 쓰는 방식이다. 양자화된 파일의 값을 F16 · BF16 으로 적은 시험 파일로 엔진의 다섯 실행
  경로와 CPU 참조의 비트를 확인했다. 공개된 F16 · BF16 파일로 하는 모델 검증은 아직 하지 않아, 앱의 모델 찾기는 Gemma 의 Q8_0 ·
  k-양자화 파일만 받게 한다. F16 · BF16 · F32 가 k-양자화 · Q8_0 과 섞인 파일은 아직 받지 않는다. 기록:
  [구조/gemma4/기록.md](../../구조/gemma4/기록.md).

---

- **Gemma 4 and Gemma 3 take unquantized files (F16, BF16, F32).** Gemma files whose matrices are all F16, BF16 or F32 load as the
  contract's 16-bit block weights (section 1.3), the way the other architectures already do. Checked with test files that store
  the values of quantized files as F16 or BF16: five bit-identical execution paths, and all-row logits equal to the CPU reference.
  Model verification with the published F16 and BF16 files has not been done yet, so the app's model search still offers only
  the Q8_0 and k-quant files of Gemma. Files that mix F16, BF16 or F32 with k-quant or Q8_0 matrices are not accepted yet. Record:
  [구조/gemma4/기록.md](../../구조/gemma4/기록.md).
