갈래: 더함

- **Qwen3.5 · Qwen3.6 의 16비트 행렬(F16 · BF16 · F32)**: Qwen3.5 계열(qwen35 · qwen35moe) 파일 안의 16비트 행렬을 받는다(계약 1.3 의
  16비트 블록 가중치 — 행렬마다). 출력 가중치가 BF16 인 k-양자화 파일(DavidAU 의 NEO · MAX 판 등), β · α(`ssm_beta` · `ssm_alpha`)가
  F16 · F32 인 파일(unsloth 의 Qwen3.6-35B-A3B UD · Qwen3.6-27B 등), 16비트만인 파일(F16 · BF16)이 실행된다. 앱의 모델 찾기는 구조마다
  받는 형식을 엔진에게서 받는다.

---

- **16-bit matrices in Qwen3.5 / Qwen3.6 files (F16 · BF16 · F32).** Qwen3.5-family files (qwen35 · qwen35moe) may now hold 16-bit
  matrices, each converted to 16-bit block weights as in contract 1.3. This runs k-quant files whose output weight is BF16 (such as
  DavidAU's NEO · MAX builds), files whose β · α (`ssm_beta` · `ssm_alpha`) are F16 or F32 (such as unsloth's Qwen3.6-35B-A3B UD and
  Qwen3.6-27B), and 16-bit-only files (F16 · BF16). The app's model finder takes the formats each architecture accepts from the
  engine.
