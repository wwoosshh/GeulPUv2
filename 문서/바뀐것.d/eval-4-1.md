갈래: 더함

- **GGUF 만들기**: 모델 평가의 작업 공간에서 원래 모델(Hugging Face 형식 폴더)을 GGUF 로 변환하고(BF16 · F16 · Q8_0) 양자화한다(Q8_0 ·
  Q6_K · Q5_K_M · Q4_K_M · IQ4_XS, imatrix — 글 파일로 만들거나 가진 파일, 행렬별 형식). 변환한 파일은 기준 모델이 되고(기준 모델이 없을 때)
  양자화한 파일은 버전이 되며, 만든 설정 · 도구의 판이 버전에 남고 변환 검증이 저절로 돈다. '기준 모델에서 양자화'로 변환을 되풀이하지 않고
  다시 만든다. 도구는 처음 한 번 '도구 준비'로 받는다 — llama.cpp b12032 의 실행 파일 · 변환 스크립트(SHA-256 확인)와, 이 PC 의 파이썬으로
  만드는 앱 전용 가상 환경의 패키지(torch CPU · transformers). 엔진이 받지 않는 섞임이면 버전에 까닭이 보인다.

---

- **Make GGUF.** A Model evaluation workspace can now convert the original model (a Hugging Face folder) to GGUF (BF16 · F16 · Q8_0) and
  quantize it (Q8_0 · Q6_K · Q5_K_M · Q4_K_M · IQ4_XS, an imatrix made from a text file or an existing one, per-matrix types). The
  converted file becomes the reference model (when there is none) and the quantized files become versions that keep their settings
  and tool versions; the conversion check runs by itself. 'Quantize the reference model' remakes files without converting again. The
  tools are downloaded once with 'Set up tools' — llama.cpp b12032 executables and conversion scripts (SHA-256 checked), plus the
  packages (torch CPU · transformers) in an app-only virtual environment made with this PC's Python. A version shows why when the
  engine does not support its mix of types.
