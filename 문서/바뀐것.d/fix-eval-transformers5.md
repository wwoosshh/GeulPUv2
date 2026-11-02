갈래: 고침

- **transformers 5 로 저장한 모델**: 지금의 transformers 5.x 로 학습 · 저장한 미세조정 폴더를 'GGUF 만들기'가 변환하지 못하고(변환 도구의
  transformers 4.57.6 이 5.x 의 tokenizer_config.json 을 읽지 못함), '원래 모델과 견주기'는 config.json 의 rope_parameters 를 몰라 경고 없이
  틀린 KL 을 냈다(#196). 이제 '도구 준비'가 transformers 5.22.1 환경을 함께 만들고(약 30 MB — torch 는 함께 씀), 폴더를 저장한 transformers 의
  판을 보고 환경을 고른다. 원래 모델과 견주기는 먼저 GGUF · config.json · transformers 가 읽은 하이퍼파라미터(층 수 · 머리 수 · rope 기준값 ·
  RMS eps …)를 견줘, transformers 가 다르게 읽었으면 KL 을 내지 않고 멈추고, 16비트 기준 모델의 KL 이 10⁻³ 보다 크면 짚어 준다. 변환 검증에
  '설정' 항목(config.json ↔ GGUF 하이퍼파라미터)을 더했고, 변환 · 견주기가 실패하면 작업 기록의 마지막 줄(까닭)을 보이며, 가중치 파일이 없는
  폴더는 변환하기 전에 그렇게 말한다(llama.cpp 는 토크나이저만 든 GGUF 를 썼다).

---

- **Models saved with transformers 5.** Make GGUF could not convert a fine-tune trained and saved with today's transformers 5.x (the
  conversion tools' transformers 4.57.6 cannot read 5.x's tokenizer_config.json), and Compare with the original model silently gave a
  wrong KL because it did not know config.json's rope_parameters (#196). 'Set up tools' now also makes a transformers 5.22.1
  environment (about 30 MB — torch is shared), and the app picks the environment from the transformers version that saved the folder.
  Compare with the original model first checks the hyperparameters (layers · heads · RoPE base · RMS eps …) in the GGUF, in config.json
  and as read by transformers; it stops without a KL when transformers read them differently, and flags a KL above 10⁻³ for a 16-bit
  reference model. The conversion check has a new 'Settings' item (config.json ↔ GGUF hyperparameters), a failed conversion or
  comparison shows the last line of its run log, and a folder without weight files is reported before converting (llama.cpp wrote a
  tokenizer-only GGUF).
