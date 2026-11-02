갈래: 더함

- **GGUF 변환 검증**: 모델 평가의 작업 공간에서 원래 모델 폴더(Hugging Face 형식 — tokenizer_config.json · tokenizer.json ·
  generation_config.json, 가중치는 읽지 않음)를 정하고 버전의 GGUF 를 검사한다. 채팅 템플릿(대화 경우 다섯 ~ 일곱을 서버와 같은 jinja 해석기로
  그려 바이트까지 견줌 — 다른 곳을 두 글로 나란히) · EOS · BOS · 특수 토큰 · 어휘와 병합을 항목마다 통과 · 문제 · 확인 필요로 보이고, 메타데이터로
  고칠 수 있는 문제는 '고친 버전 만들기'로 새 버전을 만든다.

---

- **GGUF conversion check.** In a Model evaluation workspace, set the original model folder (Hugging Face format — tokenizer_config.json ·
  tokenizer.json · generation_config.json; the weights are not read) and check a version's GGUF. The chat template (five to seven
  conversation cases rendered with the same jinja interpreter as the server and compared byte for byte — differences shown side by side),
  EOS, BOS, special tokens, and the vocabulary and merges are each marked pass, problem or to check, and problems that the metadata can
  fix become a new version with 'Create fixed version'.
