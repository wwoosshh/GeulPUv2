갈래: 바꿈

- **토크나이저의 사전 처리 표를 llama.cpp b12662 에 맞췄습니다.** 사전 처리 이름(`tokenizer.ggml.pre`) 여섯 — `inkling2` · `tekken-num3` ·
  `domyn-edge` · `nemotron-3.5-embed` · `soprano` · `zonos3` — 을 새로 알아봅니다(`soprano` 는 SentencePiece 형식이라 받지 않습니다). 엔진이
  그 구조를 받는 파일이면 이제 열립니다. 이미 있던 이름 95 개의 나누기 규칙은 그대로라 지금 모델의 토큰은 바뀌지 않습니다.

---

- **The pre-tokenizer table now follows llama.cpp b12662.** Six pre-tokenizer names (`tokenizer.ggml.pre`) are new — `inkling2`,
  `tekken-num3`, `domyn-edge`, `nemotron-3.5-embed`, `soprano` and `zonos3` (`soprano` is SentencePiece-style and stays unsupported). Files
  with these names open when the engine supports their architecture. The rules of the 95 existing names are unchanged, so current models
  get the same tokens.
