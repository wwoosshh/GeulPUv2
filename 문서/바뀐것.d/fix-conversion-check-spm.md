갈래: 고침

- **변환 검증 — SentencePiece 토크나이저와 transformers 5 의 BOS**: 원래 모델이 BOS 를 붙이는지를 `tokenizer_config.json` 에 없으면
  `tokenizer.json` 의 post_processor 로도 판단한다 — transformers 5 로 저장한 모델에서 'BOS 문제'로 잘못 알리고 BOS 를 끄는 '고친 버전'을
  권하던 것을 고쳤다(#214). 알 수 없으면 '확인 필요'로 두고 고칠 값을 권하지 않는다. SentencePiece GGUF 는 병합 대신 어휘만 견주고,
  사용자 정의 토큰의 ▁ 공백 표기와 `<unk>` 의 UNKNOWN 형식은 같은 것으로 본다. 모델의 어휘 밖 번호에 있는 특수 토큰(이미지 토큰 등)은
  따로 센다(직접 만든 SentencePiece 토크나이저 · Gemma 3 에서 나던 '문제').

---

- **Conversion check — SentencePiece tokenizers and the transformers 5 BOS.** Whether the original model adds BOS is now also
  read from the post_processor in `tokenizer.json` when `tokenizer_config.json` does not say. Models saved with transformers 5
  were flagged with a BOS problem, and the suggested fixed version turned BOS off (#214). When it cannot be told, the item is
  "check" and no fix is suggested. A SentencePiece GGUF is compared by vocabulary instead of merges, and the ▁ space mark of
  user-defined tokens and the UNKNOWN type of `<unk>` count as the same. Special tokens with IDs outside the model's vocabulary
  (image tokens and the like) are counted apart (false problems with a self-made SentencePiece tokenizer and with Gemma 3).
