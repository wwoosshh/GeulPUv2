갈래: 더함

- **GGUF 메타데이터 고치기**: 모델 평가의 버전 메뉴에서 채팅 템플릿 · 특수 토큰(BOS · EOS · EOT · PAD) · add_bos/add_eos 를 바꾼 새 GGUF 를
  만들어 새 버전으로 더한다. 텐서는 바이트 그대로 옮기고 메타데이터(키)만 바꾼다 — 원래 파일은 그대로다. 새 버전에는 만든 방법 · 부모 버전 ·
  바꾼 키가 남고, 같은 데이터 · 설정으로 다시 평가하면 부모와 전후 비교할 수 있다.

---

- **GGUF metadata fix.** From a version's menu in Model evaluation, create a new GGUF with a changed chat template, special tokens
  (BOS · EOS · EOT · PAD) or add_bos/add_eos and add it as a new version. The tensors are copied byte for byte and only the metadata
  (keys) change — the original file stays as it is. The new version records how it was made, its parent version and the changed keys,
  and evaluating it again with the same data and settings compares it with its parent.
