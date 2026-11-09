갈래: 더함

- **화면에서 모든 기능**: 기능 목록의 기능이 모두 화면에서 닿습니다 — 모델 › 파일의 모델 메뉴에 '서버 정보'(서버가 알리는 모델 이름 · 문맥
  길이 · 토크나이저), 새 실행 창에 Anthropic Messages · OpenAI Responses 형식(토큰 수 세기 포함 — 이 두 형식은 서버가 시드를 받지 않아 답의
  텍스트만 기록합니다), 스텝 · 층 화면의 점수에 토큰 id 입력(텍스트로 적을 수 없는 특수 토큰의 점수). 화면은 앱을 기능 목록의 기능 id 로만
  부르고, CI 가 화면 코드에 경로가 없는지와 모든 기능이 화면에서 닿는지를 확인합니다([화면구현.md](../화면구현.md) §6).

---

- **Every operation on a screen.** Every operation in the registry is now reachable from a screen — "Server info" in the model menu
  on Models › Files (model name, context length and tokenizer as the server reports them), Anthropic Messages and OpenAI Responses in
  the new-run form (with token counting — these two formats take no seed, so only the answer text is recorded), and token ids as score
  input on the step and layer screen (scores for special tokens that cannot be typed as text). The screens call the app only by
  operation id, and CI checks that screen code holds no paths and that every operation is reachable ([화면구현.md](../화면구현.md) §6).
