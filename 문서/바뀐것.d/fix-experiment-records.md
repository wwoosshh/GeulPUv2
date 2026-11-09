갈래: 바꿈

- **실험의 템플릿 언어 · 저자 · 점검의 경고**: 한국어 화면에서 만든 실험은 한국어 템플릿(plan.md · results.md · 질문 · 판정 기준)으로
  만들어지고, 새 실험 대화상자에서 저자를 적습니다(CLI 는 `experiment create --lang ko --authors`). `experiment update` 는 저자 · 장치 · 재현
  명령과 직접 더한 파일(`add_files` — SHA-256 · 크기는 앱이 적음)을 받고, 모르는 필드는 거절합니다. 실험 점검은 템플릿 그대로인 계획 · 질문 ·
  판정 기준 · 저자를 경고하고(통과는 그대로), 점검한 항목을 작업 기록에 적습니다. 점수 · 렌즈 실행의 기록에는 쓰지 않는 seed 를 적지
  않습니다([사용성시험.md](../사용성시험.md) §3).

---

- **Experiment template language, authors and check warnings.** Experiments created in the Korean UI start from a Korean template
  (plan.md, results.md, question, criteria), and the new-experiment dialog takes the authors (CLI: `experiment create --lang ko
  --authors`). `experiment update` takes authors, devices, reproduce commands and files you added (`add_files` — the app writes their
  SHA-256 and size), and refuses fields it does not know. The experiment check warns about a plan, question, criteria or authors left
  as the template has them (it still passes) and writes what it checked into the job log. Score and lens runs no longer record a seed
  they do not use ([사용성시험.md](../사용성시험.md) §3).
