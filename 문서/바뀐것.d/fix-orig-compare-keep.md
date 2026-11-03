갈래: 고침

- **원래 모델과 견주기 — 다시 견줘도 더 많이 견준 결과가 남음**: 같은 결과를 '견줄 대화 수'로 다시 견주면 앞서 모두 견준 결과를
  덮어써, GSM8K 1,319 개를 24 분 동안 견준 값이 '처음 200 개'의 값으로 바뀌던 것을 고쳤다(#231). 이제 견준 결과는 대화 수 · 계산
  형식(float32 · bfloat16)마다 따로 남고, 카드에는 가장 많이 견준 것이 보이며 나머지는 '다른 견주기' 줄로 보인다. API
  `GET /api/eval/original` 은 그 결과에 `file` 과 모든 견준 결과의 요약 `runs` 를 덧붙인다.

---

- **Compare with the original model — comparing again keeps the larger result.** Comparing the same result again with
  "Conversations to compare" overwrote the full comparison, so 24 minutes of GSM8K (1,319 conversations) turned into the values
  of the first 200 (#231). Each number of conversations and compute type (float32 · bfloat16) now keeps its own result; the card
  shows the one that compared the most and lists the others as "Other comparison" lines. `GET /api/eval/original` adds `file`
  and `runs` (a summary of every stored comparison) to the result.
