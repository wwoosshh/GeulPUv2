갈래: 고침

- **정답률의 95 % 구간이 대화 수가 2 의 거듭제곱이면 폭 0 이 되던 것**: 채점한 대화가 16 · 32 · 64 · 128 · 256 · 512 · 1024 개처럼 2 의
  거듭제곱이면 부트스트랩 95 % 구간이 [정답률, 정답률]로 무너졌다 — 다시 뽑을 대화의 번호를 선형 합동 생성기의 낮은 비트(x % n)로 골라,
  그 비트의 주기가 n 과 같아 다시 뽑은 표본이 늘 원래 대화의 순열이 되었기 때문이다. 이제 두 걸음의 높은 비트로 고른다(64 개 · 정답률 0.5
  의 구간 [0.39, 0.63] — 정규 근사의 폭과 맞음). 다른 대화 수의 구간도 뽑기가 바뀌어 끝자리가 조금 달라진다(같은 결과를 다시 평가하면
  같은 구간). 빠른 점검은 이제 구간을 참조와의 비트뿐 아니라 성질(점추정을 품음 · 정규 근사의 폭)로도 보고, 지수 · 로그 · KL · 꼬리 ·
  백분위수도 표준 함수 · 다른 합 · 바로 고른 순위로 다시 셈한 값과 견준다(외부 리뷰 — 참조와 같다는 검사만으로는 둘이 함께 틀린 것을
  잡지 못한다).

---

- **The 95 % accuracy interval no longer collapses when the number of conversations is a power of two.** With 16 · 32 · 64 ·
  128 · 256 · 512 · 1024 graded conversations, the bootstrap 95 % interval was [accuracy, accuracy] — the resampled conversation
  numbers came from the low bits of a linear congruential generator (x % n), whose period equals n, so every resample was a
  permutation of the original conversations. Numbers now come from the high bits of two steps (64 conversations at accuracy 0.5:
  [0.39, 0.63] — matching the width of the normal approximation). Intervals for other numbers of conversations change in the last
  digits because the draw changed (evaluating the same result again gives the same interval). The quick check now checks the
  interval's properties (contains the point estimate · width of the normal approximation) besides its bits against the reference,
  and recomputes exp · log · KL · tail · percentile with standard functions, a different sum and a direct rank (external review —
  a check against a reference alone misses errors both share).
