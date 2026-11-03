갈래: 고침

- **부호 검정의 p 가 큰 표본에서 0 이 되던 것**: 한쪽만 맞은 대화가 합쳐 1,075 개 이상이면 부호 검정의 p 가 0 이 되어(첫 항 2⁻ⁿ 이
  배정밀도 실수에서 0 으로 사라짐) 600 · 599 처럼 거의 반반인 결과도 '뚜렷한 차이'로 계산되던 것을 고쳤다(#236 — 외부 리뷰). 1,023 ~
  1,074 개에서도 정밀도를 잃었다. 이제 항을 로그로 더해, 정수로 정확히 계산한 값과의 상대 오차가 4만 개에서도 10⁻¹⁰ 안쪽이다. 결과의
  정답률 p 와 두 결과를 견주는 판정(나아짐 · 나빠짐)이 이것을 쓴다 — 판정은 견줄 때 계산하므로 곧바로 바르고, 이미 만든 결과에 적힌 p 는
  다시 평가하면 바르게 나온다. 빠른 점검은 이제 부호 검정을 정확한 값과도 견준다(n 1,000 ~ 4만의 경계 포함).

---

- **The sign test's p no longer drops to 0 for large samples.** With 1,075 or more conversations right on one side only, the
  sign test returned p = 0 (the first term 2⁻ⁿ vanished in double precision), so even a near-even 600 / 599 counted as a clear
  difference (#236 — external review); from 1,023 to 1,074 it lost precision. Terms are now summed in log space and stay within
  10⁻¹⁰ relative error of the exact value up to 40,000. The result's accuracy p and the better / worse verdict of comparing two
  results use it — the verdict is computed when comparing, so it is right at once; a p already written in a result is right
  after evaluating again. The quick check now also compares the sign test with exact values (including n 1,000 – 40,000).
