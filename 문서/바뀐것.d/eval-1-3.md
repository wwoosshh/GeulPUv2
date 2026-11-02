갈래: 더함

- **모델 평가 화면**: 왼쪽 메뉴에 '모델 평가'를 더했다. 작업 공간(기준 모델과 비교할 GGUF 버전들)을 만들고 화면에서 평가를 실행한 뒤,
  결과(판정 한 문장 · KL divergence · top-1 일치율 · 처음 갈라진 스텝의 분포와 마진이 작은 스텝에서 갈라진 몫 · 끝 토큰으로 끝난 답 · 분야 ·
  대화마다), 두 결과의 전후 비교(기준 모델과 같아진 대화 · 새로 달라진 대화 — 부호 검정), 대화 하나의 답이 달라진 곳(두 답을 나란히, 처음
  갈라진 스텝에서 두 모델이 후보 토큰에 준 확률, 그 앞뒤 스텝)을 본다. 결과의 실행 로그는 디버거로 연다.

---

- **Model evaluation screen.** A new 'Model evaluation' entry in the left menu. Create a workspace (a reference model plus the GGUF
  versions to compare), run an evaluation from the screen, then read the result (a one-sentence verdict, KL divergence, top-1 agreement,
  the distribution of first divergent steps and the share that diverged at a low-margin step, answers that ended with the end token,
  categories, every conversation), a before/after comparison of two results (conversations that now match the reference and those that
  newly diverged — sign test), and where one conversation's answers differ (both answers side by side, the probabilities both models gave
  the candidate tokens at the first divergent step, and the steps around it). A result's run log opens in the debugger.
