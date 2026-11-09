갈래: 더함

- **스텝 · 층 화면**: 실행 화면에서 스텝을 고르고 Enter(또는 '스텝 · 층 화면')로 그 스텝 하나를 자세히 봅니다 — 상위 후보(1 ~ 128 개, 후보마다
  분기), 로짓 렌즈 표(층마다 1 위 · 2 위 · 3 위와 샘플링된 토큰의 순위, 마지막 층의 1 위가 끝까지 1 위로 남는 첫 층 표시 — 모델 옆에 J 파일이
  있으면 Jacobian lens 로 바꿔 보기), 그 스텝부터 이어질 텍스트의 점수. 화면의 계산은 모두 실험의 실행으로 남아 다시 열면 그대로 보이고, 점수 ·
  로짓 렌즈 실행의 실행 화면은 그 표를 보입니다. 점수 · 렌즈 실행이 스텝부터 다른 토큰을 넣었으면 앱이 원래 실행과 개입을 적고, 개입의 스텝
  번호는 화면의 번호(분기는 원래 실행의 번호를 잇는다)와 같습니다. 기능 목록의 '층 렌즈'는 '로짓 렌즈'로 바꿨습니다([화면구현.md](../화면구현.md)
  §4).

---

- **Step and layer screen.** Pick a step on the run screen and press Enter (or "Step and layer screen") to look at that one step: the
  top candidates (1 to 128, each can be branched), the logit lens table (top three per layer and the sampled token's rank, with the
  first layer from which the last layer's top token stays first marked — switch to the Jacobian lens when a J file sits next to the
  model), and the score of a text from that step on. Every computation on the screen is recorded as a run of the experiment, so it is
  there again when the screen opens, and score and logit-lens runs show their tables on the run screen. When a score or lens puts other
  tokens on from a step, the app records the base run and the intervention; intervention steps use the screens' numbering (a branch
  continues its base run's numbers). The registry's "layer lens" wording is now "logit lens" ([화면구현.md](../화면구현.md) §4).
