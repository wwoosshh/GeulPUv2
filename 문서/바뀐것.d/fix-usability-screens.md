갈래: 고침

- **분기 · 실행 목록 · 이동 · 모델 더하기를 알기 쉽게**: 분기 실행의 설명이 '앞 4 토큰 + 새 28 토큰 = 32 토큰 · 스텝 4 부터 원래 난수로
  이음'처럼 받은 토큰과 새로 만든 토큰, 원래 난수로 이은 위치를 보입니다. 실행 목록의 생성 실행에 프롬프트의 앞부분이 보이고, 결과 탭에서도
  실험을 점검합니다. 사이드바의 '실험'은 늘 실험 목록으로 갑니다. GGUF 파일 더하기는 .gguf 파일 경로를 바로 받고, 더한 뒤 '내 모델'을
  보입니다. 비교 탭은 가장 최근 생성 실행을 그 base(없으면 첫 생성 실행)와 비교하고, 재현 파일 단추의 이름이 '재현 파일(JSON)'입니다
  ([사용성시험.md](../사용성시험.md) §3).

---

- **Clearer branches, run list, navigation and model adding.** A branch run's description shows the kept and the new tokens and where
  the original random numbers continue ("4 kept + 28 new = 32 tokens · original random numbers from step 4"). Generation runs in the
  run list show the start of their prompt, and the Results tab can check the experiment too. The sidebar's Experiments always opens the
  experiment list. Adding a GGUF file takes a .gguf file path directly and shows My models afterwards. The compare tab pairs the newest
  generation run with its base (or the first generation run), and the reproduction-file button reads "Reproduction file (JSON)"
  ([사용성시험.md](../사용성시험.md) §3).
