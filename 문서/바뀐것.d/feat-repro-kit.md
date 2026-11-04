갈래: 더함

- **재현 키트(`repro-kit/`)**: GeulPU 0.21.0 을 돌리는 다른 기계가 기준 PC 와 같은 비트를 내는지 명령 하나로 본다 — Qwen3-8B Q8_0 ·
  Qwen3.5-9B Q4_K_M 을 기본 설정과 정밀 설정으로 불러, 정해 둔 요청 12 개를 하나씩 · 한꺼번에 보내고 프롬프트 토큰 · 생성 토큰 · 걸음마다의
  로짓 해시를 기준값(RTX 4070 Ti)과 견준다. 앱의 재현성 검증도 함께 돌릴 수 있고, 결과 파일을 남긴다. 파이썬 표준 라이브러리만 쓴다
  ([README.ko.md](../../repro-kit/README.ko.md)).

---

- **Reproducibility kit (`repro-kit/`).** One command checks whether another machine running GeulPU 0.21.0 gives the same bits as
  the reference PC: it loads Qwen3-8B Q8_0 and Qwen3.5-9B Q4_K_M with the default and the exact settings, sends 12 fixed requests one
  at a time and all at once, and compares the prompt tokens, the generated tokens and every step's logit hash with the reference
  values (RTX 4070 Ti). It can also run the app's reproducibility check and writes a result file to share. Standard-library Python
  only ([README.md](../../repro-kit/README.md)).
