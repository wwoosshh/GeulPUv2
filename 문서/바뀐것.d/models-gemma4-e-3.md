갈래: 더함

- **엔진이 Gemma 4 E2B · E4B 의 k-양자화 파일을 받는다**(0.24.0): 12B · 31B 의 블록에 더해 층마다의 입력(PLE), 뒤쪽 층들의 KV 공유, 층마다
  다른 FFN 폭(E2B)을 계약 4.12 의 7 ~ 9 대로 계산한다. 층을 줄인 시험 파일 둘(E2B · E4B 의 Q4_K_M 에서 다섯 층씩)로 엔진의 다섯 실행
  경로가 같은 비트이고 모든 행 로짓이 CPU 참조와 같은 것을 확인했다. 층마다의 임베딩 표(E2B 1.6 GB · E4B 1.9 GB)는 그래픽 메모리에
  둔다. 프롬프트 처리에서 KV 를 나눠 쓰는 층은 다음 토큰에 필요한 행만 계산한다(llama.cpp 와 같은 일 — 값은 같다). 실제 파일의 모델 검증과
  앱의 모델 목록은 다음 단계라, 앱의 모델 찾기는 아직 이 파일들을 '실행 가능'으로 표시하지 않는다. Q8_0 ·
  F16 · BF16 파일은 아직 받지 않는다. 기록: [구조/gemma4/기록.md](../../구조/gemma4/기록.md).

---

- **The engine takes the k-quant files of Gemma 4 E2B and E4B.** On top of the 12B · 31B block it computes the per-layer inputs
  (PLE), the KV sharing of the later layers and the per-layer FFN width (E2B) as items 7–9 of contract section 4.12 say.
  Checked with two layer-reduced test files (five layers each from the Q4_K_M files of E2B and E4B): five bit-identical execution
  paths, and all-row logits equal to the CPU reference. The per-layer embedding table (1.6 GB for E2B, 1.9 GB for E4B) is kept in
  graphics memory. In prompt processing the layers that share a KV compute only the rows needed for the next token (as llama.cpp
  does — the values are the same). Model verification with the full files and the app's model list come in the next step, so the app's model
  search does not show these files as runnable yet. Q8_0, F16 and BF16 files are not accepted yet. Record:
  [구조/gemma4/기록.md](../../구조/gemma4/기록.md).
