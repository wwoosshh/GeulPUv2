갈래: 더함

- **Gemma 4 E2B · E4B 가 실행되고 앱의 모델 목록에 들었다**(0.24.0): 12B · 31B 의 블록에 더해 층마다의 입력(PLE), 뒤쪽 층들의 KV 공유,
  층마다 다른 FFN 폭(E2B)을 계약 4.12 의 7 ~ 9 대로 계산한다. unsloth 의 Q4_K_M 파일 둘로 다섯 실행 경로가 같은 비트인 것과 앱의 채팅 ·
  도구 호출을 확인했고, E2B 는 전체 파일의 모든 행 로짓이 CPU 참조와 같다. 층마다의 임베딩 표(E2B 1.6 GB · E4B 1.9 GB)는 그래픽 메모리에
  둔다. 프롬프트 처리에서 KV 를 나눠 쓰는 층은 다음 토큰에 필요한 행만 계산한다(llama.cpp 와 같은 일 — 값은 같다). 속도는 llama.cpp 의
  약 0.7 ~ 1.03 배다(생성 약 0.8 배). 앱의 모델 찾기도 이 두 크기의 k-양자화 파일을 '실행 가능'으로 표시한다 — Q8_0 · F16 · BF16 파일은
  아직 받지 않는다. 기록: [구조/gemma4/기록.md](../../구조/gemma4/기록.md).

---

- **Gemma 4 E2B and E4B run and are in the app's model list.** On top of the 12B · 31B block the engine computes the per-layer
  inputs (PLE), the KV sharing of the later layers and the per-layer FFN width (E2B) as items 7–9 of contract section 4.12 say.
  Checked with unsloth's two Q4_K_M files: five bit-identical execution paths and chat and tool calls in the app; for E2B the
  all-row logits of the full file equal the CPU reference. The per-layer embedding table (1.6 GB for E2B, 1.9 GB for E4B) is kept
  in graphics memory. In prompt processing the layers that share a KV compute only the rows needed for the next token (as
  llama.cpp does — the values are the same). Speed is about 0.7–1.03 times llama.cpp's (decoding about 0.8 times). The app's
  model search now shows the k-quant files of these two sizes as runnable — Q8_0, F16 and BF16 files are not accepted yet.
  Record: [구조/gemma4/기록.md](../../구조/gemma4/기록.md).
