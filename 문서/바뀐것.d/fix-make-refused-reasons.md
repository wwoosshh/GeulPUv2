갈래: 고침

- **엔진이 받지 않는 GGUF 의 까닭을 바로 알림**: 'GGUF 만들기'가 엔진이 받지 않는 파일을 만들면 작업 요약에 그 형식과 까닭이 나온다(전에는
  작업 기록 줄에만). 너비 · FFN 이 256 의 배수가 아닌 작은 모델은 k-양자화가 llama-quantize 의 대체 형식(Q5_0 · Q5_1 · IQ4_NL)으로 바뀐다는
  것을 양자화 앞에 알리고 Q8_0 · Q4_0 을 권한다. 만들기에 Q4_0 을 더했다(낱말표 · 출력도 Q4_0 — 엔진의 Q4_0 길). 엔진의 알림은 형식을
  번호 대신 이름으로 보이고, '지원하지 않는 모델 크기'는 걸린 조건 하나를 말한다(SmolLM2-360M 은 '헤드 수 15가 홀수' — 전에는 '헤드 크기
  64 · 96 · 128만'으로 보였다)(#215).

---

- **Why the engine refuses a GGUF is shown right away.** When Make GGUF produces files the engine does not support, the job
  summary now names those types and the reason (before, only a run-log line did). For small models whose hidden size or FFN
  width is not a multiple of 256, Make GGUF warns before quantizing that llama-quantize turns k-quants into fallback types
  (Q5_0 · Q5_1 · IQ4_NL) and suggests Q8_0 · Q4_0. Q4_0 joins the quantization types (token embeddings and output in Q4_0
  too — the engine's Q4_0 path). The engine's messages name formats instead of numbers, and "unsupported model size" now
  states the one condition that failed (SmolLM2-360M: the head count 15 is odd — before, it read "only head size 64 · 96 ·
  128") (#215).
