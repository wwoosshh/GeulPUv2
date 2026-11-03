갈래: 고침

- **평가의 요청이 실패한 까닭을 남김**: 추론 서버가 요청을 거절하면(예: 채팅 틀의 도구 호출 꼴을 GeulPU 가 읽지 못해 `tools` 가 든
  대화를 400 으로) 실행 로그(run-*.jsonl)의 그 줄에 서버가 돌려준 본문이 `error` 로 남고, 작업 기록과 요약에 처음 까닭이 나온다 — 전에는
  `"response": null` 뿐이었다(#218). 버전이 답을 하나도 받지 못하면 결과를 만들지 않고 까닭을 알린다.

---

- **Why an evaluation request failed is kept.** When the inference server refuses a request (for example, conversations with
  `tools` rejected with 400 because GeulPU does not read the chat template's tool-call format), that line of the run log
  (run-*.jsonl) keeps the body the server returned as `error`, and the run log and summary show the first reason — before,
  there was only `"response": null` (#218). A version that gets no answer at all no longer produces a result; it reports why.
