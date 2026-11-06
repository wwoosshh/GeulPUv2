갈래: 바꿈

- **llama.cpp 재현 검증 — 기록기가 기본으로 투명해짐**: 기록기가 요청을 바이트 그대로 llama-server 에 넘기고, 실행 중에는 서버에
  아무것도 묻지 않으며(`/slots` · `/props` · `/tokenize` 없음, 모델 파일도 읽지 않음), 토큰 없는 영수증(`geulpu-llama-receipt/2`)을
  남깁니다. 설정에 llama-server 의 실행 파일을 적으면 저장할 때 `--version` 으로 빌드를 읽어 둡니다(따로 도는 프로세스). 지금까지의
  기록기(요청에 확률 · 토큰 칸을 더하고 슬롯 상태를 묻는 것 — 토큰 있는 영수증)는 "개입" 방식으로 골라 쓸 수 있고, 무엇을 바꾸는지
  알립니다([라마재현.md](../라마재현.md) 31단계).

---

- **llama.cpp replay check — the recorder is transparent by default.** The recorder now forwards requests to llama-server byte for
  byte, never queries the server during a run (no `/slots` · `/props` · `/tokenize`, and it does not read the model file), and
  writes token-less receipts (`geulpu-llama-receipt/2`). With the llama-server executable in the settings, the build is read with
  `--version` on save (a separate process). The previous recorder (adding probability and token fields to requests and querying
  slot state — receipts with tokens) remains available as the "probing" mode, which states what it changes
  ([문서/라마재현.md](../라마재현.md), stage 31).
