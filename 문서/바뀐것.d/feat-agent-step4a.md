갈래: 더함

- **도구 오류 감지(에이전트 호환 4 단계의 첫 조각)**: 에이전트가 보낸 대화에 실패한 도구 결과가 있으면 서버가 찾아 종류(일시 · 고침 · 해없음
  · 모호)를 가르고, 서버 로그에 한 줄(`[n] 도구 오류 감지 …`)과 답의 머리 칸 `X-GeulPU-Tool-Errors` 로 적는다. 실패는 확실한 표시만 쓴다 —
  Anthropic 의 `is_error`, Cline 의 `"success": false`, OpenCode 의 도구 오류 JSON, 그리고 User-Agent 로 고른 Qwen Code · Codex · Kilo ·
  OpenCode 의 오류 글. 모델에 가는 글은 바꾸지 않는다(답은 비트까지 그대로).

---

- **Tool-error detection (first piece of agent-compatibility step 4).** When a conversation from an agent contains a failed tool
  result, the server now finds it, classifies it (transient, fixable, unrecoverable or ambiguous) and records it in the server log
  (`[n] 도구 오류 감지 …`) and in an `X-GeulPU-Tool-Errors` response header. Only reliable signals are used: Anthropic `is_error`,
  Cline's `"success": false`, OpenCode's tool-error JSON, and the error text of Qwen Code, Codex, Kilo and OpenCode selected by
  User-Agent. The text sent to the model is unchanged (replies are bit-identical).
