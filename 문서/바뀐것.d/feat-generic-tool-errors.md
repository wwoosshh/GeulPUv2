갈래: 더함

- **도구 오류 보정이 다른 클라이언트에도 듣는다.** 정해진 에이전트(Qwen Code · Codex · OpenCode · Kilo · Cline) 밖의 클라이언트(직접 만든
  에이전트 · LangChain · OpenAI Agents SDK · MCP 클라이언트)가 보낸 도구 결과도 표준 오류 글(Python 예외 · HTTP 상태 · MCP · API 오류
  JSON)로 시작하면 실패로 찾고, 시간 초과 · 잘못된 인자 · 없는 도구 · 권한 없음을 가른다. tool 메시지의 `is_error` 와 MCP 의 `isError` 도
  읽는다. 정해진 에이전트의 동작은 그대로다.

---

- **Tool-error correction now covers other clients.** Tool results from clients outside the known agents (Qwen Code · Codex · OpenCode ·
  Kilo · Cline) — custom agents, LangChain, the OpenAI Agents SDK, MCP clients — are detected when they start with a standard error text
  (Python exceptions, HTTP status, MCP, API error JSON) and classified as timeouts, wrong arguments, unknown tools or missing permissions.
  The `is_error` field on tool messages and MCP `isError` are read too. Known agents behave as before.
