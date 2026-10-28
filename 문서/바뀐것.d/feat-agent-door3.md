갈래: 더함

- **OpenAI Responses API(`/v1/responses`, 상태 없이)**: Responses API 만 쓰는 Codex CLI 를 붙일 수 있다(설정의 provider 에
  `base_url = "http://127.0.0.1:44544/v1"` · `wire_api = "responses"`). 요청은 같은 채팅 틀 · 같은 엔진으로 돌고(같은 요청 → 같은 토큰),
  답은 output 항목(message · function_call) · usage, 흘림은 response.created ~ response.completed 사건이다. web_search 같은 서버 도구는
  빼고, `previous_response_id`(서버에 남긴 대화)는 받지 않는다(400 — `store: false` 로 대화를 모두 보낸다). 생각은 내지 않는다.

---

- **OpenAI Responses API (`/v1/responses`, stateless).** Codex CLI, which only speaks the Responses API, can now attach (a provider with
  `base_url = "http://127.0.0.1:44544/v1"` and `wire_api = "responses"`). Requests run through the same chat template and engine (the
  same request gives the same tokens); replies carry output items (message, function_call) and usage, and streaming emits the
  response.created … response.completed events. Hosted tools such as web_search are dropped, and `previous_response_id` (server-stored
  conversations) is rejected with 400 — send the whole conversation with `store: false`. Thinking is not returned.
