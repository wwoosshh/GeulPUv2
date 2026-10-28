갈래: 더함

- **Anthropic Messages API(`/v1/messages` · `/v1/messages/count_tokens`)**: Claude Code 처럼 Anthropic 꼴로 말하는 에이전트를 붙일 수
  있다(`ANTHROPIC_BASE_URL=http://127.0.0.1:44544`). 요청은 같은 채팅 틀 · 같은 엔진으로 돌고(같은 요청 → 같은 토큰), 답은 Anthropic 꼴의
  content 블록(text · tool_use) · stop_reason · usage, 흘림은 message_start ~ message_stop 사건이다. 생각은 내지 않는다. 문맥을 넘으면
  `prompt is too long` 오류로 알려 에이전트가 대화를 줄인다.

---

- **Anthropic Messages API (`/v1/messages`, `/v1/messages/count_tokens`).** Agents that speak the Anthropic format, such as Claude
  Code, can now attach (`ANTHROPIC_BASE_URL=http://127.0.0.1:44544`). Requests run through the same chat template and engine (the same
  request gives the same tokens); replies use Anthropic content blocks (text, tool_use), stop_reason and usage, and streaming emits
  the message_start … message_stop events. Thinking is not returned. A prompt over the context limit is reported as
  `prompt is too long`, so the agent compacts the conversation.
