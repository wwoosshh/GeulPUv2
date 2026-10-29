갈래: 고침

- **도구 호출이 중간에 끊기지 않는다.** Qwen 꼴의 도구 호출(`<tool_call>` ~ `</tool_call>`)을 쓰는 동안 모델이 `<|im_start|>` 같은
  특수 토큰을 고르면 호출이 끊겨 글로 돌아갔고, 에이전트는 그 글을 끝 답으로 받아 멈췄다(Qwen3-32B 가 경로의 드문 한글 자리에서). 이제
  호출 안에서는 그런 토큰을 빼고 다음 후보를 고른다(llama.cpp 의 도구 호출 문법과 같은 효과). **출력 바뀜** — 호출 안에서 특수 토큰이
  골라지던 요청만, 그 밖의 답은 비트까지 그대로.

---

- **Tool calls are no longer cut off midway.** While the model wrote a Qwen-style tool call (`<tool_call>` … `</tool_call>`), picking a
  special token such as `<|im_start|>` cut the call, the reply came back as plain text, and the agent took it as its final answer and
  stopped (Qwen3-32B, at a rare Hangul character in a path). Inside a call such tokens are now skipped and the next candidate is used,
  the same effect as llama.cpp's tool-call grammar. **Output change** — only for requests where a special token was picked inside a call;
  every other reply stays the same bit for bit.
