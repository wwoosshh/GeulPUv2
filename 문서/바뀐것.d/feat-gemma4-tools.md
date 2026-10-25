갈래: 더함

- **Gemma 4 의 도구 호출 · 생각 묶음**: Gemma 4 의 채팅 틀이 쓰는 도구 호출(`<|tool_call>call:이름{열쇠:값,…}<tool_call|>`)을 받아
  OpenAI 의 `tool_calls` 로 낸다 — 인자는 llama.cpp 와 같은 규칙으로 JSON(글 `<|"|>…<|"|>` · 수 · true · false · null · 객체 · 배열).
  생각 묶음 `<|channel>thought … <channel|>` 은 도구를 보낸 요청이면 `reasoning_content` 로, 아니면 다른 모델처럼 답 앞의 글로 온다(채널
  표지 · 이름은 떼고, 생각을 끈 답 끝에 붙는 빈 채널도 뗀다 — `geulpu.special` 이면 표지까지 글로). 앱의 '생각 과정 쓰기'가 Gemma 4 에도
  듣고, 켜면 생각을 접어 보인다.

---

- **Tool calls and thinking for Gemma 4.** Gemma 4's own tool-call format (`<|tool_call>call:name{key:value,…}<tool_call|>`) is now
  read and returned as OpenAI `tool_calls`, with the arguments turned into JSON by the same rules as llama.cpp (strings `<|"|>…<|"|>`,
  numbers, true, false, null, objects and arrays). Its thinking block `<|channel>thought … <channel|>` comes as `reasoning_content` in
  requests that send tools, and otherwise in front of the answer like other models (the channel markers and name are removed, as is the
  empty channel that may follow an answer with thinking off — with `geulpu.special` the markers come as text). The app's 'Thinking'
  switch now works for Gemma 4 and folds the thinking away when it is on.
