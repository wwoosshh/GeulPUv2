갈래: 더함

- **도구 호출을 강제하는 `tool_choice`**: `/v1/chat/completions` 가 `tool_choice` 의 `"required"`(아무 호출) · 함수 지정
  (`{"type": "function", "function": {"name": …}}`)을 받는다. 생성 머리 뒤(틀의 생각 끔 꼴 뒤)에 모델 틀의 호출 여는 글을 미리 넣어 지키므로
  같은 요청이면 같은 토큰이고, 이 칸을 쓰지 않는 요청은 그대로다. 도구가 하나면 `required` 는 그 이름까지 지키고, 여럿이면 호출만 지킨다
  (이름은 모델이 고른다). Qwen Code 의 보조 요청(세션 제목 · 요약 등)이 400 으로 막히던 것이 풀린다. MiniCPM 의 XML 꼴은 아직 받지 않는다(400).

---

- **`tool_choice` that forces a tool call.** `/v1/chat/completions` accepts `tool_choice: "required"` (any call) and a named
  function (`{"type": "function", "function": {"name": …}}`). The server enforces it by placing the model template's call opening after
  the generation prompt (after the template's thinking-off form), so the same request gives the same tokens and requests without this
  field are unchanged. With a single tool, `required` also fixes the name; with several tools only the call is enforced and the model
  picks the name. Qwen Code's side requests (session titles, summaries and others), which failed with 400, now work. MiniCPM's XML call
  form is not supported yet (400).
