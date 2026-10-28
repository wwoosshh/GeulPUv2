갈래: 고침

- **중괄호가 겹친 Qwen 도구 호출도 읽는다.** GGUF 의 채팅 틀이 호출 예시를 `{{"name": …}}` 처럼 두 겹으로 적으면(Qwen2.5-7B-Instruct
  의 Q8_0 GGUF) 모델이 그대로 따라 써서, 지금까지는 도구 호출이 아닌 글로 돌려줬다. 이제 겹친 중괄호를 건너 호출로 읽는다(llama.cpp 와 같다).

---

- **Qwen tool calls with doubled braces are parsed.** When a GGUF chat template shows the tool-call example as `{{"name": …}}`
  (the Qwen2.5-7B-Instruct Q8_0 GGUF), the model copies the doubled braces and the reply came back as plain text instead of a tool call.
  The extra braces are now skipped and the call is parsed, as llama.cpp does.
