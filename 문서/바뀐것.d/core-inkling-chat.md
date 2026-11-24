갈래: 더함

- **요청의 `reasoning_effort` 를 채팅 템플릿에 넘깁니다.** OpenAI 의 `reasoning_effort`(텍스트)가 템플릿의 같은 이름 변수가 됩니다 —
  `chat_template_kwargs` 의 같은 키보다 앞섭니다(llama.cpp 와 같다). 그 변수를 쓰지 않는 템플릿의 프롬프트는 그대로입니다. 템플릿 해석기는
  `tojson(sort_keys=true)` 도 받습니다 — 전에는 이것을 쓰는 템플릿을 오류로 냈습니다.

---

- **The request's `reasoning_effort` now reaches the chat template.** OpenAI's `reasoning_effort` (a string) becomes the template variable of
  the same name and takes precedence over the same key in `chat_template_kwargs` (as in llama.cpp). Templates that do not use it render the
  same prompt. The template interpreter also accepts `tojson(sort_keys=true)` — templates using it used to fail.
