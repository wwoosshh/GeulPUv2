갈래: 더함

- **GLM-4.5 · 4.7 의 도구 호출 형식을 읽는다**: `<tool_call>이름<arg_key>키</arg_key><arg_value>값</arg_value>…</tool_call>`(마커가 모두
  특수 토큰). 인자의 값은 도구 정의의 type 으로 읽는다(string 이면 그대로, 아니면 JSON). 스트리밍 · tool_choice 의 required · 함수
  지정도 다른 형식과 같다. 그리고 엔진의 모델 확인이 MoE 의 전문가 수 · 전문가 가중치의 크기를 알려, 앱이 규칙을 모르는 구조(deepseek2)의
  MoE 파일도 고정 메모리를 어림하고 정밀 모드만이라고 보인다. 모델 찾기의 메모리 어림도 엔진이 MoE 라고 한 구조면 전문가를 RAM 으로 본다.

---

- **Tool calls in the GLM-4.5 · 4.7 format.** `<tool_call>name<arg_key>key</arg_key><arg_value>value</arg_value>…</tool_call>`, where
  every marker is a special token. Argument values are read by the type in the tool definition (as is for string, otherwise as JSON).
  Streaming and tool_choice required / a named function work as in the other formats. The engine's model check now reports the MoE
  expert count and expert-weight size, so the app estimates pinned memory and shows exact mode only for MoE files of architectures whose
  rules it does not know (deepseek2); the model search's memory estimate treats experts as RAM for every architecture the engine marks
  as MoE.
