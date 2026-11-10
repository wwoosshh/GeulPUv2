갈래: 더함

- **도구 호출 마커가 일반 텍스트인 두 형식을 읽는다**: 어휘에 `<tool_call>` 특수 토큰이 없어 호출을 텍스트로 쓰는 템플릿(KT Mi:dm 2.0 —
  llama.cpp 의 Hermes 2 Pro 형식)과 functionary v3.1 의 `<function=이름>{JSON}</function>`(카카오 Kanana 1.5). 전에는 앞의 것은 도구가
  든 요청을 400 으로 거절했고, 뒤의 것은 Llama 3 JSON 으로 잘못 골라 호출을 읽지 못했다. 마커는 생성한 텍스트에서 찾고, 스트리밍은 여는
  마커의 앞부분일 수 있는 끝부분을 마커인지 정해질 때까지 보내지 않는다. 호출 안의 특수 토큰을 빼는 규칙(0.15.0)과 tool_choice 의
  required · 함수 지정도 같다. 이전 호출의 인자를 객체로 그리면 오류가 나는 템플릿(Kanana 1.5)은 인자를 문자열 그대로 넘긴다 — 템플릿을
  시험 삼아 그려 보고 정한다(llama.cpp 와 같은 방식). 지금 받는 모델의 형식은 그대로다.

---

- **Tool calls whose markers are plain text.** Templates that write `<tool_call>` as text because the vocabulary has no such special
  token (KT's Mi:dm 2.0 — llama.cpp's Hermes 2 Pro form), and functionary v3.1's `<function=name>{JSON}</function>` (Kakao's
  Kanana 1.5). Before, the first kind answered requests with tools with 400 and the second was taken for Llama 3 JSON, so its calls
  were not read. The markers are found in the generated text, and streaming holds back a trailing piece that could start an opening
  marker until it is decided. The in-call special-token rule (0.15.0) and tool_choice required / a named function work the same way.
  Templates that fail when an earlier call's arguments are an object (Kanana 1.5) get the arguments as the original string — decided
  by rendering a probe conversation, as llama.cpp does. Models GeulPU already accepts keep their formats.
