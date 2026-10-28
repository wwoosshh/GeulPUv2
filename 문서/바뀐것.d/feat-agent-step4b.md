갈래: 더함

- **도구 오류의 꼴(에이전트 호환 4 단계의 둘째 조각 — 기본은 끔)**: 요청 칸 `geulpu_tool_errors`(`"off"` · `"c1"` · `"c2"` · `"c3"`)나
  서버 옵션 `--도구오류 c1|c2|c3` 으로, 에이전트가 방금 받은 실패한 도구 결과(일시 · 고침 · 해없음으로 가른 것)를 연구에서 잰 꼴로 바꿔
  모델에 보인다 — C1 은 결과 끝에 안내 한 문장, C2 는 실패한 호출 · 결과를 지우고 실행기 알림, C3 은 결과를 두고 다음 사용자 차례에
  알림. 글은 연구의 것을 바이트까지 그대로 쓴다. 세 API(OpenAI 채팅 · Anthropic · Responses) 모두 같고, 머리 칸
  `X-GeulPU-Tool-Errors` 에 고른 꼴과 바꾼 수를 적는다. 끄면 답은 비트까지 그대로다. 앱이 에이전트의 User-Agent 를 추론 서버로 넘겨,
  앱 포트로 붙은 에이전트에도 에이전트마다의 오류 글 규칙이 듣는다.

---

- **Tool-error forms (second piece of agent-compatibility step 4 — off by default).** With the request field
  `geulpu_tool_errors` (`"off"`, `"c1"`, `"c2"`, `"c3"`) or the server option `--도구오류 c1|c2|c3`, a failed tool result the agent
  just received (classified as transient, fixable or unrecoverable) is shown to the model in one of the forms measured in the
  research — C1 appends one guidance sentence to the result, C2 removes the failed call and result and adds a runtime notice, C3
  keeps the result and adds the notice in a following user turn. The text is the research's, byte for byte. All three APIs (OpenAI
  chat, Anthropic, Responses) behave the same, and the `X-GeulPU-Tool-Errors` header reports the form and how many failures were
  changed. When off, replies are bit-identical. The app now passes the agent's User-Agent to the inference server, so the
  per-agent error-text rules also apply to agents connected through the app's port.
