갈래: 더함

- **앱에서 코딩 에이전트 띄우기**: 새 '에이전트' 화면에서 Claude Code · Codex CLI · Qwen Code · OpenCode · Aider 가운데 하나와 작업 폴더를
  고르고 시작을 누르면, 앱이 새 터미널 창에서 그 에이전트를 지금 불러 둔 모델에 붙여 띄운다. 설정(환경 변수 · 명령줄)은 그 실행에만
  넘기고 에이전트의 설정 파일과 환경 변수는 바꾸지 않으며, 파일을 고치거나 명령을 실행하기 전의 승인 묻기는 에이전트의 것 그대로다. 설치
  여부(PATH)와 설치 명령, 직접 실행할 명령(복사), VS Code 확장(Cline · Kilo Code)에 넣을 값, 문맥 · 동시 처리 경고('에이전트에 맞게 다시
  불러오기'), 최근 감지한 도구 오류를 보여 준다. 설정에 '도구 오류의 꼴'(끔 · C1 · C2 · C3 — 기본 끔)이 생겼고, Claude Code 가 찌르는
  `HEAD /api/hello` 에 200 으로 답한다. 리눅스에서는 아직 창을 띄우지 않고 명령만 보여 준다.

---

- **Start coding agents from the app.** On the new Agents page, pick Claude Code, Codex CLI, Qwen Code, OpenCode or Aider and a working
  folder and press Start: the app opens a new terminal window with that agent attached to the loaded model. The settings (environment
  variables and command-line options) are passed to that run only — the agent's configuration files and your environment variables are
  not changed — and the agent still asks before it edits files or runs commands. The page shows whether each agent is installed (PATH)
  with its install command, the commands to run it yourself (copy), the values for the Cline and Kilo Code VS Code extensions, context
  and concurrency warnings ("Reload for agents") and recently detected tool errors. Settings gain "Tool-error form" (off, C1, C2, C3 —
  off by default), and `HEAD /api/hello`, which Claude Code probes, now returns 200. On Linux the app shows the commands but does not open
  a window yet.
