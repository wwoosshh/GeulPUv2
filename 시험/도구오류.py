#!/usr/bin/env python3
"""에이전트 호환 4 단계 — 도구 오류 층(엔진/도구오류.gl — 엔진/시험_도구오류.gl)을 GPU 없이 본다.

4a 감지: 사례는 3 단계 녹화(에이전트 일곱이 GeulPU 에 실제로 보낸 도구 결과)의 글 그대로이고, 손으로 단 답(메시지 번호 · 종류 · 규칙)과
견준다. 실패가 아닌 결과를 실패로 잡지 않는지(성공한 셸 · 출력 안의 'Error:' 줄 · 모르는 에이전트 · 표시 없는 같은 글)도 본다.
4b 꼴: 결정 스텝의 실패를 C1 · C2 · C3 로 바꾼 messages 의 JSON 글이 참조(아래 참조꼴 — errmsg2 세션.py 의 꼴 · 붙이기 · 글뽑기 ·
알림글과 자극.인자글을 그대로 옮긴 것, 실패가 여럿이면 차례로)의 json.dumps 글과 글자까지 같은가.

  python -I 시험/도구오류.py <시험_도구오류 실행 파일>

끝 줄 "도구 오류 감지: 모두 맞음 (N 가지)"이면 종료 0, 아니면 1.
"""
import copy
import json
import os
import subprocess
import sys
import tempfile

sys.stdout.reconfigure(encoding="utf-8")

QWEN = "QwenCode/0.26.2 (win32; x64)"
OPENCODE = "opencode/latest/2.2.0/cli"
KILO = "Kilo-Code/7.8.18 ai-sdk/provider-utils/4.0.27 runtime/bun/1.4.4"
CLINE = "ai-sdk-openai-compatible/3.0.67 ai-sdk-provider-utils/5.0.58 bun/1.4.2"
CODEX = "codex_exec/0.165.0 (Windows 10.0.26200; x86_64) xterm-256color (codex_exec; 0.165.0)"
CLAUDE = "claude-cli/2.1.270 (external, claude-desktop, agent-sdk/0.3.308)"
HTTPX = "python-httpx/0.27.2"


def 대화(*결과들):
    """사용자 하나 뒤에 (assistant 호출 · tool 결과) 쌍들 — tool 메시지는 2 · 4 · 6 … 번. 결과는 글(content)이거나 tool 메시지의 칸들"""
    ms = [{"role": "user", "content": "do the task"}]
    for i, r in enumerate(결과들):
        cid = f"call_{i}"
        ms.append({"role": "assistant", "content": "", "tool_calls": [
            {"id": cid, "type": "function", "function": {"name": "tool", "arguments": "{}"}}]})
        m = {"role": "tool", "tool_call_id": cid}
        m.update(r if isinstance(r, dict) else {"content": r})
        ms.append(m)
    return ms


# (아이디, User-Agent, messages, 서버 로그의 글 — 없으면 "", 머리 칸 값 — 보지 않으면 None)
사례들 = [
    # Qwen Code — 글 규칙(User-Agent 로 고름)
    ("qwen 없는 파일", QWEN, 대화(r"File not found: E:\geulpu-work\agents\work\qwen\notes.txt"), "#2 모호(qwen-notfound)", None),
    ("qwen 셸 종료 1", QWEN, 대화("Command: cd E:\\geulpu-work\\agents\\work\\qwen && python fib.py\nDirectory: (root)\nOutput: /usr/bin/bash: "
                                 "line 1: cd: E:geulpu-workagentsworkqwen: No such file or directory\nError: (none)\nExit Code: 1\n"
                                 "Signal: (none)\nProcess Group PGID: (none)"), "#2 모호(qwen-exit)", None),
    ("qwen 셸 성공", QWEN, 대화("Command: python fib.py\nDirectory: (root)\nOutput: 0\n1\n1\nError: (none)\nExit Code: 0\nSignal: (none)\n"
                               "Process Group PGID: (none)"), "", None),
    ("qwen 셸 출력 안의 Error: 줄", QWEN, 대화("Command: python t.py\nDirectory: (root)\nOutput: Error: not really\ndone\nError: (none)\n"
                                              "Exit Code: 0\nSignal: (none)\nProcess Group PGID: (none)"), "", None),
    ("qwen 셸 실행 못 함", QWEN, 대화("Command: nosuchcmd\nDirectory: (root)\nOutput: (empty)\nError: spawn nosuchcmd ENOENT\n"
                                     "Exit Code: (none)\nSignal: (none)\nProcess Group PGID: (none)"), "#2 모호(qwen-exit)", None),
    ("qwen 읽기 전 편집", QWEN, 대화(r"File E:\geulpu-work\agents\work\qwen\locked.txt has not been read in this session. Use the "
                                    "read_file tool first to load the current content before editing it"), "#2 고침(qwen-error)", None),
    ("qwen 권한 없음", QWEN, 대화(r"Error reading file E:\x\secret.txt: EACCES: permission denied, open 'E:\x\secret.txt'"),
     "#2 해없음(qwen-error)", "1; 2=unrecoverable:qwen-error"),
    ("qwen 성공한 읽기 안의 낱말", QWEN, 대화("line one\nFile not found: is just text here\n"), "", None),
    ("qwen 조각 배열 내용", QWEN, 대화({"content": [{"type": "text", "text": "File not found: E:\\x\\notes.txt"}]}),
     "#2 모호(qwen-notfound)", None),
    ("모르는 에이전트 — 글 규칙은 쓰지 않음", None, 대화(r"File not found: E:\x\notes.txt"), "", None),
    ("사용자 글의 낱말은 보지 않음", QWEN, [{"role": "user", "content": "File not found: EACCES permission denied"}], "", None),
    # OpenCode — 명시 표시(도구 오류 JSON) · 셸의 끝 줄
    ("opencode 없는 파일", OPENCODE, 대화('{"error":{"type":"tool.execution","message":"File not found: notes.txt"},"content":[]}'),
     "#2 모호(opencode)", None),
    ("opencode 쓸 수 없음", OPENCODE, 대화('{"error":{"type":"tool.execution","message":"Unable to write locked.txt"},"content":[]}'),
     "#2 해없음(opencode)", None),
    ("opencode 빈 oldString", OPENCODE, 대화('{"error":{"type":"tool.execution","message":"oldString must not be empty. Use write to '
                                           'create or overwrite a file."},"content":[]}'), "#2 고침(opencode)", None),
    ("opencode 셸 종료 1", OPENCODE, 대화("checking config...\r\nerror: config.ini not found\r\n\nExited with code 1"),
     "#2 모호(opencode-exit)", None),
    ("opencode 셸 성공 출력", OPENCODE, 대화("0\r\n1\r\n1\r\n2\r\n3\r\n5\r\n8\r\n13\r\n21\r\n34\r\n"), "", None),
    ("opencode 파일 만듦", OPENCODE, 대화("Created file successfully: fib.py"), "", None),
    ("opencode 오류 JSON 은 에이전트를 몰라도", None, 대화('{"error":{"type":"tool.execution","message":"File not found: notes.txt"},'
                                                    '"content":[]}'), "#2 모호(opencode)", None),
    # Kilo — 글 규칙
    ("kilo 없는 파일", KILO, 대화(r"File not found: E:\geulpu-work\agents\work\kilo\notes.txt"), "#2 모호(kilo-notfound)", None),
    ("kilo 쓰기 실패", KILO, 대화(r"Unknown: FileSystem.writeFile (E:\geulpu-work\agents\work\kilo\locked.txt)"), "#2 모호(kilo-fs)", None),
    ("kilo 빈 oldString", KILO, 대화("oldString cannot be empty when editing an existing file. Provide the exact text to replace, or use "
                                    "write for an intentional full-file replacement."), "#2 고침(kilo-usage)", None),
    ("kilo 쓰기 성공", KILO, 대화("Wrote file successfully."), "", None),
    ("kilo 셸 실패는 종료 코드가 없어 못 잡음(한계)", KILO, 대화("C:\\Python312\\python.exe: can't open file 'E:\\\\x\\\\fib.py': [Errno 2] "
                                                         "No such file or directory\r\n"), "", None),
    # Cline — 명시 표시(결과 JSON 의 success:false — User-Agent 는 SDK 의 것이라 꼴로 알아본다)
    ("cline 없는 파일", CLINE, 대화('[{"query":"E:\\\\x\\\\notes.txt","result":"","error":"Error reading file: ENOENT: no such file or '
                                   'directory, stat \'E:\\\\x\\\\notes.txt\'","success":false}]'), "#2 모호(cline)", None),
    ("cline 명령 종료 1", CLINE, 대화('[{"query":"python check.py","result":"[Command exited with code 1]\\nchecking config...\\r\\n\\n'
                                     '[stderr]\\nerror: config.ini not found\\r\\n","error":"Command exited with code 1","success":false}]'),
     "#2 모호(cline)", None),
    ("cline 바꿀 글 없음", CLINE, 대화('{"query":"edit:E:\\\\x\\\\check.py","result":"","error":"Editor operation failed: No replacement '
                                      'performed: text not found in E:\\\\x\\\\check.py.","success":false}'), "#2 고침(cline)", None),
    ("cline insert_line", CLINE, 대화('{"query":"insert:E:\\\\x\\\\locked.txt","result":"","error":"Editor operation failed: Invalid '
                                     'insert_line: 9. insert_line must be a positive one-based boundary line in the range 1-3. Use 3 to '
                                     'append at EOF.","success":false}'), "#2 고침(cline)", None),
    ("cline 성공", CLINE, 대화('[{"query":"python fib.py","result":"0\\r\\n1\\r\\n1\\r\\n2\\r\\n","success":true}]'), "", None),
    ("cline 일시(503)", CLINE, 대화('[{"query":"fetch","result":"","error":"Request failed: 503 Service Unavailable","success":false}]'),
     "#2 일시(cline)", None),
    ("cline 종류는 error 칸부터(결과 글의 낱말은 보지 않음)", CLINE,
     대화('[{"query":"x","result":"permission denied is only file text","error":"Command exited with code 2","success":false}]'),
     "#2 모호(cline)", None),
    ("cline 꼴은 에이전트를 몰라도", None, 대화('[{"query":"x","result":"","error":"Error reading file: ENOENT","success":false}]'),
     "#2 모호(cline)", None),
    # Claude Code — 명시 표시(Anthropic is_error → 문 2 가 geulpu_error 로)
    ("claude is_error 종료 2", CLAUDE, 대화({"content": "Exit code 2\nC:\\Python312\\python.exe: can't open file 'E:\\\\x\\\\fib.py': "
                                                     "[Errno 2] No such file or directory", "geulpu_error": True}),
     "#2 모호(is_error)", "1; 2=ambiguous:is_error"),
    ("claude 같은 글 · 표시 없음", CLAUDE, 대화("Exit code 2\nsomething failed"), "", None),
    ("claude is_error 권한", CLAUDE, 대화({"content": "EACCES: permission denied, open 'E:\\x\\secret.txt'", "geulpu_error": True}),
     "#2 해없음(is_error)", None),
    ("claude is_error 연결 거절(일시)", CLAUDE, 대화({"content": "connect ECONNREFUSED 127.0.0.1:8080", "geulpu_error": True}),
     "#2 일시(is_error)", None),
    ("claude is_error 거짓", CLAUDE, 대화({"content": "Exit code 2", "geulpu_error": False}), "", None),
    ("claude is_error · 빈 내용", CLAUDE, 대화({"content": "", "geulpu_error": True}), "#2 모호(is_error)", None),
    # Codex — 글 규칙(소스의 꼴)
    ("codex 종료 1", CODEX, 대화("Exit code: 1\nWall time: 0.1 seconds\nOutput:\nchecking config...\nerror: config.ini not found\n"),
     "#2 모호(codex-exit)", None),
    ("codex Process exited", CODEX, 대화("Process exited with code 2"), "#2 모호(codex-exit)", None),
    ("codex 시간 초과", CODEX, 대화("command timed out after 10000 milliseconds"), "#2 모호(codex-timeout)", None),
    ("codex 패치 실패", CODEX, 대화(r"apply_patch verification failed: Failed to read file to update E:\x\a.py: No such file or "
                                   "directory (os error 2)"), "#2 고침(codex-patch)", None),
    ("codex 성공", CODEX, 대화("Exit code: 0\nWall time: 0.1 seconds\nOutput:\nok\n"), "", None),
    # 여럿 · 차례
    ("실패 둘 사이에 성공 하나", OPENCODE, 대화('{"error":{"type":"tool.execution","message":"File not found: notes.txt"},"content":[]}',
                                              "Created file successfully: a.txt",
                                              '{"error":{"type":"tool.execution","message":"Unable to write locked.txt"},"content":[]}'),
     "#2 모호(opencode) · #6 해없음(opencode)", "2; 2=ambiguous:opencode, 6=unrecoverable:opencode"),
    ("messages 가 없음", QWEN, None, "", None),
    # 다른 클라이언트(모르는 User-Agent) — 표준 오류 글로 시작하면 실패(generic), 종류는 넓힌 낱말. 글은 각 출처(라이브러리 · SDK)의 꼴
    ("다른 · Python TypeError(고침)", HTTPX, 대화("TypeError: search() got an unexpected keyword argument 'q'"),
     "#2 고침(generic)", "1; 2=fixable:generic"),
    ("다른 · requests 시간 초과(일시)", HTTPX, 대화("requests.exceptions.ReadTimeout: HTTPSConnectionPool(host='api.example.com', port=443): "
                                                "Read timed out. (read timeout=10)"), "#2 일시(generic)", None),
    ("다른 · socket.timeout(일시)", HTTPX, 대화("socket.timeout: timed out"), "#2 일시(generic)", None),
    ("다른 · MCP SDK 실행 오류 · 연결 끊김(일시)", HTTPX, 대화("Error executing tool fetch_page: ConnectionResetError(104, 'Connection reset by peer')"),
     "#2 일시(generic)", None),
    ("다른 · requests 503(일시)", HTTPX, 대화("503 Server Error: Service Unavailable for url: https://api.example.com/v1/items"),
     "#2 일시(generic)", None),
    ("다른 · httpx 401(해없음)", HTTPX, 대화("Client error '401 Unauthorized' for url 'https://api.example.com/v1/items'"),
     "#2 해없음(generic)", None),
    ("다른 · urllib 403(해없음)", HTTPX, 대화("HTTP Error 403: Forbidden"), "#2 해없음(generic)", None),
    ("다른 · LangGraph 없는 도구(고침)", HTTPX, 대화("Error: get_weather is not a valid tool, try one of [search, calculator]."),
     "#2 고침(generic)", None),
    ("다른 · OpenAI Agents SDK 기본 글(일시)", HTTPX, 대화("An error occurred while running the tool. Please try again. Error: Connection refused"),
     "#2 일시(generic)", None),
    ("다른 · MCP TypeScript SDK(고침)", HTTPX, 대화("MCP error -32602: Tool lookup_user not found"), "#2 고침(generic)", None),
    ("다른 · gRPC UNAVAILABLE(일시)", HTTPX, 대화("Error: 14 UNAVAILABLE: No connection established"), "#2 일시(generic)", None),
    ("다른 · API 오류 JSON 429(일시)", HTTPX, 대화('{"error": {"code": 429, "message": "Too Many Requests"}}'), "#2 일시(generic)", None),
    ("다른 · pydantic 검사(고침)", HTTPX, 대화("Error: 1 validation error for SearchArgs\nquery\n  Field required [type=missing]"),
     "#2 고침(generic)", None),
    ("다른 · 앞의 빈칸", HTTPX, 대화("\n  ValueError: Invalid params: limit must be positive"), "#2 고침(generic)", None),
    ("다른 · 명령 시간 초과는 모호", HTTPX, 대화("Error: Command timed out after 120 seconds"), "#2 모호(generic)", None),
    ("다른 · 종류 모름은 모호", HTTPX, 대화("Error: something went wrong"), "#2 모호(generic)", None),
    ("다른 · not available 은 모호", HTTPX, 대화("RuntimeError: the service is not available"), "#2 모호(generic)", None),
    ("다른 · Traceback(모호)", HTTPX, 대화("Traceback (most recent call last):\n  File \"t.py\", line 1, in <module>\nKeyError: 'x'"),
     "#2 모호(generic)", None),
    ("다른 · 표시 없는 클라이언트도 같음", None, 대화("TypeError: f() missing 1 required positional argument: 'x'"), "#2 고침(generic)", None),
    # 다른 클라이언트 — 실패가 아닌 것
    ("다른 · 성공 JSON 의 error null", HTTPX, 대화('{"error": null, "result": 5}'), "", None),
    ("다른 · 빈 error 글", HTTPX, 대화('{"error": "", "result": 5}'), "", None),
    ("다른 · Error rate 로 시작하는 성공", HTTPX, 대화("Error rate: 0.2% over the last hour"), "", None),
    ("다른 · 낱말만 든 성공", HTTPX, 대화("The ValueError is raised when the input is negative."), "", None),
    ("다른 · 숫자로 시작하는 성공", HTTPX, 대화("503 items found"), "", None),
    ("다른 · 점으로 시작", HTTPX, 대화(".Error: hidden"), "", None),
    # 명시 표시 — 모든 클라이언트
    ("tool 메시지의 is_error(OpenAI 꼴)", HTTPX, 대화({"content": "connect ECONNREFUSED 127.0.0.1:9000", "is_error": True}),
     "#2 일시(is_error)", None),
    ("tool 메시지의 is_error 거짓", HTTPX, 대화({"content": "fine", "is_error": False}), "", None),
    ("MCP isError JSON(고침)", HTTPX, 대화('{"content": [{"type": "text", "text": "Unknown tool: foo"}], "isError": true}'),
     "#2 고침(mcp-iserror)", None),
    ("MCP isError 거짓", HTTPX, 대화('{"content": [{"type": "text", "text": "ok"}], "isError": false}'), "", None),
    # 정해진 에이전트는 넓히지 않음 — Qwen Code 셸 출력 안의 TypeError 는 스크립트의 오류다
    ("qwen 셸 안의 TypeError 는 모호 그대로", QWEN, 대화("Command: python t.py\nDirectory: (root)\nOutput: TypeError: f() got an unexpected "
                                                     "keyword argument 'x'\nError: (none)\nExit Code: 1\nSignal: (none)\n"
                                                     "Process Group PGID: (none)"), "#2 모호(qwen-exit)", None),
    ("codex 의 Error: 줄은 그대로(감지 안 함)", CODEX, 대화("Error: Connection refused"), "", None),
]

# ===== 4b — 참조(errmsg2 세션.py · 자극.py 그대로) =====
안내 = {1: "This is a temporary failure. Retrying the same call is expected to succeed.",
        2: "This call cannot succeed as written. The error says what to change; call the tool again with the corrected value.",
        3: "This cannot be fixed by retrying or by changing the arguments. Tell the user what happened."}


def 글뽑기(content):
    if isinstance(content, str):
        return content
    return "\n".join(b.get("text", "") for b in content if isinstance(b, dict))


def 인자글(args):
    return ", ".join(f"{k}={v}" for k, v in args.items())


def 알림글(이름, args, 오류, 종류):
    return f"Note from the tool runtime: the last attempt called {이름} with {인자글(args)} and it failed with: {오류}\n{안내[종류]}"


def 붙이기(msgs, k, 알림):
    if k is None:
        return msgs + [{"role": "user", "content": 알림}]
    msgs[k] = dict(msgs[k], content=글뽑기(msgs[k]["content"]) + "\n\n" + 알림)
    return msgs


def 꼴하나(msgs, i, 조건, 종류):
    """세션.꼴 — 오류는 그 tool 메시지의 내용(글로 편 것)"""
    msgs = copy.deepcopy(msgs)
    j = max(x for x in range(i) if msgs[x]["role"] == "assistant" and msgs[x].get("tool_calls"))
    tc = next(c for c in msgs[j]["tool_calls"] if c["id"] == msgs[i]["tool_call_id"])
    오류 = 글뽑기(msgs[i]["content"])
    if 조건 == "c1":
        msgs[i]["content"] = 오류 + "\n" + 안내[종류]
        return msgs
    알림 = 알림글(tc["function"]["name"], json.loads(tc["function"]["arguments"]), 오류, 종류)
    뒤사용자 = next((x for x in range(i + 1, len(msgs)) if msgs[x]["role"] == "user"), None)
    if 조건 == "c3":
        return 붙이기(msgs, 뒤사용자, 알림)
    del msgs[i]
    msgs[j]["tool_calls"] = [c for c in msgs[j]["tool_calls"] if c["id"] != tc["id"]]
    if not msgs[j]["tool_calls"] and not (msgs[j].get("content") or "").strip():
        del msgs[j]
    elif not msgs[j]["tool_calls"]:
        del msgs[j]["tool_calls"]
    return 붙이기(msgs, len(msgs) - 1 if msgs[-1]["role"] == "user" else None, 알림)


def 참조꼴(msgs, 실패들, 조건):
    """결정 스텝(마지막 assistant 뒤)의 실패 가운데 안내가 있는 종류(1 · 2 · 3)를 차례로 — (바꾼 수, messages)"""
    끝a = max(x for x, m in enumerate(msgs) if m["role"] == "assistant")
    고른 = [(msgs[i]["tool_call_id"], 종류) for i, 종류 in sorted(실패들) if i > 끝a and 종류 in 안내]
    for tid, 종류 in 고른:
        i = next(x for x, m in enumerate(msgs) if m["role"] == "tool" and m.get("tool_call_id") == tid)
        msgs = 꼴하나(msgs, i, 조건, 종류)
    return len(고른), msgs


def 호출(cid, 이름, args, 글="", 날인자=None):
    return {"role": "assistant", "content": 글, "tool_calls": [
        {"id": cid, "type": "function", "function": {"name": 이름, "arguments": 날인자 or json.dumps(args, ensure_ascii=False)}}]}


def 결과(cid, 내용):
    return {"role": "tool", "tool_call_id": cid, "content": 내용}


시스템 = {"role": "system", "content": "You are a coding agent."}
요청 = {"role": "user", "content": "Fix the build."}
알림차례 = {"role": "user", "content": [{"type": "text", "text": "<system-reminder>\nKeep going.\n</system-reminder>"}]}
권한 = "Error: EACCES: permission denied, open 'E:\\x\\secret.txt'"
병렬 = {"role": "assistant", "content": "", "tool_calls": [
    {"id": "p1", "type": "function", "function": {"name": "read_file", "arguments": '{"file_path": "a.txt"}'}},
    {"id": "p2", "type": "function", "function": {"name": "read_file", "arguments": '{"file_path": "b.txt"}'}}]}
# 겹친 값 · 따옴표 · 역슬래시 · 한국어 · NBSP · ZWSP · 이모지 · -0 · 1e20 · 참거짓 · null(repr 꼴)
겹인자 = ('{"command": ["bash", "-lc", "echo \'hi\' \\"x\\""], "opts": {"a": 1.5, "b": true, "c": null, '
         '"d": "줄\\n바꿈\\\\탭\\t", "e": "it\'s", "f": "a\\u00a0b\\u200bc😀", "g": [1, -0, 1e20, 2.5e-7]}, "n": 3, "name": "이름"}')

# (아이디, User-Agent, messages, 꼴, 실패 [(번호, 종류)] — None 이면 엔진의 감지 · 참조는 아래 '답실패')
꼴사례들 = []
for 꼴 in ("c1", "c2", "c3"):
    꼴사례들 += [
        (f"{꼴} 하나 · 끝이 tool", None, [시스템, 요청, 호출("c1", "read_file", {"file_path": "secret.txt"}), 결과("c1", 권한)], 꼴, [(3, 3)]),
        (f"{꼴} 하나 · 끝에 reminder(조각 배열)", None,
         [시스템, 요청, 호출("c1", "run_shell_command", {"command": "python check.py", "is_background": False}),
          결과("c1", [{"type": "text", "text": "Command: python check.py\nExit Code: 1"}]), 알림차례], 꼴, [(3, 2)]),
        (f"{꼴} 병렬 둘 다 실패 · 내용 없음", None, [시스템, 요청, 병렬, 결과("p1", 권한), 결과("p2", 권한)], 꼴, [(3, 3), (4, 3)]),
        (f"{꼴} 병렬 하나만 실패", None, [시스템, 요청, 병렬, 결과("p1", "hello"), 결과("p2", 권한)], 꼴, [(4, 3)]),
        (f"{꼴} assistant 에 글", None, [요청, 호출("c1", "fetch", {"url": "http://x"}, "Let me check."),
                                       결과("c1", "connect ECONNRESET")], 꼴, [(2, 1)]),
        (f"{꼴} 인자 repr", None, [요청, 호출("c1", "shell", None, 날인자=겹인자), 결과("c1", "Service Unavailable")], 꼴, [(2, 1)]),
        (f"{꼴} 같은 열쇠 둘(뒤의 값)", None, [요청, 호출("c1", "f", None, 날인자='{"a": 1, "b": 2, "a": 3}'), 결과("c1", 권한)], 꼴, [(2, 3)]),
        (f"{꼴} 병렬 둘 다 실패 · 끝에 reminder", None, [요청, 병렬, 결과("p1", 권한), 결과("p2", 권한), 알림차례], 꼴, [(2, 3), (3, 3)]),
        (f"{꼴} 앞 턴의 실패는 그대로", None, [요청, 호출("c1", "read_file", {"file_path": "a"}), 결과("c1", 권한),
                                         호출("c2", "read_file", {"file_path": "b"}), 결과("c2", 권한)], 꼴, [(2, 3), (4, 3)]),
        (f"{꼴} assistant 의 content 가 null", None, [요청, dict(호출("c1", "read_file", {"file_path": "a"}), content=None), 결과("c1", 권한)],
         꼴, [(2, 3)]),
        (f"{꼴} 모호는 바꾸지 않음", None, [요청, 호출("c1", "read_file", {"file_path": "a"}), 결과("c1", "File not found: a")], 꼴, [(2, 4)]),
        (f"{꼴} 호출 아이디를 못 찾음", None, [요청, 호출("c1", "read_file", {"file_path": "a"}), 결과("zz", 권한)], 꼴, [(2, 3)]),
        (f"{꼴} 인자를 못 풂", None, [요청, 호출("c1", "read_file", None, 날인자="{not json"), 결과("c1", 권한)], 꼴, [(2, 3)]),
        (f"{꼴} 감지와 함께(OpenCode 쓸 수 없음)", OPENCODE, [요청, 호출("c1", "write", {"path": "locked.txt", "content": "done"}),
                                                     결과("c1", '{"error":{"type":"tool.execution","message":"Unable to write locked.txt"},'
                                                              '"content":[]}')], 꼴, None),
        (f"{꼴} 감지와 함께(Qwen 없는 파일 — 모호)", QWEN, [요청, 호출("c1", "read_file", {"file_path": "n.txt"}),
                                                   결과("c1", "File not found: n.txt")], 꼴, None),
        (f"{꼴} 감지와 함께(다른 클라이언트 TypeError — 고침)", HTTPX, [요청, 호출("c1", "search", {"q": "x"}),
                                                            결과("c1", "TypeError: search() got an unexpected keyword argument 'q'")], 꼴, None),
    ]
꼴사례들.append(("off 는 바꾸지 않음", None, [요청, 호출("c1", "read_file", {"file_path": "a"}), 결과("c1", 권한)], "off", [(2, 3)]))
# 감지와 함께 — 참조가 쓸 실패(엔진의 감지가 내야 할 것)
답실패 = {"감지와 함께(OpenCode 쓸 수 없음)": [(2, 3)], "감지와 함께(Qwen 없는 파일 — 모호)": [(2, 4)],
         "감지와 함께(다른 클라이언트 TypeError — 고침)": [(2, 2)]}


def 꼴답(ms, 꼴, 실패들, 아이디):
    if 실패들 is None:
        실패들 = 답실패[아이디.split(" ", 1)[1]]
    if 꼴 not in ("c1", "c2", "c3"):
        return 0, None
    try:
        n, 바뀐 = 참조꼴(ms, 실패들, 꼴)
    except (StopIteration, ValueError, json.JSONDecodeError):
        return 0, None                       # 참조가 못 하는 것(호출을 못 찾음 · 인자를 못 풂) — 엔진은 바꾸지 않는다
    return n, (json.dumps(바뀐, ensure_ascii=False) if n else None)


def main():
    exe = sys.argv[1]
    with tempfile.TemporaryDirectory() as 임시:
        입력 = os.path.join(임시, "사례.json")
        출력 = os.path.join(임시, "출력.json")
        with open(입력, "w", encoding="utf-8", newline="\n") as f:
            json.dump([{"id": a, "agent": ua, "messages": ms} for a, ua, ms, _, _ in 사례들] +
                      [dict({"id": "꼴 " + a, "agent": ua, "messages": ms, "form": 꼴}, **({"fail": 실} if 실 is not None else {}))
                       for a, ua, ms, 꼴, 실 in 꼴사례들], f, ensure_ascii=False)
        r = subprocess.run([exe, 입력, 출력], capture_output=True)
        if r.returncode != 0:
            print(r.stdout.decode("utf-8", "replace") + r.stderr.decode("utf-8", "replace"))
            print(f"도구 오류 감지: 시험_도구오류 종료 {r.returncode}")
            return 1
        with open(출력, encoding="utf-8") as f:
            나온 = json.load(f)
    틀림 = 0
    for 아이디, _, _, 로그, 머리 in 사례들:
        n = 나온.get(아이디)
        맞음 = isinstance(n, dict) and n.get("log") == 로그 and (머리 is None or n.get("header") == 머리)
        if 맞음 and not 로그:
            맞음 = n.get("header") == ""
        if not 맞음:
            틀림 += 1
            print(f"틀림  {아이디}\n      나온 {n!r}\n      답   log={로그!r} header={머리!r}")
    for 아이디, _, ms, 꼴, 실 in 꼴사례들:
        n = 나온.get("꼴 " + 아이디) or {}
        답수, 답글 = 꼴답(ms, 꼴, 실, 아이디)
        if n.get("changed") != 답수 or n.get("messages_text") != 답글:
            틀림 += 1
            print(f"틀림  꼴 {아이디}: 바꾼 수 {n.get('changed')} (답 {답수})\n      나온 {n.get('messages_text')!r}\n      답   {답글!r}")
    모두 = len(사례들) + len(꼴사례들)
    if 틀림:
        print(f"도구 오류 감지: 틀림 {틀림} / {모두}")
        return 1
    print(f"도구 오류 감지: 모두 맞음 ({모두} 가지 — 감지 {len(사례들)} · 꼴 {len(꼴사례들)})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
