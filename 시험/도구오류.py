#!/usr/bin/env python3
"""에이전트 호환 4 단계 4a — 도구 오류 감지(엔진/도구오류.gl — 엔진/시험_도구오류.gl)를 GPU 없이 본다. 사례는 3 단계 녹화(에이전트 일곱이
GeulPU 에 실제로 보낸 도구 결과)의 글 그대로이고, 손으로 단 답(메시지 번호 · 종류 · 규칙)과 견준다. 실패가 아닌 결과를 실패로 잡지 않는지
(성공한 셸 · 출력 안의 'Error:' 줄 · 모르는 에이전트 · 표시 없는 같은 글)도 본다 — 감지는 정밀도가 먼저다.

  python -I 시험/도구오류.py <시험_도구오류 실행 파일>

끝 줄 "도구 오류 감지: 모두 맞음 (N 가지)"이면 종료 0, 아니면 1.
"""
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
]


def main():
    exe = sys.argv[1]
    with tempfile.TemporaryDirectory() as 임시:
        입력 = os.path.join(임시, "사례.json")
        출력 = os.path.join(임시, "출력.json")
        with open(입력, "w", encoding="utf-8", newline="\n") as f:
            json.dump([{"id": a, "agent": ua, "messages": ms} for a, ua, ms, _, _ in 사례들], f, ensure_ascii=False)
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
    if 틀림:
        print(f"도구 오류 감지: 틀림 {틀림} / {len(사례들)}")
        return 1
    print(f"도구 오류 감지: 모두 맞음 ({len(사례들)} 가지)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
