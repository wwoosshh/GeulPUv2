#!/usr/bin/env python3
"""앱리디자인 C1 (문서/명령줄.md §2) — GeulPU 의 하위 명령(CLI)을 GPU · 앱 없이 점검한다.

빌드한 GeulPU(.exe)를 임시 폴더에 앱/data/operations.json 과 함께 두고(CLI 는 실행 파일 옆의 data/ 에서 기능 목록을 읽는다):
  1. 부를 수 있는 모든 기능(status exists · internal 아님)을 영어 이름 · 한국어 이름 · id 로 `--dry-run` 해서, 세 출력이 같고 방법 · 경로가 기능
     목록과 같으며, 인자마다 형에 맞춘 값이 요청 본문(POST — 점이 든 이름은 안쪽 객체) 또는 쿼리 문자열(GET)에 그대로 들어가는지,
  2. 사용법 오류 다섯(없는 명령 · 목록에 없는 인자 · 빠진 필수 인자 · 형이 틀린 값 · --json 과 인자를 함께 씀)이 종료 코드 2 와 오류 JSON 을,
     앱이 없는 포트로 부르면 3 을 내는지,
  3. `ops` 가 부를 수 있는 기능을 모두 내는지 본다.
  4. 앱리디자인 E1(문서/화면구현.md §1) — 명령 창과 함께 쓰는 예(시험/명령창예.json)마다 CLI 가 적어 둔 요청 · 오류 · 도움말을 내는지 본다. 명령 창의
     명령 해석(앱/web/ui/command.js)은 같은 예를 node 시험/명령창점검.mjs 로 통과한다 — 두 쪽이 같은 요청을 만든다.

    python -I -X utf8 시험/명령줄점검.py <빌드한 GeulPU 실행 파일> [--write-console-cases]

--write-console-cases 는 예 파일의 argv 로 CLI 를 돌려 line(명령 창에 칠 한 줄)과 expect 를 다시 채운다(argv 를 고친 뒤에만).
"""
import concurrent.futures
import json
import os
import re
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import urllib.parse

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REGISTRY = os.path.join(ROOT, "앱", "data", "operations.json")
GUIDE = os.path.join(ROOT, "docs", "cli.md")                      # C3 — GeulPU guide 가 내는 안내 한 쪽
SAMPLE = {"integer": "3", "number": "0.5", "string": "x y 가", "array": "[1,2]", "object": '{"a":1}'}


def run(exe, args):
    p = subprocess.run([exe] + args, capture_output=True, stdin=subprocess.DEVNULL, timeout=60)
    out = p.stdout.decode("utf-8", "replace").strip()
    try:
        return p.returncode, json.loads(out)
    except ValueError:
        return p.returncode, {"_raw": out[:300]}


def callable_ops(reg):
    return [o for o in reg["operations"] if o.get("status") == "exists" and not o.get("internal")]


def sample_args(op):
    """--name value pairs for every parameter (skipping one that is the prefix of another — they cannot go together) and the expected
    values: {name: (json value, query text)}."""
    names = [p["name"] for p in op.get("params", [])]
    args, expect = [], {}
    for p in op.get("params", []):
        n, t = p["name"], p.get("type", "string")
        if any(m.startswith(n + ".") for m in names):
            continue
        if n == "stream":                                 # the CLI refuses stream: true (it prints whole answers)
            args += [f"--{n}", "false"]
            expect[n] = (False, "false")
        elif t == "boolean":
            args += [f"--{n}"]
            expect[n] = (True, "true")
        else:
            v = SAMPLE.get(t, '"z"')
            args += [f"--{n}", v]
            expect[n] = (v if t == "string" else json.loads(v), v)
    return args, expect


def nested(expect):
    body = {}
    for n, (v, _) in expect.items():
        d, parts = body, n.split(".")
        for part in parts[:-1]:
            d = d.setdefault(part, {})
        d[parts[-1]] = v
    return body


def check_op(exe, op):
    """Problems of one operation (empty when it passes)."""
    bad = []
    args, expect = sample_args(op)
    outs = []
    for name in (op["cli"]["en"], op["cli"]["ko"], op["id"]):
        rc, out = run(exe, name.split(" ") + args + ["--dry-run"])
        if rc != 0:
            bad.append(f"{name!r}: exit {rc} {out}")
        outs.append(out)
    if bad:
        return bad
    if not (outs[0] == outs[1] == outs[2]):
        bad.append(f"the three names give different requests: {outs}")
    out = outs[0]
    h = op["http"]
    if out.get("op") != op["id"] or out.get("method") != h["method"]:
        bad.append(f"op/method {out.get('op')}/{out.get('method')} — registry {op['id']}/{h['method']}")
    if h["method"] == "POST":
        if out.get("path") != h["path"]:
            bad.append(f"path {out.get('path')} — registry {h['path']}")
        if out.get("body") != nested(expect):
            bad.append(f"body {out.get('body')} — expected {nested(expect)}")
    else:
        path, _, query = (out.get("path") or "").partition("?")
        if path != h["path"]:
            bad.append(f"path {path} — registry {h['path']}")
        got = dict(urllib.parse.parse_qsl(query, keep_blank_values=True))
        want = {n: q for n, (_, q) in expect.items()}
        if got != want or out.get("body") is not None:
            bad.append(f"query {got} body {out.get('body')} — expected {want}")
    return bad


CASES = os.path.join(ROOT, "시험", "명령창예.json")
SAFE_WORD = re.compile(r"^[A-Za-z0-9._\-/:@%+=,가-힣]+$")       # 앱/web/ui/command.js 의 quote 와 같은 규칙


def quote(s):
    return s if s and SAFE_WORD.match(s) else "'" + s.replace("'", "'\\''") + "'"


def cli_outcome(exe, argv):
    """예 하나를 CLI 로 — 요청(--dry-run) · 오류 · 도움말 · ops 가운데 하나를 예 파일의 expect 형식으로(모르는 답이면 None)."""
    rc, out = run(exe, argv + ["--dry-run"])
    # F3-3 — --select 는 --dry-run 의 요청에서 고른 값(JSON 의 어느 형이든)
    if rc == 0 and any(x == "--select" or x.startswith("--select=") for x in argv):
        return {"selected": out}
    if not isinstance(out, dict):
        return None
    if rc == 2 and isinstance(out.get("error"), dict):
        return {"error": {"message": out["error"].get("message"), "next": out["error"].get("next")}}
    if rc != 0:
        return None
    if out.get("format") == "geulpu-cli/1":
        return {"ops": True}
    if "method" in out and "path" in out and "op" in out:
        return {"request": {"op": out["op"], "method": out["method"], "path": out["path"], "body": out.get("body")}}
    if "id" in out:
        return {"help": out["id"]}
    if "usage" in out:
        return {"help": None}
    return None


def console_cases(exe, write):
    doc = json.load(open(CASES, encoding="utf-8"))
    bad = 0
    for c in doc["cases"]:
        got = cli_outcome(exe, c["argv"])
        if write:
            if got is None:
                print(f"틀림 예 '{c['name']}': CLI 의 답을 읽지 못함")
                bad += 1
                continue
            c["line"] = " ".join(quote(a) for a in c["argv"])
            c["expect"] = got
        elif got != c.get("expect"):
            bad += 1
            print(f"틀림 예 '{c['name']}': CLI {json.dumps(got, ensure_ascii=False)[:200]} · 적어 둔 것 {json.dumps(c.get('expect'), ensure_ascii=False)[:200]}")
    if write and not bad:
        open(CASES, "w", encoding="utf-8", newline="\n").write(json.dumps(doc, ensure_ascii=False, indent=1) + "\n")
        print(f"예 파일을 다시 썼습니다: {len(doc['cases'])} 개")
    return len(doc["cases"]), bad


def free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def main():
    write = sys.argv[2:] == ["--write-console-cases"]
    if len(sys.argv) not in (2, 3) or not os.path.isfile(sys.argv[1]) or (len(sys.argv) == 3 and not write):
        print("쓰는 법: python -I -X utf8 시험/명령줄점검.py <빌드한 GeulPU 실행 파일> [--write-console-cases]")
        return 2
    t0 = time.time()
    reg = json.load(open(REGISTRY, encoding="utf-8"))
    ops = callable_ops(reg)
    tmp = tempfile.mkdtemp(prefix="명령줄점검_")
    try:
        exe = os.path.join(tmp, os.path.basename(sys.argv[1]))
        shutil.copy2(sys.argv[1], exe)
        os.makedirs(os.path.join(tmp, "data"))
        shutil.copy2(REGISTRY, os.path.join(tmp, "data", "operations.json"))
        shutil.copy2(GUIDE, os.path.join(tmp, "data", "cli-guide.md"))            # 꾸리기 가 docs/cli.md 를 이 이름으로 넣는다
        problems = 0
        with concurrent.futures.ThreadPoolExecutor(8) as pool:
            for op, bad in zip(ops, pool.map(lambda o: check_op(exe, o), ops)):
                if bad:
                    problems += 1
                    print(f"틀림 {op['id']}: " + " | ".join(bad))
        print(f"기능 {len(ops)} 개 × 이름 셋(--dry-run): {len(ops) - problems}/{len(ops)} 맞음")

        port = free_port()
        cases = [  # (what, args, exit code)
            ("없는 명령", ["no", "such", "command"], 2),
            ("목록에 없는 인자", ["run", "complete", "--prompt", "x", "--bogus", "1", "--dry-run"], 2),
            ("빠진 필수 인자", ["run", "branch", "--max_tokens", "3", "--dry-run"], 2),
            ("형이 틀린 값", ["run", "complete", "--prompt", "x", "--max_tokens", "abc", "--dry-run"], 2),
            ("--json 과 인자를 함께", ["run", "complete", "--json", "{}", "--prompt", "x", "--dry-run"], 2),
            ("앱이 없는 포트", ["system", "state", "--app-port", str(port)], 3),
        ]
        wrong = 0
        for what, args, code in cases:
            rc, out = run(exe, args)
            err = out.get("error") if isinstance(out, dict) else None
            ok = rc == code and isinstance(err, dict) and err.get("exit") == code and err.get("message")
            wrong += not ok
            print(f"{'맞음' if ok else '틀림'} {what}: 종료 {rc} — {(err or out)}")
        rc, out = run(exe, ["ops"])
        listed = {c["id"] for c in out.get("commands", [])} if rc == 0 else set()
        ops_ok = listed == {o["id"] for o in ops}
        print(f"{'맞음' if ops_ok else '틀림'} ops: {len(listed)} 개")
        # C3 — guide 는 안내 한 쪽을 그대로(영어 이름 · 한국어 이름), 200 줄 안
        want = open(GUIDE, encoding="utf-8").read().strip()
        for name in ("guide", "안내"):
            p = subprocess.run([exe, name], capture_output=True, stdin=subprocess.DEVNULL, timeout=60)
            got = p.stdout.decode("utf-8", "replace").replace("\r\n", "\n").strip()
            good = p.returncode == 0 and got == want and len(want.splitlines()) <= 200
            ops_ok = ops_ok and good
            print(f"{'맞음' if good else '틀림'} {name}: 종료 {p.returncode}, {len(got.splitlines())} 줄")
        # E1 — 명령 창과 함께 쓰는 예
        n_cases, bad_cases = console_cases(exe, write)
        print(f"{'맞음' if not bad_cases else '틀림'} 명령 창과 함께 쓰는 예: {n_cases - bad_cases}/{n_cases}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    ok = problems == 0 and wrong == 0 and ops_ok and bad_cases == 0
    print(f"명령줄 점검: 기능 {len(ops) - problems}/{len(ops)} · 오류 경우 {len(cases) - wrong}/{len(cases)} · ops {'맞음' if ops_ok else '틀림'}"
          f" · 명령 창 예 {n_cases - bad_cases}/{n_cases} ({time.time() - t0:.1f} 초)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
