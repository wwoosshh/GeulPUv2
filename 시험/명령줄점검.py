#!/usr/bin/env python3
"""앱리디자인 C1 (문서/명령줄.md §2) — GeulPU 의 하위 명령(CLI)을 GPU · 앱 없이 점검한다.

빌드한 GeulPU(.exe)를 임시 폴더에 앱/data/operations.json 과 함께 두고(CLI 는 실행 파일 옆의 data/ 에서 기능 목록을 읽는다):
  1. 부를 수 있는 모든 기능(status exists · internal 아님)을 영어 이름 · 한국어 이름 · id 로 `--dry-run` 해서, 세 출력이 같고 방법 · 경로가 기능
     목록과 같으며, 인자마다 형에 맞춘 값이 요청 본문(POST — 점이 든 이름은 안쪽 객체) 또는 쿼리 문자열(GET)에 그대로 들어가는지,
  2. 사용법 오류 다섯(없는 명령 · 목록에 없는 인자 · 빠진 필수 인자 · 형이 틀린 값 · --json 과 인자를 함께 씀)이 종료 코드 2 와 오류 JSON 을,
     앱이 없는 포트로 부르면 3 을 내는지,
  3. `ops` 가 부를 수 있는 기능을 모두 내는지 본다.

    python -I -X utf8 시험/명령줄점검.py <빌드한 GeulPU 실행 파일>
"""
import concurrent.futures
import json
import os
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


def free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def main():
    if len(sys.argv) != 2 or not os.path.isfile(sys.argv[1]):
        print("쓰는 법: python -I -X utf8 시험/명령줄점검.py <빌드한 GeulPU 실행 파일>")
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
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    ok = problems == 0 and wrong == 0 and ops_ok
    print(f"명령줄 점검: 기능 {len(ops) - problems}/{len(ops)} · 오류 경우 {len(cases) - wrong}/{len(cases)} · ops {'맞음' if ops_ok else '틀림'}"
          f" ({time.time() - t0:.1f} 초)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
