#!/usr/bin/env python3
"""앱리디자인 F1(문서/사용성시험.md §1) — 실험 폴더 둘의 기록을 비교하고, 한 실험의 명령 기록을 CLI 로 다시 실행한다(시험 도구).

  python -I -X utf8 시험/기록비교.py compare <A> <B>
  python -I -X utf8 시험/기록비교.py replay <A> --to <B> --app-port <p> --geulpu <GeulPU.exe> [--session <이름>]
  python -I -X utf8 시험/기록비교.py --self-test

compare — 같으면 종료 코드 0, 다르면 1 과 다른 곳(파일 · 키 · 두 값)의 목록. 비교하는 것과 하지 않는 것(문서/사용성시험.md §1):
  명령 기록 — 줄마다 op · method · path · params · status 와 줄의 순서. time · ms · source · session 은 빼고, params 안의 그 실험 자신의
    폴더 경로는 <experiment> 로.
  experiment.json — id · title · question · created · authors 를 빼고, 파일 목록(files)은 경로만, 요청에 시드가 없던 실행의 seed 는 빼고.
  실행 기록 — time · ms · source · session 을 빼고, 요청은 stream · stream_options 를 빼고, 응답은 그때의 앱 상태에 따른 값(id · created ·
    캐시 · 슬롯 · 걸린 시간 · 추측 디코딩의 초안 수, 요청에 시드가 없을 때 서버가 고른 시드)을 빼고 — 수는 비트까지. 스트리밍 기록(events)과
    스트리밍이 아닌 기록은 답(글 · 토큰마다의 로그확률 · 상위 후보 · 끝난 이유 · 토큰 수 · 확장 필드)으로 비교한다.
  plan.md · results.md 같은 글 파일 — 두 실험의 제목 · id 를 <title> · <id> 로 바꿔 글자까지. 나머지 파일은 바이트까지.
replay — A 의 records/commands.jsonl 을 줄마다 `GeulPU <op> --json @<본문> --experiment <B> --app-port <p>`(작업인 기능은 --wait)로
  보낸다. 본문은 명령 창의 명령 복사와 같은 규칙(stream · stream_options 를 빼고, A 자신의 폴더 경로는 B 의 것으로). 시작 상태는 A 의 첫
  실행 기록에 맞춘다 — 불러온 모델이 다르면 기록 없이(--experiment 없이) 불러오고, 계산 방식이 다르면 멈춘다. 명령마다 CLI 의 종료 코드를
  적고, 결과의 판정은 compare 가 한다(실패한 명령도 앱이 상태와 함께 명령 기록에 남긴다).
--self-test — 같은 사본 · 그때의 값만 다른 사본 · 스트리밍으로 받은 사본은 같다고, 일부러 바꾼 사본 열한 가지는 다르다고 판정하는가
  (GPU 없이, 임시 폴더).
"""
import argparse
import copy
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REGISTRY = os.path.join(ROOT, "앱", "data", "operations.json")

CMD_SKIP = {"time", "ms", "source", "session"}
EXP_SKIP = {"id", "title", "question", "created", "authors"}
REC_SKIP = {"time", "ms", "source", "session", "stream", "request", "response", "events"}
STREAM_KEYS = ("stream", "stream_options")
# 응답에서 그때의 앱 상태에 따라 달라지는 값 — 계약은 이것들이 결과를 바꾸지 않는다고 한다(문서/사용성시험.md §1)
VOLATILE = {"id", "created", "created_at", "system_fingerprint", "timings", "prompt_tokens_cached", "prompt_tokens_details", "store", "slot",
            "first_token_ms", "generate_us", "speculative", "draft_proposed", "draft_accepted", "draft_steps",
            "cache_read_input_tokens", "cache_creation_input_tokens", "input_tokens_details"}
TEXT_EXT = (".md", ".txt")


def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def read_jsonl(path):
    rows = []
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rows.append(json.loads(line))
    return rows


def short(v):
    s = json.dumps(v, ensure_ascii=False)
    return s if len(s) <= 140 else s[:137] + "…"


def same_number(a, b):
    """수는 비트까지 — 1 과 1.0 은 앱이 다르게 적은 것이라 다르다, -0.0 과 0.0 도"""
    if type(a) is not type(b):
        return False
    if isinstance(a, float):
        return (math.isnan(a) and math.isnan(b)) or repr(a) == repr(b)
    return a == b


def walk(a, b, where, out):
    if isinstance(a, dict) and isinstance(b, dict):
        for k in list(a) + [k for k in b if k not in a]:
            if k not in a or k not in b:
                out.append((f"{where}.{k}", a.get(k, "(없음)"), b.get(k, "(없음)")))
            else:
                walk(a[k], b[k], f"{where}.{k}", out)
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            out.append((f"{where} (길이)", len(a), len(b)))
        for i in range(min(len(a), len(b))):
            walk(a[i], b[i], f"{where}[{i}]", out)
    elif isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool) and not isinstance(b, bool):
        if not same_number(a, b):
            out.append((where, a, b))
    elif type(a) is not type(b) or a != b:
        out.append((where, a, b))


def subst_paths(v, own, to):
    """값 안의 그 실험 자신의 폴더 경로(같거나 그 아래)를 to 로 — 대소문자 · / 와 \\ 의 차이는 같은 경로로 본다"""
    base = os.path.normcase(os.path.normpath(own))

    def f(x):
        if isinstance(x, dict):
            return {k: f(y) for k, y in x.items()}
        if isinstance(x, list):
            return [f(y) for y in x]
        if isinstance(x, str) and x and (x[0] in "/\\" or x[1:3] in (":\\", ":/")):
            n = os.path.normcase(os.path.normpath(x))
            if n == base:
                return to
            if n.startswith(base + os.sep):
                return to + x[len(x) - (len(n) - len(base)):]
        return x
    return f(v)


def strip_stream(v):
    return {k: y for k, y in v.items() if k not in STREAM_KEYS} if isinstance(v, dict) else v


def scrub(v, keep_seed):
    """응답에서 그때의 앱 상태에 따른 값을 뺀다. Anthropic 의 input_tokens 는 캐시에서 읽은 토큰을 뺀 수라 셋을 더해 둔다"""
    if isinstance(v, dict):
        out = {}
        for k, y in v.items():
            if k in VOLATILE or (k == "seed" and not keep_seed):
                continue
            out[k] = scrub(y, keep_seed)
        if "cache_read_input_tokens" in v or "cache_creation_input_tokens" in v:
            out["input_tokens"] = sum(v.get(k) or 0 for k in ("input_tokens", "cache_read_input_tokens", "cache_creation_input_tokens"))
        return out
    if isinstance(v, list):
        return [scrub(y, keep_seed) for y in v]
    return v


def answer(rec):
    """스트리밍이든 아니든 같은 형식의 답 — 글 · 생각 글 · 토큰마다의 로그확률(상위 후보) · 끝난 이유 · 토큰 수 · 확장 필드(geulpu)"""
    req = rec.get("request") if isinstance(rec.get("request"), dict) else {}
    text, reasoning, lps = [], [], []
    finish = usage = ext = None
    events = rec.get("events") if isinstance(rec.get("events"), list) else [rec.get("response") or {}]
    for ev in events:
        if not isinstance(ev, dict):
            continue
        if isinstance(ev.get("usage"), dict):
            usage = ev["usage"]
        if isinstance(ev.get("geulpu"), dict):
            ext = ev["geulpu"]
        for ch in ev.get("choices") or []:
            if not isinstance(ch, dict):
                continue
            if ch.get("finish_reason") is not None:
                finish = ch["finish_reason"]
            msg = ch.get("message") or ch.get("delta") or {}
            if isinstance(msg, dict):
                if isinstance(msg.get("content"), str):
                    text.append(msg["content"])
                if isinstance(msg.get("reasoning_content"), str):
                    reasoning.append(msg["reasoning_content"])
            if isinstance(ch.get("text"), str):
                text.append(ch["text"])
            lp = ch.get("logprobs")
            if isinstance(lp, dict):
                if isinstance(lp.get("content"), list):
                    lps.extend(lp["content"])
                if isinstance(lp.get("token_logprobs"), list):          # 텍스트 생성 — 토큰마다 묶는다
                    toks, tops = lp.get("tokens") or [], lp.get("top_logprobs") or []
                    for i, x in enumerate(lp["token_logprobs"]):
                        lps.append({"token": toks[i] if i < len(toks) else None, "logprob": x,
                                    "top_logprobs": tops[i] if i < len(tops) else None})
    return scrub({"text": "".join(text), "reasoning": "".join(reasoning), "logprobs": lps, "finish_reason": finish,
                  "usage": usage, "geulpu": ext}, "seed" in req)


def compare_runs(ra, rb, where, out):
    walk({k: v for k, v in ra.items() if k not in REC_SKIP}, {k: v for k, v in rb.items() if k not in REC_SKIP}, where, out)
    walk(strip_stream(ra.get("request")), strip_stream(rb.get("request")), where + ".request", out)
    sa, sb = isinstance(ra.get("events"), list), isinstance(rb.get("events"), list)
    if sa != sb:                                                    # 한쪽만 스트리밍 — 답으로
        walk(answer(ra), answer(rb), where + " (답)", out)
        return
    key = "events" if sa else "response"
    keep = isinstance(ra.get("request"), dict) and "seed" in ra["request"]
    walk(scrub(ra.get(key), keep), scrub(rb.get(key), keep), f"{where}.{key}", out)


def files_of(folder):
    out = []
    for d, _, names in os.walk(folder):
        for n in names:
            out.append(os.path.relpath(os.path.join(d, n), folder).replace("\\", "/"))
    return sorted(out)


def norm_exp(e, own):
    x = {k: v for k, v in e.items() if k not in EXP_SKIP}
    if isinstance(x.get("files"), list):
        x["files"] = sorted(f.get("path") for f in x["files"] if isinstance(f, dict))
    runs = []
    for r in x.get("runs") or []:
        r = dict(r)
        st = r.get("settings") if isinstance(r.get("settings"), dict) else {}
        if "seed" not in st:
            r.pop("seed", None)
        if st:
            r["settings"] = strip_stream(st)
        runs.append(r)
    x["runs"] = runs
    return subst_paths(x, own, "<experiment>")


def norm_cmds(rows, own):
    out = []
    for r in rows:
        x = {k: v for k, v in r.items() if k not in CMD_SKIP}
        if isinstance(x.get("params"), dict):
            x["params"] = subst_paths(strip_stream(x["params"]), own, "<experiment>")
        out.append(x)
    return out


def compare(a, b):
    """두 실험 폴더 — (다른 곳의 목록, 요약 글)"""
    out = []
    ea, eb = load_json(os.path.join(a, "experiment.json")), load_json(os.path.join(b, "experiment.json"))
    fa, fb = files_of(a), files_of(b)
    for p in sorted(set(fa) ^ set(fb)):
        out.append((p, "있음" if p in fa else "(없음)", "있음" if p in fb else "(없음)"))
    walk(norm_exp(ea, a), norm_exp(eb, b), "experiment.json", out)
    ca = read_jsonl(os.path.join(a, "records", "commands.jsonl"))
    cb = read_jsonl(os.path.join(b, "records", "commands.jsonl"))
    walk(norm_cmds(ca, a), norm_cmds(cb, b), "records/commands.jsonl", out)
    # 두 실험의 제목 · id — 다시 실행한 글에는 A 의 것이 들어 있을 수 있어 둘 다 바꾼다(긴 것부터)
    subs = sorted({(e[k], ph) for e in (ea, eb) for k, ph in (("title", "<title>"), ("id", "<id>")) if isinstance(e.get(k), str) and e[k]},
                  key=lambda s: len(s[0]), reverse=True)
    for p in sorted(set(fa) & set(fb)):
        if p in ("experiment.json", "records/commands.jsonl"):
            continue
        pa, pb = os.path.join(a, p), os.path.join(b, p)
        if p.startswith("records/runs/") and p.endswith(".json"):
            compare_runs(load_json(pa), load_json(pb), p, out)
        elif p.endswith(TEXT_EXT):
            ta, tb = open(pa, encoding="utf-8").read(), open(pb, encoding="utf-8").read()
            for n, ph in subs:
                ta, tb = ta.replace(n, ph), tb.replace(n, ph)
            if ta != tb:
                out.append((p, "(글)", "(다른 글)"))
        elif open(pa, "rb").read() != open(pb, "rb").read():
            out.append((p, "(바이트)", "(다른 바이트)"))
    runs = len(ea.get("runs") or [])
    return out, f"명령 {len(ca)} 줄 · 실행 {runs} 개 · 파일 {len(fa)} 개"


def cmd_compare(a, b):
    out, summary = compare(a, b)
    if not out:
        print(f"같음 — {summary}")
        return 0
    print(f"다름 — {len(out)} 곳 ({summary}):")
    for where, x, y in out[:60]:
        print(f"  {where}: {short(x)} ≠ {short(y)}")
    if len(out) > 60:
        print(f"  … {len(out) - 60} 곳 더")
    return 1


# ---- replay ----

def cli(geulpu, args, port, timeout=1800):
    r = subprocess.run([geulpu] + args + ["--app-port", str(port)], capture_output=True, timeout=timeout, stdin=subprocess.DEVNULL)
    text = r.stdout.decode("utf-8", "replace")
    try:
        j = json.loads(text)
    except ValueError:
        j = None
    return r.returncode, j, text


def cmd_replay(a, b, port, geulpu, session):
    ops = {o["id"]: o for o in load_json(REGISTRY)["operations"]}
    rows = read_jsonl(os.path.join(a, "records", "commands.jsonl"))
    if not rows:
        print(f"명령 기록이 없다: {a}")
        return 2
    paths = []
    for folder in (a, b):
        rc, j, text = cli(geulpu, ["experiment.show", "--path", os.path.abspath(folder)], port)
        if rc != 0 or not isinstance(j, dict) or not j.get("path"):
            print(f"실험 폴더가 아니다: {folder} — {text.strip()[:300]}")
            return 2
        paths.append(j["path"])
    own_a, path_b = paths
    # 시작 상태 — A 의 첫 실행 기록의 모델 · 계산 방식
    ea = load_json(os.path.join(a, "experiment.json"))
    first = next((r for r in ea.get("runs") or [] if r.get("record")), None)
    if first:
        rec = load_json(os.path.join(a, first["record"]))
        want, srv = rec.get("model"), rec.get("server") or {}
        rc, st, _ = cli(geulpu, ["system.state"], port)
        sv = (st or {}).get("server") or {}
        if want and not (sv.get("state") == "ready" and sv.get("model") == want):
            rc, _, text = cli(geulpu, ["model.load", "--model", want, "--wait"], port)
            print(f"시작 상태: 모델 {want} 불러오기(기록 없이) → {rc}")
            rc, st, _ = cli(geulpu, ["system.state"], port)
            sv = (st or {}).get("server") or {}
        off = [k for k in ("precision", "kv8", "fast_attention", "yarn") if k in srv and (srv.get(k) or False) != (sv.get(k) or False)]
        if off:
            print("시작 상태의 계산 방식이 A 의 첫 실행과 다르다: " + ", ".join(f"{k} {srv.get(k)} ≠ {sv.get(k)}" for k in off))
            return 2
    bad = 0
    tmp = tempfile.mkdtemp(prefix="geulpu-replay-")
    try:
        for i, r in enumerate(rows, 1):
            op = r.get("op")
            params = r.get("params")
            if not isinstance(params, dict):
                print(f"{i:3d} {op}: params 가 JSON 객체가 아니다 — 보내지 못함")
                bad += 1
                continue
            body = subst_paths(strip_stream(params), own_a, path_b)
            f = os.path.join(tmp, f"{i}.json")
            with open(f, "w", encoding="utf-8", newline="\n") as fp:
                json.dump(body, fp, ensure_ascii=False, separators=(",", ":"))
            args = [op, "--json", "@" + f, "--experiment", path_b]
            if (ops.get(op) or {}).get("job"):
                args.append("--wait")
            if session:
                args += ["--session", session]
            rc, j, text = cli(geulpu, args, port)
            want = r.get("status")
            note = "" if rc == 0 or (isinstance(want, int) and want >= 400) else " — 실패: " + text.strip()[:300]
            print(f"{i:3d} {op} → 종료 {rc} (A 의 상태 {want}){note}", flush=True)
            if rc == 3:
                print("앱에 닿지 못했다 — 멈춤")
                return 3
            if note:
                bad += 1
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print(f"다시 실행: 명령 {len(rows)} 줄, 기대와 다르게 끝난 것 {bad}")
    return 1 if bad else 0


# ---- 자체 시험 ----

def _sample(root, name, title, chat_stream):
    """자체 시험의 실험 폴더 — 텍스트 생성 · 점수(서버가 고른 시드) · 분기 · 채팅(스트리밍이거나 아니거나)과 실험 갱신 한 줄"""
    d = os.path.join(root, name)
    os.makedirs(os.path.join(d, "records", "runs"))
    srv = {"sha256": "ab" * 32, "layer": "cuda", "precision": "exact", "kv8": False, "fast_attention": False, "context": 8192, "yarn": 0}
    work = {"prompt_tokens_cached": 0, "store": {"hit": 0}, "slot": 0, "first_token_ms": 1000, "generate_us": 2000, "speculative": True,
            "draft_proposed": 1, "draft_accepted": 0, "draft_steps": 1}

    def comp(rid, op, req, toks, hashes, lps, extra=None):
        g = dict(work, seed=req.get("seed", 1234567890123456789), tokens=toks, logit_hashes=hashes, prompt_tokens=[3],
                 entropy=[0.5] * len(toks), top_ids=[[t, t + 1] for t in toks], top_logprobs=[[x, x - 1.0] for x in lps])
        g.update(extra or {})
        resp = {"id": "cmpl-1", "object": "text_completion", "created": 100, "model": "m",
                "choices": [{"index": 0, "text": "".join(f" t{t}" for t in toks), "finish_reason": "length",
                             "logprobs": {"tokens": [f" t{t}" for t in toks], "token_logprobs": lps,
                                          "top_logprobs": [{f" t{t}": x} for t, x in zip(toks, lps)]}}],
                "usage": {"prompt_tokens": 1, "completion_tokens": len(toks), "total_tokens": 1 + len(toks),
                          "prompt_tokens_details": {"cached_tokens": 0}}, "geulpu": g}
        return {"run": rid, "op": op, "path": "/v1/completions", "time": 100, "ms": 5, "status": 200, "source": "gui", "model": "m",
                "server": srv, "app": "0.22.0", "request": req, "response": resp}

    runs = {
        "r1": comp("r1", "run.complete", {"prompt": "Hi", "max_tokens": 3, "temperature": 0.8, "logprobs": 2,
                                          "geulpu": {"top_logprobs": 2, "logit_hash": True, "prompt_tokens": True}, "seed": 7},
                   [5, 6, 7], ["0123456789abcdef", "1123456789abcdef", "2123456789abcdef"], [-0.5, -1.25, -0.125]),
        "r2": comp("r2", "run.score", {"prompt": [3, 5, 6, 7], "echo": True, "max_tokens": 0, "logprobs": 2,
                                       "geulpu": {"echo_from": 1, "top_logprobs": 2, "logit_hash": True}},
                   [], [], [], {"prompt_logit_hashes": ["0123456789abcdef", "1123456789abcdef", "2123456789abcdef"],
                                "prompt_top_ids": [None, [5, 8], [6, 9], [7, 10]]}),
        "r3": comp("r3", "run.branch", {"prompt": [3, 5, 9], "max_tokens": 2, "temperature": 0.8, "logprobs": 2,
                                        "geulpu": {"echo_from": 1, "step_offset": 2, "logit_hash": True}, "seed": 7},
                   [11, 12], ["3123456789abcdef", "4123456789abcdef"], [-0.75, -2.5]),
    }
    # 채팅 — 같은 답을 스트리밍(이벤트)으로 또는 한 번에
    content = [{"token": "Hel", "logprob": -0.25, "top_logprobs": [{"token": "Hel", "logprob": -0.25}, {"token": "Hi", "logprob": -1.5}]},
               {"token": "lo", "logprob": -0.0625, "top_logprobs": [{"token": "lo", "logprob": -0.0625}, {"token": "p", "logprob": -3.0}]}]
    g4 = dict(work, seed=99, tokens=[21, 22], logit_hashes=["5123456789abcdef", "6123456789abcdef"], prompt_tokens=[1, 2, 3])
    usage = {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5, "prompt_tokens_details": {"cached_tokens": 0}}
    req4 = {"messages": [{"role": "user", "content": "hi"}], "max_tokens": 4, "temperature": 0, "logprobs": True, "top_logprobs": 2,
            "geulpu": {"prompt_tokens": True, "top_logprobs": 2, "logit_hash": True}}
    r4 = {"run": "r4", "op": "run.chat", "path": "/v1/chat/completions", "time": 100, "ms": 5, "status": 200, "source": "gui", "model": "m",
          "server": srv, "app": "0.22.0"}
    if chat_stream:
        ch = lambda delta, lp, fin: {"id": "chatcmpl-1", "object": "chat.completion.chunk", "created": 100, "model": "m",  # noqa: E731
                                     "choices": [{"index": 0, "delta": delta, "logprobs": lp, "finish_reason": fin}]}
        ev = [ch({"role": "assistant", "content": ""}, None, None), ch({"content": "Hel"}, {"content": content[:1]}, None),
              ch({"content": "lo"}, {"content": content[1:]}, None), dict(ch({}, None, "stop"), geulpu=g4),
              {"id": "chatcmpl-1", "object": "chat.completion.chunk", "created": 100, "model": "m", "choices": [], "usage": usage}]
        r4.update(stream=True, request=dict(req4, stream=True, stream_options={"include_usage": True}), response=ev[-2], events=ev)
    else:
        r4.update(request=req4, response={"id": "chatcmpl-1", "object": "chat.completion", "created": 100, "model": "m",
                                          "choices": [{"index": 0, "message": {"role": "assistant", "content": "Hello"},
                                                       "logprobs": {"content": content}, "finish_reason": "stop"}],
                                          "usage": usage, "geulpu": g4})
    runs["r4"] = r4
    for rid, rec in runs.items():
        with open(os.path.join(d, "records", "runs", rid + ".json"), "w", encoding="utf-8", newline="\n") as f:
            json.dump(rec, f, ensure_ascii=False, indent=1)
    cmds = [{"time": 100, "op": "experiment.update", "method": "POST", "path": "/api/experiments/update",
             "params": {"path": d, "plan": "# Plan\n"}, "source": "gui", "status": 200, "ms": 3}]
    for rid, rec in runs.items():
        cmds.append({"time": 100, "op": rec["op"], "method": "POST", "path": rec["path"], "params": rec["request"], "source": "gui",
                     "status": 200, "ms": 5})
    with open(os.path.join(d, "records", "commands.jsonl"), "w", encoding="utf-8", newline="\n") as f:
        f.write("".join(json.dumps(c, ensure_ascii=False) + "\n" for c in cmds))
    entry = lambda rid, kind, op, **kw: dict({"id": rid, "kind": kind, "operation": op, "record": f"records/runs/{rid}.json"}, **kw)  # noqa: E731
    exp = {"format": "geulpu-experiment/1", "id": name, "title": title, "question": "q", "status": "running", "created": "2026-01-01",
           "authors": ["a"], "geulpu": {"version": "0.22.0", "mode": "exact"}, "devices": [],
           "models": [{"name": "M", "file": "m.gguf", "sha256": "ab" * 32, "bytes": 10, "source": {"local": "m.gguf"}}],
           "runs": [entry("r1", "generate", "run.complete", settings={"max_tokens": 3, "temperature": 0.8, "seed": 7}, seed=7),
                    entry("r2", "score", "run.score", settings={"echo": True, "max_tokens": 0}, seed=1234567890123456789, base="r1"),
                    entry("r3", "branch", "run.branch", settings={"max_tokens": 2, "temperature": 0.8, "seed": 7}, seed=7, base="r1",
                          intervention={"type": "token", "step": 1, "value": 9}),
                    entry("r4", "generate", "run.chat", settings={"max_tokens": 4, "temperature": 0}, seed=99)],
           "criteria": [{"id": "C1", "text": "t", "result": "pending"}],
           "files": [{"path": p, "sha256": "00" * 32, "bytes": 1} for p in ["records/commands.jsonl"] + [f"records/runs/{r}.json" for r in runs]],
           "reproduce": {"commands": []}, "links": []}
    with open(os.path.join(d, "experiment.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(exp, f, ensure_ascii=False, indent=1)
    with open(os.path.join(d, "plan.md"), "w", encoding="utf-8", newline="\n") as f:
        f.write(f"# Plan — {title}\n\nThe question.\n")
    return d


def _edit_json(path, fn):
    v = load_json(path)
    v = fn(v) or v
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(v, f, ensure_ascii=False, indent=1)


def _edit_cmds(d, fn):
    p = os.path.join(d, "records", "commands.jsonl")
    rows = fn(read_jsonl(p)) or read_jsonl(p)
    with open(p, "w", encoding="utf-8", newline="\n") as f:
        f.write("".join(json.dumps(c, ensure_ascii=False) + "\n" for c in rows))


def self_test():
    root = tempfile.mkdtemp(prefix="geulpu-recordcmp-")
    try:
        a = _sample(root, "2026-01-01-a", "Sample", True)
        ok = True

        def case(name, expect_same, make):
            nonlocal ok
            b = make()
            out, _ = compare(a, b)
            good = (not out) == expect_same
            ok = ok and good
            first = f" — {out[0][0]}" if out else ""
            print(f"{'통과' if good else '못 미침'}  {'같음' if expect_same else '다름'}: {name}{first if not expect_same or out else ''}")
            shutil.rmtree(b, ignore_errors=True)

        def copy_of(tag):
            b = os.path.join(root, "2026-01-02-" + tag)
            shutil.copytree(a, b)
            _edit_cmds(b, lambda rows: [dict(r, params=subst_paths(r["params"], a, b)) for r in rows])   # 다시 실행이 하듯 자기 경로로
            return b

        def volatile():
            b = os.path.join(root, "2026-01-02-volatile")
            _sample(root, "2026-01-02-volatile", "Sample B", True)

            def run_edit(rec):
                rec.update(time=200, ms=9, source="cli", session="s1")
                rec["response"]["id"] = "cmpl-9"
                rec["response"]["created"] = 200
                g = rec["response"]["geulpu"]
                g.update(prompt_tokens_cached=3, store={"hit": 2}, slot=1, first_token_ms=7, generate_us=8, draft_proposed=0, draft_steps=0)
                if "seed" not in rec["request"]:
                    g["seed"] = 42
            for r in ("r1", "r2", "r3"):
                _edit_json(os.path.join(b, "records", "runs", r + ".json"), run_edit)
            _edit_json(os.path.join(b, "experiment.json"), lambda e: e["runs"][1].update(seed=42))
            _edit_cmds(b, lambda rows: [dict(r, time=200, ms=1, source="cli", session="s1") for r in rows])
            return b

        def streamless():
            return _sample(root, "2026-01-02-nostream", "Sample", False)

        case("같은 사본", True, lambda: copy_of("same"))
        case("그때의 값만 다른 사본(시각 · 출처 · 캐시 · 걸린 시간 · 서버가 고른 시드 · 제목 · 자기 경로)", True, volatile)
        case("채팅을 스트리밍 없이 받은 사본", True, streamless)

        def changed(tag, fn):
            def make():
                b = copy_of(tag)
                fn(b)
                return b
            return make

        def run(b, rid, fn):
            _edit_json(os.path.join(b, "records", "runs", rid + ".json"), fn)

        def exp(b, fn):
            _edit_json(os.path.join(b, "experiment.json"), fn)

        cases = [
            ("토큰 하나", lambda b: run(b, "r1", lambda r: r["response"]["geulpu"]["tokens"].__setitem__(0, 8))),
            ("로짓 해시 하나", lambda b: run(b, "r3", lambda r: r["response"]["geulpu"]["logit_hashes"].__setitem__(0, "3123456789abcdee"))),
            ("로그확률의 마지막 비트", lambda b: run(b, "r1", lambda r: r["response"]["choices"][0]["logprobs"]["token_logprobs"].__setitem__(
                1, math.nextafter(-1.25, 0.0)))),
            ("요청 인자 하나", lambda b: run(b, "r1", lambda r: r["request"].__setitem__("temperature", 0.7))),
            ("명령의 op", lambda b: _edit_cmds(b, lambda rows: [dict(r, op="run.lens") if r["op"] == "run.score" else r for r in rows])),
            ("실행의 kind", lambda b: exp(b, lambda e: e["runs"][1].__setitem__("kind", "lens"))),
            ("실행의 base", lambda b: exp(b, lambda e: e["runs"][1].__setitem__("base", "r3"))),
            ("개입 스텝", lambda b: exp(b, lambda e: e["runs"][2]["intervention"].__setitem__("step", 2))),
            ("실행 하나 더", lambda b: (shutil.copy(os.path.join(b, "records", "runs", "r1.json"), os.path.join(b, "records", "runs", "r5.json")),
                                     exp(b, lambda e: e["runs"].append(dict(e["runs"][0], id="r5", record="records/runs/r5.json"))))),
            ("실행 하나 빠짐", lambda b: (os.remove(os.path.join(b, "records", "runs", "r3.json")),
                                      exp(b, lambda e: e.__setitem__("runs", [r for r in e["runs"] if r["id"] != "r3"])))),
            ("명령 한 줄 더", lambda b: _edit_cmds(b, lambda rows: rows + [rows[1]])),
        ]
        for i, (name, fn) in enumerate(cases):
            case(name, False, changed(f"x{i}", fn))
        print("기록 비교 자체 시험: " + ("모두 통과 — 같음 3 · 다름 11" if ok else "못 미침"))
        return 0 if ok else 1
    finally:
        shutil.rmtree(root, ignore_errors=True)


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--self-test":
        return self_test()
    p = argparse.ArgumentParser(description="실험 기록 비교 · 명령 기록 다시 실행(문서/사용성시험.md §1)")
    sub = p.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("compare", help="실험 폴더 둘의 기록 비교")
    c.add_argument("a")
    c.add_argument("b")
    r = sub.add_parser("replay", help="A 의 명령 기록을 CLI 로 B 에 다시 실행")
    r.add_argument("a")
    r.add_argument("--to", required=True)
    r.add_argument("--app-port", type=int, required=True)
    r.add_argument("--geulpu", required=True, help="GeulPU.exe(패키지의 것)")
    r.add_argument("--session", default="")
    x = p.parse_args()
    if x.cmd == "compare":
        return cmd_compare(x.a, x.b)
    return cmd_replay(x.a, x.to, x.app_port, x.geulpu, x.session)


if __name__ == "__main__":
    sys.exit(main())
