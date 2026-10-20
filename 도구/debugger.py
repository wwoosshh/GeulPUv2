#!/usr/bin/env python3
"""GeulPU 디버거 v1(문서/디버거.md) — 실행 기록 · 테스트 세트 · 재생(replay)을 스크립트로. 앱(GeulPU)의 API 와 /v1 으로 묻는다.
표준 라이브러리만.

    python 도구/debugger.py [--app http://127.0.0.1:44544] status
    python 도구/debugger.py record-start [NAME]       # 앱의 /v1 을 지나는 생성 요청을 실행 로그에(자료\\실행기록\\<NAME>.jsonl)
    python 도구/debugger.py record-stop
    python 도구/debugger.py get NAME FILE.jsonl         # 실행 로그를 받아 파일로
    python 도구/debugger.py delete NAME
    python 도구/debugger.py run-set TESTSET.jsonl NAME [--seed 1000]
    python 도구/debugger.py replay RUNLOG.jsonl|NAME [--out REPORT.json] [--requests 3,7] [--ignore-header]

실행 로그(run log, JSONL): 첫 줄은 헤더 {"format": "geulpu-run-log/1", "app", "commit", "model": {id, file, sha256, …}, "server":
{layer, precision, context, yarn, …}, "build": {server — 그 백엔드의 서버 실행 파일 sha256, module, module_sha256}}, 그 뒤 요청마다
{"i", "at", "ms", "path", "status", "model", "server", "request", "response"} — server 는 그 요청을 계산한 서버(모델 sha256 · 백엔드 ·
계산 방식 · 문맥 · YaRN 배), response 는 서버의 응답 JSON(스트리밍이면 마지막 이벤트)이고 그 "geulpu" 에 시드 · prompt_tokens ·
tokens · logit_hashes 가 있다. 첫 기록기의 헤더({"geulrun": 1, …})도 읽는다.

테스트 세트(test set, JSONL): 줄마다 채팅 요청 하나({"messages": …, "max_tokens": …, "temperature": …}) 또는 {"path":
"/v1/completions", "request": {…}}. 시드가 없으면 --seed + 차례를 넣는다(같은 테스트 세트면 같은 실행). 스트리밍은 끈다.

재생: 기록한 요청을 그대로(시드는 기록한 값으로 고정해) 다시 보내 프롬프트 토큰 · 나온 토큰 · 스텝마다 로짓 해시 · 끝난 까닭
(finish_reason)을 비교한다. 오류로 끝난 요청(문맥 초과 등)은 같은 오류(HTTP 상태 · 코드)가 나오는지 본다. 요청을 계산한 서버의 모델
sha256 · 계산 방식(정밀 · 빠른) · YaRN 배(빠른 모드면 백엔드도)가 지금과 다르면 다른 계산이라 먼저 알리고 멈춘다(--ignore-header 면
그대로). GeulPU 버전 · 서버 실행 파일 · 모듈(헤더의 build)이 다르면 알리기만 하고 돌린다 — 새 빌드가 비트를 바꿨나를 재생이 바로 본다.
"""
import argparse
import copy
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

생성길 = ("/v1/chat/completions", "/v1/completions")


def 묻기(앱, 방법, 길, 몸=None, 시간=3600, 날것=False):
    r = urllib.request.Request(앱.rstrip("/") + 길, data=None if 몸 is None else json.dumps(몸, ensure_ascii=False).encode("utf-8"),
                               method=방법, headers={"Content-Type": "application/json"})
    try:
        d = urllib.request.urlopen(r, timeout=시간).read()
    except urllib.error.HTTPError as e:
        본 = e.read().decode("utf-8", "replace")
        raise SystemExit(f"{길}: HTTP {e.code} — {본[:300]}")
    if 날것:
        return d
    return json.loads(d) if d else None


def 묻기상태(앱, 길, 몸, 시간=3600):
    """POST — (HTTP 상태, JSON 또는 None). 오류 응답도 돌려준다"""
    r = urllib.request.Request(앱.rstrip("/") + 길, data=json.dumps(몸, ensure_ascii=False).encode("utf-8"), method="POST",
                               headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(r, timeout=시간) as f:
            상태, d = f.status, f.read()
    except urllib.error.HTTPError as e:
        상태, d = e.code, e.read()
    try:
        return 상태, json.loads(d)
    except ValueError:
        return 상태, None


def 헤더인가(x):
    return isinstance(x, dict) and (str(x.get("format", "")).startswith("geulpu-run-log/") or x.get("geulrun") == 1)


def 실행읽기(본, 이름):
    줄 = [json.loads(l) for l in 본.splitlines() if l.strip()]
    if not 줄 or not 헤더인가(줄[0]):
        raise SystemExit(f"GeulPU 실행 로그가 아니다: {이름}")
    return 줄[0], 줄[1:]


def 지금헤더(앱):
    """지금 앱 버전과 서버 — 요청 줄의 "server" 와 같은 꼴(YaRN 배는 엔진의 규칙대로)"""
    st = 묻기(앱, "GET", "/api/state")
    sv = st["server"]
    m = next((x for x in st.get("models", []) if x.get("id") == sv.get("model")), None)
    s = (m or {}).get("shape") or {}
    얀 = 0
    if s.get("yarn_orig", 0) > 0 and sv.get("context", 0) > s.get("context_native", 0):
        얀 = 2 if sv["context"] <= 2 * s["yarn_orig"] else 4
    빌드 = {}
    try:                                            # 서버 실행 파일 · 불러온 모듈의 해시(출처 API — 모듈은 파일을 계산하므로 조금 걸린다)
        p = 묻기(앱, "GET", "/api/provenance")
        빌드 = {"bins": {x.get("file"): x.get("sha256") for x in p.get("binaries") or []},
                "module_sha256": (p.get("module") or {}).get("sha256")}
    except SystemExit:
        pass
    return {"app": (st.get("app") or {}).get("version"), "state": sv.get("state"), "model": sv.get("model"), "build": 빌드,
            "server": {"sha256": (m or {}).get("sha256"), "layer": sv.get("layer"), "precision": sv.get("precision"),
                       "context": sv.get("context"), "yarn": 얀}}


def 짧게(v):
    return v[:12] if isinstance(v, str) and len(v) > 12 else v


def 빌드다름(헤더, 지금):
    """알림만 — GeulPU 버전 · 서버 실행 파일 · 모듈이 기록한 때와 다른가(같은 백엔드 · 계산 방식일 때만 비교한다)"""
    다름 = []
    if 헤더.get("app") != 지금["app"]:
        다름.append(f"GeulPU 버전 {헤더.get('app')} → {지금['app']}")
    b = 헤더.get("build") or {}
    sv = 헤더.get("server") or {}
    if b and sv.get("layer") == 지금["server"]["layer"]:
        파일 = {"cuda": "bin\\server-cuda.exe", "vulkan": "bin\\server-vulkan.exe"}.get(sv.get("layer"))
        지금서버 = (지금.get("build") or {}).get("bins", {}).get(파일)
        if b.get("server") and 지금서버 and b["server"] != 지금서버:
            다름.append(f"서버 빌드 {짧게(b['server'])} → {짧게(지금서버)}")
        지금모듈 = (지금.get("build") or {}).get("module_sha256")
        if sv.get("precision") == 지금["server"]["precision"] and b.get("module_sha256") and 지금모듈 and b["module_sha256"] != 지금모듈:
            다름.append(f"모듈 {짧게(b['module_sha256'])} → {짧게(지금모듈)}")
    return 다름


def 기록서버(헤더, x):
    """요청을 계산한 서버 — 요청 줄의 "server"(없으면 헤더에서)"""
    if isinstance(x.get("server"), dict):
        return x["server"]
    sv = 헤더.get("server") or {}
    return {"sha256": (헤더.get("model") or {}).get("sha256"), "layer": sv.get("layer"), "precision": sv.get("precision"),
            "context": sv.get("context"), "yarn": sv.get("yarn")}


def 서버다름(기록, 지금):
    """비트가 같으려면 같아야 하는 것 — 모델 sha256 · 계산 방식 · YaRN 배, 빠른 모드면 백엔드도(CUDA ↔ Vulkan 은 정밀 모드만 같다)"""
    다름 = []
    for k, 이름 in (("sha256", "모델 sha256"), ("precision", "계산 방식"), ("yarn", "YaRN 배")):
        if 기록.get(k) != 지금[k]:
            다름.append(f"{이름} {짧게(기록.get(k))} → {짧게(지금[k])}")
    if 기록.get("precision") == "fast" == 지금["precision"] and 기록.get("layer") != 지금["layer"]:
        다름.append(f"백엔드 {기록.get('layer')} → {지금['layer']}(빠른 모드)")
    return 다름


def 상태(a):
    d = 묻기(a.app, "GET", "/api/debug")
    print(json.dumps(d, ensure_ascii=False, indent=1))


def 기록시작(a):
    d = 묻기(a.app, "POST", "/api/debug/record/start", {"name": a.name or ""})
    print(f"기록 시작: {d['name']} — {d['dir']}")


def 기록멈춤(a):
    d = 묻기(a.app, "POST", "/api/debug/record/stop", {})
    print(f"기록 멈춤: {d.get('name')} · 요청 {d.get('count')}")


def 받기(a):
    d = 묻기(a.app, "GET", "/api/debug/run?name=" + urllib.parse.quote(a.name), 날것=True)
    open(a.file, "wb").write(d)
    print(f"받음: {a.file} ({len(d):,} 바이트)")


def 지우기(a):
    묻기(a.app, "POST", "/api/debug/run/delete", {"name": a.name})
    print(f"지움: {a.name}")


def 테스트세트(a):
    세트 = [json.loads(l) for l in open(a.testset, encoding="utf-8") if l.strip()]
    d = 묻기(a.app, "POST", "/api/debug/record/start", {"name": a.name})
    print(f"기록 시작: {d['name']} · 요청 {len(세트)}", flush=True)
    try:
        for i, x in enumerate(세트):
            길 = x.get("path", "/v1/chat/completions") if "request" in x else "/v1/chat/completions"
            몸 = copy.deepcopy(x["request"] if "request" in x else x)
            몸["stream"] = False
            몸.setdefault("seed", a.seed + i)
            t0 = time.time()
            r = 묻기(a.app, "POST", 길, 몸)
            g = r.get("geulpu") or {}
            print(f"{i:3d} {길} · 프롬프트 {(r.get('usage') or {}).get('prompt_tokens')} · 생성 {len(g.get('tokens') or [])} · "
                  f"{(r['choices'][0].get('finish_reason'))} · {time.time() - t0:.1f} 초", flush=True)
    finally:
        d = 묻기(a.app, "POST", "/api/debug/record/stop", {})
        print(f"기록 멈춤: {d.get('name')} · 요청 {d.get('count')}", flush=True)


def 재생(a):
    if os.path.exists(a.run):
        헤더, 기록 = 실행읽기(open(a.run, encoding="utf-8").read(), a.run)
    else:                                            # 앱에 있는 실행 이름이면 받아서
        본 = 묻기(a.app, "GET", "/api/debug/run?name=" + urllib.parse.quote(a.run), 날것=True).decode("utf-8")
        헤더, 기록 = 실행읽기(본, a.run)
    지금 = 지금헤더(a.app)
    if 지금["state"] != "ready":
        raise SystemExit(f"서버가 준비되지 않았다(state {지금['state']}) — 모델을 불러온 뒤 재생한다")
    알림 = 빌드다름(헤더, 지금)
    if 알림:
        print("기록한 빌드와 지금 빌드가 다르다 — " + " · ".join(알림) + " (커널이 바뀐 빌드면 비트가 다를 수 있다 — 재생으로 확인한다)")
    다름 = {}
    for x in 기록:
        for d in 서버다름(기록서버(헤더, x), 지금["server"]):
            다름[d] = 다름.get(d, 0) + 1
    if 다름:
        print("기록한 서버와 지금 서버가 다르다 — " + " · ".join(f"{d}(요청 {n})" for d, n in 다름.items()))
        if not a.ignore_header:
            raise SystemExit("재생을 멈춘다(--ignore-header 로 그대로 돌릴 수 있다 — 비트가 다를 수 있다)")
    if a.requests:
        고른 = {int(v) for v in a.requests.split(",") if v.strip()}
        기록 = [x for x in 기록 if x.get("i") in 고른]
    보고 = []
    같음수 = 0
    for x in 기록:
        resp = x.get("response") or {}
        g = resp.get("geulpu") or {}
        if x.get("path") not in 생성길 or "request" not in x:
            보고.append({"i": x.get("i"), "result": "skipped", "reason": "not a generation path, or the request is not JSON"})
            continue
        몸 = copy.deepcopy(x["request"])
        몸["stream"] = False
        몸.pop("stream_options", None)
        if x.get("status") != 200:                  # 오류 응답이면 같은 오류(상태 · 코드)가 나오나
            if not isinstance(resp.get("error"), dict):
                보고.append({"i": x.get("i"), "result": "skipped", "reason": f"status {x.get('status')} without an error response"})
                continue
            t0 = time.time()
            상태2, r = 묻기상태(a.app, x["path"], 몸)
            걸림 = time.time() - t0
            코드1, 코드2 = resp["error"].get("code"), ((r or {}).get("error") or {}).get("code")
            틀린 = [] if 상태2 == x["status"] and 코드1 == 코드2 else [f"error {x['status']} {코드1} → {상태2} {코드2}"]
            같음수 += not 틀린
            보고.append({"i": x.get("i"), "result": "same" if not 틀린 else "different", "differences": 틀린, "error": x["status"],
                         "seconds": round(걸림, 2)})
            print(f"{x.get('i'):4d} {x['path']} · 오류 {x['status']} {코드1} · {'같음' if not 틀린 else '다름 — ' + 틀린[0]} · {걸림:.1f} 초",
                  flush=True)
            continue
        if g.get("tokens") is None:
            보고.append({"i": x.get("i"), "result": "skipped", "reason": "no tokens in the response (cut stream or no geulpu object)"})
            continue
        if "seed" in g:
            몸["seed"] = g["seed"]
        gp = 몸.get("geulpu") if isinstance(몸.get("geulpu"), dict) else {}
        gp.update({"logit_hash": True, "prompt_tokens": True})
        몸["geulpu"] = gp
        t0 = time.time()
        r = 묻기(a.app, "POST", x["path"], 몸)
        걸림 = time.time() - t0
        g2 = r.get("geulpu") or {}
        끝1 = ((resp.get("choices") or [{}])[0]).get("finish_reason")
        끝2 = ((r.get("choices") or [{}])[0]).get("finish_reason")
        틀린 = []
        if g.get("prompt_tokens") is not None and g.get("prompt_tokens") != g2.get("prompt_tokens"):
            틀린.append("prompt tokens")
        t1, t2 = g.get("tokens") or [], g2.get("tokens") or []
        h1, h2 = g.get("logit_hashes"), g2.get("logit_hashes")
        첫다른 = next((k for k in range(max(len(t1), len(t2))) if k >= len(t1) or k >= len(t2) or t1[k] != t2[k]), None)
        if 첫다른 is not None:
            틀린.append(f"tokens (step {첫다른})")
        if h1 is not None and h1 != h2:
            k2 = next((k for k in range(max(len(h1), len(h2 or []))) if h2 is None or k >= len(h1) or k >= len(h2) or h1[k] != h2[k]), None)
            틀린.append(f"logit hashes (step {k2})")
        if 끝1 is not None and 끝1 != 끝2:
            틀린.append(f"finish_reason {끝1} → {끝2}")
        같음수 += not 틀린
        보고.append({"i": x.get("i"), "result": "same" if not 틀린 else "different", "differences": 틀린, "steps": len(t1),
                     "hashes": h1 is not None, "seconds": round(걸림, 2)})
        print(f"{x.get('i'):4d} {x['path']} · 스텝 {len(t1)}{'' if h1 is not None else ' (해시 없음 — 토큰만)'} · {'같음' if not 틀린 else '다름'}"
              + (f" — {', '.join(틀린)}" if 틀린 else "") + f" · {걸림:.1f} 초", flush=True)
    돌린 = sum(1 for x in 보고 if x["result"] != "skipped")
    print(f"재생: 요청 {돌린} 중 같음 {같음수}" + (f" · 건너뜀 {len(보고) - 돌린}" if len(보고) > 돌린 else ""))
    if a.out:
        json.dump({"header": 헤더, "current": 지금, "build_differences": 알림, "server_differences": 다름, "results": 보고},
                  open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    sys.exit(0 if 같음수 == 돌린 else 1)


def main():
    p = argparse.ArgumentParser(description="GeulPU 디버거 v1 — 실행 기록 · 테스트 세트 · 재생")
    p.add_argument("--app", default=os.environ.get("GEULPU_APP", "http://127.0.0.1:44544"), help="GeulPU 앱의 주소(환경 변수 GEULPU_APP)")
    sp = p.add_subparsers(dest="command", required=True)
    sp.add_parser("status", help="기록 상태와 실행 목록")
    s = sp.add_parser("record-start", help="앱의 /v1 을 지나는 생성 요청을 실행 로그로 기록하기 시작")
    s.add_argument("name", nargs="?")
    sp.add_parser("record-stop", help="기록 멈춤")
    s = sp.add_parser("get", help="실행 로그를 받아 파일로")
    s.add_argument("name")
    s.add_argument("file")
    s = sp.add_parser("delete", help="실행 로그 지우기")
    s.add_argument("name")
    s = sp.add_parser("run-set", help="테스트 세트를 실행 하나로 기록")
    s.add_argument("testset")
    s.add_argument("name")
    s.add_argument("--seed", type=int, default=1000, help="시드가 없는 요청에 넣을 시드의 시작(+ 차례)")
    s = sp.add_parser("replay", help="실행 로그를 다시 돌려 비트까지 비교")
    s.add_argument("run", metavar="RUNLOG.jsonl|NAME")
    s.add_argument("--out", help="보고서 JSON")
    s.add_argument("--requests", help="이 차례(i)만 — 쉼표로")
    s.add_argument("--ignore-header", action="store_true", help="기록한 서버와 달라도 그대로 돌린다")
    a = p.parse_args()
    {"status": 상태, "record-start": 기록시작, "record-stop": 기록멈춤, "get": 받기, "delete": 지우기, "run-set": 테스트세트,
     "replay": 재생}[a.command](a)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
