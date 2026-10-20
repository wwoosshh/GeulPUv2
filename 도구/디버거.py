#!/usr/bin/env python3
"""GeulPU 디버거 1판(문서/디버거.md) — 실행 기록 · 시험 묶음 · 재생을 스크립트로. 앱(GeulPU)의 API 와 /v1 으로 묻는다. 표준 라이브러리만.

    python 도구/디버거.py [--앱 http://127.0.0.1:44544] 상태
    python 도구/디버거.py 기록시작 [이름]          # 앱의 /v1 을 지나는 생성 요청을 실행 파일에(자료\\실행기록\\<이름>.jsonl)
    python 도구/디버거.py 기록멈춤
    python 도구/디버거.py 받기 <이름> <파일.jsonl>
    python 도구/디버거.py 지우기 <이름>
    python 도구/디버거.py 돌리기 <시험묶음.jsonl> <이름> [--씨앗 1000]
    python 도구/디버거.py 재생 <실행.jsonl | 이름> [--결과 보고.json] [--요청 3,7] [--머리무시]

실행 파일(JSONL): 첫 줄 머리 {"geulrun": 1, "app", "commit", "model": {id, file, sha256, …}, "server": {layer, precision, context,
yarn, …}, "build": {server — 그 층의 서버 실행 파일 sha256, module, module_sha256}}, 그 뒤
요청마다 {"i", "at", "ms", "path", "status", "model", "server", "request", "response"} — server 는 그 요청을 셈한 서버(모델 sha256 · 층 ·
계산 방식 · 문맥 · YaRN 배), response 는 서버의 응답 JSON(흘림이면 마지막 사건)이고 그 "geulpu" 에 씨앗 · prompt_tokens · tokens ·
logit_hashes 가 있다.

시험 묶음(JSONL): 줄마다 채팅 요청 하나({"messages": …, "max_tokens": …, "temperature": …}) 또는 {"path": "/v1/completions",
"request": {…}}. 씨앗이 없으면 --씨앗 + 차례를 넣는다(같은 묶음이면 같은 실행). 흘림은 끈다.

재생: 기록한 요청을 그대로(씨앗은 기록한 값으로 박아) 다시 보내 프롬프트 토큰 · 나온 토큰 · 걸음마다 로짓 해시 · 끝 까닭을 견준다.
오류로 끝난 요청(문맥 초과 등)은 같은 오류(HTTP 상태 · 코드)가 나오는지 본다.
요청을 셈한 서버의 모델 sha256 · 계산 방식(정밀 · 빠른) · YaRN 배(빠른 모드면 층도)가 지금과 다르면 다른 셈이라 먼저 알리고 멈춘다
(--머리무시 면 그대로). GeulPU 판 · 서버 실행 파일 · 모듈(머리의 build)이 다르면 알리기만 하고 돌린다 — 새 빌드가 비트를 바꿨나를 재생이
바로 본다.
"""
import argparse
import copy
import json
import os
import sys
import time
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


def 실행읽기(경로):
    줄 = [json.loads(l) for l in open(경로, encoding="utf-8") if l.strip()]
    if not 줄 or 줄[0].get("geulrun") != 1:
        raise SystemExit(f"GeulPU 실행 파일이 아니다: {경로}")
    return 줄[0], 줄[1:]


def 지금머리(앱):
    """지금 앱 판과 서버 — 요청 줄의 "server" 와 같은 꼴(YaRN 배는 엔진의 규칙대로)"""
    st = 묻기(앱, "GET", "/api/state")
    sv = st["server"]
    m = next((x for x in st.get("models", []) if x.get("id") == sv.get("model")), None)
    s = (m or {}).get("shape") or {}
    얀 = 0
    if s.get("yarn_orig", 0) > 0 and sv.get("context", 0) > s.get("context_native", 0):
        얀 = 2 if sv["context"] <= 2 * s["yarn_orig"] else 4
    빌드 = {}
    try:                                            # 서버 실행 파일 · 불러온 모듈의 해시(출처 — 모듈은 파일을 셈하므로 조금 걸린다)
        p = 묻기(앱, "GET", "/api/provenance")
        빌드 = {"bins": {x.get("file"): x.get("sha256") for x in p.get("binaries") or []},
                "module_sha256": (p.get("module") or {}).get("sha256")}
    except SystemExit:
        pass
    return {"app": (st.get("app") or {}).get("version"), "state": sv.get("state"), "model": sv.get("model"), "build": 빌드,
            "server": {"sha256": (m or {}).get("sha256"), "layer": sv.get("layer"), "precision": sv.get("precision"),
                       "context": sv.get("context"), "yarn": 얀}}


def 빌드다름(머리, 지금):
    """알림만 — GeulPU 판 · 서버 실행 파일 · 모듈이 기록한 때와 다른가(같은 층 · 계산 방식일 때만 견준다). 다르면 비트가 다를 수 있다 —
    재생이 바로 그것을 본다(새 빌드가 비트를 바꿨나)"""
    짧게 = lambda v: v[:12] if isinstance(v, str) and len(v) > 12 else v
    다름 = []
    if 머리.get("app") != 지금["app"]:
        다름.append(f"GeulPU 판 {머리.get('app')} → {지금['app']}")
    b = 머리.get("build") or {}
    sv = 머리.get("server") or {}
    if b and sv.get("layer") == 지금["server"]["layer"]:
        파일 = {"cuda": "bin\\server-cuda.exe", "vulkan": "bin\\server-vulkan.exe"}.get(sv.get("layer"))
        지금서버 = (지금.get("build") or {}).get("bins", {}).get(파일)
        if b.get("server") and 지금서버 and b["server"] != 지금서버:
            다름.append(f"서버 빌드 {짧게(b['server'])} → {짧게(지금서버)}")
        지금모듈 = (지금.get("build") or {}).get("module_sha256")
        if sv.get("precision") == 지금["server"]["precision"] and b.get("module_sha256") and 지금모듈 and b["module_sha256"] != 지금모듈:
            다름.append(f"모듈 {짧게(b['module_sha256'])} → {짧게(지금모듈)}")
    return 다름


def 기록서버(머리, x):
    """요청을 셈한 서버 — 요청 줄의 "server"(없으면 머리에서)"""
    if isinstance(x.get("server"), dict):
        return x["server"]
    sv = 머리.get("server") or {}
    return {"sha256": (머리.get("model") or {}).get("sha256"), "layer": sv.get("layer"), "precision": sv.get("precision"),
            "context": sv.get("context"), "yarn": sv.get("yarn")}


def 서버다름(기록, 지금):
    """비트가 같으려면 같아야 하는 것 — 모델 sha256 · 계산 방식 · YaRN 배, 빠른 모드면 층도(CUDA ↔ Vulkan 은 정밀 모드만 같다)"""
    짧게 = lambda v: v[:12] if isinstance(v, str) and len(v) > 12 else v
    다름 = []
    for k, 이름 in (("sha256", "모델 sha256"), ("precision", "계산 방식"), ("yarn", "YaRN 배")):
        if 기록.get(k) != 지금[k]:
            다름.append(f"{이름} {짧게(기록.get(k))} → {짧게(지금[k])}")
    if 기록.get("precision") == "fast" == 지금["precision"] and 기록.get("layer") != 지금["layer"]:
        다름.append(f"층 {기록.get('layer')} → {지금['layer']}(빠른 모드)")
    return 다름


def 상태(a):
    d = 묻기(a.앱, "GET", "/api/debug")
    print(json.dumps(d, ensure_ascii=False, indent=1))


def 기록시작(a):
    d = 묻기(a.앱, "POST", "/api/debug/record/start", {"name": a.이름 or ""})
    print(f"기록 시작: {d['name']} — {d['dir']}")


def 기록멈춤(a):
    d = 묻기(a.앱, "POST", "/api/debug/record/stop", {})
    print(f"기록 멈춤: {d.get('name')} · 요청 {d.get('count')}")


def 받기(a):
    d = 묻기(a.앱, "GET", "/api/debug/run?name=" + urllib.parse.quote(a.이름), 날것=True)
    open(a.파일, "wb").write(d)
    print(f"받음: {a.파일} ({len(d):,} 바이트)")


def 지우기(a):
    묻기(a.앱, "POST", "/api/debug/run/delete", {"name": a.이름})
    print(f"지움: {a.이름}")


def 돌리기(a):
    묶음 = [json.loads(l) for l in open(a.묶음, encoding="utf-8") if l.strip()]
    d = 묻기(a.앱, "POST", "/api/debug/record/start", {"name": a.이름})
    print(f"기록 시작: {d['name']} · 요청 {len(묶음)}", flush=True)
    try:
        for i, x in enumerate(묶음):
            길 = x.get("path", "/v1/chat/completions") if "request" in x else "/v1/chat/completions"
            몸 = copy.deepcopy(x["request"] if "request" in x else x)
            몸["stream"] = False
            몸.setdefault("seed", a.씨앗 + i)
            t0 = time.time()
            r = 묻기(a.앱, "POST", 길, 몸)
            g = r.get("geulpu") or {}
            print(f"{i:3d} {길} · 프롬프트 {(r.get('usage') or {}).get('prompt_tokens')} · 생성 {len(g.get('tokens') or [])} · "
                  f"{(r['choices'][0].get('finish_reason'))} · {time.time() - t0:.1f} 초", flush=True)
    finally:
        d = 묻기(a.앱, "POST", "/api/debug/record/stop", {})
        print(f"기록 멈춤: {d.get('name')} · 요청 {d.get('count')}", flush=True)


def 재생(a):
    경로 = a.실행
    if not os.path.exists(경로):                     # 앱의 실행 이름이면 받아서
        경로 = None
        본 = 묻기(a.앱, "GET", "/api/debug/run?name=" + urllib.parse.quote(a.실행), 날것=True).decode("utf-8")
        줄 = [json.loads(l) for l in 본.splitlines() if l.strip()]
        머리, 기록 = 줄[0], 줄[1:]
    else:
        머리, 기록 = 실행읽기(경로)
    지금 = 지금머리(a.앱)
    if 지금["state"] != "ready":
        raise SystemExit(f"서버가 준비되지 않았다(state {지금['state']}) — 모델을 올린 뒤 재생한다")
    알림 = 빌드다름(머리, 지금)
    if 알림:
        print("기록한 빌드와 지금 빌드가 다르다 — " + " · ".join(알림) + " (커널이 바뀐 빌드면 비트가 다를 수 있다 — 재생으로 본다)")
    다름 = {}
    for x in 기록:
        for d in 서버다름(기록서버(머리, x), 지금["server"]):
            다름[d] = 다름.get(d, 0) + 1
    if 다름:
        print("기록한 서버와 지금 서버가 다르다 — " + " · ".join(f"{d}(요청 {n})" for d, n in 다름.items()))
        if not a.머리무시:
            raise SystemExit("재생을 멈춘다(--머리무시 로 그대로 돌릴 수 있다 — 비트가 다를 수 있다)")
    if a.요청:
        고른 = {int(v) for v in a.요청.split(",") if v.strip()}
        기록 = [x for x in 기록 if x.get("i") in 고른]
    보고 = []
    같음수 = 0
    for x in 기록:
        resp = x.get("response") or {}
        g = resp.get("geulpu") or {}
        if x.get("path") not in 생성길 or "request" not in x:
            보고.append({"i": x.get("i"), "결과": "건너뜀", "까닭": "생성 길이 아니거나 요청이 JSON 이 아니다"})
            continue
        몸 = copy.deepcopy(x["request"])
        몸["stream"] = False
        몸.pop("stream_options", None)
        if x.get("status") != 200:                  # 오류 응답이면 같은 오류(상태 · 코드)가 나오나
            if not isinstance(resp.get("error"), dict):
                보고.append({"i": x.get("i"), "결과": "건너뜀", "까닭": f"status {x.get('status')} · 오류 응답이 없다"})
                continue
            t0 = time.time()
            상태2, r = 묻기상태(a.앱, x["path"], 몸)
            걸림 = time.time() - t0
            코드1, 코드2 = resp["error"].get("code"), ((r or {}).get("error") or {}).get("code")
            틀린 = [] if 상태2 == x["status"] and 코드1 == 코드2 else [f"오류 {x['status']} {코드1} → {상태2} {코드2}"]
            같음수 += not 틀린
            보고.append({"i": x.get("i"), "결과": "같음" if not 틀린 else "다름", "틀린": 틀린, "오류": x["status"], "초": round(걸림, 2)})
            print(f"{x.get('i'):4d} {x['path']} · 오류 {x['status']} {코드1} · {'같음' if not 틀린 else '다름 — ' + 틀린[0]} · {걸림:.1f} 초",
                  flush=True)
            continue
        if g.get("tokens") is None:
            보고.append({"i": x.get("i"), "결과": "건너뜀", "까닭": "응답에 토큰이 없다(흘림이 끊겼거나 geulpu 가 없다)"})
            continue
        if "seed" in g:
            몸["seed"] = g["seed"]
        gp = 몸.get("geulpu") if isinstance(몸.get("geulpu"), dict) else {}
        gp.update({"logit_hash": True, "prompt_tokens": True})
        몸["geulpu"] = gp
        t0 = time.time()
        r = 묻기(a.앱, "POST", x["path"], 몸)
        걸림 = time.time() - t0
        g2 = r.get("geulpu") or {}
        끝1 = ((resp.get("choices") or [{}])[0]).get("finish_reason")
        끝2 = ((r.get("choices") or [{}])[0]).get("finish_reason")
        틀린 = []
        if g.get("prompt_tokens") is not None and g.get("prompt_tokens") != g2.get("prompt_tokens"):
            틀린.append("프롬프트 토큰")
        t1, t2 = g.get("tokens") or [], g2.get("tokens") or []
        h1, h2 = g.get("logit_hashes"), g2.get("logit_hashes")
        첫다른 = next((k for k in range(max(len(t1), len(t2))) if k >= len(t1) or k >= len(t2) or t1[k] != t2[k]), None)
        if 첫다른 is not None:
            틀린.append(f"토큰(걸음 {첫다른})")
        if h1 is not None and h1 != h2:
            k2 = next((k for k in range(max(len(h1), len(h2 or []))) if h2 is None or k >= len(h1) or k >= len(h2) or h1[k] != h2[k]), None)
            틀린.append(f"로짓 해시(걸음 {k2})")
        if 끝1 is not None and 끝1 != 끝2:
            틀린.append(f"끝 까닭 {끝1} → {끝2}")
        결과 = "같음" if not 틀린 else "다름"
        같음수 += not 틀린
        보고.append({"i": x.get("i"), "결과": 결과, "틀린": 틀린, "걸음": len(t1), "해시": h1 is not None, "초": round(걸림, 2)})
        print(f"{x.get('i'):4d} {x['path']} · 걸음 {len(t1)}{'' if h1 is not None else ' (해시 없음 — 토큰만)'} · {결과}"
              + (f" — {', '.join(틀린)}" if 틀린 else "") + f" · {걸림:.1f} 초", flush=True)
    돌린 = sum(1 for x in 보고 if x["결과"] != "건너뜀")
    print(f"재생: 요청 {돌린} 중 같음 {같음수}" + (f" · 건너뜀 {len(보고) - 돌린}" if len(보고) > 돌린 else ""))
    if a.결과:
        json.dump({"머리": 머리, "지금": 지금, "빌드다름": 알림, "다름": 다름, "보고": 보고}, open(a.결과, "w", encoding="utf-8"),
                  ensure_ascii=False, indent=1)
    sys.exit(0 if 같음수 == 돌린 else 1)


def main():
    p = argparse.ArgumentParser(description="GeulPU 디버거 1판 — 실행 기록 · 시험 묶음 · 재생")
    p.add_argument("--앱", default=os.environ.get("GEULPU_APP", "http://127.0.0.1:44544"))
    sp = p.add_subparsers(dest="명령", required=True)
    sp.add_parser("상태")
    s = sp.add_parser("기록시작")
    s.add_argument("이름", nargs="?")
    sp.add_parser("기록멈춤")
    s = sp.add_parser("받기")
    s.add_argument("이름")
    s.add_argument("파일")
    s = sp.add_parser("지우기")
    s.add_argument("이름")
    s = sp.add_parser("돌리기")
    s.add_argument("묶음")
    s.add_argument("이름")
    s.add_argument("--씨앗", type=int, default=1000)
    s = sp.add_parser("재생")
    s.add_argument("실행")
    s.add_argument("--결과")
    s.add_argument("--요청", help="이 차례(i)만 — 쉼표로")
    s.add_argument("--머리무시", action="store_true")
    a = p.parse_args()
    {"상태": 상태, "기록시작": 기록시작, "기록멈춤": 기록멈춤, "받기": 받기, "지우기": 지우기, "돌리기": 돌리기, "재생": 재생}[a.명령](a)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
