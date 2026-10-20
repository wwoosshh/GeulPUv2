#!/usr/bin/env python3
"""디버거 v1 4단계 — 재기: 재생 · 점수 매기기 · 분기 · 비교에 드는 시간(문서/디버거.md 의 4단계 결과).

    python 재기/디버거비용.py [--app http://127.0.0.1:44639] [--runs 묶음50-q4km,qwencode3] [--compare 묶음50-q4km:묶음50-q8]
                             [--out 보고.json]

실행마다 그 실행을 생성한 모델을 불러와(도구/debugger.py load run:…):
  1. 재생 — 프롬프트 캐시를 비우고 서버를 다시 띄운 뒤 한 번(차가운 캐시), 곧바로 한 번 더(캐시가 찬 상태). 요청 · 분기마다 걸린
     시간과, 기록한 때 그 요청이 걸린 시간(요청 줄의 ms — 스트리밍이었으면 스트리밍 시간)을 나란히 적는다. 재생이 모두 같아야 잰 값으로 친다.
  2. 점수 매기기 — `score --check`(실행의 점수 줄을 다시 계산해 비트까지 비교 — 덧붙이지 않는다)의 요청마다 시간과 답 토큰 수.
그리고 두 실행의 비교 계산(GPU 없음)을 세 번 재어 가장 짧은 시간. 기록 비용은 1단계(재기/디버거기록비용.py), GPU 없는 CPU 확인은
3단계(도구/repro_check.py)에 있다. 표준 라이브러리만.
"""
import argparse
import importlib.util
import json
import os
import re
import subprocess
import sys
import tempfile
import time
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
도구 = os.path.join(ROOT, "도구", "debugger.py")
_spec = importlib.util.spec_from_file_location("debugger", 도구)
d = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(d)


def 묻기(앱, 방법, 길, 몸=None):
    r = urllib.request.Request(앱 + 길, data=None if 몸 is None else json.dumps(몸).encode("utf-8"), method=방법,
                               headers={"Content-Type": "application/json"})
    b = urllib.request.urlopen(r, timeout=900).read()
    return json.loads(b) if b else None


def 도구돌리기(앱, *인자):
    t0 = time.perf_counter()
    p = subprocess.run([sys.executable, "-X", "utf8", 도구, "--app", 앱, *인자], capture_output=True, text=True, encoding="utf-8")
    return p.returncode, p.stdout, time.perf_counter() - t0


def 서버준비(앱, 모델):
    for _ in range(900):
        sv = 묻기(앱, "GET", "/api/state")["server"]
        if sv["state"] == "ready" and sv["model"] == 모델:
            return sv
        time.sleep(1)
    raise SystemExit("서버가 준비되지 않았다")


def 차가운캐시(앱, 모델):
    """프롬프트 캐시를 비우고 서버를 다시 띄운다 — 슬롯도 비고 저장소에도 없다"""
    묻기(앱, "POST", "/api/server/stop", {})
    print("프롬프트 캐시 비움:", 묻기(앱, "POST", "/api/store/clear", {}), flush=True)
    묻기(앱, "POST", "/api/server/start", {"model": 모델})
    서버준비(앱, 모델)


def 재생재기(앱, 이름):
    with tempfile.TemporaryDirectory() as 폴더:
        p = os.path.join(폴더, "재생.json")
        rc, 출력, 걸림 = 도구돌리기(앱, "replay", 이름, "--out", p)
        if rc != 0:
            raise SystemExit(f"재생이 같지 않다 — 잰 값으로 치지 않는다:\n{출력[-1500:]}")
        보고 = json.load(open(p, encoding="utf-8"))
    요청 = [x for x in 보고["results"] if "i" in x and x["result"] != "skipped"]
    분기 = [x for x in 보고["results"] if "b" in x and x["result"] != "skipped"]
    return {"wall_seconds": round(걸림, 2), "requests": len(요청), "request_seconds": round(sum(x["seconds"] for x in 요청), 2),
            "branches": len(분기), "branch_seconds": round(sum(x["seconds"] for x in 분기), 2),
            "steps": sum(x.get("steps", 0) for x in 요청), "all_same": all(x["result"] == "same" for x in 요청 + 분기)}


def 점수재기(앱, 이름):
    rc, 출력, 걸림 = 도구돌리기(앱, "score", 이름, "--check")
    if rc != 0:
        raise SystemExit(f"점수를 다시 계산하니 다르다:\n{출력[-1500:]}")
    줄 = re.findall(r"^\s*\d+ 스텝 (\d+) · 같음 · ([\d.]+) 초$", 출력, re.M)
    return {"wall_seconds": round(걸림, 2), "requests": len(줄), "tokens": sum(int(a) for a, _ in 줄),
            "request_seconds": round(sum(float(b) for _, b in 줄), 2)}


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description="디버거 v1 — 재생 · 점수 매기기 · 비교에 드는 시간")
    ap.add_argument("--app", default="http://127.0.0.1:44639")
    ap.add_argument("--runs", default="묶음50-q4km,qwencode3")
    ap.add_argument("--compare", default="묶음50-q4km:묶음50-q8")
    ap.add_argument("--out")
    a = ap.parse_args()
    보고 = {"app": a.app, "runs": {}}
    for 이름 in [x for x in a.runs.split(",") if x]:
        실 = d.실행불러오기(a.app, 이름)
        sv = d.기록서버(실.헤더, 실.요청들[0])
        rc, 출력, _ = 도구돌리기(a.app, "load", "run:" + 이름)
        print(출력.strip(), flush=True)
        모델 = 묻기(a.app, "GET", "/api/state")["server"]["model"]
        기록 = {"requests": len(실.요청들), "recorded_seconds": round(sum(x.get("ms", 0) for x in 실.요청들) / 1000, 2),
                "prompt_tokens": sum(len(d.응답덧(x).get("prompt_tokens") or []) for x in 실.요청들),
                "answer_tokens": sum(len(d.응답덧(x).get("tokens") or []) for x in 실.요청들), "server": sv}
        차가운캐시(a.app, 모델)
        찬 = 재생재기(a.app, 이름)
        print(f"{이름}: 재생(차가운 캐시) {찬}", flush=True)
        더 = 재생재기(a.app, 이름)
        print(f"{이름}: 재생(캐시 찬 상태) {더}", flush=True)
        점 = 점수재기(a.app, 이름)
        print(f"{이름}: 점수 다시 계산 {점}", flush=True)
        보고["runs"][이름] = {"recorded": 기록, "replay_cold": 찬, "replay_warm": 더, "score": 점}
    if a.compare:
        A이름, B이름 = a.compare.split(":")
        A, B = d.실행불러오기(a.app, A이름), d.실행불러오기(a.app, B이름)
        시간 = []
        for _ in range(3):
            t0 = time.perf_counter()
            r = d.비교계산(A, B)
            시간.append(time.perf_counter() - t0)
        보고["compare"] = {"a": A이름, "b": B이름, "seconds": round(min(시간), 4), "requests": r["summary"]["requests"],
                           "tokens": r["summary"]["tokens"]}
        print(f"비교 {A이름} ↔ {B이름}: {보고['compare']}", flush=True)
    if a.out:
        json.dump(보고, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
