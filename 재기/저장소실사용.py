"""GeulPUv2 문맥 저장소 4단계 — 실사용: Qwen Code 로 코딩 작업 셋(재기/MoE — 할 일 도구 · 가계부 · 메모 웹앱)을 GeulPU 앱의 OpenAI
호환 길(/v1)에 문맥 저장소 끔 · 켬으로 돌린다. 요청마다 재기/MoE/프록시.py 로 적고(geulpu 덧붙임 — 첫 토큰 · 이 슬롯 · 다른 슬롯 ·
저장소에서 다시 쓴 토큰), 실행 · 작업마다 요약한다. 앱(작업용 — 다른 포트 · 다른 자료 폴더)은 먼저 띄워 모델을 더해 둔다.

    python -X utf8 재기/저장소실사용.py <설정.json>

설정.json: {"app": "http://127.0.0.1:44639", "model": "<앱의 모델 아이디>", "qwen": qwen.cmd, "venv": venv 폴더, "home": 빈 홈,
            "out": 출력 폴더, "tasks": [{"name", "seed": 폴더|null, "prompt", "timeout_s"}],
            "runs": [{"name": "끔", "store": "off", "clear": true}, {"name": "켬", "store": "on", "clear": true},
                     {"name": "켬_다시", "store": "on", "clear": false}]}
실행마다: 앱 설정 context_store → (clear 면 저장소 지우기) → 서버 멈추기 · 불러오기(설정은 불러올 때 적용 — 멈추기 전에 앱이
저장소를 비운다) → 작업마다 빈 홈(Qwen Code 의 메모리가 시스템 프롬프트에 든다 — 실행끼리 같게)과 시작 저장소를
out/work/<작업>(실행끼리 같은 경로 — 시스템 프롬프트의 작업 폴더가 같게)으로 새로 복사 → 프록시 → qwen --yolo "<지시>" → 그 폴더에서
pytest(에이전트가 일을 끝냈나). 결과: out/<실행>/<작업>.jsonl(요청마다) · .out(Qwen Code 출력), out/요약.json · 요약.txt
"""
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
PX = 44652


def 앱(S, 방법, 길, 몸=None, 시간=600):
    r = urllib.request.Request(S["app"] + 길, data=None if 몸 is None else json.dumps(몸).encode("utf-8"), method=방법,
                               headers={"Content-Type": "application/json"})
    d = urllib.request.urlopen(r, timeout=시간).read()
    return json.loads(d) if d else None


def 기다리기(url, p=None, 초=60):
    for _ in range(초 * 2):
        if p is not None and p.poll() is not None:
            return False
        try:
            urllib.request.urlopen(url, timeout=2).read()
            return True
        except Exception:
            time.sleep(0.5)
    return False


def 서버다시(S):
    앱(S, "POST", "/api/server/stop", {})
    앱(S, "POST", "/api/server/start", {"model": S["model"]})
    t0 = time.time()
    while time.time() - t0 < 600:
        sv = 앱(S, "GET", "/api/state")["server"]
        if sv["state"] == "ready":
            return sv
        if sv["state"] == "failed":
            raise SystemExit(f"서버를 올리지 못함: {sv.get('reason')}")
        time.sleep(1)
    raise SystemExit("서버가 600 초 안에 준비되지 않음")


def 요약(기록):
    줄 = [json.loads(l) for l in open(기록, encoding="utf-8") if l.strip()] if os.path.exists(기록) else []
    g = [x.get("geulpu") or {} for x in 줄]
    st = [x.get("store") or {} for x in g]
    return {"요청": len(줄),
            "요청시간합초": round(sum(x["t1"] - x["t0"] for x in 줄), 1),
            "첫토큰합초": round(sum((x.get("first_token_ms") or 0) for x in g) / 1e6, 2),          # 서버의 이 칸은 µs(이름만 ms)
            "생성합초": round(sum((x.get("generate_us") or 0) for x in g) / 1e6, 2),
            "프롬프트": sum((x.get("usage") or {}).get("prompt_tokens") or 0 for x in 줄),
            "생성": sum((x.get("usage") or {}).get("completion_tokens") or 0 for x in 줄),
            "이슬롯": sum(x.get("slot_cached_tokens", 0) for x in st),
            "다른슬롯": sum(x.get("slot_copy_tokens", 0) for x in st),
            "저장소": sum(x.get("restored_tokens", 0) for x in st),
            "되살림초": round(sum(x.get("restore_ms", 0.0) for x in st) / 1000, 2),
            "첫요청": {"프롬프트": (줄[0].get("usage") or {}).get("prompt_tokens"), "첫토큰ms": round((g[0].get("first_token_ms") or 0) / 1000),
                     "저장소": st[0].get("restored_tokens", 0), "다른슬롯": st[0].get("slot_copy_tokens", 0),
                     "깨진자리": st[0].get("reuse_end"), "까닭": st[0].get("break")} if 줄 else None}


def main():
    S = json.load(open(sys.argv[1], encoding="utf-8"))
    OUT = S["out"]
    os.makedirs(os.path.join(OUT, "work"), exist_ok=True)
    홈 = S["home"]

    def 홈새로():           # 작업마다 빈 홈 — Qwen Code 의 메모리(시스템 프롬프트에 들어간다)가 실행끼리 같게
        if os.path.exists(홈):
            shutil.rmtree(홈)
        os.makedirs(os.path.join(홈, ".qwen"))
        json.dump({"privacy": {"usageStatisticsEnabled": False}, "telemetry": {"enabled": False},
                   "general": {"disableAutoUpdate": True, "disableUpdateNag": True}},
                  open(os.path.join(홈, ".qwen", "settings.json"), "w", encoding="utf-8"))
    env = dict(os.environ)
    env.update({"HOME": 홈, "USERPROFILE": 홈, "OPENAI_BASE_URL": f"http://127.0.0.1:{PX}/v1", "OPENAI_API_KEY": "local",
                "OPENAI_MODEL": S["model"], "PYTHONIOENCODING": "utf-8",
                "PATH": os.path.join(S["venv"], "Scripts") + os.pathsep + env["PATH"]})
    env.pop("VIRTUAL_ENV", None)
    결과 = json.load(open(os.path.join(OUT, "요약.json"), encoding="utf-8")) if os.path.exists(os.path.join(OUT, "요약.json")) else {}
    for 실행 in S["runs"]:
        폴더 = os.path.join(OUT, 실행["name"])
        os.makedirs(폴더, exist_ok=True)
        앱(S, "POST", "/api/settings", {"context_store": 실행["store"]})
        if 실행.get("clear"):
            앱(S, "POST", "/api/store/clear", {})
        sv = 서버다시(S)
        print(f"== {실행['name']}: 저장소 {실행['store']} · 서버 {sv['state']} ({sv.get('seconds')} 초) · 슬롯 {sv.get('slots')} · "
              f"문맥 {sv.get('context')}", flush=True)
        for t in S["tasks"]:
            홈새로()
            작업 = os.path.join(OUT, "work", t["name"])
            if os.path.exists(작업):
                shutil.rmtree(작업)
            if t.get("seed"):
                shutil.copytree(t["seed"], 작업)
            else:
                os.makedirs(작업)
            기록 = os.path.join(폴더, t["name"] + ".jsonl")
            if os.path.exists(기록):
                os.remove(기록)
            px = subprocess.Popen([sys.executable, "-X", "utf8", os.path.join(HERE, "MoE", "프록시.py"), str(PX), S["app"], 기록, "geulpu"],
                                  stdout=subprocess.DEVNULL, stderr=open(기록 + ".err", "w"))
            assert 기다리기(f"http://127.0.0.1:{PX}/v1/models", px), "프록시가 뜨지 않음"
            t0 = time.time()
            with open(os.path.join(폴더, t["name"] + ".out"), "w", encoding="utf-8") as o:
                try:
                    r = subprocess.run([S["qwen"], "--yolo", "--auth-type", "openai", "-o", "json", t["prompt"]], cwd=작업, env=env,
                                       stdout=o, stderr=subprocess.STDOUT, timeout=t.get("timeout_s", 1800))
                    code = r.returncode
                except subprocess.TimeoutExpired:
                    code = "timeout"
            걸림 = time.time() - t0
            px.kill()
            px.wait()
            시험 = subprocess.run([os.path.join(S["venv"], "Scripts", "python.exe"), "-m", "pytest", "-q"], cwd=작업, env=env,
                                capture_output=True, timeout=600)
            y = 요약(기록)
            y.update({"걸린초": round(걸림, 1), "끝": code, "pytest": 시험.returncode,
                      "pytest꼬리": 시험.stdout.decode("utf-8", "replace").strip().splitlines()[-1:]})
            결과.setdefault(실행["name"], {})[t["name"]] = y
            print(f"{실행['name']} {t['name']}: {걸림:.0f} 초 · 요청 {y['요청']} · 첫 토큰 합 {y['첫토큰합초']} 초 · 생성 합 {y['생성합초']} 초 · "
                  f"프롬프트 {y['프롬프트']:,} (이 슬롯 {y['이슬롯']:,} · 다른 슬롯 {y['다른슬롯']:,} · 저장소 {y['저장소']:,}) · "
                  f"첫 요청 {y['첫요청']} · pytest {시험.returncode} {y['pytest꼬리']}", flush=True)
            json.dump(결과, open(os.path.join(OUT, "요약.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        결과.setdefault(실행["name"], {})["저장소상태"] = 앱(S, "GET", "/api/store")
    json.dump(결과, open(os.path.join(OUT, "요약.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("끝", flush=True)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
