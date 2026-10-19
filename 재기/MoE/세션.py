#!/usr/bin/env python3
"""MoE 재기 — Qwen Code 로 코딩 작업을 llama-server 에 돌리며 기록한다(프록시.py 를 거쳐).

  python 세션.py <세션설정.json>

세션설정.json: {"server": exe, "model": gguf, "server_args": [...], "qwen": qwen.cmd, "venv": venv 폴더, "home": 빈 홈,
               "out": 출력 폴더, "tasks": [{"name", "seed": 폴더|null, "prompt", "timeout_s"}],
               "control": [{"name", "messages", "max_tokens"}]}
작업마다: 시작 저장소를 out/work/<name> 으로 복사 → 프록시(44650 → 44651, 기록 out/<name>.jsonl) → qwen --yolo "<prompt>".
끝나면 대조 글(코딩이 아닌 글)을 같은 서버에 보내 out/control.jsonl 에 적는다.
"""
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.request

S = json.load(open(sys.argv[1], encoding="utf-8"))
OUT = S["out"]
os.makedirs(os.path.join(OUT, "work"), exist_ok=True)
HERE = os.path.dirname(os.path.abspath(__file__))
UP, PX = 44651, 44650


def 기다리기(url, p=None, 초=600):
    for _ in range(초):
        if p is not None and p.poll() is not None:
            return False
        try:
            urllib.request.urlopen(url, timeout=2).read()
            return True
        except Exception:
            time.sleep(1)
    return False


def 프록시(기록):
    p = subprocess.Popen([sys.executable, "-X", "utf8", os.path.join(HERE, "프록시.py"), str(PX), f"http://127.0.0.1:{UP}",
                          기록], stdout=subprocess.DEVNULL, stderr=open(기록 + ".err", "w"))
    assert 기다리기(f"http://127.0.0.1:{PX}/health", p, 30), "프록시가 뜨지 않음"
    return p


서버로그 = open(os.path.join(OUT, "server.log"), "w", encoding="utf-8")
srv = subprocess.Popen([S["server"], "-m", S["model"], "--port", str(UP)] + S["server_args"], stdout=서버로그,
                       stderr=subprocess.STDOUT)
assert 기다리기(f"http://127.0.0.1:{UP}/health", srv), "서버가 뜨지 않음"
print("서버 뜸", flush=True)

홈 = S["home"]
os.makedirs(os.path.join(홈, ".qwen"), exist_ok=True)
json.dump({"privacy": {"usageStatisticsEnabled": False}, "telemetry": {"enabled": False},
           "general": {"disableAutoUpdate": True, "disableUpdateNag": True}},
          open(os.path.join(홈, ".qwen", "settings.json"), "w", encoding="utf-8"))
env = dict(os.environ)
env.update({"HOME": 홈, "USERPROFILE": 홈, "OPENAI_BASE_URL": f"http://127.0.0.1:{PX}/v1", "OPENAI_API_KEY": "local",
            "OPENAI_MODEL": "qwen3.6-35b-a3b", "PYTHONIOENCODING": "utf-8",
            "PATH": os.path.join(S["venv"], "Scripts") + os.pathsep + env["PATH"]})
env.pop("VIRTUAL_ENV", None)

for t in S["tasks"]:
    작업 = os.path.join(OUT, "work", t["name"])
    if os.path.exists(작업):
        shutil.rmtree(작업)
    if t.get("seed"):
        shutil.copytree(t["seed"], 작업)
    else:
        os.makedirs(작업)
    기록 = os.path.join(OUT, t["name"] + ".jsonl")
    if os.path.exists(기록):
        os.remove(기록)
    px = 프록시(기록)
    t0 = time.time()
    with open(os.path.join(OUT, t["name"] + ".out"), "w", encoding="utf-8") as o:
        try:
            r = subprocess.run([S["qwen"], "--yolo", "--auth-type", "openai", "-o", "json", t["prompt"]], cwd=작업, env=env,
                               stdout=o, stderr=subprocess.STDOUT, timeout=t.get("timeout_s", 2700))
            code = r.returncode
        except subprocess.TimeoutExpired:
            code = "timeout"
    px.kill()
    px.wait()
    n = sum(1 for _ in open(기록, encoding="utf-8")) if os.path.exists(기록) else 0
    print(f"작업 {t['name']}: {time.time() - t0:.0f} s · 끝 {code} · 요청 {n}", flush=True)

px = 프록시(os.path.join(OUT, "control.jsonl"))
for c in S.get("control", []):
    몸 = {"messages": c["messages"], "max_tokens": c.get("max_tokens", 1500), "temperature": 0.7, "stream": False}
    r = urllib.request.Request(f"http://127.0.0.1:{PX}/v1/chat/completions", json.dumps(몸).encode(),
                               {"Content-Type": "application/json"})
    j = json.loads(urllib.request.urlopen(r, timeout=3600).read())
    print("대조", c["name"], j.get("timings", {}).get("predicted_n"), flush=True)
px.kill()
srv.kill()
srv.wait()
print("끝", flush=True)
