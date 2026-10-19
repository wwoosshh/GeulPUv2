#!/usr/bin/env python3
"""MoE 재기 — llama-server 를 설정마다 띄워 같은 요청의 생성 · 프롬프트 속도를 잰다.

  python 설정재기.py <설정.json> <요청.json> <출력.jsonl>

설정.json: {"server": exe, "model": gguf, "common": [...], "configs": [{"name": ..., "args": [...]}, ...]}
요청.json: [{"name": ..., "messages" | "prompt": ..., "max_tokens": n}, ...] — 첫 요청은 데우기(기록은 하되 가운데값에서 뺌)
서버의 timings(prompt_n · cache_n · prompt_per_second · predicted_n · predicted_per_second)를 적는다.
"""
import json
import os
import statistics
import subprocess
import sys
import time
import urllib.request

설정 = json.load(open(sys.argv[1], encoding="utf-8"))
요청들 = json.load(open(sys.argv[2], encoding="utf-8"))
출력 = sys.argv[3]
PORT = 설정.get("port", 44651)
URL = f"http://127.0.0.1:{PORT}"


def 부르기(경로, 몸, 시간=3600):
    r = urllib.request.Request(URL + 경로, json.dumps(몸).encode(), {"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(r, timeout=시간).read())


def vram():
    try:
        return int(subprocess.check_output(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                                           text=True).split()[0])
    except Exception:
        return None


def 기다리기(p, 로그):
    for _ in range(600):
        if p.poll() is not None:
            return False
        try:
            if json.loads(urllib.request.urlopen(URL + "/health", timeout=2).read()).get("status") == "ok":
                return True
        except Exception:
            pass
        time.sleep(1)
    return False


for c in 설정["configs"]:
    로그 = os.path.join(os.path.dirname(출력), f"srv_{c['name']}.log")
    인자 = [설정["server"], "-m", 설정["model"], "--port", str(PORT)] + 설정.get("common", []) + c["args"]
    t0 = time.time()
    with open(로그, "w", encoding="utf-8") as lf:
        p = subprocess.Popen(인자, stdout=lf, stderr=subprocess.STDOUT, cwd=설정.get("cwd"))
    줄 = {"config": c["name"], "args": c["args"]}
    if not 기다리기(p, 로그):
        줄["error"] = "서버가 뜨지 않음: " + open(로그, encoding="utf-8", errors="replace").read()[-1500:]
        print(c["name"], "실패", flush=True)
    else:
        줄["load_s"] = round(time.time() - t0, 1)
        줄["vram_mib"] = vram()
        결과 = []
        for k, q in enumerate(요청들):
            몸 = {"max_tokens": q.get("max_tokens", 256), "temperature": 0, "cache_prompt": q.get("cache_prompt", True)}
            몸.update(q.get("extra", {}))
            if "body" in q:                      # 에이전트 기록의 요청 그대로(도구 포함) — 흘림 · 길이만 바꿈
                b = {k: v for k, v in q["body"].items() if k not in ("stream", "stream_options", "max_tokens")}
                j = 부르기("/v1/chat/completions", dict(b, **몸))
            elif "messages" in q:
                j = 부르기("/v1/chat/completions", dict(몸, messages=q["messages"]))
            else:
                j = 부르기("/completion", dict(몸, prompt=q["prompt"], n_predict=몸.pop("max_tokens")))
            t = j.get("timings", {})
            결과.append({"name": q.get("name", k), "warm": q.get("warm", k == 0), **{x: t.get(x) for x in (
                "prompt_n", "cache_n", "prompt_per_second", "predicted_n", "predicted_per_second", "draft_n",
                "draft_n_accepted")}})
            print(c["name"], 결과[-1], flush=True)
        줄["vram_after_mib"] = vram()
        줄["requests"] = 결과
        잰 = [r["predicted_per_second"] for r in 결과 if not r["warm"] and r["predicted_per_second"]]
        줄["tg_median"] = round(statistics.median(잰), 2) if 잰 else None
        print(c["name"], "생성 가운데", 줄["tg_median"], "VRAM", 줄["vram_mib"], flush=True)
    p.kill()
    p.wait()
    time.sleep(3)
    with open(출력, "a", encoding="utf-8") as f:
        f.write(json.dumps(줄, ensure_ascii=False) + "\n")
