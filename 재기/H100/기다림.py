"""서버가 답할 때까지 기다린다: python3 재기/H100/기다림.py URL PID 제한초 — 0 이면 답함, 1 이면 시간 넘김, 2 면 서버 프로세스가 죽음(PID 0 이면 보지 않음)."""
import json
import os
import sys
import time
import urllib.request

url, pid, limit = sys.argv[1], int(sys.argv[2]), float(sys.argv[3])
t0 = time.time()
while True:
    if pid > 0:
        try:
            os.kill(pid, 0)
        except OSError:
            sys.exit(2)
    try:
        body = {"model": "m", "prompt": [1000, 1001], "max_tokens": 1, "temperature": 0}
        r = urllib.request.Request(url + "/v1/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
        urllib.request.urlopen(r, timeout=10).read()
        sys.exit(0)
    except Exception:
        if time.time() - t0 > limit:
            sys.exit(1)
        time.sleep(2)
