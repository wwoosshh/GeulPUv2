#!/usr/bin/env python3
"""MoE 재기 — 코딩 에이전트와 llama-server 사이의 기록용 프록시(표준 라이브러리만).

  python 프록시.py <듣는 포트> <llama-server 주소> <기록.jsonl> [geulpu]

요청 · 응답을 그대로 넘기고, /v1/chat/completions 는 요청 JSON 과 응답(흘림이면 조각을 모은 것 — reasoning_content · content ·
tool_calls)과 서버의 timings(prompt_n · cache_n · predicted_n · 시간)를 한 줄씩 적는다. 넷째 인자 geulpu 면 GeulPU 서버 · 앱 앞에서
채팅 요청에 "geulpu": {} 를 넣어(덧붙임 — 첫 토큰 · 캐시 · 문맥 저장소) 응답의 geulpu 도 적는다(문맥 저장소 4단계).
"""
import http.client
import json
import sys
import threading
import time
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

PORT = int(sys.argv[1])
UP = urllib.parse.urlparse(sys.argv[2])
LOG = sys.argv[3]
덧붙임 = len(sys.argv) > 4 and sys.argv[4] == "geulpu"
잠금 = threading.Lock()
차례 = [0]


def 적기(줄):
    with 잠금:
        with open(LOG, "a", encoding="utf-8") as f:
            f.write(json.dumps(줄, ensure_ascii=False) + "\n")


class 손(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.0"           # 응답마다 연결을 닫는다 — 흘림의 끝을 연결 닫기로 알린다

    def log_message(self, *a):
        pass

    def 넘기기(self, 몸):
        c = http.client.HTTPConnection(UP.hostname, UP.port, timeout=3600)
        머리 = {k: v for k, v in self.headers.items() if k.lower() not in ("host", "content-length", "connection",
                                                                        "accept-encoding")}
        머리["Content-Length"] = str(len(몸 or b""))
        c.request(self.command, self.path, body=몸, headers=머리)
        return c, c.getresponse()

    def do_GET(self):
        c, r = self.넘기기(None)
        d = r.read()
        self.send_response(r.status)
        for k, v in r.getheaders():
            if k.lower() not in ("transfer-encoding", "connection", "content-length"):
                self.send_header(k, v)
        self.send_header("Content-Length", str(len(d)))
        self.end_headers()
        self.wfile.write(d)
        c.close()

    def do_POST(self):
        n = int(self.headers.get("Content-Length") or 0)
        몸 = self.rfile.read(n)
        채팅 = self.path.rstrip("/").endswith("/chat/completions")
        if 채팅 and 덧붙임:
            try:
                j0 = json.loads(몸)
                j0.setdefault("geulpu", {})
                몸 = json.dumps(j0, ensure_ascii=False).encode("utf-8")
            except ValueError:
                pass
        with 잠금:
            차례[0] += 1
            i = 차례[0]
        t0 = time.time()
        c, r = self.넘기기(몸)
        self.send_response(r.status)
        for k, v in r.getheaders():
            if k.lower() not in ("transfer-encoding", "connection", "content-length"):
                self.send_header(k, v)
        self.end_headers()
        모음 = {"reasoning_content": "", "content": "", "tool_calls": {}, "finish_reason": None}
        timings, usage, 날것, gp = None, None, b"", None
        흘림 = "text/event-stream" in (r.getheader("Content-Type") or "")
        if 흘림:
            while True:
                줄 = r.readline()
                if not 줄:
                    break
                self.wfile.write(줄)
                self.wfile.flush()
                s = 줄.strip()
                if not s.startswith(b"data:") or s == b"data: [DONE]":
                    continue
                try:
                    j = json.loads(s[5:])
                except ValueError:
                    continue
                timings = j.get("timings", timings)
                usage = j.get("usage", usage)
                gp = j.get("geulpu", gp)
                for ch in j.get("choices") or []:
                    d = ch.get("delta") or {}
                    모음["reasoning_content"] += d.get("reasoning_content") or ""
                    모음["content"] += d.get("content") or ""
                    for tc in d.get("tool_calls") or []:
                        k = tc.get("index", 0)
                        t = 모음["tool_calls"].setdefault(k, {"id": None, "name": "", "arguments": ""})
                        t["id"] = tc.get("id") or t["id"]
                        f = tc.get("function") or {}
                        t["name"] += f.get("name") or ""
                        t["arguments"] += f.get("arguments") or ""
                    if ch.get("finish_reason"):
                        모음["finish_reason"] = ch["finish_reason"]
        else:
            날것 = r.read()
            self.wfile.write(날것)
            try:
                j = json.loads(날것)
                timings, usage, gp = j.get("timings"), j.get("usage"), j.get("geulpu")
                m = (j.get("choices") or [{}])[0].get("message") or {}
                모음 = {"reasoning_content": m.get("reasoning_content") or "", "content": m.get("content") or "",
                      "tool_calls": {k: {"id": t.get("id"), "name": t["function"]["name"],
                                         "arguments": t["function"]["arguments"]}
                                     for k, t in enumerate(m.get("tool_calls") or [])},
                      "finish_reason": (j.get("choices") or [{}])[0].get("finish_reason")}
            except (ValueError, KeyError, TypeError):
                pass
        c.close()
        if 채팅:
            try:
                요청 = json.loads(몸)
            except ValueError:
                요청 = None
            적기({"i": i, "t0": t0, "t1": time.time(), "status": r.status, "stream": 흘림, "request": 요청,
                "response": 모음, "timings": timings, "usage": usage, "geulpu": gp})


ThreadingHTTPServer(("127.0.0.1", PORT), 손).serve_forever()
