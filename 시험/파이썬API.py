#!/usr/bin/env python3
"""파이썬 API(도구/pypi/geulpu/api.py) 시험 — 세 가지로 돈다.

  python -I -X utf8 시험/파이썬API.py                                       # 빠른 점검 2f — 가짜 앱으로(GPU · 앱 없이)
  python -I -X utf8 시험/파이썬API.py --앱 <GeulPU.exe> --자료 <빈 폴더>    # 빠른 점검 4t — 빌드한 앱을 띄우고 끄기(GPU 없이 · 윈도)
  python -I -X utf8 시험/파이썬API.py --앱 <GeulPU.exe> --자료 <빈 폴더> --모델 <k-양자화 .gguf>
                                                                            # 실제 모델로 한 번(GPU — 시험/GPU줄.py 로 줄을 선다)

가짜 앱은 실제 서버처럼 로짓 행을 문맥(프롬프트 + 앞 토큰)만으로 정하고 뽑기의 난수를 (시드, 스텝 번호)로 정한다 — 그래서 분기의 앞
스텝 해시 · step_offset · 자기 확인(같은 토큰으로 이으면 원래 답)이 실제와 같은 뜻으로 맞는다. 보는 것: 요청 본문(stream 끔 · geulpu 칸 ·
echo · 분기의 프롬프트 · 시드 · 남은 max_tokens · 샘플링 칸), 응답 풀기(echo 배열의 앞 null · 해시는 echo_from 부터), 모델 더하기 ·
SHA-256 기다리기 · 불러오기(같으면 다시 불러오지 않음 · 듣지 않은 설정의 경고 · 실패), 비교, 오류, close 의 설정 되돌리기, 휠의 파일 목록.
마지막 줄 '파이썬 API: 모두 통과 (N 가지)'(종료 0) 또는 '파이썬 API: 틀림 N — …'(종료 1).
"""
import argparse
import collections
import hashlib
import http.server
import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import warnings
from concurrent.futures import ThreadPoolExecutor

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.join(ROOT, "도구", "pypi"))
import geulpu  # noqa: E402
from geulpu import api  # noqa: E402

본것, 틀림 = [], []


def 봄(이름, 맞음, 덧=""):
    print(f"{'통과' if 맞음 else '틀림'}  {이름}" + (f" — {덧}" if 덧 and not 맞음 else ""), flush=True)
    본것.append(이름)
    if not 맞음:
        틀림.append(이름)


def 오류인가(할일, 글=None, 상태=None):
    try:
        할일()
    except geulpu.GeulPUError as e:
        return (글 is None or 글 in str(e)) and (상태 is None or e.status == 상태)
    return False


# ---- 가짜 앱 ----
EOS = 2
LPA, LPB, LPX, ENT = -0.25, -1.75, -9.5, 0.625       # 이진수로 딱 떨어지는 값(JSON 을 오가도 같은 실수)


def 해시(문맥):
    return hashlib.sha256(",".join(map(str, 문맥)).encode()).hexdigest()[:16]


def 위(문맥):
    """문맥의 로짓 행 — 1위 · 2위 토큰(행은 문맥만으로 정해진다)"""
    h = int(해시(문맥), 16)
    a, b = 10 + h % 50, 10 + (h // 50) % 50
    if b == a:
        b = a + 1 if a < 59 else 10
    if h % 9 == 0 and len(문맥) >= 30:       # 끝 토큰은 문맥이 길어진 뒤에만(시험의 답이 몇 토큰은 되게)
        a = EOS
    return a, b


def 뽑기(문맥, 시드, 스텝, 온도):
    a, b = 위(문맥)
    if not 온도:
        return a
    r = int(hashlib.sha256(f"{시드}:{스텝}".encode()).hexdigest()[:8], 16) / 2 ** 32
    return a if r < 0.6 else b


def 로그(문맥, t):
    a, b = 위(문맥)
    return LPA if t == a else LPB if t == b else LPX


def 채팅토큰(msgs):
    t = [1]
    for m in msgs:
        t += [ord(c) for c in f"{m['role']}:{m['content']}\n"]
    return t + [3]


def 글(t):
    return chr(t) if t >= 32 else f"<{t}>"


def 생성(P, 시드, 온도, 최대, 시작스텝, k):
    문맥, 토큰, 해시들, 로그들, 엔트, 위번호, 위값 = list(P), [], [], [], [], [], []
    끝 = "length"
    for i in range(최대):
        a, b = 위(문맥)
        t = 뽑기(문맥, 시드, 시작스텝 + i, 온도)
        토큰.append(t)
        해시들.append(해시(문맥))
        if t == EOS:
            끝 = "stop"
            break
        로그들.append(로그(문맥, t))
        엔트.append(ENT)
        위번호.append([a, b][:k])
        위값.append([LPA, LPB][:k])
        문맥.append(t)
    return 토큰, 해시들, 로그들, 엔트, 위번호, 위값, 끝


class 가짜앱:
    def __init__(self, 자료):
        self.자료 = 자료
        self.잠금 = threading.Lock()
        self.설정 = {"settings_version": 3, "gpu_layer": "auto", "vulkan_device": "", "models_dir": "", "update_check": True, "slots": 2,
                   "context": 8192, "precision": "fast", "precision_unquantized": "exact", "kv8_cache": True, "fast_attention": True,
                   "speculative": True, "tool_errors": "off"}
        self.처음설정 = dict(self.설정)
        self.모델들, self.일 = [], None
        self.서버 = {"state": "off", "model": ""}
        self.장치 = {"checked": True, "layer": "cuda", "name": "Fake GPU", "memory": 12884901888, "variant": "sm_89", "vendor": "nvidia",
                   "driver": "999.99", "cuda_ok": True, "vulkan_ok": True, "reason": ""}
        self.셈 = collections.Counter()
        self.몸들 = collections.defaultdict(list)
        self.차례 = []                 # 상태를 읽을 때마다 한 칸씩 — (남은 읽기, 할 일)
        self.번호 = 0

    def 틱(self):
        남은 = []
        for n, f in self.차례:
            if n <= 1:
                f()
            else:
                남은.append((n - 1, f))
        self.차례 = 남은

    def 모델찾기(self, mid):
        return next((m for m in self.모델들 if m["id"] == mid), None)

    def 상태(self):
        self.틱()
        return {"app": {"name": "GeulPU", "version": "0.21.0", "commit": "fake123", "built": "2026-11-05", "installer": "pip",
                        "data_dir": self.자료.rstrip("\\/") + os.sep, "port": 0},
                "settings": dict(self.설정), "device": dict(self.장치), "server": dict(self.서버), "job": self.일,
                "models": [dict(m) for m in self.모델들]}

    def 처리(self, 방법, 길, 몸):
        self.셈[길] += 1
        if 몸 is not None:
            self.몸들[길].append(몸)
        if 길 == "/api/ping":
            return 200, {"app": "geulpu", "version": "0.21.0"}
        if 길 == "/api/state":
            return 200, self.상태()
        if 길 == "/api/provenance":
            return 200, {"binaries": [{"file": "server-cuda.exe", "sha256": "ab" * 32}], "module": {"sha256": "cd" * 32}}
        if 길 == "/api/settings":
            for k, v in 몸.items():
                if k in self.설정:
                    self.설정[k] = v
            return 200, {"ok": True}
        if 길 == "/api/models/add":
            p = 몸.get("path", "")
            if not os.path.isfile(p):
                return 400, {"error": {"message": "파일이 없습니다"}}
            있던 = next((m for m in self.모델들 if m["path"] == p), None)
            if 있던:
                return 409, {"error": {"message": f"이미 추가한 파일입니다: {있던['id']}"}}
            self.번호 += 1
            이름 = os.path.basename(p)
            k = "Q4_K" in 이름
            self.모델들.append({"id": f"m{self.번호}", "name": 이름, "path": p, "size": os.path.getsize(p), "sha256": "", "verified": False,
                              "verifying": 0, "verify_bytes": 0, "catalog": "", "template": "chatml", "device_reason": "",
                              "shape": {"ok": True, "arch": "qwen3", "quant": "Q4_K_M" if k else "Q8_0", "file_type": "x",
                                        "kv8_ok": k, "fast_attention_ok": k}})
            return 200, {"ok": True, "id": f"m{self.번호}"}
        if 길 == "/api/models/verify":
            m = self.모델찾기(몸.get("id"))
            if m is None or m["verifying"]:
                return 409, {"error": {"message": "모델이 없거나 이미 확인 중입니다"}}
            m["verifying"] = 1

            def 끝(m=m):
                m["sha256"] = hashlib.sha256(open(m["path"], "rb").read()).hexdigest()
                m["verifying"] = 0
            self.차례.append((2, 끝))
            return 200, {"ok": True}
        if 길 == "/api/server/stop":
            self.서버 = {"state": "off", "model": ""}
            return 200, {"ok": True}
        if 길 == "/api/server/start":
            m = self.모델찾기(몸.get("model"))
            if m is None:
                return 400, {"error": {"message": "모델이 없습니다"}}
            if self.일 and self.일.get("state") == "running":
                return 409, {"error": {"message": "다른 작업이 실행 중입니다"}}
            self.번호 += 1
            self.일 = {"id": self.번호, "kind": "server", "state": "running", "reason": ""}
            self.서버 = {"state": "starting", "model": m["id"]}

            def 준비(m=m):
                if "bad" in m["name"]:
                    self.서버 = {"state": "failed", "model": m["id"], "reason": "가짜 실패", "log_tail": "fake log"}
                    self.일["state"], self.일["reason"] = "failed", "모델을 불러오지 못했습니다"
                    return
                s, 모 = self.설정, m["shape"]
                빠름 = s["precision"] == "fast"
                kv8 = bool(s["kv8_cache"] and 모["kv8_ok"] and 빠름 and self.장치["layer"] == "cuda")
                self.서버 = {"state": "ready", "model": m["id"], "layer": self.장치["layer"], "precision": s["precision"], "kv8": kv8,
                           "fast_attention": bool(kv8 and s["fast_attention"] and 모["fast_attention_ok"]),
                           "speculative": bool(s["speculative"]), "template": "chatml", "slots": s["slots"], "context": s["context"],
                           "yarn": 0, "reason": "", "log_tail": ""}
                self.일["state"] = "done"
            self.차례.append((2, 준비))
            return 200, {"ok": True, "job": self.번호}
        if 길 == "/api/device/check":
            self.번호 += 1
            self.일 = {"id": self.번호, "kind": "device", "state": "running"}

            def 확인():
                바란 = self.설정["gpu_layer"]
                self.장치["layer"] = 바란 if 바란 in ("cuda", "vulkan") else "cuda"
                self.일["state"] = "done"
            self.차례.append((1, 확인))
            return 200, {"ok": True, "job": self.번호}
        if 길 == "/tokenize":
            return 200, {"tokens": [ord(c) for c in 몸["prompt"]] if "prompt" in 몸 else 채팅토큰(몸["messages"])}
        if 길 == "/detokenize":
            return 200, {"prompt": "".join(글(t) for t in 몸["tokens"])}
        if 길 == "/v1/chat/completions":
            return self.채팅(몸)
        if 길 == "/v1/completions":
            return self.완성(몸)
        return 404, {"error": {"message": "없는 길"}}

    def 채팅(self, 몸):
        msgs = 몸.get("messages")
        if not msgs:
            return 400, {"error": {"message": "messages 가 없습니다", "type": "invalid_request_error"}}
        P = 채팅토큰(msgs)
        g = 몸.get("geulpu") or {}
        시드, 온도, 최대 = 몸.get("seed", 777), 몸.get("temperature", 0.8), 몸.get("max_tokens", 12)
        묻기, k = bool(몸.get("logprobs")), int(몸.get("top_logprobs") or 0)
        토큰, 해시들, 로그들, 엔트, 위번호, 위값, 끝 = 생성(P, 시드, 온도, 최대, g.get("step_offset", 0), max(k, 1))
        lp = None
        if 묻기:
            lp = {"content": [{"token": 글(t), "logprob": x, "bytes": [], "top_logprobs": [{"token": 글(a), "logprob": y}
                                                                                     for a, y in zip(ti, tl)][:k]}
                              for t, x, ti, tl in zip(토큰, 로그들, 위번호, 위값)]}
        gp = {"tokens": 토큰, "seed": 시드}
        if g.get("logit_hash"):
            gp["logit_hashes"] = 해시들
        if g.get("prompt_tokens"):
            gp["prompt_tokens"] = P
        if 묻기:
            gp.update(entropy=엔트, top_ids=위번호, top_logprobs=위값)
        return 200, {"object": "chat.completion", "model": "fake-model", "usage": {"prompt_tokens": len(P), "completion_tokens": len(토큰)},
                     "choices": [{"index": 0, "message": {"role": "assistant", "content": "".join(글(t) for t in 토큰 if t != EOS)},
                                  "finish_reason": 끝, "logprobs": lp}], "geulpu": gp}

    def 완성(self, 몸):
        pr = 몸.get("prompt")
        P = [ord(c) for c in pr] if isinstance(pr, str) else list(pr)
        g = 몸.get("geulpu") or {}
        메아리, 묻기 = bool(몸.get("echo")), 몸.get("logprobs") is not None
        k = max(1, int(몸.get("logprobs") or 0))
        시드, 온도, 최대 = 몸.get("seed", 777), 몸.get("temperature", 0.8), 몸.get("max_tokens", 12)
        토큰, 해시들, 로그들, 엔트, 위번호, 위값, 끝 = 생성(P, 시드, 온도, 최대, g.get("step_offset", 0), k)
        gp = {"tokens": 토큰, "seed": 시드}
        tlp, ttx = [], []
        if 메아리:
            n0 = max(1, int(g.get("echo_from", 0)))
            pe, pti, ptl = [], [], []
            for i, t in enumerate(P):
                ttx.append(글(t))
                if i < n0:                    # echo_from 앞은 null(첫 토큰은 늘)
                    for xs in (tlp, pe, pti, ptl):
                        xs.append(None)
                    continue
                a, b = 위(P[:i])
                tlp.append(로그(P[:i], t))
                pe.append(ENT)
                pti.append([a, b][:k])
                ptl.append([LPA, LPB][:k])
            if 묻기:
                gp.update(prompt_entropy=pe, prompt_top_ids=pti, prompt_top_logprobs=ptl)
                if g.get("logit_hash"):
                    gp["prompt_logit_hashes"] = [해시(P[:i]) for i in range(n0, len(P))]
        tlp += 로그들
        ttx += [글(t) for t in 토큰 if t != EOS]
        if g.get("logit_hash"):
            gp["logit_hashes"] = 해시들
        if g.get("prompt_tokens"):
            gp["prompt_tokens"] = P
        if 묻기:
            gp.update(entropy=엔트, top_ids=위번호, top_logprobs=위값)
        본 = ("".join(글(t) for t in P) if 메아리 else "") + "".join(글(t) for t in 토큰 if t != EOS)
        return 200, {"object": "text_completion", "model": "fake-model",
                     "choices": [{"index": 0, "text": 본, "finish_reason": 끝,
                                  "logprobs": {"tokens": ttx, "token_logprobs": tlp, "top_logprobs": [], "text_offset": []} if 묻기 else None}],
                     "geulpu": gp}


def 가짜띄우기(앱):
    class 처리기(http.server.BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def 답(self, 방법):
            길 = self.path.split("?", 1)[0]
            n = int(self.headers.get("Content-Length") or 0)
            몸 = json.loads(self.rfile.read(n).decode("utf-8")) if n else None
            with 앱.잠금:
                상태, 답 = 앱.처리(방법, 길, 몸)
            b = json.dumps(답, ensure_ascii=False).encode("utf-8")
            self.send_response(상태)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(b)))
            self.end_headers()
            self.wfile.write(b)

        def do_GET(self):
            self.답("GET")

        def do_POST(self):
            self.답("POST")

    서버 = http.server.ThreadingHTTPServer(("127.0.0.1", 0), 처리기)
    threading.Thread(target=서버.serve_forever, daemon=True).start()
    return 서버, f"http://127.0.0.1:{서버.server_address[1]}"


def 가짜시험():
    폴더 = tempfile.mkdtemp(prefix="geulpu-api-")
    앱 = 가짜앱(폴더)
    서버, 주소 = 가짜띄우기(앱)
    try:
        g = geulpu.connect(주소)
        i0 = g.info()
        봄("connect · info — 불러온 모델 없음 · 판 · 장치", i0["model"] is None and i0["geulpu"]["version"] == "0.21.0"
          and i0["device"]["name"] == "Fake GPU")
        봄("connect — GeulPU 가 아닌 주소는 오류", 오류인가(lambda: geulpu.connect("http://127.0.0.1:9")))
        봄("add_model — 없는 파일은 오류", 오류인가(lambda: g.add_model(os.path.join(폴더, "없음.gguf")), "no such file"))
        바이트 = bytes(range(256)) * 64
        파일4, 파일8, 파일bad = (os.path.join(폴더, n) for n in ("Qwen-Q4_K_M.gguf", "Qwen-Q8_0.gguf", "bad-Q4_K_M.gguf"))
        for p in (파일4, 파일8, 파일bad):
            open(p, "wb").write(바이트)
        sha = hashlib.sha256(바이트).hexdigest()
        m = g.add_model(파일4)
        봄("add_model — 앱이 SHA-256 을 셀 때까지 기다림", m["sha256"] == sha and 앱.셈["/api/models/verify"] == 1, str(m))
        더함 = 앱.셈["/api/models/add"]
        m2 = g.add_model(파일4.replace("\\", "/"))
        봄("add_model — 이미 있는 파일(다른 철자의 경로)은 다시 더하지 않음", m2["id"] == m["id"] and 앱.셈["/api/models/add"] == 더함)
        본상태, 첫 = g.state, [True]

        def 숨김():                           # 첫 읽기에서 목록을 비워 409(이미 추가함)의 길로
            s = 본상태()
            if 첫[0]:
                s["models"], 첫[0] = [], False
            return s
        g.state = 숨김
        m3 = g.add_model(파일4)
        g.state = 본상태
        봄("add_model — 409 의 글에서 아이디를 읽음", m3["id"] == m["id"])

        sv = g.load(파일4, precision="exact")
        봄("load — 설정을 보내고 서버를 켬", 앱.몸들["/api/settings"][-1] == {"precision": "exact", "precision_unquantized": "exact"}
          and sv["state"] == "ready" and sv["precision"] == "exact" and 앱.셈["/api/server/start"] == 1)
        켬 = 앱.셈["/api/server/start"]
        g.load(m["id"], precision="exact")
        봄("load — 같은 모델 · 같은 설정이면 다시 불러오지 않음", 앱.셈["/api/server/start"] == 켬)
        g.load(sha[:12])
        봄("load — SHA-256 앞자리로 고름", 앱.셈["/api/server/start"] == 켬)
        with warnings.catch_warnings(record=True) as 경고:
            warnings.simplefilter("always")
            sv8 = g.load(파일8, precision="fast", kv8=True, fast_attention=True)
        봄("load — 듣지 않은 설정(Q8_0 의 8비트 KV)은 경고 · 다른 모델이면 멈췄다 켬",
          any("kv8_cache" in str(w.message) for w in 경고) and sv8["kv8"] is False and 앱.셈["/api/server/stop"] >= 1)
        with warnings.catch_warnings(record=True) as 경고:
            warnings.simplefilter("always")
            svk = g.load(파일4, precision="fast", kv8=True, fast_attention=True)
        봄("load — k-양자화 모델은 8비트 KV · 빠른 어텐션이 들고 경고 없음", svk["kv8"] is True and svk["fast_attention"] is True and not 경고)
        g.load(파일4, layer="vulkan", precision="exact")
        봄("load — layer 를 바꾸면 하드웨어 확인 뒤 그 백엔드로", 앱.장치["layer"] == "vulkan" and 앱.서버["layer"] == "vulkan"
          and 앱.셈["/api/device/check"] == 1)
        g.load(파일4, layer="cuda", precision="exact")
        봄("load — 불러오지 못하면 까닭과 함께 오류", 오류인가(lambda: g.load(파일bad), "did not load"))
        g.load(파일4, precision="exact")

        msgs = [{"role": "user", "content": "Name one planet."}]
        a = g.chat(msgs, max_tokens=10, temperature=0.7, seed=5, top_logprobs=2)
        b0 = 앱.몸들["/v1/chat/completions"][-1]
        봄("chat — 요청: stream 끔 · logit_hash · prompt_tokens · logprobs", b0["stream"] is False and b0["logprobs"] is True
          and b0["geulpu"] == {"logit_hash": True, "prompt_tokens": True} and b0["messages"] == msgs and "model" not in b0)
        끝EOS = 1 if a.tokens and a.tokens[-1] == EOS else 0
        봄("chat — 응답 풀기(토큰 · 해시 · 프롬프트 토큰 · 시드 · 로그확률)", a.prompt_tokens == 채팅토큰(msgs) and a.seed == 5
          and len(a.logit_hashes) == len(a.tokens) and len(a.logprobs) == len(a.tokens) - 끝EOS and len(a.top_ids) == len(a.logprobs)
          and a.path == "/v1/chat/completions" and a.model == "fake-model" and len(a.tokens) >= 4, repr(a))
        b = g.rerun(a)
        봄("rerun — 같은 요청 · 같은 시드면 같은 비트", bool(geulpu.compare(a, b)) and 앱.몸들["/v1/chat/completions"][-1]["seed"] == 5)
        g.rerun(a, seed=6, temperature=0)
        봄("rerun — 바꾼 칸이 들어감", 앱.몸들["/v1/chat/completions"][-1]["seed"] == 6
          and 앱.몸들["/v1/chat/completions"][-1]["temperature"] == 0)

        s = g.score(a)
        sb = 앱.몸들["/v1/completions"][-1]
        봄("score(gen) — 요청: echo · max_tokens 0 · echo_from · 프롬프트 + 답", sb["echo"] is True and sb["max_tokens"] == 0
          and sb["geulpu"]["echo_from"] == len(a.prompt_tokens) and sb["prompt"] == a.prompt_tokens + a.tokens)
        봄("score(gen) — 로짓 해시가 생성과 같고 로그확률이 이어짐(echo 배열의 앞 null 을 뗌)", s.logit_hashes == a.logit_hashes
          and s.tokens == a.tokens and s.logprobs[:len(a.logprobs)] == a.logprobs and None not in s.logprobs
          and len(s.top_ids) == len(a.tokens), repr(s))
        st = g.score("The capital is", " Paris")
        봄("score(글, 글) — 답의 토큰은 이은 글의 토큰에서 프롬프트 몫을 뗀 것", st.tokens == [ord(c) for c in " Paris"]
          and st.prompt_tokens == [ord(c) for c in "The capital is"] and len(st.logprobs) == 6)
        sm = g.score(msgs, a.tokens)
        봄("score(메시지, 번호)", sm.prompt_tokens == a.prompt_tokens and sm.logit_hashes == a.logit_hashes)
        봄("score — 번호 프롬프트에 글 답 · 빈 프롬프트는 오류", 오류인가(lambda: g.score([1, 2], "x"), "text prompt")
          and 오류인가(lambda: g.score([], [5]), "prompt token"))

        k = 2
        nb = g.branch(a, k, token="same")
        bb = 앱.몸들["/v1/completions"][-1]
        봄("branch(same) — 요청: 프롬프트 + 앞 k 토큰 + 토큰 · step_offset k+1 · 시드 · 남은 max_tokens · temperature",
          bb["prompt"] == a.prompt_tokens + a.tokens[:k + 1] and bb["geulpu"]["step_offset"] == k + 1 and bb["seed"] == 5
          and bb["max_tokens"] == 10 - (k + 1) and bb["temperature"] == 0.7 and bb["geulpu"]["echo_from"] == len(a.prompt_tokens)
          and bb["echo"] is True, json.dumps(bb)[:300])
        봄("branch(same) — 원래 답이 그대로 나옴(자기 확인)", bool(geulpu.compare(a, nb)) and nb.prefix_same is True, repr(nb))
        점수수 = sum(1 for x in 앱.몸들["/v1/completions"] if x.get("max_tokens") == 0)
        rb = g.branch(a, k)
        대안 = next(t for t in a.top_ids[k] if t != a.tokens[k])
        c = geulpu.compare(a, rb)
        봄("branch — 원래와 다른 가장 높은 토큰(생성의 top_ids 에서 — 점수 요청 없이)", rb.token == 대안 and rb.original_token == a.tokens[k]
          and sum(1 for x in 앱.몸들["/v1/completions"] if x.get("max_tokens") == 0) == 점수수)
        봄("branch — 앞 k+1 스텝의 해시가 원래와 같고 스텝 k 에서 갈라짐", rb.prefix_same is True and c.first_diff_step == k
          and "tokens differ" in c.why and rb.tokens[:k] == a.tokens[:k] and rb.logit_hashes[:k + 1] == a.logit_hashes[:k + 1]
          and rb.new_tokens == rb.tokens[k + 1:] and len(rb.logit_hashes) == len(rb.tokens), f"{c} {rb!r}")
        봄("branch — 분기한 답의 글(토큰을 이어 푼 글)", rb.text == "".join(글(t) for t in rb.tokens))
        rr = g.rerun(rb)
        봄("rerun(branch) — 같은 분기를 다시: 같은 비트", isinstance(rr, geulpu.Branch) and bool(geulpu.compare(rb, rr))
          and rr.prefix_same is True)
        봄("rerun(branch) — 바꿀 칸은 받지 않음", _값오류(lambda: g.rerun(rb, seed=1)))
        if k + 1 < len(rb.tokens):
            b2 = g.branch(rb, k + 1, token="same")
            봄("branch(branch, same) — 남은 max_tokens 는 원래 요청의 것에서", 앱.몸들["/v1/completions"][-1]["max_tokens"] == 10 - (k + 2)
              and bool(geulpu.compare(rb, b2)))
        a2 = g.chat(msgs, max_tokens=10, temperature=0.7, seed=5)
        봄("chat — logprobs 를 묻지 않으면 로그확률 칸이 None · 비트는 같음", a2.logprobs is None and a2.top_ids is None
          and bool(geulpu.compare(a, a2)))
        rb2 = g.branch(a2, k)
        봄("branch — top_ids 가 없으면 그 스텝을 점수로 매겨 고름", rb2.token == rb.token and bool(geulpu.compare(rb, rb2))
          and sum(1 for x in 앱.몸들["/v1/completions"] if x.get("max_tokens") == 0) == 점수수 + 1)
        rb3 = g.branch(a, k, token=33)
        봄("branch(token=번호)", rb3.token == 33 and 앱.몸들["/v1/completions"][-1]["prompt"][-1] == 33)
        봄("branch — 답 밖의 스텝은 오류", 오류인가(lambda: g.branch(a, len(a.tokens)), "outside"))
        cp = g.complete("Once upon", max_tokens=5, temperature=0, logprobs=2)
        봄("complete — 글 프롬프트 · 로그확률", cp.prompt_tokens == [ord(c) for c in "Once upon"] and cp.path == "/v1/completions"
          and cp.logprobs is not None and len(cp.tokens) == 5 and bool(geulpu.compare(cp, g.rerun(cp))))
        ce = g.complete("Once upon", max_tokens=80, temperature=0, logprobs=1)
        봄("complete — 끝 토큰으로 끝나면 토큰 · 해시에는 들고 로그확률에는 없음", ce.finish_reason == "stop" and ce.tokens[-1] == EOS
          and len(ce.logit_hashes) == len(ce.tokens) and len(ce.logprobs) == len(ce.tokens) - 1, repr(ce))
        봄("tokenize · detokenize", g.tokenize("ab") == [97, 98] and g.detokenize([97, 98]) == "ab" and g.tokenize(msgs) == 채팅토큰(msgs))

        try:
            g.chat([])
            맞음 = False
        except geulpu.GeulPUError as e:
            맞음 = e.status == 400 and "messages" in str(e) and isinstance(e.body, dict)
        봄("오류 — HTTP 400 의 상태 · 글 · 본문", 맞음)
        일 = g.info(provenance=True)
        봄("info — 모델 파일 · SHA-256 · 계산 길 · 출처", 일["model"]["sha256"] == sha and 일["model"]["file"] == "Qwen-Q4_K_M.gguf"
          and 일["server"]["precision"] == "exact" and 일["provenance"]["module"]["sha256"] == "cd" * 32)

        G = geulpu.Generation
        x = G(text="", tokens=[5, 6, 7], logit_hashes=["a", "b", "c"], prompt_tokens=[9])
        봄("compare — 같음 · 로짓 · 토큰 · 길이 · 프롬프트", bool(geulpu.compare(x, G("", [5, 6, 7], ["a", "b", "c"], [9])))
          and geulpu.compare(x, G("", [5, 6, 7], ["a", "x", "c"], [9])).first_diff_step == 1
          and "logits" in geulpu.compare(x, G("", [5, 6, 7], ["a", "x", "c"], [9])).why
          and geulpu.compare(x, G("", [5, 8, 7], ["a", "b", "c"], [9])).why.startswith("tokens differ from step 1")
          and geulpu.compare(x, G("", [5, 6], ["a", "b"], [9])).first_diff_step == 2
          and geulpu.compare(x, G("", [5, 6, 7], ["a", "b", "c"], [8])).why == "prompt tokens differ")

        g.close()
        봄("close — 바꾼 설정을 처음 것으로 되돌림", all(앱.설정[k] == 앱.처음설정[k] for k in api.SETTING_KEYS if k in 앱.처음설정))
        봄("close — 두 번 불러도 됨 · 연결한 앱의 서버는 그대로", (g.close() is None) and 앱.서버["state"] == "ready")
    finally:
        서버.shutdown()
        shutil.rmtree(폴더, ignore_errors=True)
    import 휠만들기
    봄("휠 — 꾸러미의 파이썬 파일 모두(api.py 포함)", 휠만들기.꾸러미파일들() == ["__init__.py", "__main__.py", "api.py"],
      str(휠만들기.꾸러미파일들()))
    봄("도움 — 경로 맞추기 · 콘솔의 포트", api._norm_dir("C:/x/y/") == api._norm_dir("C:\\x\\y") and _콘솔포트())
    import importlib.util                     # api.py 를 파일 하나로(0.21.0 사용자 — 스크립트 옆에 둔 geulpu_api.py)
    명세 = importlib.util.spec_from_file_location("geulpu_api", os.path.join(ROOT, "도구", "pypi", "geulpu", "api.py"))
    홀로 = importlib.util.module_from_spec(명세)
    sys.modules["geulpu_api"] = 홀로          # import geulpu_api 처럼(dataclass 가 모듈을 sys.modules 에서 찾는다)
    명세.loader.exec_module(홀로)
    봄("api.py 홀로 — 설치된 geulpu 꾸러미의 앱 폴더를 찾음", 홀로._app_folder() == geulpu.app_dir() and callable(홀로.start))


def _값오류(할일):
    try:
        할일()
    except ValueError:
        return True
    return False


def _콘솔포트():
    with tempfile.NamedTemporaryFile("wb", delete=False, suffix=".log") as f:
        f.write("GeulPU 0.21.0\n  화면: http://127.0.0.1:44567/\n".encode("utf-8"))
    try:
        return api._console_port(f.name) == 44567 and api._console_port(f.name + ".없음") is None
    finally:
        os.remove(f.name)


# ---- 실제 앱 ----
def 앱시험(exe, 자료, 모델):
    shutil.rmtree(자료, ignore_errors=True)
    t0 = time.time()
    g = geulpu.start(data=자료, exe=exe, wait_device=bool(모델), timeout=120)
    p = g._proc
    자식 = []
    try:
        st = g.state()
        봄("start — 창 없이 띄우고 그 자료 폴더의 앱인지 확인", g.ping().get("app") == "geulpu" and p.poll() is None
          and api._norm_dir(st["app"]["data_dir"]) == api._norm_dir(자료), f"{st['app'].get('data_dir')} ({time.time() - t0:.1f} 초)")
        봄("start — 콘솔 출력은 자료 폴더의 기록 파일로", os.path.isfile(os.path.join(자료, "geulpu-console.log")))
        if 모델:
            모델시험(g, 모델)
            자식 = 자식들(p.pid)                   # 앱이 띄운 서버 · 도구(끝낼 때 함께 꺼져야 한다)
        else:
            os.makedirs(os.path.join(자료, "모델"), exist_ok=True)
            파일 = os.path.join(자료, "모델", "t.gguf")
            바이트 = bytes(range(256)) * 4096
            open(파일, "wb").write(바이트)
            m = g.add_model(파일)
            봄("add_model(실제 앱) — SHA-256 을 기다려 받음", m.get("sha256") == hashlib.sha256(바이트).hexdigest(), str(m.get("sha256")))
    finally:
        g.close()
    봄("close — 앱이 끝남", p.poll() is not None, f"종료 코드 {p.poll()}")
    if 자식:
        time.sleep(1)
        남은 = [x for x in 자식 if 살았나(x)]
        봄(f"close — 앱이 띄운 프로세스 {len(자식)} 개도 끝남(서버 포함)", not 남은, f"남은 {남은}")


def 자식들(pid):
    """윈도 — 그 프로세스가 띄운 프로세스의 번호(다른 체계는 빈 목록)"""
    if os.name != "nt":
        return []
    out = subprocess.run(["powershell", "-NoProfile", "-Command",
                          f"(Get-CimInstance Win32_Process -Filter 'ParentProcessId={int(pid)}').ProcessId"],
                         capture_output=True, text=True).stdout.split()
    return [int(x) for x in out if x.isdigit()]


def 살았나(pid):
    out = subprocess.run(["powershell", "-NoProfile", "-Command", f"[bool](Get-Process -Id {int(pid)} -ErrorAction SilentlyContinue)"],
                         capture_output=True, text=True).stdout.strip()
    return out == "True"


def 모델시험(g, 모델):
    """실제 모델로 한 번 — 정밀 · 빠른 설정에서 생성 · 다시 · 동시 · 점수 · 분기를 비트로"""
    t0 = time.time()
    sv = g.load(모델, precision="exact")
    일 = g.info()
    print(f"  불러옴 {일['model']['file']} · {일['server']['precision']} · {일['device']['name']} · {일['geulpu']['version']} "
          f"({time.time() - t0:.1f} 초)", flush=True)
    봄("load(exact) — 준비 · SHA-256", sv["state"] == "ready" and sv["precision"] == "exact" and bool(일["model"]["sha256"]))
    msgs = [{"role": "user", "content": "Write two sentences about the Moon."}]
    a = g.chat(msgs, max_tokens=48, temperature=0.8, seed=1234, top_logprobs=2)
    print(f"  {a!r} · {a.seconds:.2f} 초", flush=True)
    봄("chat — 해시 · 프롬프트 토큰 · 로그확률의 길이", len(a.logit_hashes) == len(a.tokens) and len(a.prompt_tokens) > 0
      and len(a.logprobs) == len(a.tokens) - (1 if a.finish_reason == "stop" else 0) and a.seed == 1234)
    봄("rerun — 같은 비트", bool(geulpu.compare(a, g.rerun(a))))
    with ThreadPoolExecutor(4) as ex:
        넷 = list(ex.map(lambda _: g.rerun(a), range(4)))
    봄("rerun 넷을 한꺼번에 — 모두 같은 비트", all(geulpu.compare(a, x) for x in 넷))
    s = g.score(a)
    봄("score(gen) — 로짓 해시가 생성과 같음", s.logit_hashes == a.logit_hashes and s.logprobs[:len(a.logprobs)] == a.logprobs,
      f"{geulpu.compare(a, s)}")
    k = len(a.tokens) // 2
    nb = g.branch(a, k, token="same")
    봄("branch(same) — 원래 답이 그대로", bool(geulpu.compare(a, nb)) and nb.prefix_same is True, f"{geulpu.compare(a, nb)}")
    rb = g.branch(a, k)
    c = geulpu.compare(a, rb)
    print(f"  {rb!r}: {c.why}", flush=True)
    봄("branch — 앞 스텝의 해시가 같고 스텝 k 에서 갈라짐 · 다시 돌려도 같음", rb.prefix_same is True and c.first_diff_step == k
      and bool(geulpu.compare(rb, g.rerun(rb))))
    cp = g.complete("The capital of France is", max_tokens=8, temperature=0)
    봄("complete — 다시 돌려도 같음", bool(geulpu.compare(cp, g.rerun(cp))), repr(cp))
    with warnings.catch_warnings(record=True) as 경고:
        warnings.simplefilter("always")
        svf = g.load(모델, precision="fast", kv8=True, fast_attention=True)
    af = g.rerun(a)
    kv8됨 = (next(m for m in g.models() if m["id"] == svf["model"]).get("shape") or {}).get("kv8_ok")
    print(f"  fast: 모델의 kv8_ok {kv8됨} · 서버 kv8 {svf.get('kv8')} · fast_attention {svf.get('fast_attention')} · "
          f"정밀과 견줌: {geulpu.compare(a, af).why}" + (f" · 경고: {경고[0].message}" if 경고 else ""), flush=True)
    봄("load(fast · kv8 · 빠른 어텐션) — 모델이 받으면 들고, 받지 않으면 까닭과 함께 경고 · 다시 돌려도 같음",
      svf["precision"] == "fast" and bool(svf.get("kv8")) == bool(kv8됨) and bool(경고) == (not kv8됨)
      and (kv8됨 or "kv8_ok is false" in str(경고[0].message)) and bool(geulpu.compare(af, g.rerun(af))))
    p = g.info(provenance=True).get("provenance") or {}
    봄("info(provenance) — 서버 · 모듈의 SHA-256", bool((p.get("module") or {}).get("sha256")) and bool(p.get("binaries")))


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--앱", help="GeulPU 실행 파일 — 주면 그 앱을 띄우고 끈다(가짜 앱 시험은 하지 않는다)")
    ap.add_argument("--자료", help="--앱 의 자료 폴더(지우고 새로 만든다)")
    ap.add_argument("--모델", help="--앱 과 함께 — 이 모델 파일로 생성 · 점수 · 분기를 한 번(GPU)")
    a = ap.parse_args()
    t0 = time.time()
    if a.앱:
        if not a.자료:
            sys.exit("--앱 에는 --자료 가 든다")
        앱시험(os.path.abspath(a.앱), os.path.abspath(a.자료), os.path.abspath(a.모델) if a.모델 else None)
    else:
        가짜시험()
    끝 = f"({len(본것)} 가지 · {time.time() - t0:.1f} 초)"
    print(f"파이썬 API: 모두 통과 {끝}" if not 틀림 else f"파이썬 API: 틀림 {len(틀림)} — {', '.join(틀림)} {끝}", flush=True)
    return 0 if not 틀림 else 1


if __name__ == "__main__":
    sys.exit(main())
