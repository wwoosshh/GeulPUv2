#!/usr/bin/env python3
"""llama.cpp 재현 4단계(문서/라마재현.md) — 꾸린 앱으로 끝에서 끝: 기록기 · 재현 검증 · 바꾼 영수증. GPU 를 쓰므로 GPU 줄 안에서.

앱은 시험 포트 · 임시 자료 폴더로 띄운다(사용자의 앱 · 자료는 건드리지 않는다). 요청은 2단계의 기록(run0_<이름>_r0.json — prompt_text ·
response)을 그대로 쓴다.

  python -I -X utf8 시험/라마앱판.py 기록 <꾸러미> <자료> <out> --서버 <llama-server.exe> --모델 <gguf> --기록폴더 <det,det2 …>
      llama-server(b12008, 2단계의 설정)를 띄워 앱의 기록기(/llama/completion — n_probs 없이 보내 기록기가 더하는지까지)로 요청을 보내고,
      응답의 토큰 · 로그 확률이 2단계의 기록과 같은지(4-1). 영수증 이름은 <out>/영수증.json.
  python -I -X utf8 시험/라마앱판.py 검증 <꾸러미> <자료> <out> --층 cuda|vulkan [--장치 <Vulkan 장치 이름의 일부>] [--이름 a,b,…]
      <out>/영수증.json 의 영수증마다 앱의 재현 검증(/api/llama/verify)을 돌린다(4-2 · 4-3). 결과는 <out>/검증-<층>-<장치>.json.
  python -I -X utf8 시험/라마앱판.py 바꾸기 <자료> <기록 out> <새 out> [--단계 4|5]
      4-3 의 바꾼 영수증 넷(모델 해시 · 토큰 하나 · n_ubatch 256 · 로그 확률 하나의 마지막 비트), 또는(--단계 5 — 온도 > 0 의 영수증에서)
      5-2 의 다섯(시드 +1 · 온도 1.2 — 다름, top_k 0 · mirostat 1 · 시드 없음 — 받지 않음)을 자료\\영수증\\ 에 쓰고 <새 out>/영수증.json 을.
  기록에 --온도 T --시드 S 를 주면(5단계) 요청마다 시드를 S, S+1, … 로 보낸다 — 2단계의 기록과 견주지 않고 영수증만 남긴다.
  python -I -X utf8 시험/라마앱판.py 대화 <꾸러미> <자료> <out> --서버 <llama-server.exe> --모델 <gguf> --기록폴더 <det …> [--시작 <이름>]
      7단계 — 기본 설정의 llama-server(-np 를 주지 않음 → 슬롯 4 · 통합 KV, 프롬프트 캐시 · RAM 캐시 기본, -c 32768 만)에 기록기로 한
      대화의 이어지는 요청 여섯(잇기 둘 · 같은 프롬프트 · 생성 잇기 · 앞을 고쳐 잇기)과 다른 대화 하나(다른 슬롯)를 보낸다.
      영수증 이름은 <out>/영수증.json(c1 … c7).
  python -I -X utf8 시험/라마앱판.py 채팅 <꾸러미> <자료> <out> --서버 <llama-server.exe> --모델 <gguf> --기록폴더 <det …>
      9단계 — 같은 기본 설정 서버에 기록기로 채팅 API(/llama/v1/chat/completions)의 한 대화(잇기 둘 · 같은 대화 · 앞을 고침 — 넷째는
      슬롯 캐시의 절반 넘게 버려 RAM 캐시 갈래)와 스트리밍 요청 하나(9-2)를 보낸다. 영수증 이름은 <out>/영수증.json(h1 … h5 · stream).
  python -I -X utf8 시험/라마앱판.py 흐름채팅 <꾸러미> <자료> <out> --서버 <llama-server.exe> --모델 <gguf> --기록폴더 <아무 폴더>
      18단계 — 2단계 설정의 서버(-np 1 · 캐시 끔)에 기록기로 스트리밍 채팅 요청 다섯(끝 토큰으로 끝나는 짧은 답 · 긴 답 · 생각 켬 ·
      온도 0.8)을 보낸다. 영수증 이름은 <out>/영수증.json(s1 … s4 · t1). 기록 에 --스트림 을 주면 /completion 을 스트리밍으로 보내고,
      조각에서 모은 칸이 2단계 기록과 같은지 본다(빠진 토큰 수도 찍는다).
  python -I -X utf8 시험/라마앱판.py 올라마 <꾸러미> <자료> <out> --올라마 <ollama.exe> --모델 <gguf> --기록폴더 <det …> --견줄 <자료>,<out>
      20단계 — Ollama(v0.40.14)를 시험 포트(11500) · 따로 둔 홈 · 모델 폴더(<out> 옆)로 띄워 모델을 싣고, 로그의 "starting llama-server"
      에서 Ollama 가 띄운 llama-server 의 포트 · 인자를 읽어 기록기를 그 서버에 붙인 뒤 2단계 프롬프트를 보낸다(cache_prompt 끔). --견줄 의
      영수증(같은 모델 · 공식 빌드)과 토큰 · 로그 확률이 같은지 본다. 영수증 이름은 <out>/영수증.json.
  python -I -X utf8 시험/라마앱판.py 바꾸기 <자료> <대화 out> <새 out> --단계 7
      7-2 의 바꾼 영수증 셋(c3 의 cache_n + 1 · c3 의 서버 포트 · c3 의 슬롯 상태에 다른 슬롯) — 7-1 의 검증 뒤에(같은 슬롯의 앞 영수증이 된다).
마지막 줄 "합계: …".
"""
import argparse
import json
import os
import struct
import subprocess
import sys
import time
import urllib.error
import urllib.request

이름들10 = ["code", "doc", "chat_ko", "py_review", "math", "tool_call", "ko_translate", "long_doc", "json_extract", "multi_turn"]
깃발 = ["-np", "1", "-fa", "on", "--no-cache-prompt", "--cache-ram", "0", "-c", "8192", "-ngl", "99"]


def 묻기(포트, 길, 몸=None, 시간=1800):
    r = urllib.request.Request(f"http://127.0.0.1:{포트}{길}", data=None if 몸 is None else json.dumps(몸, ensure_ascii=False).encode("utf-8"),
                               method="GET" if 몸 is None else "POST", headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(r, timeout=시간).read() or b"null")


class 앱:
    def __init__(self, 꾸러미, 자료, 포트):
        os.makedirs(자료, exist_ok=True)
        self.포트 = 포트
        self.기록 = open(os.path.join(자료, "앱.log"), "ab")
        self.p = subprocess.Popen([os.path.join(꾸러미, "GeulPU.exe"), "--포트", str(포트), "--자료", 자료, "--열지않음"],
                                  stdout=self.기록, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
        for _ in range(240):
            try:
                묻기(포트, "/api/ping", 시간=5)
                break
            except OSError:
                time.sleep(0.5)
        for _ in range(600):                                       # 켜질 때의 장치 확인이 끝날 때까지
            st = 묻기(포트, "/api/state")
            if st["device"].get("checked") and (st.get("job") or {}).get("state") != "running":
                break
            time.sleep(1)

    def 작업(self, 번, 한도=6 * 3600):
        t0 = time.time()
        while time.time() - t0 < 한도:
            time.sleep(2)
            j = 묻기(self.포트, "/api/state").get("job") or {}
            if j.get("id") == 번 and j.get("state") != "running":
                return j
        raise RuntimeError("작업 시간 넘김")

    def 닫기(self):
        self.p.terminate()
        try:
            self.p.wait(30)
        except subprocess.TimeoutExpired:
            self.p.kill()


def 칸들(r):
    return [(c["id"], c["logprob"], [(t["id"], t["logprob"]) for t in c["top_logprobs"]]) for c in r.get("completion_probabilities", [])]


def 흐름보내기(포트, 길, 몸):
    """18단계 — 스트리밍 요청을 보내고 SSE 몸(글)을 그대로"""
    r = urllib.request.Request(f"http://127.0.0.1:{포트}{길}", data=json.dumps(몸, ensure_ascii=False).encode("utf-8"),
                               method="POST", headers={"Content-Type": "application/json"})
    return urllib.request.urlopen(r, timeout=1800).read().decode("utf-8", errors="replace")


def 흐름모으기(글, 채팅):
    """18단계 — SSE 조각에서 알려진 토큰 · 로그 확률 칸 · 끝 조각(/completion 은 stop 참, 채팅은 __verbose)"""
    토큰, 칸, 끝 = [], [], None
    for 줄 in 글.split("\n"):
        줄 = 줄.rstrip("\r")
        if not 줄.startswith("data: {"):
            continue
        e = json.loads(줄[6:])
        if 채팅:
            if "__verbose" in e:
                끝 = e["__verbose"]
            for ch in e.get("choices", []):
                for en in ((ch.get("logprobs") or {}).get("content") or []):
                    칸.append(en)
                    토큰.append(en["id"])
        elif e.get("stop"):
            끝 = e
        else:
            토큰 += e.get("tokens", [])
            칸 += e.get("completion_probabilities", [])
    return 토큰, 칸, 끝


def 기록하기(a):
    폴더들 = a.기록폴더.split(",")

    def 기록(name):
        for d in 폴더들:
            p = os.path.join(d, f"run0_{name}_r0.json")
            if os.path.exists(p):
                return json.load(open(p, encoding="utf-8"))
        raise FileNotFoundError(name)

    os.makedirs(a.out, exist_ok=True)
    s = 앱(a.꾸러미, a.자료, a.포트)
    srv = None
    표, 같음수 = {}, 0
    try:
        log = open(os.path.join(a.out, "llama-server.log"), "w", encoding="utf-8", errors="replace")
        srv = subprocess.Popen([a.서버, "-m", a.모델, *깃발, "--host", "127.0.0.1", "--port", str(a.라마포트)], stdout=log, stderr=subprocess.STDOUT)
        t0 = time.time()
        while True:
            try:
                if 묻기(a.라마포트, "/health", 시간=5).get("status") == "ok":
                    break
            except (urllib.error.URLError, ConnectionError, TimeoutError, json.JSONDecodeError):
                pass
            if srv.poll() is not None or time.time() - t0 > 600:
                raise RuntimeError("llama-server 가 준비되지 않았다")
            time.sleep(0.5)
        묻기(a.포트, "/api/llama/recorder", {"port": a.라마포트, "args": "llama-server -m " + a.모델 + " " + " ".join(깃발)})
        영수증폴더 = os.path.join(a.자료, "영수증")
        for q, name in enumerate(a.이름):
            want = 기록(name)
            앞 = set(os.listdir(영수증폴더)) if os.path.isdir(영수증폴더) else set()
            seed = a.시드 + q if a.온도 > 0 else 0
            몸 = {"prompt": want["prompt_text"], "n_predict": 256, "temperature": a.온도, "seed": seed, "stream": a.스트림}
            빠짐 = 0
            if a.스트림:                                          # 18단계 — 조각에서 모은 칸이 2단계 기록(비스트리밍)과 같은가
                토큰, 칸, 끝 = 흐름모으기(흐름보내기(a.포트, "/llama/completion", 몸), False)
                빠짐 = (끝 or {}).get("tokens_predicted", -1) - len(토큰)
                r = {"tokens": 토큰, "completion_probabilities": 칸}
                같음 = a.온도 > 0 or (칸들(r) == 칸들(want["response"]) and (끝 or {}).get("tokens_predicted") == len(want["response"]["tokens"]))
            else:
                r = 묻기(a.포트, "/llama/completion", 몸)
                같음 = a.온도 > 0 or (r.get("tokens") == want["response"]["tokens"] and 칸들(r) == 칸들(want["response"]))   # 온도 > 0 은 견줄 기록이 없다
            같음수 += 같음
            새것 = None
            for _ in range(120):                                   # 영수증은 스레드가 모델 해시 뒤에 쓴다 — 다 쓴 .json 이름만
                새 = sorted(f for f in set(os.listdir(영수증폴더)) - 앞 - {"기록기.json"} if f.endswith(".json")) if os.path.isdir(영수증폴더) else []
                if 새:
                    새것 = 새[-1][:-5]
                    break
                time.sleep(1)
            표[name] = 새것
            흐름글 = f" · 스트리밍(빠진 토큰 {빠짐})" if a.스트림 else ""
            if a.온도 > 0:
                print(f"{name}: 온도 {a.온도} · 시드 {seed} · 토큰 {len(r.get('tokens', []))}{흐름글} · 영수증 {새것}", flush=True)
            else:
                print(f"{name}: 응답이 2단계 기록과 {'같음' if 같음 else '다름'}{흐름글} · 영수증 {새것}", flush=True)
        json.dump(표, open(os.path.join(a.out, "영수증.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    finally:
        if srv is not None:
            srv.terminate()
        s.닫기()
    print(f"합계: 요청 {len(a.이름)} · 영수증 {sum(1 for v in 표.values() if v)} · 2단계 기록과 같음 {같음수}")
    return 0 if 같음수 == len(a.이름) and all(표.values()) else 1


def 서버띄우기(a, 깃발들, log경로):
    log = open(log경로, "w", encoding="utf-8", errors="replace")
    srv = subprocess.Popen([a.서버, "-m", a.모델, *깃발들, "--host", "127.0.0.1", "--port", str(a.라마포트)], stdout=log, stderr=subprocess.STDOUT)
    t0 = time.time()
    while True:
        try:
            if 묻기(a.라마포트, "/health", 시간=5).get("status") == "ok":
                return srv
        except (urllib.error.URLError, ConnectionError, TimeoutError, json.JSONDecodeError):
            pass
        if srv.poll() is not None or time.time() - t0 > 600:
            raise RuntimeError("llama-server 가 준비되지 않았다")
        time.sleep(0.5)


def 새영수증(폴더, 앞):
    for _ in range(120):                                           # 영수증은 스레드가 모델 해시 뒤에 쓴다 — 다 쓴 .json 이름만
        새 = sorted(f for f in set(os.listdir(폴더)) - 앞 - {"기록기.json"} if f.endswith(".json")) if os.path.isdir(폴더) else []
        if 새:
            return 새[-1][:-5]
        time.sleep(1)
    return None


def 대화하기(a):
    """7단계 — 기본 설정 서버에 한 대화(같은 슬롯)의 이어지는 요청과 다른 대화 하나"""
    폴더들 = a.기록폴더.split(",")
    시작 = None
    for d in 폴더들:
        p = os.path.join(d, f"run0_{a.시작}_r0.json")
        if os.path.exists(p):
            시작 = json.load(open(p, encoding="utf-8"))["prompt_text"]
            break
    if 시작 is None:
        raise FileNotFoundError(a.시작)
    깃발대화 = ["-c", "32768"]
    os.makedirs(a.out, exist_ok=True)
    s = 앱(a.꾸러미, a.자료, a.포트)
    srv = None
    표 = {}
    try:
        srv = 서버띄우기(a, 깃발대화, os.path.join(a.out, "llama-server.log"))
        묻기(a.포트, "/api/llama/recorder", {"port": a.라마포트, "args": "llama-server -m " + a.모델 + " " + " ".join(깃발대화)})
        영수증폴더 = os.path.join(a.자료, "영수증")
        꼬리 = "<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n<think>\n"
        프롬프트, 응답 = {}, {}

        def 보내기(이름, p, 씨앗):
            앞 = set(os.listdir(영수증폴더)) if os.path.isdir(영수증폴더) else set()
            r = 묻기(a.포트, "/llama/completion", {"prompt": p, "n_predict": 64, "temperature": 0.0, "seed": 씨앗, "stream": False})
            표[이름] = 새영수증(영수증폴더, 앞)
            프롬프트[이름], 응답[이름] = p, r
            tm = r.get("timings", {})
            print(f"{이름}: 슬롯 {r.get('id_slot')} · 프롬프트 토큰 {r.get('tokens_evaluated')} · cache_n {tm.get('cache_n')} · prompt_n {tm.get('prompt_n')}"
                  f" · 생성 {len(r.get('tokens', []))} · 영수증 {표[이름]}", flush=True)

        보내기("c1", 시작, 1)
        보내기("c2", 프롬프트["c1"] + 응답["c1"]["content"] + 꼬리.format("방금 정리한 것을 한 문장으로 줄여 줘."), 2)
        보내기("c3", 프롬프트["c2"] + 응답["c2"]["content"] + 꼬리.format("그 문장을 영어로 옮겨 줘."), 3)
        보내기("c4", 프롬프트["c3"], 4)                                               # 같은 프롬프트 — 체크포인트 되살리기
        보내기("c5", 프롬프트["c4"] + 응답["c4"]["content"], 5)                       # 생성 잇기 — 되살리지 않고 이어 감
        보내기("c6", 프롬프트["c2"] + 응답["c2"]["content"] + 꼬리.format("대신 세 줄로 나눠 줘."), 6)   # 앞을 고쳐 잇기 — 앞 체크포인트
        보내기("c7", "<|im_start|>user\n짧게 인사해 줘.<|im_end|>\n<|im_start|>assistant\n<think>\n", 7)   # 다른 대화 — 다른 슬롯
        json.dump(표, open(os.path.join(a.out, "영수증.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    finally:
        if srv is not None:
            srv.terminate()
        s.닫기()
    print(f"합계: 요청 {len(표)} · 영수증 {sum(1 for v in 표.values() if v)}")
    return 0 if all(표.values()) else 1


def 채팅하기(a):
    """9단계 — 기본 설정 서버에 채팅 API 의 한 대화와 스트리밍 요청 하나"""
    깃발대화 = ["-c", "32768"]
    os.makedirs(a.out, exist_ok=True)
    s = 앱(a.꾸러미, a.자료, a.포트)
    srv = None
    표 = {}
    try:
        srv = 서버띄우기(a, 깃발대화, os.path.join(a.out, "llama-server.log"))
        묻기(a.포트, "/api/llama/recorder", {"port": a.라마포트, "args": "llama-server -m " + a.모델 + " " + " ".join(깃발대화)})
        영수증폴더 = os.path.join(a.자료, "영수증")
        S = {"role": "system", "content": "너는 GPU 커널을 잘 아는 동료다. 질문의 언어로 짧게 답한다."}
        U = ["CUDA 에서 warp 하나는 스레드 몇 개야? 한 문단으로.", "그럼 AMD 는? 두 문장으로.", "둘의 차이가 커널 작성에 주는 영향 하나만."]

        def 보내기(이름, msgs, 씨앗, 흐름=False):
            앞 = set(os.listdir(영수증폴더)) if os.path.isdir(영수증폴더) else set()
            몸 = {"messages": msgs, "max_tokens": 64, "temperature": 0.0, "seed": 씨앗, "stream": 흐름,
                  "chat_template_kwargs": {"enable_thinking": False}}
            if 흐름:
                r = urllib.request.Request(f"http://127.0.0.1:{a.포트}/llama/v1/chat/completions", data=json.dumps(몸, ensure_ascii=False).encode("utf-8"),
                                           method="POST", headers={"Content-Type": "application/json"})
                urllib.request.urlopen(r, timeout=600).read()
                답 = ""
            else:
                r = 묻기(a.포트, "/llama/v1/chat/completions", 몸)
                답 = r["choices"][0]["message"].get("content") or ""
                v = r.get("__verbose") or {}
                tm = v.get("timings", {})
                print(f"{이름}: 슬롯 {v.get('id_slot')} · 프롬프트 {v.get('tokens_evaluated')} · cache_n {tm.get('cache_n')} · 생성 {len(v.get('tokens', []))}"
                      f" · 답 {답[:30]!r}", flush=True)
            표[이름] = 새영수증(영수증폴더, 앞)
            print(f"{이름}: 영수증 {표[이름]}", flush=True)
            return 답

        m = [S, {"role": "user", "content": U[0]}]
        d1 = 보내기("h1", m, 1)
        m = m + [{"role": "assistant", "content": d1}, {"role": "user", "content": U[1]}]
        d2 = 보내기("h2", m, 2)
        m = m + [{"role": "assistant", "content": d2}, {"role": "user", "content": U[2]}]
        보내기("h3", m, 3)
        보내기("h4", list(m), 4)                                                     # 같은 대화 — 체크포인트 되살리기
        보내기("h5", m[:3] + [{"role": "user", "content": "다른 질문 — Intel GPU 는?"}], 5)   # 앞을 고침 — RAM 캐시 갈래 · 사용자 자리 체크포인트
        보내기("stream", m[:3] + [{"role": "user", "content": "한 줄로."}], 6, 흐름=True)       # 9-2 — 받지 않음
        json.dump(표, open(os.path.join(a.out, "영수증.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    finally:
        if srv is not None:
            srv.terminate()
        s.닫기()
    print(f"합계: 요청 {len(표)} · 영수증 {sum(1 for v in 표.values() if v)}")
    return 0 if all(표.values()) else 1


def 흐름채팅하기(a):
    """18단계 — 2단계 설정 서버에 스트리밍 채팅 요청 다섯"""
    os.makedirs(a.out, exist_ok=True)
    s = 앱(a.꾸러미, a.자료, a.포트)
    srv = None
    표 = {}
    try:
        srv = 서버띄우기(a, 깃발, os.path.join(a.out, "llama-server.log"))
        묻기(a.포트, "/api/llama/recorder", {"port": a.라마포트, "args": "llama-server -m " + a.모델 + " " + " ".join(깃발)})
        영수증폴더 = os.path.join(a.자료, "영수증")
        생각끔 = {"enable_thinking": False}
        요청들 = [("s1", "Answer with one word: what color is the sky on a clear day?", 64, 0.0, 생각끔),
                ("s2", "한 낱말로만 답하세요: 맑은 날 하늘은 무슨 색인가요?", 64, 0.0, 생각끔),
                ("s3", "Explain in three sentences why the sky is blue.", 96, 0.0, None),
                ("s4", "하늘이 파란 까닭을 세 문장으로 설명해 주세요.", 96, 0.0, 생각끔),
                ("t1", "Answer with one word: what color is grass?", 64, 0.8, 생각끔)]
        if a.온도 > 0:                                           # 22단계 — 모두 이 온도로(시드는 요청마다 다르게)
            요청들 = [(f"{이름}t", 글, 192, a.온도, 틀) for 이름, 글, n, 온도, 틀 in 요청들]
        for q, (이름, 글, n, 온도, 틀) in enumerate(요청들):
            앞 = set(os.listdir(영수증폴더)) if os.path.isdir(영수증폴더) else set()
            몸 = {"messages": [{"role": "user", "content": 글}], "max_tokens": n, "temperature": 온도, "seed": 7 + q * (a.온도 > 0), "stream": True}
            if 틀:
                몸["chat_template_kwargs"] = 틀
            토큰, 칸, 끝 = 흐름모으기(흐름보내기(a.포트, "/llama/v1/chat/completions", 몸), True)
            표[이름] = 새영수증(영수증폴더, 앞)
            print(f"{이름}: 칸 {len(칸)} · 생성 {(끝 or {}).get('tokens_predicted')} · 끝 {(끝 or {}).get('stop_type')} · 영수증 {표[이름]}", flush=True)
        json.dump(표, open(os.path.join(a.out, "영수증.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    finally:
        if srv is not None:
            srv.terminate()
        s.닫기()
    print(f"합계: 요청 {len(표)} · 영수증 {sum(1 for v in 표.values() if v)}")
    return 0 if all(표.values()) else 1


def 올라마하기(a):
    """20단계 — Ollama 가 띄운 llama-server 에 기록기를 붙여 2단계 프롬프트를 보낸다"""
    import re
    폴더들 = a.기록폴더.split(",")

    def 기록(name):
        for d in 폴더들:
            q = os.path.join(d, f"run0_{name}_r0.json")
            if os.path.exists(q):
                return json.load(open(q, encoding="utf-8"))
        raise FileNotFoundError(name)

    견자료, 견out = a.견줄.split(",")
    견표 = json.load(open(os.path.join(견out, "영수증.json"), encoding="utf-8"))
    os.makedirs(a.out, exist_ok=True)
    밑 = os.path.dirname(os.path.abspath(a.out))
    홈, 모델들 = os.path.join(밑, "ollama-home"), os.path.join(밑, "ollama-models")
    os.makedirs(홈, exist_ok=True)
    os.makedirs(모델들, exist_ok=True)
    올포트 = 11500
    env = dict(os.environ)
    env.update({"OLLAMA_HOST": f"127.0.0.1:{올포트}", "OLLAMA_MODELS": 모델들, "OLLAMA_DEBUG": "1", "USERPROFILE": 홈, "HOME": 홈,
                "OLLAMA_KEEP_ALIVE": "60m"})
    로그경로 = os.path.join(a.out, "ollama.log")
    올 = subprocess.Popen([a.올라마, "serve"], env=env, stdout=open(로그경로, "w", encoding="utf-8", errors="replace"), stderr=subprocess.STDOUT)
    s = None
    표, 같음수 = {}, 0
    try:
        for _ in range(240):
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{올포트}/api/version", timeout=3).read()
                break
            except (urllib.error.URLError, ConnectionError, TimeoutError):
                time.sleep(0.5)
        mf = os.path.join(a.out, "Modelfile")
        open(mf, "w", encoding="utf-8").write(f"FROM {a.모델}\n")
        r = subprocess.run([a.올라마, "create", "replay", "-f", mf], env=env, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=3600)
        print(f"ollama create: {r.returncode}", flush=True)
        묻기(올포트, "/api/generate", {"model": "replay", "prompt": "Hi", "raw": True, "stream": False, "options": {"num_predict": 1, "temperature": 0}})
        cmd = [m for m in re.findall(r'msg="starting llama-server" cmd="([^"]+)"', open(로그경로, encoding="utf-8", errors="replace").read())][-1]
        cmd = cmd.replace("\\\\", "\\")
        라마포트 = int(re.search(r"--port (\d+)", cmd).group(1))
        인자 = cmd.split("llama-server.exe", 1)[1].strip()
        print(f"Ollama 의 llama-server: 포트 {라마포트} · 인자 {인자}", flush=True)
        s = 앱(a.꾸러미, a.자료, a.포트)
        묻기(a.포트, "/api/llama/recorder", {"port": 라마포트, "args": "llama-server " + 인자})
        영수증폴더 = os.path.join(a.자료, "영수증")
        for name in a.이름:
            want = 기록(name)
            견 = json.load(open(os.path.join(견자료, "영수증", 견표[name] + ".json"), encoding="utf-8"))
            if len(견["prompt_tokens"]) + 256 > 4096:
                print(f"{name}: 건너뜀 — 프롬프트 {len(견['prompt_tokens'])} 토큰 + 생성 256 이 Ollama 의 문맥 4096 을 넘음", flush=True)
                continue
            앞 = set(os.listdir(영수증폴더)) if os.path.isdir(영수증폴더) else set()
            r2 = 묻기(a.포트, "/llama/completion", {"prompt": want["prompt_text"], "n_predict": 256, "temperature": 0, "seed": 0, "stream": False,
                                                 "cache_prompt": False})
            같음 = r2.get("tokens") == 견["response"]["tokens"] and 칸들(r2) == 칸들(견["response"])
            같음수 += 같음
            표[name] = 새영수증(영수증폴더, 앞)
            print(f"{name}: 공식 빌드의 영수증과 {'같음' if 같음 else '다름'} · 영수증 {표[name]}", flush=True)
        json.dump(표, open(os.path.join(a.out, "영수증.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    finally:
        if s is not None:
            s.닫기()
        subprocess.run(["taskkill", "/PID", str(올.pid), "/T", "/F"], capture_output=True)
    print(f"합계: 요청 {len(표)} · 영수증 {sum(1 for v in 표.values() if v)} · 공식 빌드와 같음 {같음수}")
    return 0 if 같음수 == len(표) and all(표.values()) else 1


def 검증하기(a):
    표 = json.load(open(os.path.join(a.out, "영수증.json"), encoding="utf-8"))
    이름 = list(표) if a.이름 is 이름들10 else a.이름                  # --이름 이 없으면 영수증.json 의 모두(바꾼 영수증이면 alter_…)
    s = 앱(a.꾸러미, a.자료, a.포트)
    결과 = {}
    try:
        for name in 이름:
            try:
                j = 묻기(a.포트, "/api/llama/verify", {"receipt": 표[name], "layer": a.층, "device": a.장치})
            except urllib.error.HTTPError as e:                     # 영수증이 없음 등 — 적고 다음으로
                결과[name] = {"state": "error", "reason": f"HTTP {e.code}"}
                print(f"{name}: error — HTTP {e.code}", flush=True)
                continue
            끝 = s.작업(j["job"])
            sm = {}
            if 끝.get("result"):
                p = os.path.join(a.자료, "결과", 끝["result"], "summary.json")
                if os.path.exists(p):
                    sm = json.load(open(p, encoding="utf-8"))
            결과[name] = {"state": 끝.get("state"), "summary": 끝.get("summary"), "reason": 끝.get("reason"),
                         **{k: sm.get(k) for k in ("steps", "values", "mismatches", "first_divergence", "steps_without_probs", "same", "seconds",
                                                   "run_log", "cache", "differences")}}
            print(f"{name}: {끝.get('state')} — {끝.get('summary') or 끝.get('reason')}", flush=True)
    finally:
        s.닫기()
    json.dump(결과, open(os.path.join(a.out, f"검증-{a.층}-{a.장치 or 'default'}.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"합계: 요청 {len(결과)} · 같음 {sum(1 for x in 결과.values() if x.get('same'))} · 값 {sum(x.get('values') or 0 for x in 결과.values())}"
          f" · 다른 값 {sum(x.get('mismatches') or 0 for x in 결과.values())}")
    return 0


def 바꾸기(a):
    폴더 = os.path.join(a.자료, "영수증")
    표0 = json.load(open(os.path.join(a.기록out, "영수증.json"), encoding="utf-8"))

    def 읽기(rid):
        return json.load(open(os.path.join(폴더, rid + ".json"), encoding="utf-8"))

    def 쓰기(d, rid):
        with open(os.path.join(폴더, rid + ".json"), "w", encoding="utf-8", newline="\n") as f:
            json.dump(d, f, ensure_ascii=False)
            f.write("\n")

    def ascii칸(e):
        return len(e["bytes"]) > 0 and all(b < 128 for b in e["bytes"])

    표 = {}
    if a.단계 == 9:                                                # 9-2 — 채팅 영수증 h2 의 사용자 구분 토큰을 지운 것
        base = 표0["h2"]
        d = 읽기(base)
        d.setdefault("chat", {})["user_start_tokens"] = []
        쓰기(d, base + "-nodelim")
        표["alter9_nodelim"] = base + "-nodelim"
        표["stream"] = 표0["stream"]
        print("alter9_nodelim: chat.user_start_tokens → []")
        os.makedirs(a.out, exist_ok=True)
        json.dump(표, open(os.path.join(a.out, "영수증.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        print(f"합계: 바꾼 영수증 {len(표)}")
        return 0
    if a.단계 == 7:                                                # 7-2 — 대화의 c3 에서(같은 슬롯의 사슬)
        base = 표0["c3"]

        def 고침7(suffix, f):
            d = 읽기(base)
            설명 = f(d)
            쓰기(d, base + suffix)
            표["alter7_" + suffix.lstrip("-")] = base + suffix
            print(f"alter7_{suffix.lstrip('-')}: {설명}")

        def 캐시(d):
            d["response"]["timings"]["cache_n"] += 1
            return f"cache_n → {d['response']['timings']['cache_n']}"

        def 포트(d):
            d["server"]["port"] += 1
            return f"서버 포트 → {d['server']['port']}(같은 서버의 앞 영수증이 없다)"

        def 슬롯(d):
            for x in d.get("slots_after", []):
                if x["id"] != d["response"]["id_slot"]:
                    x["id_task"] = 999
                    return f"슬롯 {x['id']} 에 id_task"
            return "?"
        고침7("-cache", 캐시)                                       # 받지 않음 — 흉내의 cache_n 이 다르다
        고침7("-port", 포트)                                        # 받지 않음 — 처음 요청이 없다
        고침7("-slots", 슬롯)                                       # 받지 않음 — 다른 슬롯도 쓰였다
        os.makedirs(a.out, exist_ok=True)
        json.dump(표, open(os.path.join(a.out, "영수증.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        print(f"합계: 바꾼 영수증 {len(표)}")
        return 0
    base = 표0["math"]
    if a.단계 == 5:                                                # 5-2 — 온도 > 0 의 영수증에서
        def 고침(suffix, **kv):
            d = 읽기(base)
            gs, body = d["response"]["generation_settings"], d["request"]["body"]
            for k, v in kv.items():
                gs[k] = v
                if k in ("seed", "temperature", "top_k", "mirostat"):
                    body[k] = v
            쓰기(d, base + suffix)
            표["alter5_" + suffix.lstrip("-")] = base + suffix
            print(f"alter5_{suffix.lstrip('-')}: {kv}")
        g0 = 읽기(base)["response"]["generation_settings"]
        고침("-seed", seed=g0["seed"] + 1)                         # 다름(처음 갈리는 걸음부터)
        고침("-temp", temperature=1.2)                              # 다름
        고침("-topk0", top_k=0)                                     # 받지 않음
        고침("-mirostat", mirostat=1)                               # 받지 않음
        고침("-noseed", seed=4294967295)                            # 받지 않음
        os.makedirs(a.out, exist_ok=True)
        json.dump(표, open(os.path.join(a.out, "영수증.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        print(f"합계: 바꾼 영수증 {len(표)}")
        return 0
    d = 읽기(base)                                                 # 1. 모델 해시의 한 글자 — 시작 전에 멈춘다
    h = d["model"]["sha256"]
    d["model"]["sha256"] = ("0" if h[0] != "0" else "1") + h[1:]
    쓰기(d, base + "-model")
    표["alter_model"] = base + "-model"
    d = 읽기(base)                                                 # 2. 걸음 k 의 토큰을 2위 후보로(tokens · 그 칸의 id · 글 · logprob)
    cps, toks = d["response"]["completion_probabilities"], d["response"]["tokens"]
    k = next(i for i in range(40, len(toks)) if cps[i]["id"] == toks[i] and ascii칸(cps[i]) and len(cps[i]["top_logprobs"]) > 1
             and ascii칸(cps[i]["top_logprobs"][1]) and cps[i]["top_logprobs"][1]["id"] != toks[i])
    alt = cps[k]["top_logprobs"][1]
    print(f"alter_token: 걸음 {k} — {toks[k]} → {alt['id']}")
    toks[k] = alt["id"]
    cps[k].update({"id": alt["id"], "token": alt["token"], "bytes": alt["bytes"], "logprob": alt["logprob"]})
    쓰기(d, base + "-token")
    표["alter_token"] = base + "-token"
    longest = max(표0.values(), key=lambda r: len(읽기(r)["prompt_tokens"]))   # 3. 프롬프트가 가장 긴 영수증의 n_ubatch 256
    d = 읽기(longest)
    print(f"alter_ubatch: 프롬프트 {len(d['prompt_tokens'])} 토큰 · n_ubatch {d['server']['n_ubatch']} → 256")
    d["server"]["n_ubatch"] = 256
    쓰기(d, longest + "-ubatch")
    표["alter_ubatch"] = longest + "-ubatch"
    d = 읽기(base)                                                 # 4. 걸음 77 · 3위 로그 확률의 마지막 비트
    e = d["response"]["completion_probabilities"][77]["top_logprobs"][2]
    b = struct.unpack("<I", struct.pack("<f", e["logprob"]))[0] ^ 1
    e["logprob"] = struct.unpack("<f", struct.pack("<I", b))[0]
    print(f"alter_bit: 걸음 77 · 3위 → 0x{b:08x}")
    쓰기(d, base + "-bit")
    표["alter_bit"] = base + "-bit"
    os.makedirs(a.out, exist_ok=True)
    json.dump(표, open(os.path.join(a.out, "영수증.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"합계: 바꾼 영수증 {len(표)}")
    return 0


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="할일", required=True)
    for n in ("기록", "검증", "대화", "채팅", "흐름채팅", "올라마"):
        q = sub.add_parser(n)
        q.add_argument("꾸러미")
        q.add_argument("자료")
        q.add_argument("out")
        q.add_argument("--포트", type=int, default=44641)
        q.add_argument("--이름", type=lambda s: s.split(","), default=이름들10)
        if n == "올라마":
            q.add_argument("--올라마", required=True)
            q.add_argument("--모델", required=True)
            q.add_argument("--기록폴더", required=True)
            q.add_argument("--견줄", required=True)
        elif n in ("기록", "대화", "채팅", "흐름채팅"):
            q.add_argument("--서버", required=True)
            q.add_argument("--모델", required=True)
            q.add_argument("--기록폴더", required=True)
            q.add_argument("--라마포트", type=int, default=18181)
            q.add_argument("--온도", type=float, default=0.0)
            q.add_argument("--시드", type=int, default=1)
            q.add_argument("--시작", default="multi_turn")
            q.add_argument("--스트림", action="store_true")                 # 18단계 — 기록: /completion 을 스트리밍으로
        else:
            q.add_argument("--층", choices=["cuda", "vulkan"], required=True)
            q.add_argument("--장치", default="")
    q = sub.add_parser("바꾸기")
    q.add_argument("자료")
    q.add_argument("기록out")
    q.add_argument("out")
    q.add_argument("--단계", type=int, choices=[4, 5, 7, 9], default=4)
    a = p.parse_args()
    return {"기록": 기록하기, "검증": 검증하기, "대화": 대화하기, "채팅": 채팅하기, "흐름채팅": 흐름채팅하기, "올라마": 올라마하기,
            "바꾸기": 바꾸기}[a.할일](a)


if __name__ == "__main__":
    sys.exit(main())
