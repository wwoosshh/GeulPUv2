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
            r = 묻기(a.포트, "/llama/completion", {"prompt": want["prompt_text"], "n_predict": 256, "temperature": a.온도, "seed": seed, "stream": False})
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
            if a.온도 > 0:
                print(f"{name}: 온도 {a.온도} · 시드 {seed} · 토큰 {len(r.get('tokens', []))} · 영수증 {새것}", flush=True)
            else:
                print(f"{name}: 응답이 2단계 기록과 {'같음' if 같음 else '다름'} · 영수증 {새것}", flush=True)
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
    for n in ("기록", "검증", "대화"):
        q = sub.add_parser(n)
        q.add_argument("꾸러미")
        q.add_argument("자료")
        q.add_argument("out")
        q.add_argument("--포트", type=int, default=44641)
        q.add_argument("--이름", type=lambda s: s.split(","), default=이름들10)
        if n in ("기록", "대화"):
            q.add_argument("--서버", required=True)
            q.add_argument("--모델", required=True)
            q.add_argument("--기록폴더", required=True)
            q.add_argument("--라마포트", type=int, default=18181)
            q.add_argument("--온도", type=float, default=0.0)
            q.add_argument("--시드", type=int, default=1)
            q.add_argument("--시작", default="multi_turn")
        else:
            q.add_argument("--층", choices=["cuda", "vulkan"], required=True)
            q.add_argument("--장치", default="")
    q = sub.add_parser("바꾸기")
    q.add_argument("자료")
    q.add_argument("기록out")
    q.add_argument("out")
    q.add_argument("--단계", type=int, choices=[4, 5, 7], default=4)
    a = p.parse_args()
    return {"기록": 기록하기, "검증": 검증하기, "대화": 대화하기, "바꾸기": 바꾸기}[a.할일](a)


if __name__ == "__main__":
    sys.exit(main())
