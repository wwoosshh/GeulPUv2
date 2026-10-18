"""28단계 시험 — 글(geul) 언어로 쓴 서버(호스트/서버.gl)를 띄워 28a(함께 돌아도 같은 비트)와 28c(API)를 본다. 표준 라이브러리만.

    python -I 시험/서버28.py <서버.exe> <모델.gguf> <작업폴더> [--슬롯 4] [--문맥 4096] [--포트 8093] [--일 a,c] [--출력 결과.json]
                             [--시험7B 시험_7B.exe --ptx 모듈.ptx]

28a: 요청 열둘(채팅 탐욕 넷 · 채팅 뽑기 다섯 · 토큰 프롬프트 탐욕 둘 · 두 차례 대화의 둘째)을
  (가) 슬롯 1 에 하나씩, (나) 슬롯 S 에 모두 한꺼번에, (다) 슬롯 S 에 거꾸로 · 씨앗을 둔 무작위 틈(0 ~ 2 초), (라) 슬롯 S 에 요청마다 둘씩 한꺼번에
  보내 요청마다 뽑은 토큰과 걸음마다의 로짓 해시(geulpu.logit_hash)가 모두 같은지. (마) 토큰 프롬프트 요청의 걸음마다 로짓(기록)이
  시험_7B 의 한 번에 길((나) 길 — 22a · 23a 에서 CPU 참조와 대조한 길)의 해당 행과 비트까지 같은지. (바) 둘째 차례가 프롬프트 캐시를 쓸 때와
  cache_prompt false 일 때 같은지. (가)의 기록(뽑기 요청의 로짓 · 토큰)은 28b (3) 이 쓴다.
28c: API 열세 가지(문서/28단계.md).
"""
import argparse, hashlib, http.client, json, os, random, socket, struct, subprocess, sys, threading, time, io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")


# ---------------------------------------------------------------- 서버 띄우기
class 서버:
    def __init__(self, exe, gguf, 포트, 슬롯, 문맥, 기록=None, ptx=None, 로그=None):
        self.포트 = 포트
        인자 = [exe, gguf, "--포트", str(포트), "--슬롯", str(슬롯), "--문맥", str(문맥)]
        if 기록:
            os.makedirs(기록, exist_ok=True)
            인자 += ["--기록", 기록]
        if ptx:
            인자 += ["--ptx", ptx]
        self.로그 = open(로그, "w", encoding="utf-8") if 로그 else subprocess.DEVNULL
        self.p = subprocess.Popen(인자, stdout=self.로그, stderr=subprocess.STDOUT)
        t0 = time.time()
        while time.time() - t0 < 300:
            if self.p.poll() is not None:
                raise SystemExit(f"서버가 멈췄다(종료 {self.p.returncode})")
            try:
                s, d = 요청(포트, "GET", "/health")
                if s == 200:
                    return
            except OSError:
                pass
            time.sleep(0.5)
        raise SystemExit("서버가 뜨지 않는다")

    def 끄기(self):
        self.p.kill()
        self.p.wait()
        if self.로그 is not subprocess.DEVNULL:
            self.로그.close()


def 요청(포트, 방법, 길, 몸=None, 머리=None, 시간=600):
    c = http.client.HTTPConnection("127.0.0.1", 포트, timeout=시간)
    본 = None
    if 몸 is not None:
        본 = 몸 if isinstance(몸, bytes) else json.dumps(몸, ensure_ascii=False).encode("utf-8")
    h = {"Content-Type": "application/json"}
    h.update(머리 or {})
    c.request(방법, 길, body=본, headers=h)
    r = c.getresponse()
    d = r.read()
    c.close()
    try:
        return r.status, json.loads(d)
    except ValueError:
        return r.status, d


def 흘림요청(포트, 길, 몸):
    """SSE — (조각 JSON 들, [DONE] 이 있었나)"""
    c = http.client.HTTPConnection("127.0.0.1", 포트, timeout=600)
    c.request("POST", 길, body=json.dumps(dict(몸, stream=True), ensure_ascii=False).encode("utf-8"),
              headers={"Content-Type": "application/json"})
    r = c.getresponse()
    raw = r.read().decode("utf-8")
    c.close()
    조각 = []
    끝 = False
    for 사건 in raw.split("\n\n"):
        사건 = 사건.strip()
        if not 사건.startswith("data: "):
            continue
        자료 = 사건[6:]
        if 자료 == "[DONE]":
            끝 = True
        else:
            조각.append(json.loads(자료))
    return r.status, 조각, 끝


# ---------------------------------------------------------------- 요청 모음
def 모음(작업):
    r = random.Random(28)
    R1 = [r.randrange(0, 151000) for _ in range(37)]
    R2 = [r.randrange(0, 151000) for _ in range(600)]
    해시 = {"logit_hash": True}
    def 채팅(글, **kw):
        return {"messages": [{"role": "user", "content": 글}], **kw}
    요들 = [
        ("G1", "/v1/chat/completions", 채팅("Write a short poem about the sea.", temperature=0, max_tokens=128)),
        ("G2", "/v1/chat/completions", 채팅("한국의 수도는 어디인가요? 두 문장으로 설명해 주세요.", temperature=0, max_tokens=128)),
        ("G3", "/v1/chat/completions", 채팅("Write a Python function that checks whether a number is prime.", temperature=0, max_tokens=128)),
        ("G4", "/v1/chat/completions", 채팅("What is 17 * 23? Show your work briefly.", temperature=0, max_tokens=128)),
        ("S1", "/v1/chat/completions", 채팅("Write a short poem about the sea.", temperature=0.8, top_k=40, top_p=0.95, min_p=0.05, seed=1, max_tokens=128)),
        ("S2", "/v1/chat/completions", 채팅("Write a short poem about the sea.", temperature=0.8, top_k=40, top_p=0.95, min_p=0.05, seed=2, max_tokens=128)),
        ("S3", "/v1/chat/completions", 채팅("Tell me a fun fact about octopuses.", temperature=1.2, top_k=0, top_p=1.0, min_p=0.0, seed=3, max_tokens=128)),
        ("S4", "/v1/chat/completions", 채팅("Describe a sunset in three sentences.", temperature=0.5, top_k=10, top_p=0.9, min_p=0.0, seed=4, max_tokens=128)),
        ("S5", "/v1/chat/completions", 채팅("Give me three names for a cat.", seed=5, max_tokens=128)),
        ("R1", "/v1/completions", {"prompt": R1, "temperature": 0, "max_tokens": 64, "ignore_eos": True}),
        ("R2", "/v1/completions", {"prompt": R2, "temperature": 0, "max_tokens": 64, "ignore_eos": True}),
    ]
    for _, _, 몸 in 요들:
        몸["geulpu"] = dict(해시)
    return 요들, R1, R2


T1글 = "My name is Mina. What is a good hobby for a rainy day? Answer briefly."
T2글 = "Can you remind me what my name is?"


def T2몸(답, 캐시=True):
    return {"messages": [{"role": "user", "content": T1글}, {"role": "assistant", "content": 답},
                         {"role": "user", "content": T2글}],
            "temperature": 0, "max_tokens": 48, "cache_prompt": 캐시, "geulpu": {"logit_hash": True}}


def 요약(d):
    g = d.get("geulpu", {})
    ch = d["choices"][0]
    글 = ch.get("message", {}).get("content") if "message" in ch else ch.get("text")
    return {"tokens": g.get("tokens"), "hashes": g.get("logit_hashes"), "finish": ch["finish_reason"], "text": 글,
            "seed": g.get("seed"), "cached": g.get("prompt_tokens_cached"), "slot": g.get("slot")}


def 보내기묶음(포트, 목록, 틈=None):
    """(이름, 길, 몸) 목록을 함께 — 틈[i] 초 뒤에 보낸다. 이름 → 요약"""
    out = {}
    잠금 = threading.Lock()
    시작 = time.time()
    def 일(i, 이름, 길, 몸):
        if 틈:
            남음 = 틈[i] - (time.time() - 시작)
            if 남음 > 0:
                time.sleep(남음)
        s, d = 요청(포트, "POST", 길, 몸)
        if s != 200:
            raise SystemExit(f"{이름}: {s} {d}")
        with 잠금:
            out.setdefault(이름, []).append(요약(d))
    ts = [threading.Thread(target=일, args=(i, n, l, b)) for i, (n, l, b) in enumerate(목록)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    return out


def 일28a(a):
    요들, R1, R2 = 모음(a.작업)
    기록 = os.path.join(a.작업, "기록28")
    결과 = {"차례": {}}
    # (가) 슬롯 1 — 하나씩(뽑기 · 토큰 프롬프트 요청은 기록)
    sv = 서버(a.exe, a.gguf, a.포트, 1, a.문맥, 기록=기록, 로그=os.path.join(a.작업, "서버_가.log"))
    try:
        가 = {}
        for 이름, 길, 몸 in 요들:
            b = json.loads(json.dumps(몸))
            if 이름[0] in "SR":
                b["geulpu"]["record"] = f"rec_{이름}"
            s, d = 요청(a.포트, "POST", 길, b)
            assert s == 200, (이름, s, d)
            가[이름] = [요약(d)]
        s, d = 요청(a.포트, "POST", "/v1/chat/completions", {"messages": [{"role": "user", "content": T1글}], "temperature": 0, "max_tokens": 64})
        assert s == 200
        T1답 = d["choices"][0]["message"]["content"]
        s, d = 요청(a.포트, "POST", "/v1/chat/completions", T2몸(T1답))
        가["T2"] = [요약(d)]
        s, d = 요청(a.포트, "POST", "/v1/chat/completions", T2몸(T1답, 캐시=False))
        T2새로 = 요약(d)
    finally:
        sv.끄기()
    결과["차례"]["가"] = 가
    결과["T1답"] = T1답
    결과["뽑기요청"] = {이름: {"body": 몸, "seed": 가[이름][0]["seed"]} for 이름, _, 몸 in 요들 if 이름[0] == "S"}
    결과["기록폴더"] = 기록
    전체 = 요들 + [("T2", "/v1/chat/completions", T2몸(T1답))]
    # (나) (다) (라) — 슬롯 S
    sv = 서버(a.exe, a.gguf, a.포트, a.슬롯, a.문맥, 로그=os.path.join(a.작업, "서버_나다라.log"))
    try:
        결과["차례"]["나"] = 보내기묶음(a.포트, 전체)
        r = random.Random(2028)
        거꾸로 = list(reversed(전체))
        틈 = sorted(r.uniform(0, 2.0) for _ in 거꾸로)
        결과["차례"]["다"] = 보내기묶음(a.포트, 거꾸로, 틈)
        결과["차례"]["라"] = 보내기묶음(a.포트, 전체 + 전체)
    finally:
        sv.끄기()
    # 견주기 — (가)가 기준
    다름 = 0
    견준 = 0
    for 이름 in 가:
        기준 = 가[이름][0]
        for 차례 in ("나", "다", "라"):
            for x in 결과["차례"][차례][이름]:
                견준 += 1
                if x["tokens"] != 기준["tokens"] or x["hashes"] != 기준["hashes"]:
                    다름 += 1
                    print(f"  다름: {이름} ({차례})")
    결과["견준"] = 견준
    결과["다름"] = 다름
    print(f"28a (가 ~ 라): 요청 {len(가)} · 견준 {견준} · 다름 {다름}")
    # (바) 캐시를 쓸 때 · 안 쓸 때
    바같음 = T2새로["tokens"] == 가["T2"][0]["tokens"] and T2새로["hashes"] == 가["T2"][0]["hashes"]
    결과["바"] = {"같음": 바같음, "캐시": 가["T2"][0]["cached"], "새로캐시": T2새로["cached"]}
    print(f"28a (바) 둘째 차례 캐시 {가['T2'][0]['cached']} 토큰 대 캐시 끔 {T2새로['cached']}: 같음 {바같음}")
    # 뽑기 요청: 씨앗이 다르면 다른 답(S1 · S2)
    결과["씨앗다름"] = 가["S1"][0]["tokens"] != 가["S2"][0]["tokens"]
    # (마) 시험_7B 의 한 번에 길
    if a.시험7B:
        결과["마"] = 마견주기(a, 기록, R1, R2, 가)
    결과["통과"] = 다름 == 0 and 바같음 and (not a.시험7B or 결과["마"]["다름"] == 0)
    return 결과


def 마견주기(a, 기록, R1, R2, 가):
    입력 = os.path.join(a.작업, "입력28마.bin")
    출력 = os.path.join(a.작업, "시험7B_28마")
    os.makedirs(출력, exist_ok=True)
    with open(입력, "wb") as f:
        f.write(struct.pack("<qqq", a.문맥, len(R2) + 63, 2))       # 로짓행 = 잡을 행(q − 잡기시작)의 가장 큰 것
        for R in (R1, R2):
            f.write(struct.pack("<qqq", len(R), 64, 0))
            f.write(struct.pack(f"<{len(R)}q", *R))
    subprocess.run([a.시험7B, a.gguf, a.ptx, 입력, 출력], check=True, stdout=subprocess.DEVNULL)
    V = None
    다름 = 0
    행 = 0
    for i, (이름, R) in enumerate((("R1", R1), ("R2", R2))):
        우리 = open(os.path.join(기록, f"rec_{이름}_로짓.bin"), "rb").read()
        저쪽 = open(os.path.join(출력, f"경우{i}_로짓.bin"), "rb").read()
        q = len(R) + 63
        V = len(저쪽) // (4 * q)
        p = len(R)
        for s in range(64):
            a1 = 우리[s * 4 * V:(s + 1) * 4 * V]
            b1 = 저쪽[(p - 1 + s) * 4 * V:(p + s) * 4 * V]
            행 += 1
            if a1 != b1:
                다름 += 1
        토큰 = list(struct.unpack(f"<{len(R) + 63}q", open(os.path.join(출력, f"경우{i}_토큰.bin"), "rb").read()))
        if 토큰[len(R):] != 가[이름][0]["tokens"][:63]:
            다름 += 1
            print(f"  (마) {이름}: 토큰이 다르다")
    print(f"28a (마) 시험_7B 의 한 번에 길과: 행 {행} · 다름 {다름}")
    return {"행": 행, "다름": 다름}


# ---------------------------------------------------------------- 28c
def 일28c(a):
    sv = 서버(a.exe, a.gguf, a.포트, a.슬롯, a.문맥, 로그=os.path.join(a.작업, "서버_다.log"))
    결과 = {}
    def 판(번, 이름, 맞음, 덧=""):
        결과[f"{번} {이름}"] = bool(맞음)
        print(f"28c {번:2d} {'통과' if 맞음 else '실패'} — {이름} {덧}")
    try:
        p = a.포트
        s, d = 요청(p, "GET", "/health")
        판(1, "health", s == 200 and d == {"status": "ok"})
        s, d = 요청(p, "GET", "/v1/models")
        판(2, "models", s == 200 and d["object"] == "list" and d["data"][0]["object"] == "model" and len(d["data"][0]["id"]) > 0)
        몸 = {"model": "x", "messages": [{"role": "system", "content": "You are terse."}, {"role": "user", "content": "Name three colors."}],
             "temperature": 0, "max_tokens": 40, "geulpu": {}}
        s, d = 요청(p, "POST", "/v1/chat/completions", 몸)
        ch = d["choices"][0]
        u = d["usage"]
        토큰 = d["geulpu"]["tokens"]
        생성수 = len(토큰) - (1 if ch["finish_reason"] == "stop" and len(토큰) > u["completion_tokens"] else 0)
        판(3, "채팅", s == 200 and d["object"] == "chat.completion" and d["model"] == "x" and ch["message"]["role"] == "assistant"
           and len(ch["message"]["content"]) > 0 and ch["finish_reason"] in ("stop", "length")
           and u["total_tokens"] == u["prompt_tokens"] + u["completion_tokens"] and u["completion_tokens"] == 생성수
           and d["id"].startswith("chatcmpl-"))
        온글 = ch["message"]["content"]
        s2, 조각, 끝 = 흘림요청(p, "/v1/chat/completions", dict(몸, stream_options={"include_usage": True}))
        내용 = [x for x in 조각 if x.get("choices")]
        이은 = "".join((x["choices"][0]["delta"].get("content") or "") for x in 내용)
        사용 = [x for x in 조각 if not x.get("choices") and x.get("usage")]
        판(4, "흘려보내기", s2 == 200 and 끝 and 내용[0]["choices"][0]["delta"].get("role") == "assistant" and 이은 == 온글
           and 내용[-1]["choices"][0]["finish_reason"] == ch["finish_reason"] and len(사용) == 1 and 사용[0]["usage"] == u
           and all(x["object"] == "chat.completion.chunk" for x in 조각), f"(조각 {len(조각)})")
        s3, d3 = 요청(p, "POST", "/v1/completions", {"prompt": "The quick brown fox", "max_tokens": 8, "temperature": 0, "geulpu": {}})
        s4, d4 = 요청(p, "POST", "/v1/completions", {"prompt": [785, 3974, 13876, 38835], "max_tokens": 8, "temperature": 0, "geulpu": {}})
        판(5, "completions 글 · 토큰", s3 == 200 and s4 == 200 and d3["object"] == "text_completion" and isinstance(d3["choices"][0]["text"], str)
           and d4["choices"][0]["text"] is not None)
        # 6 stop — 먼저 멈춤 없이 낸 글에서 멈춤 글을 고른다
        기본 = {"messages": [{"role": "user", "content": "Count from 1 to 20, separated by commas."}], "temperature": 0, "max_tokens": 80}
        s, d = 요청(p, "POST", "/v1/chat/completions", 기본)
        전글 = d["choices"][0]["message"]["content"]
        멈춤 = "7"
        s, d6 = 요청(p, "POST", "/v1/chat/completions", dict(기본, stop=[멈춤, "zzz"]))
        s6, 조각6, _ = 흘림요청(p, "/v1/chat/completions", dict(기본, stop=[멈춤, "zzz"]))
        흘린 = "".join((x["choices"][0]["delta"].get("content") or "") for x in 조각6 if x.get("choices"))
        글6 = d6["choices"][0]["message"]["content"]
        판(6, "stop", 멈춤 in 전글 and 글6 == 전글[:전글.index(멈춤)] and d6["choices"][0]["finish_reason"] == "stop" and 흘린 == 글6, repr(글6[-20:]))
        s, d7 = 요청(p, "POST", "/v1/chat/completions", dict(기본, max_tokens=5))
        판(7, "max_tokens", d7["choices"][0]["finish_reason"] == "length" and d7["usage"]["completion_tokens"] == 5)
        한글 = "한국어와 이모지 🙂 를 섞은 글: 서울은 어디?"
        몸8 = {"messages": [{"role": "user", "content": 한글}], "temperature": 0, "max_tokens": 16, "geulpu": {}}
        s, d8a = 요청(p, "POST", "/v1/chat/completions", json.dumps(몸8, ensure_ascii=False).encode("utf-8"))
        s, d8b = 요청(p, "POST", "/v1/chat/completions", json.dumps(몸8, ensure_ascii=True).encode("ascii"))
        판(8, "한글 · 이모지(날글 · \\u)", d8a["geulpu"]["tokens"] == d8b["geulpu"]["tokens"] and d8a["usage"] == d8b["usage"]
           and d8a["choices"][0]["message"]["content"] == d8b["choices"][0]["message"]["content"])
        e1 = 요청(p, "POST", "/v1/chat/completions", b"{bad json")
        e2 = 요청(p, "GET", "/nope")
        e3 = 요청(p, "POST", "/v1/chat/completions", {"messages": [{"role": "user", "content": "hi"}], "n": 2})
        e4 = 요청(p, "POST", "/v1/chat/completions", {"messages": [{"role": "user", "content": "hi"}], "logprobs": True})
        e5 = 요청(p, "POST", "/v1/completions", {"prompt": [1] * a.문맥, "max_tokens": 1})
        오류맞음 = lambda r, c: r[0] == c and isinstance(r[1], dict) and "message" in r[1].get("error", {})
        판(9, "오류 다섯", 오류맞음(e1, 400) and 오류맞음(e2, 404) and 오류맞음(e3, 400) and 오류맞음(e4, 400) and 오류맞음(e5, 400))
        c = http.client.HTTPConnection("127.0.0.1", p, timeout=120)
        c.request("GET", "/health")
        r1 = c.getresponse(); r1.read()
        c.request("POST", "/v1/completions", body=json.dumps({"prompt": "Hi", "max_tokens": 2, "temperature": 0}),
                  headers={"Content-Type": "application/json"})
        r2 = c.getresponse(); r2.read()
        c.request("GET", "/health", headers={"Connection": "close"})
        r3 = c.getresponse(); r3.read()
        닫힘 = r3.getheader("Connection", "").lower() == "close"
        c.close()
        판(10, "이어 쓰기 · Connection: close", r1.status == 200 and r2.status == 200 and r3.status == 200 and 닫힘)
        # 11 흘리는 중에 끊기
        so = socket.create_connection(("127.0.0.1", p), timeout=60)
        몸11 = json.dumps({"messages": [{"role": "user", "content": "Write a long story."}], "temperature": 0, "max_tokens": 400,
                          "stream": True}).encode()
        so.sendall(b"POST /v1/chat/completions HTTP/1.1\r\nHost: x\r\nContent-Type: application/json\r\nContent-Length: "
                   + str(len(몸11)).encode() + b"\r\n\r\n" + 몸11)
        받음 = b""
        while 받음.count(b"data: ") < 5:
            받음 += so.recv(4096)
        so.close()
        time.sleep(0.5)
        s, d11 = 요청(p, "POST", "/v1/chat/completions", {"messages": [{"role": "user", "content": "Say hi."}], "temperature": 0, "max_tokens": 8})
        판(11, "흘리는 중에 끊은 연결 뒤", s == 200 and len(d11["choices"][0]["message"]["content"]) > 0)
        몸파일 = os.path.join(a.작업, "curl_몸.json")
        json.dump({"messages": [{"role": "user", "content": "Say hello in Korean."}], "temperature": 0, "max_tokens": 16},
                  open(몸파일, "w", encoding="utf-8"), ensure_ascii=False)
        cr = subprocess.run(["curl", "-s", "-m", "120", f"http://127.0.0.1:{p}/v1/chat/completions", "-H", "Content-Type: application/json",
                             "--data-binary", "@" + 몸파일], capture_output=True)
        try:
            dc = json.loads(cr.stdout.decode("utf-8"))
            맞음12 = dc["object"] == "chat.completion" and len(dc["choices"][0]["message"]["content"]) > 0
        except Exception:
            맞음12 = False
        판(12, "curl", 맞음12)
        c = http.client.HTTPConnection("127.0.0.1", p, timeout=60)
        c.request("OPTIONS", "/v1/chat/completions", headers={"Origin": "http://example.com", "Access-Control-Request-Method": "POST"})
        r = c.getresponse(); r.read()
        판(13, "OPTIONS 의 CORS", r.status == 204 and r.getheader("Access-Control-Allow-Origin") == "*")
        c.close()
    finally:
        sv.끄기()
    결과["통과"] = all(v for k, v in 결과.items() if k != "통과")
    return 결과


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("exe")
    ap.add_argument("gguf")
    ap.add_argument("작업")
    ap.add_argument("--슬롯", type=int, default=4)
    ap.add_argument("--문맥", type=int, default=4096)
    ap.add_argument("--포트", type=int, default=8093)
    ap.add_argument("--일", default="a,c")
    ap.add_argument("--시험7B")
    ap.add_argument("--ptx")
    ap.add_argument("--출력")
    a = ap.parse_args()
    a.exe = os.path.abspath(a.exe)
    a.gguf = os.path.abspath(a.gguf)
    a.작업 = os.path.abspath(a.작업)
    if a.시험7B:
        a.시험7B = os.path.abspath(a.시험7B)
        a.ptx = os.path.abspath(a.ptx)
    os.makedirs(a.작업, exist_ok=True)
    결과 = {"모델": os.path.basename(a.gguf), "슬롯": a.슬롯, "문맥": a.문맥}
    for 일 in a.일.split(","):
        if 일 == "a":
            결과["28a"] = 일28a(a)
        elif 일 == "c":
            결과["28c"] = 일28c(a)
    if a.출력:
        json.dump(결과, open(a.출력, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("모두 통과" if all(v.get("통과") for k, v in 결과.items() if k.startswith("28")) else "통과하지 못한 것이 있다")


if __name__ == "__main__":
    main()
