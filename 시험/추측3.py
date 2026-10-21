"""GeulPUv2 3단계 시험 — 추측 디코딩(프롬프트 찾기)을 켜도 답이 비트까지 같은가. 표준 라이브러리만.

    python -I 시험/추측3.py <서버.exe> <모델.gguf> <작업폴더> [--ptx 모듈] [--포트 8095] [--문맥 4096] [--슬롯 4] [--출력 결과.json]

서버를 --추측시험 으로 띄워(geulpu.draft_tokens 를 받는다) 요청마다 다섯 모드를 견준다 — 기준은 추측 끔(geulpu.speculative false):
  켬(프롬프트 찾기 — 기본), 정답(앞선 끔 실행의 토큰을 초안으로 — 받아들임 끝까지), 반쯤(정답의 세 번째마다 틀리게), 틀림(모두 틀리게).
뽑은 토큰 · 걸음마다의 로짓 해시 · 글 · 끝 까닭 · 로그확률(물었으면) · echo 점수가 모두 같아야 한다.
  (가) 슬롯 1 — 요청 하나씩(채팅 탐욕 · 뽑기, 토큰 프롬프트, 되풀이가 많은 일(베끼기 · 코드 고치기), 로그확률, echo, 최대 생성 5,
       버스트 안의 멈춤 글, 끝 토큰, 문맥 끝(4,096 — 되풀이 토큰열), 조각(64) · 묶음(1,024) 경계를 지나는 생성), 두 차례 대화의 캐시.
  (나) 슬롯 S — 모든 요청의 켬 · 정답 · 틀림을 한꺼번에(행 8 을 슬롯끼리 나눔). (가)의 끔과 같아야 한다.
초안 셈(geulpu.draft_proposed · draft_accepted · draft_steps)을 모드마다 적는다 — 버스트 길이 실제로 돌았는지.
"""
import argparse, http.client, io, json, os, random, subprocess, sys, threading, time

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")


class 서버:
    def __init__(self, exe, gguf, 포트, 슬롯, 문맥, ptx=None, 로그=None, 더=()):
        self.포트 = 포트
        인자 = [os.path.abspath(exe), gguf, "--포트", str(포트), "--슬롯", str(슬롯), "--문맥", str(문맥), "--추측시험", *더]
        if ptx:
            인자 += ["--ptx", ptx]
        self.로그 = open(로그, "w", encoding="utf-8") if 로그 else subprocess.DEVNULL
        self.p = subprocess.Popen(인자, stdout=self.로그, stderr=subprocess.STDOUT)
        t0 = time.time()
        while time.time() - t0 < 600:
            if self.p.poll() is not None:
                raise SystemExit(f"서버가 멈췄다(종료 {self.p.returncode})")
            try:
                s, _ = 요청(포트, "GET", "/health")
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


def 요청(포트, 방법, 길, 몸=None, 시간=900):
    c = http.client.HTTPConnection("127.0.0.1", 포트, timeout=시간)
    본 = None if 몸 is None else json.dumps(몸, ensure_ascii=False).encode("utf-8")
    c.request(방법, 길, body=본, headers={"Content-Type": "application/json"})
    r = c.getresponse()
    d = r.read()
    c.close()
    try:
        return r.status, json.loads(d)
    except ValueError:
        return r.status, d


# ---------------------------------------------------------------- 요청 모음
베낄글 = ("The workshop opened at dawn. Every bench held a different machine: a loom that wove copper thread, a clock that "
         "counted backwards, and a small engine that hummed whenever someone spoke near it. The keeper wrote each repair in a "
         "ledger, line by line, so that the next keeper could follow the same steps without guessing. On the last page she "
         "left a note: keep the oil warm, keep the gears clean, and never wind the backward clock past midnight.")
코드글 = '''def summarize(values):
    total = 0
    count = 0
    smallest = None
    largest = None
    for v in values:
        total += v
        count += 1
        if smallest is None or v < smallest:
            smallest = v
        if largest is None or v > largest:
            largest = v
    mean = total / count if count else 0
    return {"total": total, "count": count, "mean": mean, "min": smallest, "max": largest}
'''


def 모음(어휘):
    r = random.Random(3)
    위 = min(어휘, 120000)
    R1 = [r.randrange(0, 위) for _ in range(37)]
    R2 = [r.randrange(0, 위) for _ in range(990)]            # 생성이 묶음 경계 1,024 를 지난다
    무늬 = [r.randrange(0, 위) for _ in range(97)]
    R3 = (무늬 * 42)[:4000]                                       # 되풀이 — 생성이 문맥 끝(4,096)까지 간다
    def 채팅(글, **kw):
        return {"messages": [{"role": "user", "content": 글}], **kw}
    요들 = [
        ("G1", "/v1/chat/completions", 채팅("Write a short poem about the sea.", temperature=0, max_tokens=128)),
        ("G2", "/v1/chat/completions", 채팅("한국의 수도는 어디인가요? 두 문장으로 설명해 주세요.", temperature=0, max_tokens=128)),
        ("G3", "/v1/chat/completions", 채팅("Write a Python function that checks whether a number is prime.", temperature=0, max_tokens=160)),
        ("S1", "/v1/chat/completions", 채팅("Write a short poem about the sea.", temperature=0.8, top_k=40, top_p=0.95, min_p=0.05, seed=1, max_tokens=128)),
        ("S2", "/v1/chat/completions", 채팅("Tell me a fun fact about octopuses.", temperature=1.2, top_k=0, top_p=1.0, min_p=0.0, seed=3, max_tokens=128)),
        ("C1", "/v1/chat/completions", 채팅("Copy the following text exactly, word for word, and nothing else:\n\n" + 베낄글, temperature=0, max_tokens=220)),
        ("C2", "/v1/chat/completions", 채팅("Add type hints to this Python function. Keep every other line exactly the same and output only the code:\n\n" + 코드글, temperature=0, max_tokens=300)),
        ("C3", "/v1/chat/completions", 채팅("Copy the following text exactly, word for word, and nothing else:\n\n" + 베낄글, temperature=0.7, top_k=40, top_p=0.95, min_p=0.05, seed=11, max_tokens=220)),
        ("L1", "/v1/chat/completions", 채팅("Copy this sentence exactly: " + 베낄글[:160], temperature=0, max_tokens=64, logprobs=True, top_logprobs=3)),
        ("L2", "/v1/completions", {"prompt": "The keeper wrote each repair in a ledger, line by line, so that", "temperature": 0.9, "seed": 7, "max_tokens": 48, "logprobs": 2}),
        ("E1", "/v1/completions", {"prompt": 베낄글, "echo": True, "logprobs": 1, "max_tokens": 16, "temperature": 0}),
        ("M1", "/v1/chat/completions", 채팅("Copy the following text exactly:\n\n" + 베낄글, temperature=0, max_tokens=5)),
        ("P1", "/v1/chat/completions", 채팅("Copy the following text exactly, word for word, and nothing else:\n\n" + 베낄글, temperature=0, max_tokens=220, stop=["small engine"])),
        ("Q1", "/v1/chat/completions", 채팅("Answer with one word only: what color is a clear daytime sky?", temperature=0, max_tokens=64)),
        ("R1", "/v1/completions", {"prompt": R1, "temperature": 0, "max_tokens": 80, "ignore_eos": True}),
        ("R2", "/v1/completions", {"prompt": R2, "temperature": 0, "max_tokens": 80, "ignore_eos": True}),
        ("R3", "/v1/completions", {"prompt": R3, "temperature": 0, "max_tokens": 200, "ignore_eos": True}),
    ]
    for _, _, 몸 in 요들:
        몸["geulpu"] = {"logit_hash": True}
    return 요들, R1, R2


def 요약(d):
    g = d.get("geulpu", {})
    ch = d["choices"][0]
    글 = ch.get("message", {}).get("content") if "message" in ch else ch.get("text")
    return {"tokens": g.get("tokens"), "hashes": g.get("logit_hashes"), "finish": ch["finish_reason"], "text": 글,
            "logprobs": ch.get("logprobs"), "prompt_hashes": g.get("prompt_logit_hashes"), "entropy": g.get("entropy"),
            "usage": d.get("usage"),
            "셈": (g.get("draft_proposed"), g.get("draft_accepted"), g.get("draft_steps"), g.get("speculative"))}


같을것 = ("tokens", "hashes", "finish", "text", "logprobs", "prompt_hashes", "entropy", "usage")


def 모드몸(몸, 모드, 기준토큰, 어휘):
    b = json.loads(json.dumps(몸))
    g = b.setdefault("geulpu", {})
    if 모드 == "끔":
        g["speculative"] = False
    elif 모드 == "켬":
        g["speculative"] = True
    else:
        g["speculative"] = True
        t = list(기준토큰)
        if 모드 == "반쯤":
            t = [(x + 1) % 어휘 if i % 3 == 2 else x for i, x in enumerate(t)]
        elif 모드 == "틀림":
            t = [(x + 1) % 어휘 for x in t]
        g["draft_tokens"] = t
    return b


def 보내기(포트, 이름, 길, 몸):
    s, d = 요청(포트, "POST", 길, 몸)
    if s != 200:
        raise SystemExit(f"{이름}: {s} {d}")
    return 요약(d)


def 견주기(기준, 이것):
    return [k for k in 같을것 if 기준.get(k) != 이것.get(k)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("exe"); ap.add_argument("gguf"); ap.add_argument("작업")
    ap.add_argument("--ptx"); ap.add_argument("--포트", type=int, default=8095)
    ap.add_argument("--문맥", type=int, default=4096); ap.add_argument("--슬롯", type=int, default=4)
    ap.add_argument("--출력")
    ap.add_argument("--far", help="GeulPUv2 ③c — 서버의 앞쪽 그룹 한도 '행 하나,행 여럿'(0,0 이면 다 찬 그룹을 모두 새 커널로)")
    a = ap.parse_args()
    os.makedirs(a.작업, exist_ok=True)
    더 = ("--far", a.far) if a.far else ()
    모드들 = ("켬", "정답", "반쯤", "틀림")
    결과 = {"가": {}, "나": {}, "다름": []}
    다름 = 0
    # (가) 슬롯 1
    sv = 서버(a.exe, a.gguf, a.포트, 1, a.문맥, ptx=a.ptx, 로그=os.path.join(a.작업, "서버_추측_가.log"), 더=더)
    try:
        s, ti = 요청(a.포트, "GET", "/tokenizer_info")
        어휘 = ti["vocab_size"] if s == 200 and isinstance(ti, dict) else 32000
        요들, R1, R2 = 모음(어휘)
        기준들 = {}
        for 이름, 길, 몸 in 요들:
            기준 = 보내기(a.포트, 이름, 길, 모드몸(몸, "끔", None, 어휘))
            기준들[이름] = 기준
            줄 = {"끔": 기준["셈"]}
            for 모드 in 모드들:
                이것 = 보내기(a.포트, 이름, 길, 모드몸(몸, 모드, 기준["tokens"], 어휘))
                d = 견주기(기준, 이것)
                줄[모드] = 이것["셈"]
                if d:
                    다름 += 1
                    결과["다름"].append({"차례": "가", "요청": 이름, "모드": 모드, "다른것": d})
            결과["가"][이름] = 줄
            print(f"(가) {이름}: 토큰 {len(기준['tokens'] or [])} · 끝 {기준['finish']} · 초안(낸 · 받은 · 걸음) "
                  + " · ".join(f"{모드} {줄[모드][0]}/{줄[모드][1]}/{줄[모드][2]}" for 모드 in 모드들), flush=True)
        # 두 차례 대화 — 둘째가 첫째(추측 켬 · 틀린 초안으로 거절이 많았던)의 캐시를 쓸 때와 cache_prompt false 일 때
        T1 = "My name is Mina. What is a good hobby for a rainy day? Answer briefly."
        첫 = {"messages": [{"role": "user", "content": T1}], "temperature": 0, "max_tokens": 64, "geulpu": {"logit_hash": True}}
        첫답 = 보내기(a.포트, "T1", "/v1/chat/completions", 모드몸(첫, "끔", None, 어휘))
        보내기(a.포트, "T1", "/v1/chat/completions", 모드몸(첫, "반쯤", 첫답["tokens"], 어휘))
        def 둘(캐시):
            return {"messages": [{"role": "user", "content": T1}, {"role": "assistant", "content": 첫답["text"]},
                                 {"role": "user", "content": "Can you remind me what my name is?"}],
                    "temperature": 0, "max_tokens": 48, "cache_prompt": 캐시, "geulpu": {"logit_hash": True}}
        둘캐시 = 보내기(a.포트, "T2", "/v1/chat/completions", 모드몸(둘(True), "켬", None, 어휘))
        둘새로 = 보내기(a.포트, "T2", "/v1/chat/completions", 모드몸(둘(False), "끔", None, 어휘))
        d = 견주기(둘새로, 둘캐시)
        print(f"(가) T2 캐시 · 새로: {'같음' if not d else d}", flush=True)
        if d:
            다름 += 1
            결과["다름"].append({"차례": "가", "요청": "T2", "다른것": d})
        # 짧은 꼬리 — 캐시를 쓴 프롬프트의 끝 2 · 3 · 4 토큰은 표 걸음(같은 슬롯의 행 여럿 — 덩이하나 · 메아리덩이)으로 들어간다.
        # 캐시 끔(프롬프트 길)과 같은가 — 생성(탐욕 · 뽑기)과 echo 점수
        for 꼬리 in (2, 3, 4):
            for 이름, 몸 in (("꼬리생성", {"temperature": 0, "max_tokens": 16, "ignore_eos": True}),
                             ("꼬리뽑기", {"temperature": 0.8, "seed": 5, "max_tokens": 16, "ignore_eos": True}),
                             ("꼬리echo", {"echo": True, "logprobs": 2, "max_tokens": 1, "temperature": 0})):
                보내기(a.포트, 이름, "/v1/completions", {"prompt": R1, "max_tokens": 2, "temperature": 0, "geulpu": {"logit_hash": True}})
                p2 = R1 + R2[:꼬리]
                g = {"logit_hash": True}
                if 이름 == "꼬리echo":
                    g["echo_from"] = len(R1)                  # echo 는 echo_from 앞만 캐시를 쓴다 — 점수 행이 꼬리(메아리덩이)
                b1 = dict(몸, prompt=p2, cache_prompt=True, geulpu=dict(g))
                b2 = dict(몸, prompt=p2, cache_prompt=False, geulpu=dict(g))
                캐 = 보내기(a.포트, 이름, "/v1/completions", b1)
                새 = 보내기(a.포트, 이름, "/v1/completions", b2)
                d = 견주기(새, 캐)
                print(f"(가) {이름} 꼬리 {꼬리}: {'같음' if not d else d}", flush=True)
                if d:
                    다름 += 1
                    결과["다름"].append({"차례": "가", "요청": f"{이름}{꼬리}", "다른것": d})
    finally:
        sv.끄기()
    # (나) 슬롯 S — 켬 · 정답 · 틀림을 모두 한꺼번에
    sv = 서버(a.exe, a.gguf, a.포트, a.슬롯, a.문맥, ptx=a.ptx, 로그=os.path.join(a.작업, "서버_추측_나.log"), 더=더)
    try:
        일들 = []
        for 이름, 길, 몸 in 요들:
            for 모드 in ("켬", "정답", "틀림"):
                일들.append((이름, 모드, 길, 모드몸(몸, 모드, 기준들[이름]["tokens"], 어휘)))
        random.Random(33).shuffle(일들)
        받음 = {}
        잠금 = threading.Lock()
        def 일(이름, 모드, 길, 몸):
            r = 보내기(a.포트, 이름, 길, 몸)
            with 잠금:
                받음[(이름, 모드)] = r
        ts = [threading.Thread(target=일, args=x) for x in 일들]
        for t in ts:
            t.start()
        for t in ts:
            t.join()
        for (이름, 모드), r in sorted(받음.items()):
            d = 견주기(기준들[이름], r)
            결과["나"].setdefault(이름, {})[모드] = r["셈"]
            if d:
                다름 += 1
                결과["다름"].append({"차례": "나", "요청": 이름, "모드": 모드, "다른것": d})
        print(f"(나) 슬롯 {a.슬롯} · 요청 {len(일들)} 한꺼번에 — 다른 것 {sum(1 for x in 결과['다름'] if x['차례'] == '나')}", flush=True)
    finally:
        sv.끄기()
    결과["모두같음"] = 다름 == 0
    print(f"끝 — 다른 것 {다름} · 모두 같음 {다름 == 0}")
    for x in 결과["다름"]:
        print("  다름:", x)
    if a.출력:
        with open(a.출력, "w", encoding="utf-8") as f:
            json.dump(결과, f, ensure_ascii=False, indent=1)
    sys.exit(0 if 다름 == 0 else 1)


if __name__ == "__main__":
    main()
