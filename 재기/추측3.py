"""GeulPUv2 3단계 재기 — 생성 속도: 추측 디코딩(프롬프트 찾기) 켬 · 끔, 그리고 llama.cpp 의 llama-server(추측 없음 · ngram-simple ·
--spec-default)와. 표준 라이브러리만.

    python -I 재기/추측3.py <모델.gguf> <결과 폴더> --글서버 <서버.exe> --ptx <모듈> --라마 <llama-b… 폴더> [--바퀴 3] [--포트 8096]
                            [--일 이름,…] [--라마끔]

일 여덟(문서/3단계.md 견줄 것 C): 코드 고치기 둘 · 문서 다시 쓰기 · 요약(되풀이가 많은 묶음 — 프롬프트에 답의 조각이 있다), 채팅 탐욕 ·
채팅 뽑기 · 수학(보통 채팅 묶음), 번역(따로 — 반복 묶음에서 뺀다). 번갈아 세 바퀴: 바퀴마다 글 서버(슬롯 1 · 덩이 512 — 앱과 같다)에
[끔 · 켬](바퀴마다 차례를 바꾼다), llama-server 셋(-ngl 99 -fa on · 슬롯 1)에 같은 요청.
잰 값: 생성 토큰/초 — 글은 (생성 토큰 − 1) / geulpu.generate_us(첫 토큰 뒤), llama 는 timings.predicted_per_second.
받아들임 — 글 draft_accepted / draft_proposed, llama timings.draft_n_accepted / draft_n. 답 — 글은 켬 · 끔이 같아야 하고(다르면 실패),
llama 는 추측 없음과 다른 답의 수를 적는다.
"""
import argparse, http.client, io, json, os, statistics, subprocess, sys, time

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")


def 요청(포트, 방법, 길, 몸=None, 시간=1800):
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


class 띄움:
    def __init__(self, 인자, 포트, 로그):
        self.포트 = 포트
        self.로그 = open(로그, "w", encoding="utf-8")
        self.p = subprocess.Popen(인자, stdout=self.로그, stderr=subprocess.STDOUT)
        t0 = time.time()
        while time.time() - t0 < 900:
            if self.p.poll() is not None:
                raise SystemExit(f"서버가 멈췄다(종료 {self.p.returncode}) — {로그}")
            try:
                s, d = 요청(포트, "GET", "/health", 시간=5)
                if s == 200:
                    return
            except OSError:
                pass
            time.sleep(0.5)
        raise SystemExit("서버가 뜨지 않는다")

    def 끄기(self):
        self.p.kill()
        self.p.wait()
        self.로그.close()


# ---------------------------------------------------------------- 일
코드 = '''import json
from dataclasses import dataclass, field


@dataclass
class Item:
    name: str
    quantity: int
    price: float
    tags: list = field(default_factory=list)


class Inventory:
    def __init__(self):
        self.items = {}
        self.history = []

    def add(self, name, quantity, price, tags=None):
        if name in self.items:
            item = self.items[name]
            item.quantity += quantity
            item.price = price
        else:
            self.items[name] = Item(name, quantity, price, list(tags or []))
        self.history.append(("add", name, quantity))

    def remove(self, name, quantity):
        if name not in self.items:
            raise KeyError(name)
        item = self.items[name]
        if quantity > item.quantity:
            raise ValueError("not enough stock")
        item.quantity -= quantity
        if item.quantity == 0:
            del self.items[name]
        self.history.append(("remove", name, quantity))

    def total_value(self):
        total = 0.0
        for item in self.items.values():
            total += item.quantity * item.price
        return total

    def find_by_tag(self, tag):
        result = []
        for item in self.items.values():
            if tag in item.tags:
                result.append(item)
        return result

    def low_stock(self, threshold):
        return [item.name for item in self.items.values() if item.quantity < threshold]

    def save(self, path):
        data = {name: [item.quantity, item.price, item.tags] for name, item in self.items.items()}
        with open(path, "w") as f:
            json.dump(data, f)

    def load(self, path):
        with open(path) as f:
            data = json.load(f)
        self.items = {}
        for name, (quantity, price, tags) in data.items():
            self.items[name] = Item(name, quantity, price, tags)
'''

오타글 = ("The village libary was founded in 1902 by a group of farmers who beleived that every child should have acess to books. "
         "At first it was only a single room above the bakery, with three shelfs and a long wooden table. Each winter the farmers "
         "would collect money to buy new books, and the baker's wife kept a careful record of every volume that was borrowed. "
         "By 1930 the colection had grown so large that the libary moved into the old school building, which had been empty since "
         "the new school opened on the hill. The move took two weeks, becuase every book was carried by hand. Children formed a long "
         "line from the bakery to the school and passed the books from one person to the next. Today the libary still stands in the "
         "same building. It has computers, a reading garden, and a small museum about the history of the village, but the long "
         "wooden table from the first room is still there, and visitors are welcome to sit at it and read.")

기사 = ("Cities around the world are rethinking how they use their streets. For most of the last century, streets were designed "
        "mainly to move cars as quickly as possible, and space for walking, cycling, and sitting was squeezed into the margins. "
        "In the last two decades, however, a growing number of cities have begun to reallocate that space. Some have closed "
        "central streets to cars on weekends, others have painted protected bicycle lanes, and many have turned parking spaces "
        "into small parks or outdoor seating for cafes. Supporters argue that these changes make cities healthier and more "
        "pleasant: people walk and cycle more, local shops see more customers, and air quality improves. Critics worry about "
        "delivery access, longer car trips, and the cost of construction. Studies from several cities suggest that the effects "
        "depend strongly on how the changes are made. Projects that were planned with local residents and businesses, and that "
        "were tested first with temporary materials, tended to gain support over time. Projects imposed quickly without "
        "consultation often met strong resistance, and some were reversed. Another lesson is that small, connected changes can "
        "matter more than a single large project. A network of safe cycling routes, for example, encourages more people to cycle "
        "than one impressive boulevard that does not connect to anything. Finally, cities have learned to measure results: "
        "counting pedestrians and cyclists, tracking shop revenue, and surveying residents before and after each change. These "
        "measurements help planners adjust designs and explain the benefits to the public.")

번역글 = ("오래된 시계방은 골목의 끝에 있었다. 주인은 아침마다 가게 문을 열고 시계 하나하나의 태엽을 감았다. 손님은 많지 않았지만, "
         "그는 고장 난 시계를 고치는 일을 서두르지 않았다. 작은 부품을 하나씩 꺼내 닦고, 맞지 않는 톱니를 새로 깎아 끼웠다. "
         "그렇게 고친 시계는 다시 몇십 년을 갔다. 사람들은 그 가게를 지나며 시계 소리를 들었고, 그 소리로 시간을 알았다.")


def 일들():
    def 채팅(글, **kw):
        return {"messages": [{"role": "user", "content": 글}], **kw}
    탐 = dict(temperature=0)
    return [
        ("코드1", "반복", 채팅("Add a one-line docstring to every method of the Inventory class below. Do not change anything else. "
                              "Output the complete updated file only.\n\n```python\n" + 코드 + "```", max_tokens=1024, **탐)),
        ("코드2", "반복", 채팅("In the code below, rename the method `total_value` to `inventory_value` and the variable `item` to `entry` "
                              "everywhere. Output the complete updated file only.\n\n```python\n" + 코드 + "```", max_tokens=1024, **탐)),
        ("고쳐쓰기", "반복", 채팅("Correct the spelling mistakes in the following text. Output the full corrected text and nothing else.\n\n"
                                 + 오타글, max_tokens=512, **탐)),
        ("요약", "반복", 채팅("Summarize the following article in five bullet points, quoting the key phrases from the text.\n\n" + 기사,
                             max_tokens=400, **탐)),
        ("채팅탐욕", "보통", 채팅("Explain how a hash table handles collisions, with a short example in Python.", max_tokens=400, **탐)),
        ("채팅뽑기", "보통", 채팅("Write a short story about a lighthouse keeper who finds a message in a bottle.", max_tokens=400,
                                 temperature=0.7, top_k=40, top_p=0.95, min_p=0.05, seed=42)),
        ("수학", "보통", 채팅("A shop sells pencils at 3 for $1.20 and notebooks at $2.75 each. Mina buys 12 pencils and 4 notebooks and "
                             "pays with a $20 bill. How much change does she get? Solve step by step.", max_tokens=400, **탐)),
        ("번역", "번역", 채팅("Translate the following Korean text into natural English.\n\n" + 번역글, max_tokens=300, **탐)),
    ]


def 글재기(포트, 몸, 켬):
    b = json.loads(json.dumps(몸))
    b["geulpu"] = {"speculative": 켬, "logit_hash": True}
    s, d = 요청(포트, "POST", "/v1/chat/completions", b)
    if s != 200:
        raise SystemExit(f"글 서버: {s} {d}")
    g = d["geulpu"]
    n = len(g["tokens"])
    us = g["generate_us"]
    return {"토큰/초": (n - 1) / (us / 1e6) if us > 0 and n > 1 else 0.0, "토큰": n, "글": d["choices"][0]["message"]["content"],
            "해시": g.get("logit_hashes"), "토큰열": g["tokens"], "낸": g.get("draft_proposed"), "받은": g.get("draft_accepted"),
            "걸음": g.get("draft_steps")}


def 라마재기(포트, 몸):
    b = json.loads(json.dumps(몸))
    b["cache_prompt"] = True
    s, d = 요청(포트, "POST", "/v1/chat/completions", b)
    if s != 200:
        raise SystemExit(f"llama-server: {s} {d}")
    t = d.get("timings", {})
    return {"토큰/초": t.get("predicted_per_second", 0.0), "토큰": t.get("predicted_n"), "글": d["choices"][0]["message"]["content"],
            "낸": t.get("draft_n"), "받은": t.get("draft_n_accepted")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("gguf"); ap.add_argument("결과")
    ap.add_argument("--글서버", required=True); ap.add_argument("--ptx", required=True); ap.add_argument("--라마")
    ap.add_argument("--바퀴", type=int, default=3); ap.add_argument("--포트", type=int, default=8096)
    ap.add_argument("--일"); ap.add_argument("--라마끔", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.결과, exist_ok=True)
    목록 = 일들()
    if a.일:
        고른 = set(a.일.split(","))
        목록 = [x for x in 목록 if x[0] in 고른]
    라마들 = [] if (a.라마끔 or not a.라마) else [
        ("llama 추측없음", []),
        ("llama ngram-simple", ["--spec-type", "ngram-simple"]),
        ("llama spec-default", ["--spec-default"]),
    ]
    기록 = {"일": [x[0] for x in 목록], "바퀴": []}
    for 바퀴 in range(a.바퀴):
        한바퀴 = {}
        # 글 서버 — 끔 · 켬 (바퀴마다 차례를 바꾼다)
        sv = 띄움([os.path.abspath(a.글서버), a.gguf, "--포트", str(a.포트), "--슬롯", "1", "--문맥", "8192", "--ptx", a.ptx],
                  a.포트, os.path.join(a.결과, f"글서버_{바퀴}.log"))
        try:
            for 이름, 묶음, 몸 in 목록:
                차례 = (False, True) if 바퀴 % 2 == 0 else (True, False)
                r = {}
                for 켬 in 차례:
                    r["켬" if 켬 else "끔"] = 글재기(a.포트, 몸, 켬)
                같음 = r["켬"]["토큰열"] == r["끔"]["토큰열"] and r["켬"]["해시"] == r["끔"]["해시"]
                한바퀴.setdefault(이름, {})["글 끔"] = r["끔"]
                한바퀴[이름]["글 켬"] = r["켬"]
                한바퀴[이름]["글 같음"] = 같음
                print(f"[{바퀴}] {이름}: 글 끔 {r['끔']['토큰/초']:.1f} · 켬 {r['켬']['토큰/초']:.1f} 토큰/초 "
                      f"(받은 {r['켬']['받은']}/{r['켬']['낸']} · 토큰 {r['켬']['토큰']}) · 같음 {같음}", flush=True)
                if not 같음:
                    print("  !! 추측 켬 · 끔의 답이 다르다", flush=True)
        finally:
            sv.끄기()
        # llama-server 셋
        for 라이름, 더 in 라마들:
            인자 = [os.path.join(a.라마, "llama-server.exe"), "-m", a.gguf, "--port", str(a.포트), "-ngl", "99", "-fa", "on",
                   "-c", "8192", "--parallel", "1", *더]
            sv = 띄움(인자, a.포트, os.path.join(a.결과, f"{라이름.replace(' ', '_')}_{바퀴}.log"))
            try:
                for 이름, 묶음, 몸 in 목록:
                    r = 라마재기(a.포트, 몸)
                    한바퀴[이름][라이름] = r
                    print(f"[{바퀴}] {이름}: {라이름} {r['토큰/초']:.1f} 토큰/초 (받은 {r['받은']}/{r['낸']} · 토큰 {r['토큰']})", flush=True)
            finally:
                sv.끄기()
        기록["바퀴"].append(한바퀴)
        with open(os.path.join(a.결과, "추측3.json"), "w", encoding="utf-8") as f:
            json.dump(기록, f, ensure_ascii=False, indent=1)
    # 요약 — 일마다 바퀴의 가운데 값
    설정들 = ["글 끔", "글 켬"] + [x[0] for x in 라마들]
    print("\n일 · 묶음 | " + " | ".join(설정들) + " | 켬/끔 | 받아들임(글)")
    묶음비 = {}
    for 이름, 묶음, _ in 목록:
        값 = {k: statistics.median(b[이름][k]["토큰/초"] for b in 기록["바퀴"]) for k in 설정들}
        받 = [b[이름]["글 켬"] for b in 기록["바퀴"]]
        비 = 값["글 켬"] / 값["글 끔"] if 값["글 끔"] else 0.0
        묶음비.setdefault(묶음, []).append(비)
        print(f"{이름} · {묶음} | " + " | ".join(f"{값[k]:.1f}" for k in 설정들) + f" | {비:.3f} | {받[0]['받은']}/{받[0]['낸']}")
        if 라마들:
            다른 = sum(1 for b in 기록["바퀴"] for k in 설정들[3:] if b[이름][k]["글"] != b[이름]["llama 추측없음"]["글"])
            print(f"    llama 추측 켬의 답이 추측없음과 다른 경우: {다른}")
    for 묶음, 비들 in 묶음비.items():
        print(f"묶음 {묶음}: 켬/끔 기하평균 {statistics.geometric_mean(비들):.3f}")
    print("글 켬 · 끔 같음:", all(b[n]["글 같음"] for b in 기록["바퀴"] for n in 기록["일"]))


if __name__ == "__main__":
    main()
