"""GeulPUv2 문맥 저장소 2단계 시험 — 서버(문서/문맥저장소.md 의 맞음 2 · 3 · 5): 기록해 둔 에이전트 요청열(Qwen Code 작업 셋 —
재기/MoE 의 세션 기록 jsonl)을 저장소 끔 · 켬(빈 저장소 · 찬 저장소 · 다시 켠 서버 · 슬롯 하나로 대화를 번갈아 · Vulkan)으로 보내
요청마다 토큰과 걸음마다 로짓 해시가 같은지 본다. 계약상 KV 는 어느 길로 채워도(이 슬롯 · 다른 슬롯 · 저장소 · 다시 셈) 같은 바이트라
같아야 한다. 표준 라이브러리만.

    python -I 시험/저장소서버.py 돌리기 <이름> <서버.exe> <모델.gguf> <모듈> <작업폴더> <기록.jsonl …>
          [--저장소 폴더] [--저장소크기 GB] [--저장소쓰기 모두|비울때|끔] [--슬롯 2] [--문맥 32768] [--포트 8099]
          [--최대 48] [--섞기] [--처음 N] [--끔 3,7] [--망가뜨리기 N] [--비우기] [--캐시끔]
    python -I 시험/저장소서버.py 견주기 <결과 a.json> <결과 b.json>

돌리기: 서버를 띄워 요청을 차례로(--섞기면 작업마다 번갈아) 보낸다 — 흘림 끔, max_tokens = --최대, 온도 0.7 · 씨앗 = 1000 + 요청
번호(뽑기를 거침), geulpu.logit_hash. --끔 의 번호는 geulpu.store: false, --캐시끔 이면 모든 요청이 cache_prompt: false(이 슬롯 ·
다른 슬롯 · 저장소 없이 다시 셈 — 기준). --망가뜨리기 N 이면 띄우기 전에 저장소의 덩이 파일 N 개의
내용 바이트 하나를 바꾼다(맞음 5 — 걸러져 다시 셈). 끝나면(--비우기) POST /geulpu/store/flush 뒤 GET /geulpu/store 를 적는다.
결과: <작업폴더>/<이름>.json(요청마다 토큰 · 해시 · 덧붙임 · 시간)과 <이름>.log(서버 기록).
견주기: 같은 요청(번호)의 토큰 · 해시 · 끝 까닭이 모두 같은가 — 하나라도 다르면 종료 코드 1.
"""
import argparse
import http.client
import io
import json
import os
import random
import subprocess
import sys
import time

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")


def 요청(포트, 방법, 길, 몸=None, 시간=3600):
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


class 서버:
    def __init__(self, exe, gguf, 모듈, 포트, 슬롯, 문맥, 덧인자, 로그):
        self.포트 = 포트
        인자 = [os.path.abspath(exe), gguf, "--ptx", 모듈, "--포트", str(포트), "--슬롯", str(슬롯), "--문맥", str(문맥)] + 덧인자
        self.로그 = open(로그, "w", encoding="utf-8")
        self.p = subprocess.Popen(인자, stdout=self.로그, stderr=subprocess.STDOUT)
        t0 = time.time()
        while time.time() - t0 < 900:
            if self.p.poll() is not None:
                raise SystemExit(f"서버가 멈췄다(종료 {self.p.returncode}) — {로그}")
            try:
                s, _ = 요청(포트, "GET", "/health", 시간=5)
                if s == 200:
                    self.올림초 = time.time() - t0
                    return
            except OSError:
                pass
            time.sleep(0.5)
        raise SystemExit("서버가 뜨지 않는다")

    def 끄기(self):
        self.p.kill()
        self.p.wait()
        self.로그.close()


def 요청열(파일들, 섞기, 처음):
    작업 = []
    for f in 파일들:
        줄 = [json.loads(l) for l in open(f, encoding="utf-8") if l.strip()]
        작업.append([(os.path.splitext(os.path.basename(f))[0], i, r["request"]) for i, r in enumerate(줄)])
    if 처음:
        작업 = [w[:처음] for w in 작업]
    if not 섞기:
        return [x for w in 작업 for x in w]
    차례, k = [], 0
    while any(k < len(w) for w in 작업):
        for w in 작업:
            if k < len(w):
                차례.append(w[k])
        k += 1
    return 차례


def 망가뜨리기(폴더, n):
    파일 = []
    for 둘 in sorted(os.listdir(폴더)):
        하위 = os.path.join(폴더, 둘)
        if len(둘) == 2 and os.path.isdir(하위):
            파일 += [os.path.join(하위, f) for f in sorted(os.listdir(하위)) if f.endswith(".kv")]
    고른 = random.Random(7).sample(파일, min(n, len(파일)))
    for p in 고른:
        with open(p, "r+b") as f:
            f.seek(4096 + 12345)
            b = f.read(1)
            f.seek(4096 + 12345)
            f.write(bytes([b[0] ^ 0x5A]))
    return [os.path.basename(p) for p in 고른]


def 돌리기(a):
    os.makedirs(a.작업폴더, exist_ok=True)
    덧 = []
    if a.저장소:
        덧 += ["--저장소", a.저장소, "--저장소크기", a.저장소크기, "--저장소쓰기", a.저장소쓰기]
    망친 = 망가뜨리기(a.저장소, a.망가뜨리기) if a.저장소 and a.망가뜨리기 else []
    끔 = set(int(x) for x in a.끔.split(",") if x) if a.끔 else set()
    목록 = 요청열(a.기록, a.섞기, a.처음)
    로그 = os.path.join(a.작업폴더, a.이름 + ".log")
    s = 서버(a.서버, a.모델, a.모듈, a.포트, a.슬롯, a.문맥, 덧, 로그)
    결과 = {"이름": a.이름, "인자": sys.argv[2:], "올림초": s.올림초, "망친": 망친, "요청": []}
    try:
        for 차례, (작업, i, 원본) in enumerate(목록):
            번호 = f"{작업}:{i}"
            몸 = dict(원본)
            몸.pop("stream_options", None)
            몸["stream"] = False
            몸["max_tokens"] = a.최대
            몸["temperature"] = 0.7
            몸["seed"] = 1000 + hash_번호(번호)
            몸["geulpu"] = {"logit_hash": True}
            if 차례 in 끔:
                몸["geulpu"]["store"] = False
            if a.캐시끔:
                몸["cache_prompt"] = False
            t0 = time.time()
            상태, r = 요청(a.포트, "POST", "/v1/chat/completions", 몸)
            걸림 = time.time() - t0
            if 상태 != 200:
                raise SystemExit(f"{번호}: HTTP {상태} {r}")
            g = r.get("geulpu", {})
            m = r["choices"][0]["message"]
            줄 = {"번호": 번호, "차례": 차례, "끔": 차례 in 끔, "프롬프트": r["usage"]["prompt_tokens"],
                 "토큰": g.get("tokens"), "해시": g.get("logit_hashes"), "끝": r["choices"][0]["finish_reason"],
                 "글": m.get("content"), "호출": m.get("tool_calls"), "슬롯": g.get("slot"),
                 "캐시": g.get("prompt_tokens_cached"), "저장소": g.get("store"), "첫토큰ms": g.get("first_token_ms"),
                 "걸림초": round(걸림, 3)}
            결과["요청"].append(줄)
            st = 줄["저장소"] or {}
            print(f"{차례:3d} {번호:16s} 프롬프트 {줄['프롬프트']:6d} · 캐시 {줄['캐시']:6d} (슬롯 {st.get('slot_cached_tokens', 0):6d} · "
                  f"다른 슬롯 {st.get('slot_copy_tokens', 0):6d} · 저장소 {st.get('restored_tokens', 0):6d} "
                  f"{st.get('restore_ms', 0.0):7.1f} ms) · 첫 토큰 {줄["첫토큰ms"] / 1000:8.1f} ms · {걸림:6.2f} 초 · 슬롯 {줄['슬롯']}",
                  flush=True)
        if a.저장소:
            if a.비우기:
                t0 = time.time()
                _, 결과["비우기"] = 요청(a.포트, "POST", "/geulpu/store/flush", {})
                결과["비우기초"] = round(time.time() - t0, 3)
            _, 결과["저장소상태"] = 요청(a.포트, "GET", "/geulpu/store")
            print("저장소:", json.dumps(결과["저장소상태"], ensure_ascii=False), flush=True)
    finally:
        s.끄기()
        json.dump(결과, open(os.path.join(a.작업폴더, a.이름 + ".json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)


def hash_번호(번호):
    h = 0
    for c in 번호.encode("utf-8"):
        h = (h * 131 + c) % 1000003
    return h


def 견주기(a):
    A = json.load(open(a.a, encoding="utf-8"))
    B = json.load(open(a.b, encoding="utf-8"))
    ra = {x["번호"]: x for x in A["요청"]}
    rb = {x["번호"]: x for x in B["요청"]}
    공통 = [k for k in ra if k in rb]
    다름 = []
    for k in 공통:
        x, y = ra[k], rb[k]
        if x["토큰"] != y["토큰"] or x["해시"] != y["해시"] or x["끝"] != y["끝"]:
            다름.append(k)
    걸음 = sum(len(ra[k]["해시"] or []) for k in 공통)
    print(f"{A['이름']} 대 {B['이름']}: 같은 요청 {len(공통)} 개 · 걸음 {걸음} — 다름 {len(다름)}" + (f" {다름[:10]}" if 다름 else ""))
    sys.exit(1 if 다름 or not 공통 else 0)


def main():
    p = argparse.ArgumentParser()
    sp = p.add_subparsers(dest="명령", required=True)
    d = sp.add_parser("돌리기")
    d.add_argument("이름")
    d.add_argument("서버")
    d.add_argument("모델")
    d.add_argument("모듈")
    d.add_argument("작업폴더")
    d.add_argument("기록", nargs="+")
    d.add_argument("--저장소")
    d.add_argument("--저장소크기", default="20")
    d.add_argument("--저장소쓰기", default="모두")
    d.add_argument("--슬롯", type=int, default=2)
    d.add_argument("--문맥", type=int, default=32768)
    d.add_argument("--포트", type=int, default=8099)
    d.add_argument("--최대", type=int, default=48)
    d.add_argument("--섞기", action="store_true")
    d.add_argument("--처음", type=int, default=0)
    d.add_argument("--끔", default="")
    d.add_argument("--망가뜨리기", type=int, default=0)
    d.add_argument("--비우기", action="store_true")
    d.add_argument("--캐시끔", action="store_true")
    c = sp.add_parser("견주기")
    c.add_argument("a")
    c.add_argument("b")
    a = p.parse_args()
    돌리기(a) if a.명령 == "돌리기" else 견주기(a)


if __name__ == "__main__":
    main()
