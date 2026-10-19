"""GeulPUv2 4b단계 시험 — 서버의 긴 문맥(생성 길 접기의 예전 한도 10,240 토큰 너머): 생성 걸음(표 걸음)이 프롬프트 길과 같은 로짓을
내는가, 추측을 켜도(행 여럿) 같은가, 백엔드를 건너(CUDA · Vulkan) 같은가. 표준 라이브러리만.

    python -I 시험/긴문맥서버.py 돌리기 <서버.exe> <모델.gguf> <재기 입력(긴 글)> <작업폴더> [--ptx 모듈] [--포트 8099]
                                     [--문맥 32768] [--길이 10300,20000,32000] [--생성 48] [--출력 결과.json]
    python -I 시험/긴문맥서버.py 견주기 <결과 a.json> <결과 b.json>

문맥 n 마다(토큰 프롬프트 — 재기 입력의 긴 글의 앞 n 토큰):
  (가) 탐욕 생성 G 토큰, 추측 끔(걸음마다 표 걸음 행 1) — 토큰 · 걸음마다 로짓 해시.
  (나) 같은 요청을 추측 켬 · 정답 초안(geulpu.draft_tokens — 행 8 까지의 검증 걸음)으로 — (가)와 토큰 · 해시가 같아야 한다.
  (다) echo: 프롬프트 = n 토큰 + (가)의 생성 토큰 앞 G − 1 개, max_tokens 1 — 프롬프트 행들의 로짓 해시(프롬프트 길로 셈) 가운데 행
       n − 1 … n + G − 2 가 (가)의 걸음 0 … G − 1 의 해시와 같아야 한다(생성 길 = 프롬프트 길 — 계약).
서버는 --추측시험 · 슬롯 1 로 띄운다. 견주기: 두 결과(예: CUDA 정밀 · Vulkan 정밀)의 토큰 · 해시가 n 마다 모두 같은가.
"""
import argparse
import http.client
import io
import json
import os
import struct
import subprocess
import sys
import time

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


class 서버:
    def __init__(self, exe, gguf, 포트, 문맥, ptx=None, 로그=None):
        self.포트 = 포트
        인자 = [os.path.abspath(exe), gguf, "--포트", str(포트), "--슬롯", "1", "--문맥", str(문맥), "--추측시험"]
        if ptx:
            인자 += ["--ptx", ptx]
        self.로그 = open(로그, "w", encoding="utf-8") if 로그 else subprocess.DEVNULL
        self.p = subprocess.Popen(인자, stdout=self.로그, stderr=subprocess.STDOUT)
        t0 = time.time()
        while time.time() - t0 < 900:
            if self.p.poll() is not None:
                raise SystemExit(f"서버가 멈췄다(종료 {self.p.returncode})")
            try:
                s, _ = 요청(포트, "GET", "/health", 시간=5)
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


def 긴글(경로):
    """재기 입력(재기_7B 꼴): [pc][프롬프트 pc 토큰][pl][긴 글 pl 토큰] — 모두 int64."""
    b = open(경로, "rb").read()
    pc = struct.unpack_from("<q", b, 0)[0]
    o = 8 + pc * 8
    pl = struct.unpack_from("<q", b, o)[0]
    return list(struct.unpack_from(f"<{pl}q", b, o + 8))


def 보내기(포트, 몸):
    s, d = 요청(포트, "POST", "/v1/completions", 몸)
    if s != 200:
        raise SystemExit(f"요청 실패: {s} {str(d)[:300]}")
    g = d.get("geulpu", {})
    return {"tokens": g.get("tokens"), "hashes": g.get("logit_hashes"), "prompt_hashes": g.get("prompt_logit_hashes"),
            "finish": d["choices"][0]["finish_reason"],
            "셈": (g.get("draft_proposed"), g.get("draft_accepted"), g.get("draft_steps"))}


def 돌리기(a):
    os.makedirs(a.작업, exist_ok=True)
    긴 = 긴글(a.입력)
    길이들 = [int(x) for x in a.길이.split(",") if x]
    G = a.생성
    결과 = {"서버": os.path.basename(a.exe), "ptx": a.ptx, "문맥": a.문맥, "생성": G, "경우": {}, "다름": 0}
    sv = 서버(a.exe, a.gguf, a.포트, a.문맥, ptx=a.ptx, 로그=os.path.join(a.작업, "서버_긴문맥.log"))
    try:
        for n in 길이들:
            assert n + G <= min(len(긴), a.문맥), (n, G, len(긴), a.문맥)
            p = 긴[:n]
            t0 = time.time()
            가 = 보내기(a.포트, {"prompt": p, "max_tokens": G, "temperature": 0, "ignore_eos": True,
                              "geulpu": {"logit_hash": True, "speculative": False}})
            t1 = time.time()
            나 = 보내기(a.포트, {"prompt": p, "max_tokens": G, "temperature": 0, "ignore_eos": True,
                              "geulpu": {"logit_hash": True, "speculative": True, "draft_tokens": 가["tokens"]}})
            다 = 보내기(a.포트, {"prompt": p + 가["tokens"][:G - 1], "max_tokens": 1, "temperature": 0, "echo": True, "logprobs": 1,
                              "geulpu": {"logit_hash": True, "speculative": False}})
            # echo 의 해시는 다음 프롬프트 토큰이 있는 행 0 … pn − 2 의 것(pn = n + G − 1) — 걸음 i(위치 n − 1 + i 의 행)는 i < G − 1 이면
            # 그 행, 걸음 G − 1 은 echo 요청의 생성 걸음 0(프롬프트의 끝 행 — 프롬프트 길)
            ph = 다["prompt_hashes"] or []
            줄 = {"(가) 토큰 수": len(가["tokens"] or []), "(가) 초": round(t1 - t0, 1),
                  "(나) 다름": [k for k in ("tokens", "hashes", "finish") if 가[k] != 나[k]], "(나) 초안": 나["셈"],
                  "(다) 행 수": len(ph)}
            if len(ph) == n + G - 2 and 다["hashes"]:
                프롬해시 = ph[n - 1:n + G - 2] + 다["hashes"][:1]
                줄["(다) 다른 걸음"] = [i for i in range(G) if 프롬해시[i] != 가["hashes"][i]]
            else:
                줄["(다) 다른 걸음"] = [f"echo 의 행 수 {len(ph)} (n + G − 2 = {n + G - 2} 이어야)"]
            줄["토큰"] = 가["tokens"]
            줄["해시"] = 가["hashes"]
            다름 = len(줄["(나) 다름"]) + len(줄["(다) 다른 걸음"])
            결과["다름"] += 다름
            결과["경우"][str(n)] = 줄
            print(f"n {n}: (가) 토큰 {줄['(가) 토큰 수']} · {줄['(가) 초']} 초 · (나) 다름 {줄['(나) 다름']} 초안 {줄['(나) 초안']} · "
                  f"(다) 다른 걸음 {줄['(다) 다른 걸음'][:8]} — {'같음' if 다름 == 0 else '다름'}", flush=True)
    finally:
        sv.끄기()
    결과["통과"] = 결과["다름"] == 0
    print(f"끝 — 다름 {결과['다름']} · 통과 {결과['통과']}", flush=True)
    if a.출력:
        json.dump(결과, open(a.출력, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    return 0 if 결과["통과"] else 1


def 견주기(a):
    x = json.load(open(a.a, encoding="utf-8"))
    y = json.load(open(a.b, encoding="utf-8"))
    다름 = 0
    for n in sorted(set(x["경우"]) | set(y["경우"]), key=int):
        p, q = x["경우"].get(n), y["경우"].get(n)
        if p is None or q is None:
            print(f"n {n}: 한쪽에만 있다")
            다름 += 1
            continue
        d = [k for k in ("토큰", "해시") if p[k] != q[k]]
        다름 += len(d)
        print(f"n {n}: {'같음' if not d else '다름 ' + str(d)}")
    print(f"끝 — 다름 {다름}")
    return 0 if 다름 == 0 else 1


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="일", required=True)
    p = sub.add_parser("돌리기")
    p.add_argument("exe"); p.add_argument("gguf"); p.add_argument("입력"); p.add_argument("작업")
    p.add_argument("--ptx"); p.add_argument("--포트", type=int, default=8099); p.add_argument("--문맥", type=int, default=32768)
    p.add_argument("--길이", default="10300,20000,32000"); p.add_argument("--생성", type=int, default=48); p.add_argument("--출력")
    q = sub.add_parser("견주기")
    q.add_argument("a"); q.add_argument("b")
    a = ap.parse_args()
    return 돌리기(a) if a.일 == "돌리기" else 견주기(a)


if __name__ == "__main__":
    sys.exit(main())
