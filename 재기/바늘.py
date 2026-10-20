"""긴 문맥의 바늘 찾기(GeulPUv2 ③a — 문서/긴문맥.md 의 맞음 3): 긴 글(wikitext-2 test) 속 깊이 10 · 50 · 90 % 에 숨긴 문장
("The secret password is NNNN.")을 글 끝의 질문으로 묻는다. GeulPU 서버 · 앱의 /v1 과 llama-server 의 /v1 에 같은 글을 보내 맞은 수를
견준다(온도 0 · 최대 16 토큰, 답에 그 번호가 있으면 맞음). 표준 라이브러리만.

    python -I 재기/바늘.py 짓기 <토큰화 주소(GeulPU /tokenize)> <wiki.test.raw> <글묶음.json> [--길이 16000,32000,64000,80000] [--깊이 10,50,90]
    python -I 재기/바늘.py 묻기 <주소(/v1 이 있는 곳)> <글묶음.json> <결과.json> [--이름 GeulPU]

짓기: 길이(채팅 틀을 뺀 글의 토큰 수 — GeulPU 의 /tokenize, 같은 GGUF 의 토크나이저라 llama.cpp 도 같은 토큰)마다 글의 앞부분을 잘라
깊이마다 문단 사이에 바늘을 넣는다. 묻기: 글마다 채팅 요청 하나(messages — 사용자 말 하나) — 답 · 맞음 · 걸린 시간 · 프롬프트 토큰을 적는다.
"""
import argparse
import json
import sys
import time
import urllib.request

질문 = "\n\nWhat is the secret password mentioned in the text above? Answer with the number only."


def 보내기(주소, 길, 몸, 시간=3600):
    r = urllib.request.Request(주소.rstrip("/") + 길, data=json.dumps(몸, ensure_ascii=False).encode("utf-8"), method="POST",
                               headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(r, timeout=시간).read())


def 토큰수(주소, 글):
    return len(보내기(주소, "/tokenize", {"prompt": 글})["tokens"])


def 짓기(a):
    글 = open(a.글, encoding="utf-8").read()
    문단 = [p for p in 글.split("\n") if p.strip()]
    전체 = "\n".join(문단)
    묶음 = []
    for 길이 in [int(x) for x in a.길이.split(",")]:
        lo, hi = 0, len(전체)                       # 토큰이 길이를 넘지 않는 가장 긴 앞부분(문단 끝에서 자름)
        while hi - lo > 200:
            mid = (lo + hi) // 2
            if 토큰수(a.토큰화, 전체[:mid]) <= 길이:
                lo = mid
            else:
                hi = mid
        끝 = 전체.rfind("\n", 0, lo)
        더미 = 전체[:끝]
        for 깊이 in [int(x) for x in a.깊이.split(",")]:
            번호 = 1000 + (길이 // 1000 * 37 + 깊이 * 13) % 9000
            바늘 = f"The secret password is {번호}."
            자리 = 더미.rfind("\n", 0, len(더미) * 깊이 // 100)
            자리 = 0 if 자리 < 0 else 자리
            본문 = 더미[:자리] + ("\n" if 자리 else "") + 바늘 + "\n" + 더미[자리:].lstrip("\n")
            묶음.append({"길이": 길이, "깊이": 깊이, "번호": 번호, "글": 본문 + 질문, "토큰": 토큰수(a.토큰화, 본문 + 질문)})
            print(f"길이 {길이:,} · 깊이 {깊이} % — 토큰 {묶음[-1]['토큰']:,} · 번호 {번호}", flush=True)
    json.dump(묶음, open(a.묶음, "w", encoding="utf-8"), ensure_ascii=False)


def 묻기(a):
    묶음 = json.load(open(a.묶음, encoding="utf-8"))
    결과 = []
    for x in 묶음:
        몸 = {"messages": [{"role": "user", "content": x["글"]}], "max_tokens": 16, "temperature": 0, "stream": False}
        t0 = time.time()
        try:
            r = 보내기(a.주소, "/v1/chat/completions", 몸)
            답 = r["choices"][0]["message"].get("content") or ""
            프롬프트 = (r.get("usage") or {}).get("prompt_tokens")
            틀림 = None
        except Exception as e:          # 문맥 넘침 등
            답, 프롬프트, 틀림 = "", None, str(e)[:200]
        걸림 = time.time() - t0
        맞음 = str(x["번호"]) in 답
        결과.append({"길이": x["길이"], "깊이": x["깊이"], "번호": x["번호"], "답": 답, "맞음": 맞음, "프롬프트": 프롬프트, "초": round(걸림, 2),
                   "틀림": 틀림})
        print(f"{a.이름} 길이 {x['길이']:,} · 깊이 {x['깊이']} % — {'맞음' if 맞음 else '틀림'} · 답 {답.strip()[:40]!r} · 프롬프트 {프롬프트} · "
              f"{걸림:.1f} 초" + (f" · {틀림}" if 틀림 else ""), flush=True)
    json.dump({"이름": a.이름, "주소": a.주소, "결과": 결과}, open(a.결과, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    길이들 = sorted({r["길이"] for r in 결과})
    print(f"{a.이름}: " + " · ".join(f"{L // 1000}k {sum(r['맞음'] for r in 결과 if r['길이'] == L)}/{sum(1 for r in 결과 if r['길이'] == L)}"
                                    for L in 길이들), flush=True)


def main():
    p = argparse.ArgumentParser()
    sp = p.add_subparsers(dest="명령", required=True)
    j = sp.add_parser("짓기")
    j.add_argument("토큰화")
    j.add_argument("글")
    j.add_argument("묶음")
    j.add_argument("--길이", default="16000,32000,64000,80000")
    j.add_argument("--깊이", default="10,50,90")
    m = sp.add_parser("묻기")
    m.add_argument("주소")
    m.add_argument("묶음")
    m.add_argument("결과")
    m.add_argument("--이름", default="GeulPU")
    a = p.parse_args()
    짓기(a) if a.명령 == "짓기" else 묻기(a)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
