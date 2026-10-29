"""커널 프로파일(10-29) — 모델의 앱 기본 모듈(앱/일.gl 의 모듈 준비와 같은 깃발)을 짓고 PTX 경로를 표준 출력의 마지막 줄로 낸다.

  python3 -I 판시간_모듈.py <bin 폴더> <gguf> <모듈 폴더>

정밀 전용 구조가 아니면 --빠른(정밀이면 앞쪽 그룹을 받을 때 --먼), k-양자화이고 8비트 KV 를 받으면 --KV8, 그 위에 F2B 를 받으면 --F2B,
빠른 어텐션을 받으면 --F2. 이름 끝은 엔진이 읽는 꼴(_빠른 · _F2 · _F2B · _KV8). 있으면 다시 짓지 않는다.
"""
import json
import os
import subprocess
import sys


def main():
    B, gguf, D = sys.argv[1:4]
    r = subprocess.run([f"{B}/server-cuda", "--모델확인", gguf], capture_output=True, text=True)
    줄 = [l for l in r.stdout.splitlines() if l.startswith("{")]
    if not 줄:
        print(f"모델 확인 실패: {r.stdout[-400:]} {r.stderr[-400:]}", file=sys.stderr)
        return 1
    j = json.loads(줄[-1])
    빠른 = not j.get("exact_only", False)
    K = j.get("quant") == "K"
    KV8 = 빠른 and K and j.get("kv8_ok", False)
    F2B = KV8 and j.get("f2b", False)
    F2 = KV8 and j.get("fast_attention_ok", False) and not F2B
    깃발 = []
    if 빠른:
        깃발.append("--빠른")
    elif j.get("far_ok", False):
        깃발.append("--먼")
    if KV8:
        깃발.append("--KV8")
    if F2B:
        깃발.append("--F2B")
    elif F2:
        깃발.append("--F2")
    이름 = j["module"] + ("_빠른" if 빠른 else "") + ("_F2B" if F2B else "_F2" if F2 else "") + ("_KV8" if KV8 else "")
    os.makedirs(D, exist_ok=True)
    gl = os.path.join(D, 이름 + ".gl")
    ptx = os.path.join(D, 이름 + ".ptx")
    print(f"모델 확인: {줄[-1][:300]}", file=sys.stderr)
    print(f"깃발 {' '.join(깃발) or '(없음)'}", file=sys.stderr)
    if not os.path.isfile(ptx):
        g = subprocess.run([f"{B}/kernel-gen", *j["gen"], gl, *j.get("gen_opts", []), *깃발], capture_output=True, text=True)
        if g.returncode != 0:
            print(f"생성기 실패: {g.stdout[-400:]} {g.stderr[-400:]}", file=sys.stderr)
            return 1
        p = subprocess.run([f"{B}/ptx-backend", gl, "-o", ptx], capture_output=True, text=True)
        if p.returncode != 0 or not os.path.isfile(ptx):
            print(f"PTX 실패: {p.stdout[-400:]} {p.stderr[-400:]}", file=sys.stderr)
            return 1
    print(ptx)
    return 0


if __name__ == "__main__":
    sys.exit(main())
