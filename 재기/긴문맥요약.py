"""GeulPUv2 — 긴 문맥 재기(재기/긴문맥.sh)의 요약: 글 엔진(재기_긴문맥 의 글_<이름>.txt)과 llama.cpp(llama-bench JSON)를 같은 표로.

  python -X utf8 재기/긴문맥요약.py <결과 폴더>

llama.cpp 는 (n_prompt, n_gen, n_depth) 마다 ub 셋 가운데 samples_ns 의 가운데 값이 가장 작은 것(프롬프트 — ub 는 llama.cpp 가 고를 수 있는
설정). 생성은 ub 와 상관없으므로 ub 셋의 가운데 값들의 가운데(가장 작은 것을 고르면 같은 셈 셋 중 운 좋은 것을 고르게 된다) — tg128 의 토큰/초
(128 걸음 평균).
"""
import glob
import json
import os
import re
import statistics
import sys


def 글값(경로):
    빈, 새, 생성 = {}, {}, {}
    for 줄 in open(경로, encoding="utf-8"):
        m = re.match(r"빈 (\d+) ms ([\d.]+)", 줄)
        if m:
            빈[int(m.group(1))] = float(m.group(2))
        m = re.match(r"깊이 (\d+) 새 1024 ms ([\d.]+)", 줄)
        if m:
            새[int(m.group(1))] = float(m.group(2))
        m = re.match(r"깊이 (\d+) 생성 (\d+) ms ([\d.]+)", 줄)
        if m:
            생성[(int(m.group(1)), int(m.group(2)))] = float(m.group(3))
    return 빈, 새, 생성


def 라마값(경로):
    """(n_prompt, n_gen, n_depth) → (ms, 설정)"""
    out = {}
    if not os.path.exists(경로):
        return out
    txt = open(경로, encoding="utf-8").read().strip()
    if not txt:
        return out
    rows = json.loads(txt) if txt.startswith("[") else [json.loads(l) for l in txt.splitlines() if l.strip()]
    모음 = {}
    for r in rows:
        k = (int(r["n_prompt"]), int(r["n_gen"]), int(r.get("n_depth", 0)))
        모음.setdefault(k, []).append((statistics.median(r["samples_ns"]) / 1e6, f"ub{r['n_ubatch']}"))
    for k, 값들 in 모음.items():
        if k[1] > 0:
            out[k] = (statistics.median(v for v, _ in 값들), "ub 셋의 가운데")
        else:
            out[k] = min(값들)
    return out


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    폴더 = sys.argv[1]
    for 경로 in sorted(glob.glob(os.path.join(폴더, "글_*.txt"))):
        이름 = os.path.basename(경로)[2:-4]
        빈, 새, 생성 = 글값(경로)
        라 = 라마값(os.path.join(폴더, f"라마빈_{이름}.json"))
        라.update(라마값(os.path.join(폴더, f"라마깊이_{이름}.json")))
        print(f"\n{이름}")
        print("  (가) 빈 캐시 프롬프트 N — ms (토큰/초), 글/라마")
        for N in sorted(빈):
            g = 빈[N]
            l = 라.get((N, 0, 0))
            비 = f"{g / l[0]:.3f}" if l else "-"
            print(f"    {N:6d}  글 {g:9.1f} ({N / g * 1000:7.0f})  라마 {l[0] if l else 0:9.1f} ({(N / l[0] * 1000) if l else 0:7.0f}, {l[1] if l else '-'})  비 {비}")
        print("  (나) 깊이 d 에서 새 입력 1,024 — ms, 글/라마")
        for d in sorted(새):
            g = 새[d]
            l = 라.get((1024, 0, d))
            비 = f"{g / l[0]:.3f}" if l else "-"
            print(f"    {d:6d}  글 {g:8.1f}  라마 {l[0] if l else 0:8.1f} ({l[1] if l else '-'})  비 {비}")
        print("  (다) 깊이 d 에서 생성 — 토큰/초(글은 행 1 걸음, 라마는 tg128), 글/라마의 시간 비 · 글의 행 8 걸음 ms")
        for d in sorted({k[0] for k in 생성}):
            g1 = 생성.get((d, 1))
            g8 = 생성.get((d, 8))
            l = 라.get((0, 128, d))
            라토 = 128 / (l[0] / 1000) if l else 0
            글토 = 1000 / g1 if g1 else 0
            비 = f"{라토 / 글토:.3f}" if (l and g1) else "-"
            print(f"    {d:6d}  글 {글토:6.1f} 토큰/초 ({g1:.2f} ms · 행 8 {g8:.2f} ms)  라마 {라토:6.1f} 토큰/초  시간 비 {비}")


if __name__ == "__main__":
    main()
