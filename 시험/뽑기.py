"""28단계 — 뽑기 계약의 파이썬판(문서/계약.md "뽑기")과 시험 28b (1) · (2).

    python -I 시험/뽑기.py 대조 <시험_뽑기.exe> [--실제 로짓.bin ...] [--출력 결과.json]
    python -I 시험/뽑기.py 분포 <시험_뽑기.exe> [--횟수 1000000] [--출력 결과.json]

대조: 합성 로짓 행(정규 σ 3 · 뾰족한 것 · 같은 값이 많은 것)과 (주면) 실제 로짓 행에서, 매개변수 열두 벌로 글 쪽(시험_뽑기)과 이 파일이
같은 토큰을 내는지. 분포: 토큰 열둘의 고정 로짓에서 세 벌(T 0.7 · 1.0 · 1.5)마다 걸음 0 … N − 1 을 뽑은 빈도가 계약의 확률 w_k / c_n 과
맞는지(카이제곱 — 기준 p ≥ 0.001). 표준 라이브러리만 쓴다(카이제곱의 p 는 정규 근사가 아닌 불완전 감마 함수로).
"""
import argparse, heapq, json, math, os, random, struct, subprocess, sys, tempfile, io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
M64 = (1 << 64) - 1
INV_LN2 = struct.unpack("<d", (4609176140021203710).to_bytes(8, "little", signed=True))[0]
LN2_HI = struct.unpack("<d", (4604418534311723008).to_bytes(8, "little", signed=True))[0]
LN2_LO = struct.unpack("<d", (4461442080421002358).to_bytes(8, "little", signed=True))[0]
역계승 = []
_f = 1
for _n in range(14):
    if _n > 0:
        _f *= _n
    역계승.append(1.0 / float(_f))


def 분할섞기64(s, i):
    x = (s + (i + 1) * 0x9E3779B97F4A7C15) & M64
    x = ((x ^ (x >> 30)) * 0xBF58476D1CE4E5B9) & M64
    x = ((x ^ (x >> 27)) * 0x94D049BB133111EB) & M64
    return x ^ (x >> 31)


def 적힌지수(a):
    y = a * INV_LN2 + 0.5
    k = math.floor(y)
    kf = float(k)
    r = (a - kf * LN2_HI) - kf * LN2_LO
    p = 역계승[13]
    for n in range(12, -1, -1):
        p = p * r + 역계승[n]
    return p * math.ldexp(1.0, k)


def 뽑기(z, T, K, P, Q, s, i):
    """z: 파이썬 float 의 목록(짧은실수 값 그대로). 고른 번호와 n. 계약의 차례 그대로 — 늘어놓기는 후보(바닥 안)만 한다."""
    V = len(z)
    큰 = max(range(V), key=z.__getitem__)          # 같은 값이면 앞의 것(가장 작은 번호)
    if T == 0.0 or K == 1:
        return 큰, 1
    zm = z[큰]
    후보 = [j for j in range(V) if (z[j] - zm) / T >= -40.0]      # 늘어놓은 차례의 앞부분(a 는 차례대로 줄어든다)
    열쇠 = lambda j: (-(z[j] + 0.0), j)                          # −0 은 +0 과 같은 값 — 번호로 가른다
    if K > 0 and K < len(후보):
        순서 = heapq.nsmallest(K, 후보, key=열쇠)
    else:
        순서 = sorted(후보, key=열쇠)
    nK = len(순서)
    c = []
    합 = 0.0
    nQ = 0
    for k in range(nK):
        a = (z[순서[k]] - zm) / T
        w = 적힌지수(a)
        합 = 합 + w
        c.append(합)
        if w >= Q:
            nQ += 1
    if Q == 0.0:
        nQ = nK
    nP = nK
    if P < 1.0:
        목표 = P * c[nK - 1]
        nP = 0
        while c[nP] < 목표:
            nP += 1
        nP += 1
    n = max(1, min(nP, nQ))
    u = float(분할섞기64(s, i) >> 11) * 2.0 ** -53
    t = u * c[n - 1]
    고름 = n - 1
    for k in range(n):
        if c[k] > t:
            고름 = k
            break
    return 순서[고름], n


def f32(x):
    return struct.unpack("<f", struct.pack("<f", x))[0]


매개벌 = [  # (T, K, P, Q)
    (0.0, 40, 0.95, 0.05), (0.3, 40, 0.95, 0.05), (0.8, 40, 0.95, 0.05), (1.0, 40, 0.95, 0.05),
    (1.7, 40, 0.95, 0.05), (0.8, 0, 1.0, 0.0), (0.8, 1, 1.0, 0.0), (1.0, 0, 0.95, 0.0),
    (1.0, 0, 0.5, 0.0), (1.0, 40, 1.0, 0.05), (0.8, 0, 1.0, 0.05), (1.7, 0, 0.95, 0.0),
]


def 합성행들(V, n, 씨앗=28):
    r = random.Random(씨앗)
    행들 = []
    for k in range(n):
        종류 = k % 3
        if 종류 == 0:
            행 = [f32(r.gauss(0.0, 3.0)) for _ in range(V)]
        elif 종류 == 1:
            행 = [f32(r.gauss(-5.0, 2.0)) for _ in range(V)]
            for _ in range(5):
                행[r.randrange(V)] = f32(r.uniform(5.0, 15.0))
        else:
            행 = [f32(float(r.randrange(-8, 3))) for _ in range(V)]     # 같은 값이 많다(± 0 포함)
            행[r.randrange(V)] = -0.0
        행들.append(행)
    return 행들


def 글돌리기(exe, V, 행들, 벌들, 씨앗들, 걸음들):
    """시험_뽑기 대조 — 입력 파일: [V][행 수][벌 수] 행들(짧은실수) 벌들(T, K, P, Q, 씨앗, 걸음)."""
    d = tempfile.mkdtemp(prefix="geulpu_bp_")
    입 = os.path.join(d, "입력.bin")
    출 = os.path.join(d, "출력.bin")
    with open(입, "wb") as f:
        f.write(struct.pack("<qqq", V, len(행들), len(벌들)))
        for 행 in 행들:
            f.write(struct.pack(f"<{V}f", *행))
        for (T, K, P, Q), s, i in zip(벌들, 씨앗들, 걸음들):
            f.write(struct.pack("<dqddQq", T, K, P, Q, s, i))
    subprocess.run([exe, "대조", 입, 출], check=True)
    data = open(출, "rb").read()
    return list(struct.unpack(f"<{len(행들) * len(벌들)}q", data))


def 대조(a):
    V = 4096
    행들 = 합성행들(V, a.합성)
    실제수 = 0
    for 경로 in a.실제 or []:                      # 실제 로짓 행 — 파일 차례로, 모두 합해 실제행 개까지
        raw = open(경로, "rb").read()
        Vr = a.어휘
        n = len(raw) // (4 * Vr)
        for k in range(n):
            if 실제수 >= a.실제행:
                break
            행들.append(list(struct.unpack_from(f"<{Vr}f", raw, k * 4 * Vr)))
            실제수 += 1
    결과 = {"행": len(행들), "벌": len(매개벌), "다름": 0, "경우": 0}
    # 어휘가 다른 행은 따로 돌린다
    무리 = {}
    for 행 in 행들:
        무리.setdefault(len(행), []).append(행)
    for Vg, 행g in 무리.items():
        r = random.Random(Vg)
        씨앗들 = [r.getrandbits(64) for _ in 매개벌]
        걸음들 = [r.randrange(0, 5000) for _ in 매개벌]
        글 = 글돌리기(a.exe, Vg, 행g, 매개벌, 씨앗들, 걸음들)
        for ri, 행 in enumerate(행g):
            for bi, (T, K, P, Q) in enumerate(매개벌):
                py, _ = 뽑기(행, T, K, P, Q, 씨앗들[bi], 걸음들[bi] + ri)
                결과["경우"] += 1
                if py != 글[ri * len(매개벌) + bi]:
                    결과["다름"] += 1
                    if 결과["다름"] <= 5:
                        print(f"  다름: 어휘 {Vg} 행 {ri} 벌 {bi}: 파이썬 {py} 글 {글[ri * len(매개벌) + bi]}")
    print(f"28b (1) 대조: 경우 {결과['경우']} · 다름 {결과['다름']}")
    return 결과


def 감마Q(s, x):
    """위 정칙 불완전 감마 Q(s, x) — 급수 / 연분수(Numerical Recipes 의 꼴)."""
    if x <= 0:
        return 1.0
    gln = math.lgamma(s)
    if x < s + 1:
        ap, 합, d = s, 1.0 / s, 1.0 / s
        for _ in range(10000):
            ap += 1
            d *= x / ap
            합 += d
            if abs(d) < abs(합) * 1e-15:
                break
        return 1.0 - 합 * math.exp(-x + s * math.log(x) - gln)
    b, c, d = x + 1 - s, 1e300, 1.0 / (x + 1 - s)
    h = d
    for i in range(1, 10000):
        an = -i * (i - s)
        b += 2
        d = an * d + b
        d = 1e-300 if abs(d) < 1e-300 else d
        c = b + an / c
        c = 1e-300 if abs(c) < 1e-300 else c
        d = 1.0 / d
        de = d * c
        h *= de
        if abs(de - 1.0) < 1e-15:
            break
    return math.exp(-x + s * math.log(x) - gln) * h


def 분포(a):
    z = [f32(x) for x in (2.0, 1.5, 1.4, 0.9, 0.3, 0.0, -0.2, -0.7, -1.1, -2.0, -3.5, -6.0)]
    V = len(z)
    결과 = {"횟수": a.횟수, "벌": []}
    for T in (0.7, 1.0, 1.5):
        d = tempfile.mkdtemp(prefix="geulpu_bp_")
        입 = os.path.join(d, "입력.bin")
        출 = os.path.join(d, "출력.bin")
        with open(입, "wb") as f:
            f.write(struct.pack("<qqq", V, 1, 1))
            f.write(struct.pack(f"<{V}f", *z))
            f.write(struct.pack("<dqddQq", T, 0, 1.0, 0.0, 20261012, 0))
        subprocess.run([a.exe, "분포", 입, 출, str(a.횟수)], check=True)
        수 = list(struct.unpack(f"<{V}q", open(출, "rb").read()))
        # 계약의 확률 — 같은 셈(늘어놓은 차례의 w, c_n)
        순서 = sorted(range(V), key=lambda j: (-z[j], j))
        zm = z[순서[0]]
        w = [적힌지수((z[j] - zm) / T) for j in 순서]
        합 = 0.0
        for x in w:
            합 = 합 + x
        기대 = [0.0] * V
        for k, j in enumerate(순서):
            기대[j] = w[k] / 합 * a.횟수
        카이 = sum((수[j] - 기대[j]) ** 2 / 기대[j] for j in range(V))
        p = 감마Q((V - 1) / 2.0, 카이 / 2.0)
        print(f"28b (2) 분포 T {T}: 카이제곱 {카이:.2f} (자유도 {V - 1}) · p {p:.4f} · 빈도 {수}")
        결과["벌"].append({"T": T, "빈도": 수, "기대": 기대, "카이제곱": 카이, "p": p})
    return 결과


def 기록(a):
    """28b (3) — 서버가 (가)에서 기록한 뽑기 요청(로짓 · 뽑은 토큰)을 이 계약판이 다시 낸다"""
    결과 = {"요청": {}, "걸음": 0, "다름": 0}
    for 경로 in a.결과:
        r = json.load(open(경로, encoding="utf-8"))
        폴더 = r["28a"].get("기록폴더") or os.path.join(os.path.dirname(os.path.abspath(경로)), "기록28")
        for 이름, x in r["28a"]["뽑기요청"].items():
            몸 = x["body"]
            T = float(몸.get("temperature", 0.8))
            K = int(몸.get("top_k", 40))
            P = float(몸.get("top_p", 0.95))
            Q = float(몸.get("min_p", 0.05))
            s = int(x["seed"])
            raw = open(os.path.join(폴더, f"rec_{이름}_로짓.bin"), "rb").read()
            토큰 = list(struct.unpack_from(f"<{len(raw) // (4 * a.어휘)}q", open(os.path.join(폴더, f"rec_{이름}_토큰.bin"), "rb").read()))
            n = len(raw) // (4 * a.어휘)
            다름 = 0
            for i in range(n):
                z = list(struct.unpack_from(f"<{a.어휘}f", raw, i * 4 * a.어휘))
                t, _ = 뽑기(z, T, K, P, Q, s, i)
                if t != 토큰[i]:
                    다름 += 1
            결과["요청"][f"{r['모델']}:{이름}"] = {"걸음": n, "다름": 다름}
            결과["걸음"] += n
            결과["다름"] += 다름
            print(f"28b (3) {r['모델']} {이름}: 걸음 {n} · 다름 {다름}")
    print(f"28b (3) 기록: 걸음 {결과['걸음']} · 다름 {결과['다름']}")
    return 결과


def 시간(a):
    """기록 — 실제 로짓 행(앞에서 100 개)으로 글 뽑기 모듈의 한 걸음 시간(µs)"""
    Vr = a.어휘
    행들 = []
    for 경로 in a.실제 or []:
        raw = open(경로, "rb").read()
        for k in range(len(raw) // (4 * Vr)):
            if len(행들) < 100:
                행들.append(raw[k * 4 * Vr:(k + 1) * 4 * Vr])
    벌들 = [(0.8, 40, 0.95, 0.05), (1.0, 0, 1.0, 0.0), (0.8, 0, 0.95, 0.0), (0.0, 40, 0.95, 0.05)]
    d = tempfile.mkdtemp(prefix="geulpu_bp_")
    입 = os.path.join(d, "입력.bin")
    출 = os.path.join(d, "출력.txt")
    with open(입, "wb") as f:
        f.write(struct.pack("<qqq", Vr, len(행들), len(벌들)))
        for 행 in 행들:
            f.write(행)
        for T, K, P, Q in 벌들:
            f.write(struct.pack("<dqddQq", T, K, P, Q, 7, 0))
    subprocess.run([a.exe, "시간", 입, 출], check=True)
    글 = open(출, encoding="utf-8").read()
    print(글, end="")
    return {"행": len(행들), "글": 글}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("일", choices=["대조", "분포", "기록", "시간"])
    ap.add_argument("exe", nargs="?")
    ap.add_argument("--결과", nargs="*")
    ap.add_argument("--합성", type=int, default=1000)
    ap.add_argument("--실제", nargs="*")
    ap.add_argument("--실제행", type=int, default=1000)
    ap.add_argument("--어휘", type=int, default=152064)
    ap.add_argument("--횟수", type=int, default=1000000)
    ap.add_argument("--출력")
    a = ap.parse_args()
    if a.exe:
        a.exe = os.path.abspath(a.exe)
    r = {"대조": 대조, "분포": 분포, "기록": 기록, "시간": 시간}[a.일](a)
    if a.출력:
        json.dump(r, open(a.출력, "w", encoding="utf-8"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
