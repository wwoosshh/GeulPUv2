"""문맥 저장소의 GPU 없는 시험(문서/문맥저장소.md 4단계) — 시험_저장소CPU(엔진/시험_저장소CPU.gl — 가짜 장치 위의 저장소.gl)를 돌려
그 줄과 파일을 파이썬 참조와 견준다: 씨앗 · 이름 사슬(SHA-256), 덩이 파일의 머리말 · 토큰 · 내용 합(64 비트) · 내용 배치(층마다 키 ·
키 지수 · 값 · 값 지수의 줄), 조각 둘로 쓴 파일 = 조각 하나로 쓴 파일, 되살리기 · 망가진 덩이 거르기 · 다른 슬롯 복사(프로그램이 가짜
장치에서 견준 값), 뒤에서 쓰기, 색인의 때 · 차례(새 덩이는 앞 덩이보다 하나 이르게, 함께 쓰인 사슬은 앞일수록 늦게, 색인.txt 에 없는
파일은 맨 앞), 크기 한도로 지우는 차례, 지우기. 빠른점검이 부른다. 표준 라이브러리만.

    python -I 시험/저장소CPU.py <시험_저장소CPU 실행 파일>
"""
import hashlib
import os
import shutil
import struct
import subprocess
import sys
import tempfile

덩이 = 256
층수, KV머리수, 머리크기 = 3, 2, 128
M = 1 << 64
때폭 = 1048576
틀림 = []


def 봄(조건, 말):
    if not 조건:
        틀림.append(말)


def 가짜값(t, l, r, x):
    v = (x * 2654435761 + r * 40503 + l * 9973 + t * 31337) & 0xFFFFFFFF
    v ^= v >> 13
    v = (v * 1540483477) & 0xFFFFFFFF
    v ^= v >> 15
    return v & 255


def 줄수(t):
    return 2 * KV머리수 if t == 0 else KV머리수


def 자리폭(t):
    return 머리크기 if t == 0 else 2 * 머리크기 if t == 2 else 머리크기 // 32


def 내용(k):
    """슬롯 0 의 위치 [256k, 256k + 256) — 층마다 키 · 키 지수 · 값 · 값 지수, 종류마다 줄(KV 머리 · 자리)마다 위치의 바이트"""
    p0 = k * 덩이
    out = bytearray()
    for l in range(층수):
        for t in range(4):
            u = 자리폭(t)
            for r in range(줄수(t)):
                out += bytes(가짜값(t, l, r, x) for x in range(p0 * u, (p0 + 덩이) * u))
    return bytes(out)


def 합(b):
    w = struct.unpack(f"<{len(b) // 8}Q", b)
    n = len(b) // 32
    a, c1, c2, c3 = 1469598103934665603, 7046029254386353131, 3141592653589793238, 2718281828459045235
    P = 1099511628211
    for i in range(n):
        a = ((a ^ w[4 * i]) * P) % M
        c1 = ((c1 ^ w[4 * i + 1]) * P) % M
        c2 = ((c2 ^ w[4 * i + 2]) * P) % M
        c3 = ((c3 ^ w[4 * i + 3]) * P) % M
    h = a ^ (c1 * 31 % M) ^ (c2 * 961 % M) ^ (c3 * 29791 % M)
    for j in range(4 * n, len(b) // 8):
        h = ((h ^ w[j]) * P) % M
    s = h - M if h >> 63 else h                     # 글의 >> 는 부호 있는 정수의 산술 이동
    return h ^ ((s >> 29) % M)


def 이름들(씨앗, 토큰, n):
    out, 앞 = [], 씨앗
    for k in range(n // 덩이):
        앞 = hashlib.sha256(앞.encode() + struct.pack(f"<{덩이}I", *토큰[k * 덩이:(k + 1) * 덩이])).hexdigest()
        out.append(앞)
    return out


def 경로(폴더, 이름):
    return os.path.join(폴더, 이름[:2], 이름 + ".kv")


def 덩이파일(폴더, 씨앗, 토큰, 이름목록, k, 바이트):
    p = 경로(폴더, 이름목록[k])
    if not os.path.exists(p):
        봄(False, f"덩이 파일이 없다: {p}")
        return None
    b = open(p, "rb").read()
    봄(len(b) == 4096 + 바이트, f"덩이 {k} 크기 {len(b)}")
    h = struct.unpack("<10Q", b[:80])
    봄(b[:4] == b"GKV1" and b[4:8] == bytes(4), f"덩이 {k} 표시")
    봄(h[1:9] == (1, 덩이, 층수, KV머리수, 머리크기, k, 바이트, 0), f"덩이 {k} 머리말 {h[1:9]}")
    앞 = 씨앗 if k == 0 else 이름목록[k - 1]
    봄(b[80:144] == 씨앗.encode() and b[144:208] == 앞.encode() and b[208:272] == 이름목록[k].encode(), f"덩이 {k} 씨앗 · 앞 이름 · 이름")
    봄(b[272:272 + 4 * 덩이] == struct.pack(f"<{덩이}I", *토큰[k * 덩이:(k + 1) * 덩이]), f"덩이 {k} 토큰")
    봄(b[272 + 4 * 덩이:4096] == bytes(4096 - 272 - 4 * 덩이), f"덩이 {k} 머리말 나머지")
    봄(b[4096:] == 내용(k), f"덩이 {k} 내용 배치")
    봄(h[9] == 합(b[4096:]), f"덩이 {k} 내용 합 {h[9]:x} ≠ {합(b[4096:]):x}")
    return b


class 색인:
    """엔진/저장소.gl 의 때 규칙 — 쓴 때 = 시계 × 때폭 − 덩이 번호"""

    def __init__(self):
        self.때 = {}
        self.시계 = 0

    def 읽기(self, 파일들, 색인줄):
        모름 = [n for n in 파일들 if n not in 색인줄]           # 수정 시각 차례(여기서는 하나뿐)
        차례 = {n: i for i, n in enumerate(dict.fromkeys(색인줄))}
        self.때 = {}
        for r, n in enumerate(모름):
            self.때[n] = (r + 1) * 때폭
        for n in 파일들:
            if n in 차례:
                self.때[n] = (len(모름) + 1 + 차례[n]) * 때폭
        self.시계 = len(모름) + len(차례) + 1

    def 새때(self, 앞, k):
        if k > 0 and 앞 in self.때:
            return self.때[앞] - 1
        self.시계 += 1
        return self.시계 * 때폭 - k

    def 앞쪽쓴때(self, 이름목록, m):
        self.시계 += 1
        for k in range(m):
            if 이름목록[k] in self.때:
                self.때[이름목록[k]] = self.시계 * 때폭 - k

    def 차례(self):
        return sorted(self.때.items(), key=lambda x: x[1])


def main():
    exe = os.path.abspath(sys.argv[1])
    뿌리 = tempfile.mkdtemp(prefix="저장소CPU_")
    try:
        r = subprocess.run([exe, 뿌리], capture_output=True, timeout=300)
        글 = r.stdout.decode("utf-8", "replace")
        봄(r.returncode == 0, f"종료 코드 {r.returncode}: {r.stderr.decode('utf-8', 'replace')[-300:]}")
        줄 = [l.split() for l in 글.splitlines() if l.strip()]
        값 = {}
        목록 = {}
        for x in 줄:
            if x[0] in ("이름", "A", "B", "C"):
                목록.setdefault(x[0], []).append(x[2])
            elif x[0] in ("차례", "다시", "남은"):
                목록.setdefault(x[0], []).append((x[1], int(x[2])))
            else:
                값[x[0]] = x[1:]
        봄("끝" in 값, "끝 줄이 없다(중간에 멈춤)")
        토큰 = [(i * 7919 + 13) % 151936 for i in range(2048)]
        B = [토큰[i] if i < 512 else (i * 104729 + 7) % 151936 for i in range(2048)]
        C = [(i * 31 + 5) % 151936 for i in range(2048)]
        씨앗 = hashlib.sha256(f"GeulPU-KV/1|시험모델|정밀|덩이 {덩이}|층 {층수}|KV머리 {KV머리수}|머리 {머리크기}".encode()).hexdigest()
        바이트 = 층수 * KV머리수 * 덩이 * (4 * 머리크기 + 2 * (머리크기 // 32))
        봄(값.get("씨앗") == [씨앗], f"씨앗 {값.get('씨앗')}")
        봄(값.get("덩이바이트") == [str(바이트)], f"덩이바이트 {값.get('덩이바이트')}")
        이름 = 이름들(씨앗, 토큰, 1000)
        봄(목록.get("이름") == 이름, "이름 사슬")
        봄(값.get("쓴") == ["3"], f"쓴 {값.get('쓴')}")
        봄(값.get("되살림") == ["768", "1", "1"], f"되살림 {값.get('되살림')}")
        첫 = [덩이파일(os.path.join(뿌리, "가"), 씨앗, 토큰, 이름, k, 바이트) for k in range(3)]
        # 조각 둘: 같은 파일, 되살림, 망가진 덩이 1 은 걸러져 지워짐
        조각2 = [l for l in 글.splitlines() if l.startswith("조각2 ")]
        봄(조각2 == ["조각2 쓴 3", "조각2 되살림 768 1"], f"조각2 {조각2}")
        for k in (0, 2):
            p = 경로(os.path.join(뿌리, "나"), 이름[k])
            봄(os.path.exists(p) and 첫[k] is not None and open(p, "rb").read() == 첫[k], f"조각 둘로 쓴 덩이 {k} 가 조각 하나의 것과 다르다")
        봄(not os.path.exists(경로(os.path.join(뿌리, "나"), 이름[1])), "망가진 덩이 1 이 지워지지 않았다")
        봄(값.get("망가짐") == ["256", "1", "0"], f"망가짐 {값.get('망가짐')}")
        봄(값.get("복사") == ["1", "1", "1"], f"복사 {값.get('복사')}")
        # 뒤에서 쓰기 · 색인
        이름A, 이름B, 이름C = 이름들(씨앗, 토큰, 6 * 덩이), 이름들(씨앗, B, 5 * 덩이), 이름들(씨앗, C, 덩이)
        봄(목록.get("A") == 이름A and 목록.get("B") == 이름B and 목록.get("C") == 이름C, "사슬 A · B · C 의 이름")
        봄(값.get("뒤에서") == ["9", "0", "9", str(9 * (4096 + 바이트))], f"뒤에서 {값.get('뒤에서')}")
        s = 색인()
        s.읽기([], [])
        for k in range(6):
            s.때[이름A[k]] = s.새때(이름A[k - 1] if k else 씨앗, k)
        s.앞쪽쓴때(이름B, 2)
        for k in range(2, 5):
            s.때[이름B[k]] = s.새때(이름B[k - 1], k)
        차례 = s.차례()
        봄(목록.get("차례") == 차례, f"색인 차례 {목록.get('차례')} ≠ {차례}")
        s.읽기([n for n, _ in 차례] + [이름C[0]], [n for n, _ in 차례])
        다시 = s.차례()
        봄(목록.get("다시") == 다시, f"다시 읽은 차례 {목록.get('다시')} ≠ {다시}")
        남은 = 다시[len(다시) - 5:]
        봄(값.get("한도") == [str(len(다시) - 5)] and 목록.get("남은") == 남은, f"한도 {값.get('한도')} · 남은 {목록.get('남은')}")
        봄(값.get("지우기") == ["5", "0"], f"지우기 {값.get('지우기')}")
        남은파일 = [f for _, _, fs in os.walk(os.path.join(뿌리, "다")) for f in fs]
        봄(not 남은파일, f"지운 뒤 남은 파일 {남은파일}")
    finally:
        shutil.rmtree(뿌리, ignore_errors=True)
    if 틀림:
        print("문맥 저장소(GPU 없이): 못 미침 — " + " / ".join(틀림[:8]))
        return 1
    print("문맥 저장소(GPU 없이): 통과 — 이름 사슬 · 머리말 · 내용 합 · 배치(조각 하나 · 둘) · 되살리기 · 망가진 덩이 · 다른 슬롯 복사 · "
          "뒤에서 쓰기 · 색인 차례 · 크기 한도 · 지우기")
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.exit(main())
