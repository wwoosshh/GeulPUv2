#!/usr/bin/env python3
"""0.12.0 12-G2 — llama.cpp 의 어휘 시험 파일(시험/어휘시험.txt 의 판 · 이름 · 바이트 · sha256)을 받는다(CI — 빠른 점검의 --어휘).
이미 있고 sha256 이 같은 파일은 그대로 둔다. 하나라도 sha256 이 다르면 종료 1(받은 것은 지운다).

  python -I 시험/어휘받기.py <폴더>
"""
import hashlib
import os
import sys
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    sys.stdout.reconfigure(encoding="utf-8")          # -I 는 PYTHONUTF8 을 보지 않는다 — 윈도 CI 의 cp1252
    폴더 = sys.argv[1]
    os.makedirs(폴더, exist_ok=True)
    판, 목록 = None, []
    for 줄 in open(os.path.join(HERE, "어휘시험.txt"), encoding="utf-8"):
        줄 = 줄.strip()
        if not 줄 or 줄.startswith("#"):
            continue
        if 줄.startswith("판 "):
            판 = 줄.split()[1]
            continue
        이름, 크기, 해시 = 줄.split()
        목록.append((이름, int(크기), 해시))
    assert 판 and 목록
    틀림 = 0
    for 이름, 크기, 해시 in 목록:
        p = os.path.join(폴더, 이름)
        if os.path.exists(p) and hashlib.sha256(open(p, "rb").read()).hexdigest() == 해시:
            continue
        url = f"https://raw.githubusercontent.com/ggml-org/llama.cpp/{판}/models/{이름}"
        자료 = urllib.request.urlopen(url, timeout=120).read()
        if len(자료) != 크기 or hashlib.sha256(자료).hexdigest() != 해시:
            print(f"sha256 이 다르다: {이름} ({len(자료)} 바이트)")
            틀림 += 1
            continue
        open(p, "wb").write(자료)
    print(f"어휘 시험 파일 {len(목록)} 개(llama.cpp {판}) — {'모두 확인' if not 틀림 else f'다름 {틀림}'}")
    sys.exit(1 if 틀림 else 0)


if __name__ == "__main__":
    main()
