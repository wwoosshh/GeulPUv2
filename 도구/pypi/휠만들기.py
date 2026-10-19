"""GeulPU 의 PyPI 휠 만들기 — 릴리스 zip(GeulPU-<판>-windows-x64.zip)을 geulpu-<판>-py3-none-win_amd64.whl 로. 파이썬 표준 라이브러리만.

  python -X utf8 도구/pypi/휠만들기.py <릴리스 zip> <출력 폴더> [--sha256sums SHA256SUMS.txt]

- SHA256SUMS.txt 를 주면 zip 의 SHA-256 을 견준다(다르면 멈춘다).
- zip 안의 GeulPU-<판>/ 폴더를 geulpu/app/ 으로 그대로, 실행기(이 폴더의 geulpu/*.py — __version__ 을 판으로)와 dist-info(METADATA ·
  WHEEL · entry_points.txt · licenses/LICENSE · RECORD)를 쓴다. 파일 차례와 시각을 고정해 같은 zip · 같은 파이썬(zlib)이면 바이트까지 같은
  휠이 나온다 — 릴리스 CI 가 꾸리기 바로 뒤에 한 번 만들어 초안에 붙이고, PyPI 에는 그 휠을 그대로 올린다(문서/배포.md).
- 판은 zip 의 폴더 이름과 version.json 에서 읽어 서로 견주고, --판(태그)을 주면 그것과도 견준다(앱 판 = 꾸러미 판).
- 다 쓴 휠을 다시 읽어 RECORD 의 해시 · 크기가 모든 파일과 맞는지, 디렉터리 항목 · 같은 이름이 없는지 본다(PyPI 가 거절하는 꼴).
"""
import argparse
import base64
import hashlib
import io
import json
import os
import re
import sys
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
시각 = (2026, 1, 1, 0, 0, 0)                   # 휠 안 파일의 시각(고정 — 다시 만들어도 같은 바이트)
태그 = "py3-none-win_amd64"
요약 = ("Local AI chat on your own GPU — a bit-reproducible GGUF LLM engine written in the geul language "
        "(Windows x64 app, CUDA or Vulkan)")


def 레코드해시(b):
    return "sha256=" + base64.urlsafe_b64encode(hashlib.sha256(b).digest()).rstrip(b"=").decode("ascii")


def 메타데이터(판, 설명):
    """핵심 메타데이터 2.4 — 라이선스는 PEP 639(License-Expression · License-File, License 분류자 · License 칸은 쓰지 않는다)."""
    머리 = [
        "Metadata-Version: 2.4",
        "Name: geulpu",
        f"Version: {판}",
        f"Summary: {요약}",
        "Author: wwoosshh",
        "License-Expression: MIT",
        "License-File: LICENSE",
        "Project-URL: Homepage, https://github.com/wwoosshh/GeulPUv2",
        "Project-URL: Changelog, https://github.com/wwoosshh/GeulPUv2/blob/main/%EB%AC%B8%EC%84%9C/%EB%B0%94%EB%80%90%EA%B2%83.md",
        "Project-URL: Issues, https://github.com/wwoosshh/GeulPUv2/issues",
        "Keywords: llm,gguf,local-llm,chat,cuda,vulkan,openai-compatible,reproducible",
        "Classifier: Development Status :: 4 - Beta",
        "Classifier: Environment :: Console",
        "Classifier: Environment :: GPU :: NVIDIA CUDA",
        "Classifier: Intended Audience :: End Users/Desktop",
        "Classifier: Operating System :: Microsoft :: Windows :: Windows 10",
        "Classifier: Operating System :: Microsoft :: Windows :: Windows 11",
        "Classifier: Programming Language :: Python :: 3",
        "Classifier: Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Requires-Python: >=3.9",
        "Description-Content-Type: text/markdown",
    ]
    return ("\n".join(머리) + "\n\n" + 설명).encode("utf-8")


def 검증(휠):
    """다 쓴 휠을 다시 읽어 — 디렉터리 항목 · 같은 이름 · 위험한 경로가 없고, RECORD 의 해시 · 크기가 모든 파일과 맞는가"""
    with zipfile.ZipFile(휠) as w:
        이름들 = w.namelist()
        if len(이름들) != len(set(이름들)):
            raise SystemExit("휠에 같은 이름의 항목이 있다")
        if any(n.endswith("/") for n in 이름들):
            raise SystemExit("휠에 디렉터리 항목이 있다")
        rec = [n for n in 이름들 if n.endswith(".dist-info/RECORD")]
        if len(rec) != 1:
            raise SystemExit("휠의 RECORD 가 하나가 아니다")
        줄들 = w.read(rec[0]).decode("utf-8").splitlines()
        적힌 = {}
        for 줄 in 줄들:
            p, h, s = 줄.rsplit(",", 2)
            적힌[p] = (h, s)
        if set(적힌) != set(이름들):
            raise SystemExit("RECORD 와 휠의 파일 목록이 다르다")
        for n in 이름들:
            if n == rec[0]:
                if 적힌[n] != ("", ""):
                    raise SystemExit("RECORD 자신의 줄은 해시 · 크기가 비어야 한다")
                continue
            b = w.read(n)
            if 적힌[n] != (레코드해시(b), str(len(b))):
                raise SystemExit(f"RECORD 가 {n} 과 맞지 않는다")


def 만들기(zip경로, 출력, 합파일=None, 바란판=None):
    원 = open(zip경로, "rb").read()
    if 합파일:
        합 = {}
        for 줄 in open(합파일, encoding="utf-8"):
            칸 = 줄.split()
            if len(칸) == 2:
                합[칸[1].lstrip("*")] = 칸[0].lower()
        이름 = os.path.basename(zip경로)
        if 이름 not in 합:
            raise SystemExit(f"SHA256SUMS 에 {이름} 이 없다")
        if 합[이름] != hashlib.sha256(원).hexdigest():
            raise SystemExit(f"{이름} 의 SHA-256 이 SHA256SUMS 와 다르다")
    z = zipfile.ZipFile(io.BytesIO(원))
    원이름 = {n.replace("\\", "/"): n for n in z.namelist()}       # .NET 판에 따라 zip 항목 이름에 역슬래시가 든다 — / 로 맞춘다
    if len(원이름) != len(z.namelist()):
        raise SystemExit("zip 에 / · \\ 만 다른 같은 이름이 있다")
    for n in 원이름:
        if n.startswith("/") or ":" in n or ".." in n.split("/"):
            raise SystemExit(f"zip 에 위험한 경로가 있다: {n}")
    이름들 = sorted(n for n in 원이름 if not n.endswith("/"))
    뿌리들 = {n.split("/", 1)[0] for n in 이름들}
    if len(뿌리들) != 1:
        raise SystemExit(f"zip 의 맨 위 폴더가 하나가 아니다: {sorted(뿌리들)}")
    뿌리 = 뿌리들.pop()
    m = re.fullmatch(r"GeulPU-(\d+\.\d+\.\d+)", 뿌리)
    if not m:
        raise SystemExit(f"zip 의 폴더 이름에서 판을 읽지 못했다: {뿌리}")
    판 = m.group(1)
    if 바란판 and 바란판.lstrip("v") != 판:
        raise SystemExit(f"zip 의 판 {판} 이 바란 판 {바란판} 과 다르다")
    vj = json.loads(z.read(원이름[f"{뿌리}/version.json"]).decode("utf-8-sig"))
    if vj.get("version") != 판 or vj.get("platform") != "windows-x64":
        raise SystemExit(f"version.json 의 판 · 꼴이 다르다: {vj.get('version')} · {vj.get('platform')}")
    if f"{뿌리}/GeulPU.exe" not in 이름들:
        raise SystemExit("zip 에 GeulPU.exe 가 없다")
    파일들 = []                                   # (휠 안의 경로, 바이트)
    처음 = open(os.path.join(HERE, "geulpu", "__init__.py"), encoding="utf-8").read()
    if 처음.count('__version__ = "0.0.0"') != 1:
        raise SystemExit("geulpu/__init__.py 의 판 자리를 찾지 못했다")
    파일들.append(("geulpu/__init__.py", 처음.replace('__version__ = "0.0.0"', f'__version__ = "{판}"').encode("utf-8")))
    파일들.append(("geulpu/__main__.py", open(os.path.join(HERE, "geulpu", "__main__.py"), "rb").read()))
    for n in 이름들:
        파일들.append(("geulpu/app/" + n[len(뿌리) + 1:], z.read(원이름[n])))
    di = f"geulpu-{판}.dist-info"
    설명 = open(os.path.join(HERE, "README.md"), encoding="utf-8").read()
    파일들.append((f"{di}/METADATA", 메타데이터(판, 설명)))
    파일들.append((f"{di}/WHEEL", ("Wheel-Version: 1.0\nGenerator: GeulPUv2 도구/pypi/휠만들기.py\nRoot-Is-Purelib: false\n"
                                f"Tag: {태그}\n").encode("utf-8")))
    파일들.append((f"{di}/entry_points.txt", b"[console_scripts]\ngeulpu = geulpu.__main__:main\n"))
    파일들.append((f"{di}/licenses/LICENSE", z.read(원이름[f"{뿌리}/LICENSE"])))
    레코드 = "".join(f"{p},{레코드해시(b)},{len(b)}\n" for p, b in 파일들) + f"{di}/RECORD,,\n"
    파일들.append((f"{di}/RECORD", 레코드.encode("utf-8")))
    os.makedirs(출력, exist_ok=True)
    휠 = os.path.join(출력, f"geulpu-{판}-{태그}.whl")
    with zipfile.ZipFile(휠, "w", zipfile.ZIP_DEFLATED) as w:
        for p, b in 파일들:
            zi = zipfile.ZipInfo(p, date_time=시각)
            zi.compress_type = zipfile.ZIP_DEFLATED
            zi.external_attr = (0o100644 << 16)
            w.writestr(zi, b)
    검증(휠)
    return 휠, 판


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("zip")
    ap.add_argument("출력")
    ap.add_argument("--sha256sums")
    ap.add_argument("--판", help="바란 판(태그 v<판> 도 받음) — zip 의 판과 다르면 멈춘다")
    a = ap.parse_args()
    휠, 판 = 만들기(a.zip, a.출력, a.sha256sums, a.판)
    print(f"휠: {휠} (판 {판} · {os.path.getsize(휠):,} 바이트 · SHA-256 {hashlib.sha256(open(휠, 'rb').read()).hexdigest()})")


if __name__ == "__main__":
    main()
