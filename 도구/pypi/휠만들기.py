"""GeulPU 의 PyPI 휠 만들기 — 릴리스 꾸러미를 플랫폼 휠로. 파이썬 표준 라이브러리만.

  python -X utf8 도구/pypi/휠만들기.py <릴리스 꾸러미> <출력 폴더> [--sha256sums SHA256SUMS.txt] [--판 <판>]

  GeulPU-<판>-windows-x64.zip   → geulpu-<판>-py3-none-win_amd64.whl
  GeulPU-<판>-linux-x64.tar.gz  → geulpu-<판>-py3-none-manylinux_2_34_x86_64.whl — 실행 파일이 판이 붙은 glibc 기호를 가져오지 않고
                                  pthread 를 libc.so.6 에서 가져오므로 glibc 2.34 이상(문서/H100.md 의 1)

- SHA256SUMS.txt 를 주면 꾸러미의 SHA-256 을 견준다(다르면 멈춘다).
- 꾸러미 안의 GeulPU-<판>/ 폴더를 geulpu/app/ 으로 그대로, 실행기(이 폴더의 geulpu/*.py — __version__ 을 판으로)와 dist-info(METADATA ·
  WHEEL · entry_points.txt · licenses/LICENSE · RECORD)를 쓴다. 파일 차례와 시각을 고정해 같은 꾸러미 · 같은 파이썬(zlib)이면 바이트까지
  같은 휠이 나온다 — 릴리스 CI 가 꾸리기 바로 뒤에 한 번 만들어 초안에 붙이고, PyPI 에는 그 휠을 그대로 올린다(문서/배포.md).
- 리눅스 꾸러미(tar)의 실행 파일(GeulPU · bin/*)은 휠 항목의 권한을 0o755 로(tar 에서도 실행 권한이 있어야 한다 — 없으면 멈춘다),
  나머지는 0o644 로 쓴다 — pip 가 깔 때 그대로 둔다. tar 에는 보통 파일과 폴더만 있어야 한다(링크 · 장치 파일이면 멈춘다).
- 판은 꾸러미의 폴더 이름과 version.json 에서 읽어 서로 견주고, --판(태그)을 주면 그것과도 견준다(앱 판 = 꾸러미 판).
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
import tarfile
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
시각 = (2026, 1, 1, 0, 0, 0)                   # 휠 안 파일의 시각(고정 — 다시 만들어도 같은 바이트)
꼴들 = {                                        # version.json 의 platform → 휠 태그 · 앱 실행 파일
    "windows-x64": ("py3-none-win_amd64", "GeulPU.exe"),
    "linux-x64": ("py3-none-manylinux_2_34_x86_64", "GeulPU"),
}
요약 = ("A deterministic local LLM engine and debugger — bit-reproducible GGUF inference on your own GPU, written in the "
        "geul language (Windows x64 app with CUDA or Vulkan, Linux x86-64 app with CUDA)")


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
        "Project-URL: Documentation, https://github.com/wwoosshh/GeulPUv2/blob/main/docs/user-guide.md",
        "Project-URL: Changelog, https://github.com/wwoosshh/GeulPUv2/blob/main/CHANGELOG.md",
        "Project-URL: Issues, https://github.com/wwoosshh/GeulPUv2/issues",
        "Keywords: llm,gguf,local-llm,chat,cuda,vulkan,openai-compatible,reproducible,deterministic,debugger",
        "Classifier: Development Status :: 4 - Beta",
        "Classifier: Environment :: Console",
        "Classifier: Environment :: GPU :: NVIDIA CUDA",
        "Classifier: Intended Audience :: Developers",
        "Classifier: Intended Audience :: End Users/Desktop",
        "Classifier: Intended Audience :: Science/Research",
        "Classifier: Operating System :: Microsoft :: Windows :: Windows 10",
        "Classifier: Operating System :: Microsoft :: Windows :: Windows 11",
        "Classifier: Operating System :: POSIX :: Linux",
        "Classifier: Programming Language :: Python :: 3",
        "Classifier: Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Classifier: Topic :: Software Development :: Debuggers",
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


def 항목읽기(원, 꾸러미이름):
    """꾸러미(zip · tar.gz)의 파일 항목 — {/ 로 맞춘 이름: (바이트, 실행 권한인가)}. 디렉터리 항목은 뺀다."""
    항목 = {}
    if 꾸러미이름.endswith(".tar.gz"):
        with tarfile.open(fileobj=io.BytesIO(원), mode="r:gz") as t:
            for m in t.getmembers():
                if m.isdir():
                    continue
                if not m.isfile():
                    raise SystemExit(f"tar 에 보통 파일이 아닌 항목이 있다: {m.name}")
                n = m.name[2:] if m.name.startswith("./") else m.name
                if n in 항목:
                    raise SystemExit(f"tar 에 같은 이름이 둘 있다: {n}")
                항목[n] = (t.extractfile(m).read(), bool(m.mode & 0o111))
    else:
        z = zipfile.ZipFile(io.BytesIO(원))
        for n in z.namelist():
            if n.endswith("/") or n.endswith("\\"):
                continue
            k = n.replace("\\", "/")                  # .NET 판에 따라 zip 항목 이름에 역슬래시가 든다 — / 로 맞춘다
            if k in 항목:
                raise SystemExit("zip 에 / · \\ 만 다른 같은 이름이 있다")
            항목[k] = (z.read(n), False)              # 윈도 휠은 모든 항목이 0o644(예전 휠과 같은 바이트)
    for n in 항목:
        if n.startswith("/") or ":" in n or ".." in n.split("/"):
            raise SystemExit(f"꾸러미에 위험한 경로가 있다: {n}")
    return 항목


def 만들기(꾸러미경로, 출력, 합파일=None, 바란판=None):
    원 = open(꾸러미경로, "rb").read()
    이름 = os.path.basename(꾸러미경로)
    if 합파일:
        합 = {}
        for 줄 in open(합파일, encoding="utf-8"):
            칸 = 줄.split()
            if len(칸) == 2:
                합[칸[1].lstrip("*")] = 칸[0].lower()
        if 이름 not in 합:
            raise SystemExit(f"SHA256SUMS 에 {이름} 이 없다")
        if 합[이름] != hashlib.sha256(원).hexdigest():
            raise SystemExit(f"{이름} 의 SHA-256 이 SHA256SUMS 와 다르다")
    항목 = 항목읽기(원, 이름)
    이름들 = sorted(항목)
    뿌리들 = {n.split("/", 1)[0] for n in 이름들}
    if len(뿌리들) != 1:
        raise SystemExit(f"꾸러미의 맨 위 폴더가 하나가 아니다: {sorted(뿌리들)}")
    뿌리 = 뿌리들.pop()
    m = re.fullmatch(r"GeulPU-(\d+\.\d+\.\d+)", 뿌리)
    if not m:
        raise SystemExit(f"꾸러미의 폴더 이름에서 판을 읽지 못했다: {뿌리}")
    판 = m.group(1)
    if 바란판 and 바란판.lstrip("v") != 판:
        raise SystemExit(f"꾸러미의 판 {판} 이 바란 판 {바란판} 과 다르다")
    vj = json.loads(항목[f"{뿌리}/version.json"][0].decode("utf-8-sig"))
    꼴 = vj.get("platform")
    if vj.get("version") != 판 or 꼴 not in 꼴들:
        raise SystemExit(f"version.json 의 판 · 꼴이 다르다: {vj.get('version')} · {꼴}")
    if not 이름.endswith(".tar.gz" if 꼴 == "linux-x64" else ".zip"):
        raise SystemExit(f"{꼴} 꾸러미의 파일 이름이 맞지 않는다: {이름}")
    태그, 실행 = 꼴들[꼴]
    if f"{뿌리}/{실행}" not in 항목:
        raise SystemExit(f"꾸러미에 {실행} 이 없다")
    파일들 = []                                   # (휠 안의 경로, 바이트, 실행 권한인가)
    처음 = open(os.path.join(HERE, "geulpu", "__init__.py"), encoding="utf-8").read()
    if 처음.count('__version__ = "0.0.0"') != 1:
        raise SystemExit("geulpu/__init__.py 의 판 자리를 찾지 못했다")
    파일들.append(("geulpu/__init__.py", 처음.replace('__version__ = "0.0.0"', f'__version__ = "{판}"').encode("utf-8"), False))
    파일들.append(("geulpu/__main__.py", open(os.path.join(HERE, "geulpu", "__main__.py"), "rb").read(), False))
    for n in 이름들:
        상대 = n[len(뿌리) + 1:]
        바이트, 실행됨 = 항목[n]
        실행파일 = 꼴 == "linux-x64" and (상대 == 실행 or 상대.startswith("bin/"))
        if 실행파일 and not 실행됨:
            raise SystemExit(f"꾸러미의 {상대} 에 실행 권한이 없다")
        파일들.append(("geulpu/app/" + 상대, 바이트, 실행파일))
    di = f"geulpu-{판}.dist-info"
    설명 = open(os.path.join(HERE, "README.md"), encoding="utf-8").read()
    파일들.append((f"{di}/METADATA", 메타데이터(판, 설명), False))
    파일들.append((f"{di}/WHEEL", ("Wheel-Version: 1.0\nGenerator: GeulPUv2 도구/pypi/휠만들기.py\nRoot-Is-Purelib: false\n"
                                f"Tag: {태그}\n").encode("utf-8"), False))
    파일들.append((f"{di}/entry_points.txt", b"[console_scripts]\ngeulpu = geulpu.__main__:main\n", False))
    파일들.append((f"{di}/licenses/LICENSE", 항목[f"{뿌리}/LICENSE"][0], False))
    레코드 = "".join(f"{p},{레코드해시(b)},{len(b)}\n" for p, b, _ in 파일들) + f"{di}/RECORD,,\n"
    파일들.append((f"{di}/RECORD", 레코드.encode("utf-8"), False))
    os.makedirs(출력, exist_ok=True)
    휠 = os.path.join(출력, f"geulpu-{판}-{태그}.whl")
    with zipfile.ZipFile(휠, "w", zipfile.ZIP_DEFLATED) as w:
        for p, b, x in 파일들:
            zi = zipfile.ZipInfo(p, date_time=시각)
            zi.compress_type = zipfile.ZIP_DEFLATED
            zi.external_attr = ((0o100755 if x else 0o100644) << 16)
            w.writestr(zi, b)
    검증(휠)
    return 휠, 판


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("꾸러미", help="GeulPU-<판>-windows-x64.zip 또는 GeulPU-<판>-linux-x64.tar.gz")
    ap.add_argument("출력")
    ap.add_argument("--sha256sums")
    ap.add_argument("--판", help="바란 판(태그 v<판> 도 받음) — 꾸러미의 판과 다르면 멈춘다")
    a = ap.parse_args()
    휠, 판 = 만들기(a.꾸러미, a.출력, a.sha256sums, a.판)
    print(f"휠: {휠} (판 {판} · {os.path.getsize(휠):,} 바이트 · SHA-256 {hashlib.sha256(open(휠, 'rb').read()).hexdigest()})")


if __name__ == "__main__":
    main()
