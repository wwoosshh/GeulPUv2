#!/usr/bin/env python3
"""모델 평가 4-1(문서/모델평가.md 의 '4-1 의 판정') — 앱이 받는 llama.cpp 의 도구 목록(앱/data/tools.json)을 만든다.

  python -I -X utf8 도구/llama도구목록.py <llama.cpp 의 태그 — 예: b12032> 앱/data/tools.json [transformers 5 의 판 — 없으면 PyPI 의 최신]

- 실행 파일: 그 태그의 GitHub 릴리스에서 윈도(win-vulkan-x64 zip) · 리눅스(ubuntu-vulkan-x64 tar.gz)의 이름 · 주소 · 크기 · SHA-256(릴리스의
  digest). 양자화(llama-quantize)는 CPU 로, imatrix(llama-imatrix)는 Vulkan 으로 GPU 를 쓴다 — 어느 회사의 GPU 든.
- 변환 스크립트: 그 태그의 convert_hf_to_gguf.py · conversion/ · gguf-py/ · requirements/ · LICENSE — 파일마다 받아 크기 · SHA-256 을 적는다(앱은
  raw.githubusercontent.com 에서 받아 이 값과 견준다). 소스 전체(약 170 MB)는 받지 않는다.
- transformers 5 환경(#196): llama.cpp 는 transformers 를 4.x 에 묶어 두었는데 4.x 는 transformers 5 로 저장한 모델 폴더를 바르게 읽지 못한다.
  관리자의 권고(ggml-org/llama.cpp#24460)대로 transformers 만 올린 두 번째 환경을 둔다 — requirements5 = 그 태그의 변환 requirements(-r 을
  따라감)에서 torch(첫 환경의 것을 함께 씀)를 빼고 transformers 를 그 판으로.
llama.cpp 는 MIT(Copyright (c) 2023-2026 The ggml authors).
"""
import hashlib
import json
import re
import sys
import urllib.request

API = "https://api.github.com/repos/ggml-org/llama.cpp"
RAW = "https://raw.githubusercontent.com/ggml-org/llama.cpp/{}/"
PYPI = "https://pypi.org/pypi/transformers/json"
빌드 = {"windows": "llama-{}-bin-win-vulkan-x64.zip", "linux": "llama-{}-bin-ubuntu-vulkan-x64.tar.gz"}
변환 = ("convert_hf_to_gguf.py", "LICENSE", "conversion/", "gguf-py/", "requirements/")


def 받기(u):
    q = urllib.request.Request(u, headers={"User-Agent": "GeulPU-tools-list", "Accept": "application/vnd.github+json"})
    with urllib.request.urlopen(q, timeout=60) as r:
        return r.read()


def 요구줄들(글받기, 이름, 본):
    """requirements 파일의 줄 — -r 은 따라가고, --extra-index-url 따위 선택지 · 주석 · 빈 줄은 뺀다. 글받기(이름) → 파일의 글"""
    if 이름 in 본:
        return []
    본.add(이름)
    줄들 = []
    for l in 글받기(이름).splitlines():
        s = l.split("#", 1)[0].strip()
        if s.startswith("-r "):
            줄들 += 요구줄들(글받기, s[3:].strip().removeprefix("./"), 본)
        elif s and not s.startswith("-"):
            줄들.append(s)
    return 줄들


def 요구목록5(글받기, 판5):
    """transformers 5 환경의 목록 — 변환 requirements 에서 torch · transformers 를 빼고 transformers 를 판5 로"""
    줄들 = 요구줄들(글받기, "requirements-convert_hf_to_gguf.txt", set())
    return [l for l in 줄들 if re.match(r"[A-Za-z0-9_.\-]+", l).group(0).lower() not in ("torch", "transformers")] + [f"transformers=={판5}"]


def main(태그, 출력, 판5=None):
    rel = json.loads(받기(f"{API}/releases/tags/{태그}"))
    자산 = {a["name"]: a for a in rel["assets"]}
    builds = {}
    for 체계, 꼴 in 빌드.items():
        a = 자산[꼴.format(태그)]
        d = a.get("digest") or ""
        assert d.startswith("sha256:"), a["name"]
        builds[체계] = {"file": a["name"], "url": a["browser_download_url"], "size": a["size"], "sha256": d[7:]}
    트리 = json.loads(받기(f"{API}/git/trees/{태그}?recursive=1"))
    assert not 트리.get("truncated")
    files = []
    for e in sorted(트리["tree"], key=lambda e: e["path"]):
        if e["type"] != "blob" or not any(e["path"] == p or (p.endswith("/") and e["path"].startswith(p)) for p in 변환):
            continue
        b = 받기(RAW.format(태그) + e["path"])
        assert len(b) == e["size"], e["path"]
        files.append({"path": e["path"], "size": len(b), "sha256": hashlib.sha256(b).hexdigest()})
    req = 받기(RAW.format(태그) + "requirements/requirements-convert_hf_to_gguf.txt").decode("utf-8")
    torch = [l.strip() for l in req.splitlines() if l.strip().startswith("torch")]
    판5 = 판5 or json.loads(받기(PYPI))["info"]["version"]
    assert 판5.startswith("5."), 판5
    요5 = 요구목록5(lambda 이름: 받기(RAW.format(태그) + "requirements/" + 이름).decode("utf-8"), 판5)
    out = {"llama_cpp": {"tag": 태그, "license": "MIT", "source": "https://github.com/ggml-org/llama.cpp", "builds": builds,
                         "convert": {"base": RAW.format(태그), "requirements": "requirements/requirements-convert_hf_to_gguf.txt",
                                     "torch": torch[0] if torch else "", "transformers5": 판5, "requirements5": 요5, "files": files}}}
    with open(출력, "w", encoding="utf-8", newline="\n") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
        f.write("\n")
    print(f"{출력}: llama.cpp {태그} — 실행 파일 {', '.join(f'{k} {v['size'] / 1048576:.1f} MB' for k, v in builds.items())} · 변환 파일 "
          f"{len(files)}개 {sum(x['size'] for x in files) / 1048576:.2f} MB · {torch[0] if torch else 'torch ?'} · transformers 5 환경 {판5}")


if __name__ == "__main__":
    if len(sys.argv) not in (3, 4):
        print(__doc__)
        sys.exit(2)
    main(*sys.argv[1:4])
