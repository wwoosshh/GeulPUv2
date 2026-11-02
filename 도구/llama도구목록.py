#!/usr/bin/env python3
"""모델 평가 4-1(문서/모델평가.md 의 '4-1 의 판정') — 앱이 받는 llama.cpp 의 도구 목록(앱/data/tools.json)을 만든다.

  python -I -X utf8 도구/llama도구목록.py <llama.cpp 의 태그 — 예: b12032> 앱/data/tools.json

- 실행 파일: 그 태그의 GitHub 릴리스에서 윈도(win-vulkan-x64 zip) · 리눅스(ubuntu-vulkan-x64 tar.gz)의 이름 · 주소 · 크기 · SHA-256(릴리스의
  digest). 양자화(llama-quantize)는 CPU 로, imatrix(llama-imatrix)는 Vulkan 으로 GPU 를 쓴다 — 어느 회사의 GPU 든.
- 변환 스크립트: 그 태그의 convert_hf_to_gguf.py · conversion/ · gguf-py/ · requirements/ · LICENSE — 파일마다 받아 크기 · SHA-256 을 적는다(앱은
  raw.githubusercontent.com 에서 받아 이 값과 견준다). 소스 전체(약 170 MB)는 받지 않는다.
llama.cpp 는 MIT(Copyright (c) 2023-2026 The ggml authors).
"""
import hashlib
import json
import sys
import urllib.request

API = "https://api.github.com/repos/ggml-org/llama.cpp"
RAW = "https://raw.githubusercontent.com/ggml-org/llama.cpp/{}/"
빌드 = {"windows": "llama-{}-bin-win-vulkan-x64.zip", "linux": "llama-{}-bin-ubuntu-vulkan-x64.tar.gz"}
변환 = ("convert_hf_to_gguf.py", "LICENSE", "conversion/", "gguf-py/", "requirements/")


def 받기(u):
    q = urllib.request.Request(u, headers={"User-Agent": "GeulPU-tools-list", "Accept": "application/vnd.github+json"})
    with urllib.request.urlopen(q, timeout=60) as r:
        return r.read()


def main(태그, 출력):
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
    out = {"llama_cpp": {"tag": 태그, "license": "MIT", "source": "https://github.com/ggml-org/llama.cpp", "builds": builds,
                         "convert": {"base": RAW.format(태그), "requirements": "requirements/requirements-convert_hf_to_gguf.txt",
                                     "torch": torch[0] if torch else "", "files": files}}}
    with open(출력, "w", encoding="utf-8", newline="\n") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
        f.write("\n")
    print(f"{출력}: llama.cpp {태그} — 실행 파일 {', '.join(f'{k} {v['size'] / 1048576:.1f} MB' for k, v in builds.items())} · 변환 파일 "
          f"{len(files)}개 {sum(x['size'] for x in files) / 1048576:.2f} MB · {torch[0] if torch else 'torch ?'}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(2)
    main(*sys.argv[1:3])
