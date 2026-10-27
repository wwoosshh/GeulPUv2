#!/bin/bash
# GeulPU 앱 꾸리기 — 리눅스 판(x86-64 · NVIDIA CUDA). 앱/꾸리기.ps1(윈도)과 같은 차례로 소스에서 릴리스 폴더와 tar.gz 를 만든다.
#
#   bash 앱/꾸리기.sh <글 저장소(고정 커밋 — 시험/글저장소커밋.txt)> [출력 폴더] [--묶지않음]
#
# 1. 글 컴파일러: 파이썬 참조 구현 → 글로 쓴 컴파일러 → 같은 소스로 한 번 더(고정점).
# 2. 빌드: GeulPU(앱/관리자.gl) · bin/ 의 server-cuda · device-cuda · diag-cuda · kernel-gen · ptx-backend, kernels/diag.ptx.
#    Vulkan(server-vulkan · device-vulkan)은 리눅스에서 아직 없다(#135).
# 3. web/ · data/(받을 모델 목록 · 교차검증 기준) · LICENSE · README.txt, version.json(판 · 커밋 · 빌드 때 · 파일마다 SHA-256).
# 4. GeulPU-<판>-linux-x64.tar.gz 와 SHA256SUMS.txt.
# 윈도의 꾸리기가 견주는 교차검증 기준의 빌드 해시(expected.json 의 build)는 윈도 실행 파일의 것이라 견주지 않는다.
set -euo pipefail
if [ $# -lt 1 ]; then echo "쓰는 법: bash 앱/꾸리기.sh <글 저장소> [출력 폴더] [--묶지않음]" >&2; exit 1; fi
GEUL=$(cd "$1" && pwd)
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/.." && pwd)
OUT=${2:-$ROOT/빌드/앱-리눅스}
NOTAR=0
for a in "$@"; do [ "$a" = "--묶지않음" ] && NOTAR=1; done
[ "$OUT" = "--묶지않음" ] && OUT=$ROOT/빌드/앱-리눅스
mkdir -p "$OUT"
OUT=$(cd "$OUT" && pwd)
LOGS=$OUT/logs
TMP=$OUT/tmp
mkdir -p "$LOGS" "$TMP"
say() { echo "[$(date +%H:%M:%S)] $*"; }
run() {  # run <이름> <폴더> <명령…>
  local name=$1 cwd=$2
  shift 2
  if ! (cd "$cwd" && "$@") > "$LOGS/$name.log" 2>&1; then
    tail -20 "$LOGS/$name.log" >&2
    echo "멈춤: $name 실패 — $LOGS/$name.log" >&2
    exit 1
  fi
}

VER=$(sed -n 's/.*앱판 = "\([0-9][0-9.]*[^"]*\)".*/\1/p' "$HERE/판.gl" | head -1)
[ -n "$VER" ] || { echo "멈춤: 앱/판.gl 에서 판을 읽지 못했습니다" >&2; exit 1; }
COMMIT=$(git -C "$ROOT" rev-parse HEAD 2> /dev/null || true)
DIRTY=false
[ -n "$(git -C "$ROOT" status --porcelain -- 엔진 커널 앱 교차검증/expected.json 교차검증/requests.json 2> /dev/null || true)" ] && DIRTY=true
PIN=$(tr -d '[:space:]' < "$ROOT/시험/글저장소커밋.txt")
GEULHEAD=$(git -C "$GEUL" rev-parse HEAD 2> /dev/null || true)
say "GeulPU $VER · 커밋 $COMMIT$([ "$DIRTY" = true ] && echo ' (고친 파일 있음)') · 글 저장소 $GEULHEAD (고정 커밋 $PIN)"
[ -n "$GEULHEAD" ] && [ "$GEULHEAD" != "$PIN" ] && echo "경고: 글 저장소가 고정 커밋이 아닙니다" >&2
export GEUL_ROOT=$GEUL PYTHONUTF8=1

say "글 컴파일러를 빌드합니다(파이썬 참조 구현 → 글로 쓴 컴파일러 → 고정점)..."
run compiler-s1 "$GEUL" python3 ref/geulc.py self/컴파일러.gl -o "$TMP/geulc_s1"
run compiler-s2 "$GEUL" "$TMP/geulc_s1" self/컴파일러.gl -o "$TMP/geulc"
COMP=$TMP/geulc

DIST=$OUT/GeulPU-$VER
rm -rf "$DIST"
mkdir -p "$DIST/bin" "$DIST/kernels" "$DIST/web" "$DIST/data/crosscheck/pair"
say "빌드..."
run ptx-backend "$ROOT" "$COMP" 백엔드/PTX/PTX.gl -o "$DIST/bin/ptx-backend"
run app "$ROOT" "$COMP" 앱/관리자.gl -o "$DIST/GeulPU"
run server-cuda "$ROOT" "$COMP" 엔진/서버.gl -o "$DIST/bin/server-cuda"
run device-cuda "$ROOT" "$COMP" 엔진/장치C.gl -o "$DIST/bin/device-cuda"
run diag-cuda "$ROOT" "$COMP" 엔진/진단.gl -o "$DIST/bin/diag-cuda"
run kernel-gen "$ROOT" "$COMP" 커널/커널생성.gl -o "$DIST/bin/kernel-gen"
run kernel-diag-ptx "$ROOT" "$DIST/bin/ptx-backend" 커널/진단.gl -o "$DIST/kernels/diag.ptx"

cp -r "$HERE/web/." "$DIST/web/"
cp "$HERE/data/models.json" "$DIST/data/"
cp "$ROOT/교차검증/expected.json" "$ROOT/교차검증/requests.json" "$DIST/data/crosscheck/"
for f in models.json requests.json expected.json; do cp "$ROOT/교차검증/쌍/$f" "$DIST/data/crosscheck/pair/"; done
cp "$ROOT/LICENSE" "$DIST/"
cp "$HERE/읽어보기.txt" "$DIST/README.txt"

say "version.json..."
python3 - "$DIST" "$VER" "$COMMIT" "$DIRTY" "$GEULHEAD" "$ROOT" << 'EOF'
import datetime, hashlib, json, os, sys
dist, ver, commit, dirty, geul, root = sys.argv[1:7]
def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()
files = {}
for d, _, fs in os.walk(dist):
    for f in fs:
        p = os.path.join(d, f)
        files[os.path.relpath(p, dist).replace(os.sep, "/")] = sha(p)
v = {"name": "GeulPU", "version": ver, "commit": commit, "dirty": dirty == "true",
     "built": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), "geul_commit": geul,
     "platform": "linux-x64", "contract": sha(os.path.join(root, "계약", "계약.md")),
     "contract_reference": sha(os.path.join(root, "계약", "정수계약.py")), "files": dict(sorted(files.items()))}
with open(os.path.join(dist, "version.json"), "w", encoding="utf-8", newline="\n") as f:
    json.dump(v, f, ensure_ascii=False, indent=2)
EOF
rm -rf "$TMP"
if [ "$NOTAR" = 1 ]; then say "꾸림: $DIST"; exit 0; fi
TGZ=GeulPU-$VER-linux-x64.tar.gz
# 권한은 파일 시스템과 상관없이 정한다(WSL 의 /mnt/* 에서는 모든 파일이 0777 로 보이고, 빌드한 실행 파일에 실행 권한이 없을 수도 있다):
# 실행 파일(GeulPU · bin/*) 0755, 나머지 0644, 폴더 0755, 주인 0:0, 이름 차례. 휠(도구/pypi/휠만들기.py)이 이 권한을 그대로 옮긴다
python3 - "$OUT" "GeulPU-$VER" "$TGZ" << 'EOF'
import os, sys, tarfile
out, top, tgz = sys.argv[1:4]
def norm(ti):
    ti.uid = ti.gid = 0
    ti.uname = ti.gname = ""
    rel = ti.name.split("/", 1)[1] if "/" in ti.name else ""
    ti.mode = 0o755 if ti.isdir() or rel == "GeulPU" or rel.startswith("bin/") else 0o644
    return ti
with tarfile.open(os.path.join(out, tgz), "w:gz") as t:
    for d, ds, fs in os.walk(os.path.join(out, top)):
        ds.sort()
        rel = os.path.relpath(d, out).replace(os.sep, "/")
        t.add(d, arcname=rel, recursive=False, filter=norm)
        for f in sorted(fs):
            t.add(os.path.join(d, f), arcname=rel + "/" + f, recursive=False, filter=norm)
EOF
(cd "$OUT" && sha256sum "$TGZ" > SHA256SUMS.txt)
say "꾸림: $OUT/$TGZ"
cat "$OUT/SHA256SUMS.txt"
