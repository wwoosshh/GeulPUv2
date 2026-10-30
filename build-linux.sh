#!/bin/bash
# build-linux.sh — build GeulPU's command-line programs from source with ASCII names only (#137).
# For terminals that cannot show or type Korean (SSH, JupyterHub). The full app (GeulPU + bin/ + web/ + data/ + tar.gz)
# is 앱/꾸리기.sh; this script builds just the pieces a server needs:
#   bin/geulc        the geul compiler (from the pinned geul-lang commit in 시험/글저장소커밋.txt)
#   bin/ptx-backend  geul kernel source → PTX
#   bin/kernel-gen   kernel generator (one module per model shape and format)
#   bin/server-cuda  the inference server (OpenAI-compatible API)
#   bin/runner-cuda  the terminal runner (optional, same module as the server)
#
#   bash build-linux.sh [geul repo dir — default ../geul-lang-v2] [out dir — default bin]
#
# Needs: git, python3 (the compiler is bootstrapped from its Python reference implementation), an NVIDIA driver at run time
# (the binaries import only libc.so.6 and libcuda.so.1). Works in Git Bash on Windows too (.exe, python).
# Then, for example (Qwen2.5-7B, k-quants, the app's default fast mode):
#   bin/kernel-gen qwen2 K 3584 28 4 18944 152064 qwen2K_3584x28x4x18944x152064.gl --fast --KV8 --F2
#   bin/ptx-backend qwen2K_3584x28x4x18944x152064.gl -o qwen2K_3584x28x4x18944x152064.ptx
#   bin/server-cuda model.gguf --slots 4 --ctx 4096 --port 8080        # English names are aliases of the Korean ones: see --help
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
GEUL=${1:-$HERE/../geul-lang-v2}
OUT=${2:-$HERE/bin}
PIN=$(tr -d '[:space:]' < "$HERE/시험/글저장소커밋.txt")
EXE=""
PY=python3
case "$(uname -s)" in MINGW*|MSYS*|CYGWIN*) EXE=.exe; PY=python ;; esac
say() { echo "[$(date +%H:%M:%S)] $*"; }

if [ ! -d "$GEUL/.git" ]; then
  say "cloning geul-lang into $GEUL and checking out the pinned commit $PIN"
  git clone -q https://github.com/wwoosshh/geul-lang "$GEUL"
  git -C "$GEUL" checkout -q "$PIN"
else
  # An existing checkout is left as it is (it may be someone's working copy): only warn when it is not at the pinned commit.
  HEAD=$(git -C "$GEUL" rev-parse HEAD 2> /dev/null || echo unknown)
  [ "$HEAD" = "$PIN" ] || echo "warning: $GEUL is at $HEAD, not the pinned geul commit $PIN (시험/글저장소커밋.txt) — the binaries may differ from the release" >&2
fi
GEUL=$(cd "$GEUL" && pwd)
mkdir -p "$OUT"
OUT=$(cd "$OUT" && pwd)
export GEUL_ROOT=$GEUL PYTHONUTF8=1

GEULC=$OUT/geulc$EXE
if [ ! -x "$GEULC" ] || [ "$(cat "$OUT/geulc.commit" 2> /dev/null || true)" != "$PIN" ]; then
  say "building the geul compiler (Python reference implementation → self-hosted compiler)…"
  (cd "$GEUL" && $PY ref/geulc.py self/컴파일러.gl -o "$OUT/geulc_s1$EXE")
  (cd "$GEUL" && "$OUT/geulc_s1$EXE" self/컴파일러.gl -o "$GEULC")     # fixed point: the same source compiled by the compiler itself
  rm -f "$OUT/geulc_s1$EXE"
  echo "$PIN" > "$OUT/geulc.commit"
else
  say "geul compiler already built for $PIN — $GEULC"
fi

build() {  # build <name> <source.gl>
  say "building $1…"
  (cd "$HERE" && "$GEULC" "$2" -o "$OUT/$1$EXE")
}
build ptx-backend 백엔드/PTX/PTX.gl
build kernel-gen  커널/커널생성.gl
build server-cuda 엔진/서버.gl
build runner-cuda 엔진/실행기.gl
say "done: $OUT — kernel-gen --help · server-cuda --help list the options (Korean names with English aliases)"
