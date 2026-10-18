#!/usr/bin/env bash
# 2단계 판정 B′ — 정밀 모드를 기준으로 F1 과 llama.cpp 의 숨은 손실(모든 행의 KL · 1위 일치)을 모델마다 잰다(시험/품질.py).
#   bash 재기/품질2.sh <작업 폴더(F:)> <결과 폴더>
# 큰 파일(로짓 — 모델마다 약 7 GB)은 작업 폴더에 두고, 견준 뒤 지운다. 결과 JSON · 기록만 결과 폴더에.
set -u
W=$1; OUT=$2
ROOT=$(cd "$(dirname "$0")/.." && pwd)
B=$ROOT/빌드/글; P=$ROOT/빌드/점검; LB=F:/geulpu-models/llama-b11719; M=F:/geulpu-models
mkdir -p "$OUT"
# 이름 | GGUF | 정밀 모듈 | 빠른 모듈 | 입력 폴더(품질.py 입력 — 토크나이저가 같은 모델끼리 같다)
MODELS=(
  "qwen7b-q8|$M/Qwen2.5-7B-Instruct-Q8_0.gguf|$P/qwen2Q8_7b.ptx|$P/qwen2Q8_7b_빠른.ptx|$W/qwen"
  "qwen7b-q4km|$M/Qwen2.5-7B-Instruct-Q4_K_M.gguf|$P/qwen2K_3584x28x4x18944x152064.ptx|$P/qwen2K_3584x28x4x18944x152064_빠른.ptx|$W/qwen"
  "llama8b-q4km|$M/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf|$P/llamaK_4096x32x8x14336x128256.ptx|$P/llamaK_4096x32x8x14336x128256_빠른.ptx|$W/llama"
)
for row in "${MODELS[@]}"; do
  IFS='|' read -r NAME GG PE PF IN <<< "$row"
  D=$W/$NAME; mkdir -p "$D/라마" "$D/정밀" "$D/빠른"
  cp "$IN/토큰.json" "$IN/입력.bin" "$D/라마/"
  echo "[$(date +%H:%M:%S)] $NAME · llama.cpp 로짓"
  python -I -X utf8 "$ROOT/시험/품질.py" 라마 "$GG" "$LB" "$D/라마" > "$OUT/라마_$NAME.log" 2>&1 || echo "  llama.cpp 실패 — $OUT/라마_$NAME.log"
  echo "[$(date +%H:%M:%S)] $NAME · 글 정밀"
  "$B/시험_7B.exe" "$GG" "$PE" "$IN/입력.bin" "$D/정밀" > "$OUT/정밀_$NAME.log" 2>&1 || echo "  정밀 실패 — $OUT/정밀_$NAME.log"
  echo "[$(date +%H:%M:%S)] $NAME · 글 빠른"
  "$B/시험_7B.exe" "$GG" "$PF" "$IN/입력.bin" "$D/빠른" > "$OUT/빠른_$NAME.log" 2>&1 || echo "  빠른 실패 — $OUT/빠른_$NAME.log"
  echo "[$(date +%H:%M:%S)] $NAME · 견주기"
  python -I -X utf8 "$ROOT/시험/품질.py" 견주기 "$D/라마" "$D/정밀" "$D/빠른" "$OUT/품질_$NAME.json" > "$OUT/견주기_$NAME.log" 2>&1 || echo "  견주기 실패 — $OUT/견주기_$NAME.log"
  tail -3 "$OUT/견주기_$NAME.log"
  rm -rf "$D"
done
echo "[$(date +%H:%M:%S)] 끝"
