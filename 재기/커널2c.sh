#!/usr/bin/env bash
# 2c단계 2c-0 — 커널마다 시간(엔진/재기_커널 — 계획의 띄움을 하나씩 이벤트 사이에, 세 번의 가운데): 정밀 · 빠른(F1) 모듈, 모델마다
# 512 · 2048 · 8192 토큰. 행렬곱(qkv · o · gate·up · down)이 int8 텐서 코어 바닥에서 얼마나 먼지를 본다(문서/2c단계.md).
#   bash 재기/커널2c.sh <출력 폴더>
# 모델 · 모듈 · 입력은 재기/견주기2.sh 와 같다(정밀 모듈은 빌드/점검, 빠른 모듈은 빌드/글, 모델 폴더는 GEULPU_MODELS — 기본 F:/geulpu-models).
set -u
OUT=$1
ROOT=$(cd "$(dirname "$0")/.." && pwd)
B=$ROOT/빌드/글; P=$ROOT/빌드/점검; IN=$ROOT/빌드/재기입력; M=${GEULPU_MODELS:-F:/geulpu-models}
mkdir -p "$OUT"
MODELS=(
  "qwen7b-q4km|$M/Qwen2.5-7B-Instruct-Q4_K_M.gguf|$P/qwen2K_3584x28x4x18944x152064.ptx|$B/qwen2K_3584x28x4x18944x152064_빠른.ptx|$IN/재기Q.bin"
  "qwen7b-q8|$M/Qwen2.5-7B-Instruct-Q8_0.gguf|$P/qwen2Q8_7b.ptx|$B/qwen2Q8_7b_빠른.ptx|$IN/재기Q.bin"
  "llama8b-q4km|$M/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf|$P/llamaK_4096x32x8x14336x128256.ptx|$B/llamaK_4096x32x8x14336x128256_빠른.ptx|$IN/재기L.bin"
)
for row in "${MODELS[@]}"; do
  IFS='|' read -r NAME GG PE PF IN1 <<< "$row"
  for MODE in 정밀 빠른; do
    PT=$PE; [ "$MODE" = 빠른 ] && PT=$PF
    echo "[$(date +%H:%M:%S)] $NAME · $MODE"
    "$B/재기_커널.exe" "$GG" "$PT" "$IN1" "$OUT/커널${MODE}_$NAME.txt" 512 2048 8192 > "$OUT/커널${MODE}_$NAME.log" 2>&1 || echo "  실패 — $OUT/커널${MODE}_$NAME.log"
  done
done
echo "[$(date +%H:%M:%S)] 끝"
