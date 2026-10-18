#!/usr/bin/env bash
# 2단계 판정 C 의 생성 쪽 — 정밀 모드와 빠른 모드(F1)의 생성 ms/토큰(엔진/재기_7B: 채팅 질문 · 문맥 2048 · 8192, 일곱 번 × 세 묶음의
# 가운데)을 모델마다, 바퀴마다 차례를 바꿔 잰다. 판정: F1 이 정밀의 ±2 % 안(생성은 가중치를 읽는 시간이 정한다).
#   bash 재기/생성2.sh <출력 폴더> <바퀴 수>
# 모델 · 모듈 · 입력은 재기/견주기2.sh 와 같다.
set -u
OUT=$1; ROUNDS=${2:-2}
ROOT=$(cd "$(dirname "$0")/.." && pwd)
B=$ROOT/빌드/글; P=$ROOT/빌드/점검; IN=$ROOT/빌드/재기입력
M=F:/geulpu-models
mkdir -p "$OUT"
# 이름 | GGUF | 정밀 모듈 | 빠른 모듈 | 재기 입력
MODELS=(
  "qwen7b-q4km|$M/Qwen2.5-7B-Instruct-Q4_K_M.gguf|$P/qwen2K_3584x28x4x18944x152064.ptx|$B/qwen2K_3584x28x4x18944x152064_빠른.ptx|$IN/재기Q.bin"
  "qwen7b-q8|$M/Qwen2.5-7B-Instruct-Q8_0.gguf|$P/qwen2Q8_7b.ptx|$B/qwen2Q8_7b_빠른.ptx|$IN/재기Q.bin"
  "llama8b-q4km|$M/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf|$P/llamaK_4096x32x8x14336x128256.ptx|$B/llamaK_4096x32x8x14336x128256_빠른.ptx|$IN/재기L.bin"
)
글() {  # 이름 모드 GGUF 모듈 입력 바퀴
  echo "[$(date +%H:%M:%S)] 바퀴 $6 · $1 · 글 $2"
  "$B/재기_7B.exe" "$3" "$4" "$5" "$OUT/생성${2}_${1}_$6.txt" > "$OUT/생성${2}_${1}_$6.log" 2>&1 || echo "  글 실패 — $OUT/생성${2}_${1}_$6.log"
}
for r in $(seq 1 "$ROUNDS"); do
  for row in "${MODELS[@]}"; do
    IFS='|' read -r NAME GG PE PF IN1 <<< "$row"
    if [ $((r % 2)) -eq 1 ]; then
      글 "$NAME" 정밀 "$GG" "$PE" "$IN1" "$r"; 글 "$NAME" 빠른 "$GG" "$PF" "$IN1" "$r"
    else
      글 "$NAME" 빠른 "$GG" "$PF" "$IN1" "$r"; 글 "$NAME" 정밀 "$GG" "$PE" "$IN1" "$r"
    fi
  done
done
echo "[$(date +%H:%M:%S)] 끝"
