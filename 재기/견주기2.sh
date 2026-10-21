#!/usr/bin/env bash
# 2단계 — 프롬프트 시간 견주기: GeulPUv2 정밀 모드 · 빠른 모드(F1) · llama.cpp(최신 공식 판)를 모델마다, 바퀴마다 차례를 바꿔 잰다.
#   bash 재기/견주기2.sh <출력 폴더> <바퀴 수>
# - 글(geul) 쪽: 엔진/재기_프롬프트.gl — 프롬프트 + 첫 토큰(마지막 행의 탐욕 토큰), 일곱 번 × 세 묶음의 가운데.
# - llama.cpp: llama-bench -o json, -fa 0 · 1 × -ub 512 · 1024 · 2048(-b 8192), 다섯 번 — 요약(재기/견주기요약.py)이 표본(samples_ns)의
#   가운데 값으로 길이마다 가장 빠른 설정을 고른다.
# - 모델 폴더는 GEULPU_MODELS(기본 F:/geulpu-models), llama.cpp 는 GEULPU_LLAMA_BENCH(기본 F:/geulpu-models/llama-b11719) — 둘 다
#   ASCII 경로(llama.cpp 도구가 한글 경로를 못 연다). 재기 입력은 빌드/재기입력/.
set -u
OUT=$1; ROUNDS=${2:-1}
ROOT=$(cd "$(dirname "$0")/.." && pwd)
B=$ROOT/빌드/글; P=$ROOT/빌드/점검; IN=$ROOT/빌드/재기입력
LB=${GEULPU_LLAMA_BENCH:-F:/geulpu-models/llama-b11719}
M=${GEULPU_MODELS:-F:/geulpu-models}
mkdir -p "$OUT"
# 이름 | GGUF | 정밀 모듈(빌드/점검) | 빠른 모듈(빌드/글) | 재기 입력
MODELS=(
  "qwen7b-q4km|$M/Qwen2.5-7B-Instruct-Q4_K_M.gguf|$P/qwen2K_3584x28x4x18944x152064.ptx|$B/qwen2K_3584x28x4x18944x152064_빠른.ptx|$IN/재기Q.bin"
  "qwen7b-q8|$M/Qwen2.5-7B-Instruct-Q8_0.gguf|$P/qwen2Q8_7b.ptx|$B/qwen2Q8_7b_빠른.ptx|$IN/재기Q.bin"
  "llama8b-q4km|$M/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf|$P/llamaK_4096x32x8x14336x128256.ptx|$B/llamaK_4096x32x8x14336x128256_빠른.ptx|$IN/재기L.bin"
)
글() {  # 이름 모드 GGUF 모듈 입력 바퀴
  echo "[$(date +%H:%M:%S)] 바퀴 $6 · $1 · 글 $2"
  "$B/재기_프롬프트.exe" "$3" "$4" "$5" "$OUT/글${2}_${1}_$6.txt" > "$OUT/글${2}_${1}_$6.log" 2>&1 || echo "  글 실패 — $OUT/글${2}_${1}_$6.log"
}
라마() {  # 이름 GGUF 바퀴
  echo "[$(date +%H:%M:%S)] 바퀴 $3 · $1 · llama.cpp"
  "$LB/llama-bench.exe" -m "$2" -p 512,2048,4096,8192 -n 0 -ngl 99 -fa 0,1 -ub 512,1024,2048 -b 8192 -r 5 -o json \
    > "$OUT/라마_${1}_$3.json" 2> "$OUT/라마_${1}_$3.log" || echo "  llama.cpp 실패 — $OUT/라마_${1}_$3.log"
}
for r in $(seq 1 "$ROUNDS"); do
  for row in "${MODELS[@]}"; do
    IFS='|' read -r NAME GG PE PF IN1 <<< "$row"
    if [ $((r % 2)) -eq 1 ]; then
      글 "$NAME" 정밀 "$GG" "$PE" "$IN1" "$r"; 글 "$NAME" 빠른 "$GG" "$PF" "$IN1" "$r"; 라마 "$NAME" "$GG" "$r"
    else
      라마 "$NAME" "$GG" "$r"; 글 "$NAME" 빠른 "$GG" "$PF" "$IN1" "$r"; 글 "$NAME" 정밀 "$GG" "$PE" "$IN1" "$r"
    fi
  done
done
echo "[$(date +%H:%M:%S)] 끝"
