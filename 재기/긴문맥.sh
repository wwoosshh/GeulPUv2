#!/usr/bin/env bash
# GeulPUv2 — 긴 문맥 실사용 재기(문서/긴문맥재기.md): 글 엔진(엔진/재기_긴문맥 — F1 모듈)과 llama.cpp(llama-bench)를 같은 꼴로.
#   (가) 빈 캐시 프롬프트 N, (나) 깊이 d 에서 새 입력 1,024 토큰, (다) 깊이 d 에서 생성(글: 표 걸음 행 1 · 8, llama.cpp: tg128).
#   bash 재기/긴문맥.sh <결과 폴더> [모델 이름 …]
# 모델 폴더는 GEULPU_MODELS(기본 F:/geulpu-models), llama.cpp 는 GEULPU_LLAMA_BENCH(기본 F:/geulpu-models/llama-b11721) — 둘 다 ASCII
# 경로(llama.cpp 도구가 한글 경로를 못 연다). 깊이는 32k 안(Qwen2.5-7B 의 GGUF 위치 수가 32,768 — 글 엔진은 그 너머를
# 받지 않는다(YaRN 없음), Llama-3.1-8B 는 KV 가 토큰당 133 KB 라 12 GB 에 32k 남짓) — 두 모델 같은 칸: 빈 16k · 32k, 깊이 8k · 16k · 31k.
# 글 엔진의 생성 걸음은 모든 깊이에서(4b단계 — 고치기 앞의 모듈은 생성 길의 접기가 위치 10,240 부터 틀려 --생성끝 10240 으로 쟀다).
set -u
OUT=$1; shift
ROOT=$(cd "$(dirname "$0")/.." && pwd)
B=$ROOT/빌드/글; IN=$ROOT/빌드/재기입력; M=${GEULPU_MODELS:-F:/geulpu-models}
LB=${GEULPU_LLAMA_BENCH:-F:/geulpu-models/llama-b11721}
mkdir -p "$OUT"
# 이름 | GGUF | 빠른 모듈 | 재기 입력 | 최대길이 | 빈 N | 깊이 d
MODELS=(
  "qwen7b-q4km|$M/Qwen2.5-7B-Instruct-Q4_K_M.gguf|$B/qwen2K_3584x28x4x18944x152064_빠른.ptx|$IN/재기Q_긴.bin|32768|16384,32768|8192,16384,31744"
  "llama8b-q4km|$M/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf|$B/llamaK_4096x32x8x14336x128256_빠른.ptx|$IN/재기L_긴.bin|33856|16384,32768|8192,16384,31744"
)
for row in "${MODELS[@]}"; do
  IFS='|' read -r NAME GG PT IN1 L PN DD <<< "$row"
  if [ $# -gt 0 ] && ! printf '%s\n' "$@" | grep -qx "$NAME"; then continue; fi
  echo "[$(date +%H:%M:%S)] $NAME · 글"
  "$B/재기_긴문맥.exe" "$GG" "$PT" "$IN1" "$OUT/글_$NAME.txt" "$L" --빈 "$PN" --깊이 "$DD" > "$OUT/글_$NAME.log" 2>&1 || echo "  글 실패 — $OUT/글_$NAME.log"
  echo "[$(date +%H:%M:%S)] $NAME · llama.cpp 빈"
  "$LB/llama-bench.exe" -m "$GG" -p "$PN" -n 0 -ngl 99 -fa 1 -ub 512,1024,2048 -b 8192 -r 3 -o json \
    > "$OUT/라마빈_$NAME.json" 2> "$OUT/라마빈_$NAME.log" || echo "  llama.cpp 빈 실패 — $OUT/라마빈_$NAME.log"
  echo "[$(date +%H:%M:%S)] $NAME · llama.cpp 깊이"
  "$LB/llama-bench.exe" -m "$GG" -p 1024 -n 128 -d "$DD" -ngl 99 -fa 1 -ub 512,1024,2048 -b 8192 -r 3 -o json \
    > "$OUT/라마깊이_$NAME.json" 2> "$OUT/라마깊이_$NAME.log" || echo "  llama.cpp 깊이 실패 — $OUT/라마깊이_$NAME.log"
done
echo "[$(date +%H:%M:%S)] 끝"
