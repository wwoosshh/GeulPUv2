#!/usr/bin/env bash
# 2b단계 2b-1 — F1A 에뮬레이션의 품질(판정 B″ — 문서/2b단계.md): 모델마다 정밀 · F1 · F1+KV(· F1+QKV) 모듈로 엔진/시험_7B --한길 을 돌려,
# 정밀 모드 대비 KL · 1위 일치를 llama.cpp b11719(f16 KV · 참고로 q8_0 KV)와 견준다(시험/품질.py — 2,048 토큰 글 넷 + 8,192 창 하나).
#   bash 재기/품질2b.sh <작업 폴더(빠른 디스크 — ASCII 경로)> <결과 폴더> [<모델 이름> …]
# 모듈은 MODDIR(기본 작업 폴더/모듈)에 만든다(빌드/글/커널생성.exe + 글 저장소의 자체PTX — 있으면 그대로 쓴다). 로짓(모델마다 수십 GB)과
# 모델의 사본(작업 폴더 — 느린 디스크에서 한 번만 읽는다, COPY_MODEL=0 이면 그대로)은 견준 뒤 지운다.
# 모델은 고정 경로(F:\geulpu-models — llama.cpp 도구가 한글 경로를 못 연다). 변수 이름은 ASCII(윈도의 Git Bash).
# GeulPUv2 ③b-1(문서/긴문맥.md) — 변형 KV8 · V8 은 8비트 KV 에뮬레이션(커널생성 --어텐션한자리 KV8 · V8). VARIANTS="F1 KV8 V8" 처럼 주면
# 모델마다의 변형 목록 대신 그것을 쓴다.
set -u
W=$1; OUT=$2; shift 2
ROOT=$(cd "$(dirname "$0")/.." && pwd)
B=$ROOT/빌드/글; LB=F:/geulpu-models/llama-b11719; M=F:/geulpu-models
GL=${GEUL_REPO:-$ROOT/../geul-lang-v2}
MD=${MODDIR:-$W/모듈}; mkdir -p "$OUT" "$MD"
WT=$ROOT/빌드/wikitext-2/wiki.test.raw
# 이름 | GGUF | 커널생성의 모양 인자 | 변형(F1 · KV · QKV — 정밀은 늘)
MODELS=(
  "qwen7b-q4km|$M/Qwen2.5-7B-Instruct-Q4_K_M.gguf|qwen2큰 K 3584 28 4 18944 152064|F1 KV QKV"
  "llama8b-q4km|$M/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf|llama큰 K 4096 32 8 14336 128256|F1 KV QKV"
  "mistral7b-q4km|$M/Mistral-7B-Instruct-v0.3-Q4_K_M.gguf|llama큰 K 4096 32 8 14336 32768|F1 KV"
  "qwen3-8b-q4km|$M/Qwen3-8B-Q4_K_M.gguf|qwen3큰 K 4096 32 8 12288 151936|F1 KV"
  "qwen7b-q8|$M/Qwen2.5-7B-Instruct-Q8_0.gguf|qwen2Q8_7b|F1 KV"
  "llama8b-q8|$M/Meta-Llama-3.1-8B-Instruct-Q8_0.gguf|llama큰 Q8_0 4096 32 8 14336 128256|F1 KV"
)
mod() {  # 이름 모양인자 변형 → 모듈 경로(표준 출력)
  local p=$MD/$1_$3.ptx g=$MD/$1_$3.gl
  if [ ! -f "$p" ]; then
    local f=()
    case $3 in F1) f=(--빠른);; KV|QKV|KV8|V8) f=(--빠른 --어텐션한자리 "$3");; esac
    # shellcheck disable=SC2086  # 모양 인자는 낱말로 나뉘어야 한다
    if ! { "$B/커널생성.exe" $2 "$g" "${f[@]}" > "$MD/$1_$3.생성.log" 2>&1 && "$GL/build/자체PTX.exe" "$g" -o "$p" > "$MD/$1_$3.ptx.log" 2>&1; }; then
      echo "  모듈 실패 — $MD/$1_$3.*.log" >&2
      rm -f "$p"
    fi
  fi
  echo "$p"
}
for row in "${MODELS[@]}"; do
  IFS='|' read -r NAME GG ARGS VARS <<< "$row"
  VARS=${VARIANTS:-$VARS}
  if [ $# -gt 0 ] && ! printf '%s\n' "$@" | grep -qx "$NAME"; then continue; fi
  D=$W/$NAME; mkdir -p "$D/라마"
  MCOPY=$W/$NAME.gguf
  if [ "${COPY_MODEL:-1}" = 1 ]; then
    echo "[$(date +%H:%M:%S)] $NAME · 모델 사본"
    cp "$GG" "$MCOPY" && GG=$MCOPY
  fi
  echo "[$(date +%H:%M:%S)] $NAME · 입력 · llama.cpp(f16 KV · q8_0 KV)"
  python -I -X utf8 "$ROOT/시험/품질.py" 입력 "$GG" "$LB" "$D/라마" --글 "$WT" --글 "$WT@400000" --글 "$WT@800000" \
    --글 "$ROOT/연구자안내.md" --긴 "$WT@1100000" > "$OUT/입력_$NAME.log" 2>&1 || echo "  입력 실패 — $OUT/입력_$NAME.log"
  python -I -X utf8 "$ROOT/시험/품질.py" 라마 "$GG" "$LB" "$D/라마" > "$OUT/라마_$NAME.log" 2>&1 || echo "  llama.cpp 실패 — $OUT/라마_$NAME.log"
  python -I -X utf8 "$ROOT/시험/품질.py" 라마 "$GG" "$LB" "$D/라마" --kv q8_0 > "$OUT/라마q8_$NAME.log" 2>&1 || echo "  llama.cpp q8_0 KV 실패 — $OUT/라마q8_$NAME.log"
  PAIRS=()
  for V in 정밀 $VARS; do
    P=$(mod "$NAME" "$ARGS" "$V")
    echo "[$(date +%H:%M:%S)] $NAME · 글 $V"
    mkdir -p "$D/$V"
    "$B/시험_7B.exe" "$GG" "$P" "$D/라마/입력.bin" "$D/$V" --한길 > "$OUT/글${V}_$NAME.log" 2>&1 || echo "  글 실패 — $OUT/글${V}_$NAME.log"
    if [ "$V" != 정밀 ]; then PAIRS+=("$V=$D/$V"); fi
  done
  echo "[$(date +%H:%M:%S)] $NAME · 견주기"
  python -I -X utf8 "$ROOT/시험/품질.py" 견주기 "$D/라마" "$D/정밀" "$OUT/품질_$NAME.json" "${PAIRS[@]}" > "$OUT/견주기_$NAME.log" 2>&1 || echo "  견주기 실패 — $OUT/견주기_$NAME.log"
  grep "^판정" "$OUT/견주기_$NAME.log"
  rm -rf "$D"
  rm -f "$MCOPY"
done
echo "[$(date +%H:%M:%S)] 끝"
