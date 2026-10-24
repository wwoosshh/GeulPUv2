#!/usr/bin/env bash
# 2b단계 2b-1 — F1A 에뮬레이션의 품질(판정 B″ — 문서/2b단계.md): 모델마다 정밀 · F1 · F1+KV(· F1+QKV) 모듈로 엔진/시험_7B --한길 을 돌려,
# 정밀 모드 대비 KL · 1위 일치를 llama.cpp b11719(f16 KV · 참고로 q8_0 KV)와 견준다(시험/품질.py — 2,048 토큰 글 넷 + 8,192 창 하나).
#   bash 재기/품질2b.sh <작업 폴더(빠른 디스크 — ASCII 경로)> <결과 폴더> [<모델 이름> …]
# 모듈은 MODDIR(기본 작업 폴더/모듈)에 만든다(빌드/글/커널생성.exe + 빌드/글/ptx-backend.exe — 문서/개발.md 의 '소스에서 빌드하기' 2. 있으면 그대로 쓴다). 로짓(모델마다 수십 GB)과
# 모델의 사본(작업 폴더 — 느린 디스크에서 한 번만 읽는다, COPY_MODEL=0 이면 그대로)은 견준 뒤 지운다.
# 모델 폴더는 GEULPU_MODELS(기본 F:/geulpu-models), llama.cpp 는 GEULPU_LLAMA_BENCH(기본 F:/geulpu-models/llama-b11719) — 둘 다 ASCII
# 경로(llama.cpp 도구가 한글 경로를 못 연다). 변수 이름은 ASCII(윈도의 Git Bash).
# GeulPUv2 ③b-2(문서/긴문맥.md) — 변형 KV8 은 실제 8비트 KV 모듈(커널생성 --빠른 --KV8 — 모듈 이름 끝 _KV8), KV8EMU · V8EMU 는 ③b-1 의
# 에뮬레이션(커널생성 --어텐션한자리 KV8 · V8 — 이름 끝이 _KV8 이 아니어서 엔진이 16비트 KV 로 올린다). VARIANTS="F1 KV8" 처럼 주면
# 모델마다의 변형 목록 대신 그것을 쓴다.
# 빠른 어텐션 F2(문서/빠른어텐션.md) — 변형 F2_KV8 은 F2 모듈(커널생성 --빠른 --F2 — 이름 끝 _F2_KV8).
# 1.0.0 1-4 — 변형 F2B_KV8 은 F2B 모듈(커널생성 --빠른 --F2B — q 의 지수를 블록마다, 이름 끝 _F2B_KV8).
set -u
W=$1; OUT=$2; shift 2
ROOT=$(cd "$(dirname "$0")/.." && pwd)
B=$ROOT/빌드/글; LB=${GEULPU_LLAMA_BENCH:-F:/geulpu-models/llama-b11719}; M=${GEULPU_MODELS:-F:/geulpu-models}
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
  # 0.12.0 12-1 — 채팅 템플릿으로 열린 모델(앱의 기본 — 8비트 KV · 빠른 어텐션)
  "r1llama8b-q4km|$M/DeepSeek-R1-Distill-Llama-8B-Q4_K_M.gguf|llama큰 K 4096 32 8 14336 128256|F1 KV8 F2_KV8"
  "hcx1.5b-q4km|$M/hyperclovax-seed-text-instruct-1.5b-q4_k_m.gguf|llama큰 K 2048 16 8 7168 110592|F1 KV8 F2_KV8"
  "minicpm5-2b-q4km|$M/MiniCPM5-2B-Q4_K_M.gguf|llamae6큰 K 2048 16 2 6144 130560|F1 KV8 F2_KV8"     # 12-1b — eps 10⁻⁶(모듈 llamae6)
  # 0.12.0 12-2 — 모델 목록에 넣는 모델(qwen3 구조의 빠른 어텐션은 F2B — 1.0.0 1-4)
  "qwen3-0.6b-q8|$M/Qwen3-0.6B-Q8_0.gguf|qwen3큰 Q8_0 1024 16 8 3072 151936|F1 KV8 F2B_KV8"
  "qwen3-4b-q4km|$M/Qwen3-4B-Q4_K_M.gguf|qwen3큰 K 2560 32 8 9728 151936|F1 KV8 F2B_KV8"
  "qwen3-4b-q8|$M/Qwen3-4B-Q8_0.gguf|qwen3큰 Q8_0 2560 32 8 9728 151936|F1 KV8 F2B_KV8"
  "qwen2.5-1.5b-q8|$M/qwen2.5-1.5b-instruct-q8_0.gguf|qwen2큰 Q8_0 1536 12 2 8960 151936|F1 KV8 F2_KV8"
  "qwen2.5-3b-q8|$M/qwen2.5-3b-instruct-q8_0.gguf|qwen2큰 Q8_0 2048 16 2 11008 151936|F1 KV8 F2_KV8"
  # 14B 는 덩이 1024 행이면 12 GB 에 문맥 8,192 가 들지 않는다(2,688 로 줄어 긴 창이 넘친다 — 256 행도 16비트 KV 는 7,936) — 덩이 128 행
  # (다섯째 칸 — 비트는 덩이와 상관없다)
  "qwen2.5-coder14b-q4km|$M/qwen2.5-coder-14b-instruct-q4_k_m.gguf|qwen2큰 K 5120 40 8 13824 152064|F1 KV8 F2_KV8|128"
  # Qwen3.5(구조 qwen35 — 빠른 모드 F1, 구조/qwen35/기록.md): 헤드 256 의 8비트 KV · 빠른 어텐션 판이 없어 F1 만
  "qwen3.5-0.8b-q8|$M/Qwen3.5-0.8B-Q8_0.gguf|qwen35큰 Q8_0 1024 8 2 3584 248320|F1"
  "qwen3.5-4b-q4km|$M/Qwen3.5-4B-Q4_K_M.gguf|qwen35큰 K 2560 16 4 9216 248320|F1"
  "qwen3.5-9b-q4km|$M/Qwen3.5-9B-Q4_K_M.gguf|qwen35큰 K 4096 16 4 12288 248320|F1"
)
mod() {  # 이름 모양인자 변형 → 모듈 경로(표준 출력)
  local p=$MD/$1_$3.ptx g=$MD/$1_$3.gl
  if [ ! -f "$p" ]; then
    local f=()
    case $3 in F1) f=(--빠른);; KV8) f=(--빠른 --KV8);; KV|QKV) f=(--빠른 --어텐션한자리 "$3");; KV8EMU|V8EMU) f=(--빠른 --어텐션한자리 "${3%EMU}");;
      F2_KV8) f=(--빠른 --F2);; F2B_KV8) f=(--빠른 --F2B);; esac
    # shellcheck disable=SC2086  # 모양 인자는 낱말로 나뉘어야 한다
    if ! { "$B/커널생성.exe" $2 "$g" "${f[@]}" > "$MD/$1_$3.생성.log" 2>&1 && "$B/ptx-backend.exe" "$g" -o "$p" > "$MD/$1_$3.ptx.log" 2>&1; }; then
      echo "  모듈 실패 — $MD/$1_$3.*.log" >&2
      rm -f "$p"
    fi
  fi
  echo "$p"
}
for row in "${MODELS[@]}"; do
  IFS='|' read -r NAME GG ARGS VARS ROWS <<< "$row"
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
    "$B/시험_7B.exe" "$GG" "$P" "$D/라마/입력.bin" "$D/$V" --한길 ${ROWS:+--최대행 "$ROWS"} > "$OUT/글${V}_$NAME.log" 2>&1 || echo "  글 실패 — $OUT/글${V}_$NAME.log"
    if [ "$V" != 정밀 ]; then PAIRS+=("$V=$D/$V"); fi
  done
  echo "[$(date +%H:%M:%S)] $NAME · 견주기"
  python -I -X utf8 "$ROOT/시험/품질.py" 견주기 "$D/라마" "$D/정밀" "$OUT/품질_$NAME.json" "${PAIRS[@]}" > "$OUT/견주기_$NAME.log" 2>&1 || echo "  견주기 실패 — $OUT/견주기_$NAME.log"
  grep "^판정" "$OUT/견주기_$NAME.log"
  rm -rf "$D"
  rm -f "$MCOPY"
done
echo "[$(date +%H:%M:%S)] 끝"
