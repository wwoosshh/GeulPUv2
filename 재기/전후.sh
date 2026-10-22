#!/bin/bash
# 1.0.0 의 측정 묶음(문서/1.0.0.md) — 1-0 에서 0.10.2 를, 1-5 에서 1.0.0 과 0.10.2 를 같은 것으로 잰다. RTX 4070 Ti 기준.
#
#   bash 재기/전후.sh <도구 폴더> <이름표> <할 것: E,M,L 가운데> [llama.cpp 폴더]
#
# 도구 폴더: 서버 · 재기_긴문맥 · 커널생성 · ptx-backend 와 mod/<정식 모듈 이름>.ptx(정밀 · _빠른 · _빠른_KV8 · _빠른_F2_KV8).
# 결과: $W/<이름표>/ — E(에이전트 재생 json) · M(재기_긴문맥 출력) · L(llama-bench json) · log · 진행.txt. 요약은 재기/전후요약.py.
# 경로는 환경 변수로 바꾼다: GP_MODELS(Q4_K_M 둘) · GP_Q8(Qwen2.5-7B Q8_0) · GP_SESS(에이전트 기록 셋) · GP_WORK(결과의 위).
# 변수 · 함수 이름은 ASCII(이 PC 의 Git Bash 는 한글 이름을 받지 않는다).
set -u
T=$1
TAG=$2
WHAT=$3
LB=${4:-}
V2=$(cd "$(dirname "$0")/.." && pwd)
MD=${GP_MODELS:-/e/geulpu-work/f2/models}
Q8=${GP_Q8:-/e/geulpu-work/v1/models/Qwen2.5-7B-Instruct-Q8_0.gguf}
SESS=${GP_SESS:-/e/geulpu-work/moe/sess}
W=${GP_WORK:-/e/geulpu-work/v1}/$TAG
IN=$V2/빌드/재기입력
export PYTHONUTF8=1
mkdir -p "$W/E" "$W/M" "$W/L" "$W/log"
step() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$W/진행.txt"; }
QG=$MD/Qwen2.5-7B-Instruct-Q4_K_M.gguf
LG=$MD/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf
QM=qwen2K_3584x28x4x18944x152064
LM=llamaK_4096x32x8x14336x128256
SESSF=("$SESS/t1_todo.jsonl" "$SESS/t2_ledger.jsonl" "$SESS/t3_memo.jsonl")

PID=
wait_up() {  # port
  for i in $(seq 1 200); do
    sleep 3
    if curl -sf -m 2 "http://127.0.0.1:$1/v1/models" > /dev/null 2>&1; then return 0; fi
    if ! kill -0 $PID 2>/dev/null; then step "  서버가 끝났다"; return 1; fi
  done
  step "  서버 준비 시간 초과"; return 1
}
stop_srv() { kill $PID 2>/dev/null; sleep 3; kill -9 $PID 2>/dev/null; wait $PID 2>/dev/null; sleep 2; }

# ---- E 에이전트 재생 ----
replay() {  # name port extra-args...
  local name=$1 port=$2; shift 2
  step "E $name"
  python -I -X utf8 "$V2/재기/에이전트재생.py" "http://127.0.0.1:$port" "$W/E/$name.json" "${SESSF[@]}" --최대 1024 --이름 qwen "$@" \
    > "$W/log/E_$name.log" 2>&1 || step "  실패 — $W/log/E_$name.log"
  tail -1 "$W/log/E_$name.log" | tee -a "$W/진행.txt"
}
if [[ $WHAT == *E* ]]; then
  for v in 빠른 빠른_F2_KV8; do
    "$T/서버.exe" "$QG" --ptx "$T/mod/${QM}_$v.ptx" --슬롯 1 --문맥 40960 --포트 8095 --이름 qwen > "$W/log/server_E_$v.log" 2>&1 &
    PID=$!
    if wait_up 8095; then replay "geulpu_$v" 8095 --geulpu; fi
    stop_srv
  done
  if [ -n "$LB" ]; then
    for kv in f16 q8_0; do
      "$LB/llama-server.exe" -m "$QG" -c 40960 -np 1 -ngl 99 -fa on --jinja -ctk $kv -ctv $kv --port 8096 > "$W/log/llama_server_E_$kv.log" 2>&1 &
      PID=$!
      if wait_up 8096; then replay "llama_$kv" 8096; fi
      stop_srv
    done
  fi
fi

# ---- M 엔진 측정(재기_긴문맥 — 청크 512 행) ----
bench() {  # name gguf module input maxlen empties depths
  step "M $1"
  "$T/재기_긴문맥.exe" "$2" "$T/mod/$3.ptx" "$4" "$W/M/$1.txt" "$5" --빈 "$6" --깊이 "$7" --rows 512 > "$W/log/M_$1.log" 2>&1 \
    || step "  실패 — $W/log/M_$1.log"
}
if [[ $WHAT == *M* ]]; then
  FULLE=512,2048,8192,32768,65536; FULLD=1024,8192,32768,65536
  bench qwenK_정밀 "$QG" "$QM" "$IN/재기Q_긴.bin" 66560 512,2048,8192,32768 1024,8192,32768
  bench qwenK_빠름 "$QG" "${QM}_빠른" "$IN/재기Q_긴.bin" 66560 $FULLE $FULLD
  bench qwenK_KV8 "$QG" "${QM}_빠른_KV8" "$IN/재기Q_긴.bin" 131072 $FULLE $FULLD
  bench qwenK_F2 "$QG" "${QM}_빠른_F2_KV8" "$IN/재기Q_긴.bin" 131072 $FULLE $FULLD
  bench llamaK_정밀 "$LG" "$LM" "$IN/재기L_긴.bin" 33856 512,2048,8192,32768 1024,8192,32768
  bench llamaK_빠름 "$LG" "${LM}_빠른" "$IN/재기L_긴.bin" 33856 512,2048,8192,32768 1024,8192,32768
  bench llamaK_KV8 "$LG" "${LM}_빠른_KV8" "$IN/재기L_긴.bin" 66560 $FULLE $FULLD
  bench llamaK_F2 "$LG" "${LM}_빠른_F2_KV8" "$IN/재기L_긴.bin" 66560 $FULLE $FULLD
  bench qwenQ8_정밀 "$Q8" qwen2Q8_7b "$IN/재기Q_긴.bin" 16384 512,2048,8192 1024,8192
  bench qwenQ8_빠름 "$Q8" qwen2Q8_7b_빠른 "$IN/재기Q_긴.bin" 16384 512,2048,8192 1024,8192
fi

# ---- L llama.cpp llama-bench ----
lbench() {  # name gguf kv prompts depths
  step "L $1"
  "$LB/llama-bench.exe" -m "$2" -p "$4" -n 0 -ngl 99 -fa 1 -ub 512,1024,2048 -b 8192 -r 3 -ctk $3 -ctv $3 -o json \
    > "$W/L/$1_빈.json" 2> "$W/log/L_$1_빈.log" || step "  실패 빈"
  "$LB/llama-bench.exe" -m "$2" -p 1024 -n 0 -d "$5" -ngl 99 -fa 1 -ub 512,1024,2048 -b 8192 -r 3 -ctk $3 -ctv $3 -o json \
    > "$W/L/$1_깊이.json" 2> "$W/log/L_$1_깊이.log" || step "  실패 깊이"
  "$LB/llama-bench.exe" -m "$2" -p 0 -n 128 -d "$5" -ngl 99 -fa 1 -r 3 -ctk $3 -ctv $3 -o json \
    > "$W/L/$1_생성.json" 2> "$W/log/L_$1_생성.log" || step "  실패 생성"
}
if [[ $WHAT == *L* ]] && [ -n "$LB" ]; then
  lbench qwenK_f16 "$QG" f16 512,2048,8192,32768,65536 1024,8192,32768,65536
  lbench qwenK_q8_0 "$QG" q8_0 512,2048,8192,32768,65536 1024,8192,32768,65536
  lbench llamaK_f16 "$LG" f16 512,2048,8192,32768 1024,8192,32768
  lbench llamaK_q8_0 "$LG" q8_0 512,2048,8192,32768,65536 1024,8192,32768,65536
  lbench qwenQ8_f16 "$Q8" f16 512,2048,8192 1024,8192
fi
step "끝"
