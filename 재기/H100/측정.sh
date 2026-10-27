#!/bin/bash
# 서버 GPU 의 속도 측정(문서/H100.md) — GeulPU 0.12.1 의 빠른 · 정밀 모드와 llama.cpp b11900 의 KV f16 · q8_0 을 한 번에 하나씩 띄워
# 요청.py 로 측정한다(한 번에 하나만 GPU 를 쓴다). 먼저 준비.sh.
#   W=<작업 폴더> G=<GeulPU 의 bin> bash 재기/H100/측정.sh
# 끝난 설정은 $W/done/ 에 표시하고 건너뛴다 — 끊겨도 다시 돌리면 이어간다.
# single = 프롬프트 처리 · 깊이별 생성(슬롯 1, 문맥 32768). conc = 동시 1 · 4 · 8 요청(슬롯 8). 추측 디코딩은 끈다.
set -u
S=$(cd "$(dirname "$0")" && pwd)
W=${W:-$HOME/geulpu-h100}
G=${G:-$W/GeulPUv2/bin}
M=$W/models
export CUDA_CACHE_PATH=${CUDA_CACHE_PATH:-$W/cache/nv} CUDA_CACHE_MAXSIZE=4294967296     # PTX 를 JIT 컴파일한 결과
source $W/llama_env.sh
R=$W/results.jsonl
L=$W/log
mkdir -p $L $W/done
PORT=${PORT:-44700}
URL=http://127.0.0.1:$PORT
echo "== 시작 $(date '+%F %T')"

run() {  # run 이름 종류 모드(single|conc) 동시목록 서버명령...
  local name=$1 kind=$2 mode=$3 conc=$4
  shift 4
  if [ -f $W/done/$name ]; then echo "== $name 건너뜀"; return; fi
  echo "== $name $(date +%T)"
  setsid "$@" > $L/$name.server.log 2>&1 &
  local pid=$!
  python3 $S/기다림.py $URL $pid 1800
  local rc=$?
  if [ $rc -eq 0 ]; then
    local ok=1
    if [ "$mode" != conc ]; then python3 $S/요청.py --url $URL --tag $name --mode single --out $R --kind $kind >> $L/$name.bench.log 2>&1 || ok=0; fi
    if [ "$mode" != single ]; then python3 $S/요청.py --url $URL --tag $name --mode conc --conc $conc --out $R --kind $kind >> $L/$name.bench.log 2>&1 || ok=0; fi
    if [ $ok -eq 1 ]; then touch $W/done/$name; echo "   끝 $(date +%T)"; else echo "   측정 실패 — $L/$name.bench.log"; tail -3 $L/$name.bench.log; fi
  else
    echo "   서버가 뜨지 않음(rc $rc) — $L/$name.server.log"
    tail -4 $L/$name.server.log
  fi
  kill -TERM -- -$pid 2> /dev/null
  sleep 5
  kill -KILL -- -$pid 2> /dev/null
  wait $pid 2> /dev/null
  for i in $(seq 1 30); do python3 $S/기다림.py $URL 0 1 > /dev/null 2>&1 || break; sleep 2; done
  sleep 3
}

geulpu() {  # geulpu 이름 GGUF 모듈 single|conc
  if [ $4 = single ]; then
    run $1 geulpu single 1 $G/server-cuda $2 --ptx $W/mod/$3.ptx --슬롯 1 --문맥 32768 --포트 $PORT --이름 m --추측끔
  else
    run $1 geulpu conc 1,4,8 $G/server-cuda $2 --ptx $W/mod/$3.ptx --슬롯 8 --문맥 1024 --포트 $PORT --이름 m --추측끔
  fi
}

llama() {  # llama 이름 GGUF KV형식 single|conc
  if [ $4 = single ]; then
    run $1 llama single 1 $LLAMA_DIR/llama-server -m $2 -ngl 999 -fa on -c 32768 -np 1 -ctk $3 -ctv $3 --port $PORT -a m -t 16
  else
    run $1 llama conc 1,4,8 $LLAMA_DIR/llama-server -m $2 -ngl 999 -fa on -c 8192 -np 8 -ctk $3 -ctv $3 --port $PORT -a m -t 16
  fi
}

for spec in "q3_8b|$M/Qwen3-8B-Q8_0.gguf|qwen3Q8_4096x32x8x12288x151936|_빠른" \
            "q25_7b|$M/Qwen2.5-7B-Instruct-Q4_K_M.gguf|qwen2K_3584x28x4x18944x152064|_빠른_F2_KV8" \
            "q3_32b|$M/Qwen3-32B-Q8_0.gguf|qwen3Q8_5120x64x8x25600x151936|_빠른"; do
  IFS='|' read -r tag gguf mod fast <<< "$spec"
  geulpu ${tag}_geulpu_fast_single $gguf $mod$fast single
  geulpu ${tag}_geulpu_fast_conc $gguf $mod$fast conc
  geulpu ${tag}_geulpu_exact_single $gguf $mod single
  geulpu ${tag}_geulpu_exact_conc $gguf $mod conc
  llama ${tag}_llama_f16_single $gguf f16 single
  llama ${tag}_llama_f16_conc $gguf f16 conc
  llama ${tag}_llama_q8_single $gguf q8_0 single
  llama ${tag}_llama_q8_conc $gguf q8_0 conc
done
echo "== 끝 $(date '+%F %T')"
