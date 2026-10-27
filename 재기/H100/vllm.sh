#!/bin/bash
# 서버 GPU 의 속도 측정(문서/H100.md) — vLLM 0.32.0 과 메모리 대역폭. 측정.sh 와 같은 작업 폴더 · 결과 파일을 쓴다.
#   W=<작업 폴더> bash 재기/H100/vllm.sh
# vLLM 의 FlashInfer 와 FP8 커널은 그 자리에서 nvcc 로 JIT 컴파일한다. 그래서 vLLM 의 PyTorch(cu130)와 같은 CUDA 13.0 의 nvcc · 헤더를
# conda-forge 에서 받아 CUDA_HOME 으로 준다. 측정한 서버의 컨테이너는 다시 켜면 루트가 지워져서 모두 작업 폴더(홈의 NFS)에 두었다.
# vLLM 0.32 는 GGUF 를 열지 못해(설정을 JSON 으로 읽으려 함) Qwen3-8B 를 BF16 과 FP8(불러올 때 바꿈)로 측정한다.
# 다시 돌리면 이어간다: 깔린 것은 건너뛰고, 받은 파일은 이어 받고, 끝난 설정(done/)은 건너뛴다.
set -u
S=$(cd "$(dirname "$0")" && pwd)
W=${W:-$HOME/geulpu-h100}
M=$W/models
C=$W/cuda13
V=$W/vllm-venv
K=$W/cache
MAMBA=${MAMBA:-$(command -v mamba || command -v conda)}
mkdir -p $K $M
export PIP_USER=0                            # 측정한 서버의 pip 는 늘 --user 로 설치하게 돼 있어 가상환경에서 거부됐다
export XDG_CACHE_HOME=$K HF_HOME=$K/hf TRITON_CACHE_DIR=$K/triton TORCHINDUCTOR_CACHE_DIR=$K/inductor VLLM_CACHE_ROOT=$K/vllm
export CUDA_CACHE_PATH=$K/nv CUDA_CACHE_MAXSIZE=4294967296
echo "== 시작 $(date '+%F %T')"

echo "== 1. CUDA 13.0 의 nvcc · 헤더 (conda-forge → $C)"
if [ ! -x $C/bin/nvcc ]; then
  rm -rf $C
  "$MAMBA" create -y -q -p $C -c conda-forge cuda-version=13.0 cuda-nvcc cuda-cudart-dev cuda-cccl libcublas-dev libcurand-dev cuda-nvrtc-dev cuda-profiler-api > $W/conda13.log 2>&1 || { tail -20 $W/conda13.log; exit 1; }
fi
T=$C/targets/x86_64-linux/lib
if [ ! -x $C/bin/ninja ] || [ ! -e $T/stubs/libcuda.so ]; then       # FlashInfer 의 JIT 는 ninja 로 빌드하고 libcuda 에 링크한다
  "$MAMBA" install -y -q -p $C -c conda-forge ninja cuda-driver-dev > $W/conda13-more.log 2>&1 || { tail -10 $W/conda13-more.log; exit 1; }
fi
[ -e $C/lib64 ] || ln -s $T $C/lib64                                # FlashInfer 는 CUDA_HOME/lib64 에서 -lcudart 를 찾는다
export CUDA_HOME=$C PATH=$C/bin:$PATH LD_LIBRARY_PATH=$C/lib:$T:${LD_LIBRARY_PATH:-} LIBRARY_PATH=$T:$T/stubs:$C/lib:${LIBRARY_PATH:-}
export FLASHINFER_WORKSPACE_BASE=$K DG_JIT_CACHE_DIR=$K/deep_gemm TORCH_EXTENSIONS_DIR=$K/torch_ext   # JIT 결과를 ~/.cache 가 아니라 작업 폴더에
export VLLM_USE_FLASHINFER_SAMPLER=0                                 # 샘플링은 PyTorch 경로(측정은 그리디)
echo "  $(nvcc --version | grep release) · ninja $(ninja --version) · $(ls $C/lib64/libcudart.so 2>&1)"

echo "== 2. vLLM 0.32.0 ($V)"
if ! $V/bin/python -c "import vllm" 2> /dev/null; then
  [ -x $V/bin/python ] || python3 -m venv $V || exit 1
  $V/bin/pip install -q --no-cache-dir --upgrade pip && $V/bin/pip install -q --no-cache-dir vllm==0.32.0 || exit 1
fi
$V/bin/python -c "import vllm, torch; print('  vllm', vllm.__version__, '· torch', torch.__version__, '· cuda', torch.version.cuda)"

echo "== 3. Qwen3-8B (BF16, 이어 받기)"
$V/bin/python -c "import sys; from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen3-8B', local_dir=sys.argv[1])" $M/Qwen3-8B || exit 1
du -sh $M/Qwen3-8B

echo "== 4. 측정"
export HF_HUB_OFFLINE=1
R=$W/results.jsonl
L=$W/log
PORT=${PORT:-44700}
URL=http://127.0.0.1:$PORT
mkdir -p $L $W/done
[ -f $W/done/bandwidth ] || { $V/bin/python $S/대역폭.py $R && touch $W/done/bandwidth; }      # 메모리 대역폭(복사) — 정리.py 의 상한
run() {  # run 이름 모델 [vLLM 옵션...] — 한 서버로 single · conc(1 · 4 · 8 · 32)
  local name=$1 model=$2
  shift 2
  if [ -f $W/done/$name ]; then echo "== $name 건너뜀"; return; fi
  echo "== $name $(date +%T)"
  # 최대 문맥 31232: 측정한 값. 가장 긴 요청(30720 + 257)이 들어가고, Qwen3-32B FP8 의 KV 캐시가 남은 메모리(7.65 GiB)에 들어가게 정했다
  setsid $V/bin/vllm serve $model --max-model-len 31232 --gpu-memory-utilization 0.92 --no-enable-prefix-caching --port $PORT \
    --served-model-name m --seed 0 "$@" > $L/$name.server.log 2>&1 &
  local pid=$!
  python3 $S/기다림.py $URL $pid 3600
  local rc=$?
  if [ $rc -eq 0 ]; then
    if python3 $S/요청.py --url $URL --tag $name --mode single --out $R --kind vllm >> $L/$name.bench.log 2>&1 \
       && python3 $S/요청.py --url $URL --tag $name --mode conc --conc 1,4,8,32 --out $R --kind vllm >> $L/$name.bench.log 2>&1; then
      touch $W/done/$name
      echo "   끝 $(date +%T)"
    else
      echo "   측정 실패 — $L/$name.bench.log"
      tail -3 $L/$name.bench.log
    fi
  else
    echo "   서버가 뜨지 않음(rc $rc) — $L/$name.server.log"
    grep -E "Error|error:" $L/$name.server.log | tail -3
  fi
  kill -TERM -- -$pid 2> /dev/null
  sleep 5
  kill -KILL -- -$pid 2> /dev/null
  wait $pid 2> /dev/null
  for i in $(seq 1 30); do python3 $S/기다림.py $URL 0 1 > /dev/null 2>&1 || break; sleep 2; done
  sleep 3
}
run q3_8b_vllm_bf16 $M/Qwen3-8B
run q3_8b_vllm_fp8 $M/Qwen3-8B --quantization fp8
# 공식 Qwen/Qwen3-32B-FP8 은 블록 FP8 커널의 JIT 컴파일이 CUDA 13.0 헤더와 맞지 않아 실패했다(문서/H100.md 의 4):
# run q3_32b_vllm_fp8 $M/Qwen3-32B-FP8
echo "== 끝 $(date '+%F %T')"
