#!/bin/bash
# 서버 GPU 의 속도 측정(문서/H100.md) — 준비. 한 번 돌린다(다시 돌리면 있는 것은 건너뛴다).
#   W=<작업 폴더> G=<GeulPU 의 bin> bash 재기/H100/준비.sh
# 1. llama.cpp b11900 을 소스로 빌드한다(sm_90). 릴리스의 CUDA 바이너리는 Ubuntu 24.04 에서 빌드돼 GLIBC 2.38 을 요구한다 — 측정한
#    서버는 22.04(GLIBC 2.35)였다. CUDA 12.8 의 컴파일러 · 헤더 · cmake · ninja 는 conda-forge 에서 받는다(드라이버는 건드리지 않는다).
# 2. 모델 파일의 SHA-256 을 확인한다. 파일은 미리 $W/models 에 받아 둔다(주소는 문서/H100.md 의 5).
# 3. GeulPU 의 커널 모듈: 정밀(--먼 — 앱의 정밀 모듈) · 빠른(Q8_0 은 --빠른, k-양자화는 --빠른 --F2 — 앱의 기본 빠른 모드).
set -u
W=${W:-$HOME/geulpu-h100}                    # 작업 폴더: models/ · mod/ · 결과 · 로그
G=${G:-$W/GeulPUv2/bin}                      # 문서/H100.md 의 1 로 빌드한 server-cuda · kernel-gen · ptx-backend
O=${O:-$W/opt}                               # CUDA 12.8 도구와 llama.cpp(측정 때는 컨테이너의 로컬 디스크 /opt — 다시 켜면 지워진다)
MAMBA=${MAMBA:-$(command -v mamba || command -v conda)}
export XDG_CACHE_HOME=${XDG_CACHE_HOME:-$W/cache}
M=$W/models
mkdir -p $W $M $O
echo "== 준비 시작 $(date '+%F %T')"

echo "== 1. llama.cpp b11900 — 소스 빌드(sm_90)"
P=$O/cuda12
D=$O/llama
if [ ! -x $D/build/bin/llama-server ]; then
  if [ ! -x $P/bin/nvcc ]; then
    EXTRA=""
    command -v g++ > /dev/null || EXTRA="gxx=11"          # nvcc 가 쓸 호스트 C++ 컴파일러가 없으면 함께 받는다
    echo "  conda-forge 에서 CUDA 12.8 의 nvcc · 헤더 · cmake · ninja 를 받는다 $EXTRA"
    "$MAMBA" create -y -q -p $P -c conda-forge cuda-version=12.8 cuda-nvcc cuda-cudart-dev libcublas-dev cuda-cccl cuda-driver-dev cmake ninja $EXTRA > $W/conda12.log 2>&1 || { tail -20 $W/conda12.log; exit 1; }
  fi
  echo "  nvcc: $($P/bin/nvcc --version | grep release) · cmake $($P/bin/cmake --version | head -1 | awk '{print $3}') · g++ $(PATH=$P/bin:$PATH g++ -dumpversion)"
  [ -d $D ] || git -c advice.detachedHead=false clone -q --depth 1 --branch b11900 https://github.com/ggml-org/llama.cpp $D || exit 1
  echo "  llama.cpp 커밋 $(git -C $D -c safe.directory='*' rev-parse HEAD)"
  PATH=$P/bin:$PATH cmake -S $D -B $D/build -G Ninja -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON \
    -DCMAKE_CUDA_ARCHITECTURES=90 -DCMAKE_CUDA_COMPILER=$P/bin/nvcc -DCUDAToolkit_ROOT=$P -DLLAMA_CURL=OFF -DBUILD_SHARED_LIBS=OFF > $W/llama-cmake.log 2>&1 || { tail -20 $W/llama-cmake.log; exit 1; }
  PATH=$P/bin:$PATH cmake --build $D/build --target llama-server -j $(nproc) > $W/llama-build.log 2>&1 || { tail -20 $W/llama-build.log; exit 1; }
fi
echo "export LLAMA_DIR=$D/build/bin; export LD_LIBRARY_PATH=$P/lib:$P/targets/x86_64-linux/lib:\${LD_LIBRARY_PATH:-}" > $W/llama_env.sh
source $W/llama_env.sh
$LLAMA_DIR/llama-server --version 2>&1 | tail -2
$LLAMA_DIR/llama-server --list-devices 2>&1 | tail -3

echo "== 2. 모델 파일(SHA-256)"
ok=1
while read -r f sha; do
  if [ ! -f "$M/$f" ]; then echo "  없음: $M/$f"; ok=0
  elif [ "$(sha256sum "$M/$f" | cut -c1-64)" = "$sha" ]; then echo "  $f 같음"
  else echo "  $f 의 SHA-256 이 다르다"; ok=0; fi
done << 'EOF'
Qwen3-8B-Q8_0.gguf 408b955510e196121c1c375201744783b5c9a43c7956d73fc78df54c66e883d6
Qwen2.5-7B-Instruct-Q4_K_M.gguf 65b8fcd92af6b4fefa935c625d1ac27ea29dcb6ee14589c55a8f115ceaaa1423
Qwen3-32B-Q8_0.gguf 2c50eb8aad05047dbf24fa014eb621adf552e14176cabe0c5db4ef38c91e2169
EOF
[ $ok -eq 1 ] || { echo "  모델 파일을 $M 에 받아 두고 다시 돌린다(주소는 문서/H100.md 의 5)"; exit 1; }

echo "== 3. GeulPU 모듈(정밀 --먼 · 빠른)"
mkdir -p $W/mod && cd $W/mod
mk() {  # mk 이름 구조 형식 너비 헤드 KV 확장 어휘 [옵션...]
  local n=$1 s=$2 f=$3 w=$4 h=$5 k=$6 x=$7 v=$8
  shift 8
  if [ -f "$n.ptx" ]; then echo "  $n.ptx 있음"; return 0; fi
  if "$G/kernel-gen" "$s" "$f" "$w" "$h" "$k" "$x" "$v" "$n.gl" "$@" > "$n.gen.log" 2>&1 && "$G/ptx-backend" "$n.gl" -o "$n.ptx" > "$n.ptx.log" 2>&1; then
    rm -f "$n.gl"
    echo "  $n.ptx 만듦 $(sha256sum "$n.ptx" | cut -c1-16)"
  else
    echo "  $n 실패"
    tail -3 "$n.gen.log" "$n.ptx.log"
  fi
}
mk qwen3Q8_4096x32x8x12288x151936 qwen3큰 Q8_0 4096 32 8 12288 151936 --먼
mk qwen3Q8_4096x32x8x12288x151936_빠른 qwen3큰 Q8_0 4096 32 8 12288 151936 --빠른
mk qwen2K_3584x28x4x18944x152064 qwen2큰 K 3584 28 4 18944 152064 --먼
mk qwen2K_3584x28x4x18944x152064_빠른_F2_KV8 qwen2큰 K 3584 28 4 18944 152064 --빠른 --F2
mk qwen3Q8_5120x64x8x25600x151936 qwen3큰 Q8_0 5120 64 8 25600 151936 --먼
mk qwen3Q8_5120x64x8x25600x151936_빠른 qwen3큰 Q8_0 5120 64 8 25600 151936 --빠른
echo "== 준비 끝 $(date '+%F %T')"
