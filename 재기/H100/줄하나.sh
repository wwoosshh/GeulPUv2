#!/bin/bash
# 생성의 행렬곱 G2(문서/생성행렬곱.md) — 1행 GEMV 처방 v2 를 0.15.0 커널과 ABBA 로 견준다(2026-10-30 의 H100 NVL MIG 측정 — 원자료는
# 재기/H100/줄하나/H100_*). 이 PC 에서 만든 묶음(재기/줄하나재기.py · 시험/쿠다드라이버.py · cut 으로 뗀 작은 모듈 m/base.ptx · m/g2.ptx ·
# v2 모듈 소스 m/g2_full.gl(Llama-3.1-8B) · m/l70_g2_full.gl(Llama-3.3-70B))을 $W 에 풀고:
#   1) 커널만: 재기/줄하나재기.py run — DRAM 길, Llama-3.1-8B · Llama-3.3-70B 의 모양 열둘
#   2) 생성 스텝: 재기_걸음 깊이 1,024 · 행 1 — 앱의 기본 모듈(0.15.0 — $BASE) 대 v2 모듈(소스를 여기 PTX 뒷단으로)
#   W=<작업 폴더> G=<GeulPU 의 bin — ptx-backend · jaegi-step> BASE=<0.15.0 모듈 폴더> M=<모델 폴더> I=<재기 입력> bash 재기/H100/줄하나.sh
# 결과는 $W/out. 다시 돌리면 끝난 것은 건너뛴다. 묶음은 tar --no-same-owner 로 푼다(NFS 홈은 소유권을 바꾸지 못한다).
set -u
W=${W:?작업 폴더(W)를 준다}
G=${G:?GeulPU 의 bin(G)을 준다}
BASE=${BASE:?0.15.0 모듈 폴더(BASE)를 준다}
M=${M:?모델 폴더(M)를 준다}
I=${I:?재기 입력(I — 재기/판시간.py 입력)을 준다}
mkdir -p $W/out $W/mod/v2
cd $W
echo "== 시작 $(date -u '+%F %T') UTC"
for f in m/base.ptx m/g2.ptx m/g2_full.gl m/l70_g2_full.gl $I; do echo "  $(sha256sum $f | cut -c1-16) $f"; done
C8="--case Q4_K:none:4096x6144 --case Q4_K:residual:4096x4096 --case Q4_K:swiglu:4096x28672 --case Q4_K:residual:14336x4096 --case Q6_K:residual:14336x4096 --case Q6_K:none:4096x1024"
C70="--case Q4_K:none:8192x10240 --case Q4_K:residual:8192x8192 --case Q4_K:swiglu:8192x57344 --case Q4_K:residual:28672x8192 --case Q6_K:residual:28672x8192 --case Q5_K:none:8192x1024"
for r in A1 B1 B2 A2; do
  case $r in A*) m=m/base.ptx ;; *) m=m/g2.ptx ;; esac
  o=out/micro_$r
  [ -f $o.json ] && { echo "== micro $r: 이미 잼"; continue; }
  echo "== micro $r $(date -u '+%T') — $m"
  python3 -I -X utf8 재기/줄하나재기.py run $m --path dram --peak-gbps 1808 $C8 $C70 --json $o.json > $o.txt 2>&1 || { echo "  종료 1 micro $r"; tail -n 3 $o.txt; exit 1; }
done
n8=llamaK_4096x32x8x14336x128256_빠른_F2_KV8.ptx
n70=llamaK_8192x64x8x28672x128256_빠른_F2_KV8.ptx
[ -f mod/v2/$n8 ] || $G/ptx-backend m/g2_full.gl -o mod/v2/$n8 > out/ptx8.txt 2>&1 || { echo "  종료 1 PTX 8B"; exit 1; }
[ -f mod/v2/$n70 ] || $G/ptx-backend m/l70_g2_full.gl -o mod/v2/$n70 > out/ptx70.txt 2>&1 || { echo "  종료 1 PTX 70B"; exit 1; }
for f in $BASE/$n8 mod/v2/$n8 $BASE/$n70 mod/v2/$n70; do [ -f $f ] || { echo "  종료 1 모듈 없음 $f"; exit 1; }; echo "  $(sha256sum $f | cut -c1-16) $f"; done
for g in Meta-Llama-3.1-8B-Instruct-Q4_K_M:$n8 Llama-3.3-70B-Instruct-Q4_K_M:$n70; do
  mdl=${g%%:*}; n=${g#*:}
  for r in A1 B1 B2 A2; do
    case $r in A*) ptx=$BASE/$n ;; *) ptx=mod/v2/$n ;; esac
    o=out/step_${mdl}_$r.txt
    if [ -f $o.rc ] && [ "$(cat $o.rc)" = 0 ]; then echo "== $mdl $r: 이미 잼"; continue; fi
    echo "== $mdl $r $(date -u '+%T')"
    $G/jaegi-step $M/$mdl.gguf $ptx $I 8704 --깊이 1024 --행 1 --번 10 > $o 2>&1
    rc=$?
    echo $rc > $o.rc
    echo "  종료 $rc $(date -u '+%T') — $(grep '^깊이 1024 행 1' $o) · $(grep '^오류칸' $o)"
    [ $rc = 0 ] || tail -n 3 $o
  done
done
echo "== 끝 $(date -u '+%F %T') UTC"
