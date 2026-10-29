#!/bin/bash
# 3.6 의 측정(문서/H100.md · docs/h100.md) — 모델마다 앱의 기본 모듈로 엔진/재기_걸음 을 돌린다: 생성 깊이 1,024 · 8,192(행 1 · 8 —
# 표 판이 없는 구조는 행 1), 프롬프트 512 · 2,048 · 8,192, 최대길이 8,704, 같은 무작위 토큰(씨앗 1).
#   W=<작업 폴더> G=<GeulPU 저장소 — bin/ 에 server-cuda · kernel-gen · ptx-backend · jaegi-step> M=<모델 폴더> bash 재기/H100/판시간.sh
# 결과는 $W/out/<모델>.txt(+ .rc), 모듈은 $W/mod. 다시 돌리면 끝난 측정은 건너뛴다. 2026-10-29 의 측정은 이 스크립트 앞에 저장소 받기 ·
# 0.15.0(5ab9c35) 위에 이 가지의 변경 얹기 · 빌드를 붙여 돌렸다(판시간/H100_실행기록.log).
set -u
W=${W:-~/geulpu-h100}
G=${G:?GeulPU 저장소(G)를 준다}
M=${M:-$W/models}
mkdir -p $W/out $W/mod
[ -f $W/input32k.bin ] || python3 -I $G/재기/판시간.py 입력 $W/input32k.bin --길이 32768 --씨앗 1
echo "  입력 $(sha256sum $W/input32k.bin | cut -c1-16)"
L=8704
for m in Meta-Llama-3.1-8B-Instruct-Q4_K_M Qwen3.5-9B-Q4_K_M Qwen3-32B-Q8_0 Qwen3.5-27B-Q4_K_M gemma-4-31B-it-Q4_K_M Llama-3.3-70B-Instruct-Q4_K_M; do
  o=$W/out/$m.txt
  [ -f $M/$m.gguf ] || { echo "== $m: 모델 파일 없음"; continue; }
  if [ -f $o.rc ] && [ "$(cat $o.rc)" = 0 ]; then echo "== $m: 이미 잼"; continue; fi
  echo "== $m $(date -u '+%T')"
  ptx=$(python3 -I $G/재기/H100/판시간_모듈.py $G/bin $M/$m.gguf $W/mod 2> $W/out/$m.모듈.txt | tail -n 1)
  if [ -z "$ptx" ] || [ ! -f "$ptx" ]; then echo "  종료 1 모듈 실패 $m: $(tail -n 2 $W/out/$m.모듈.txt)"; continue; fi
  echo "  모듈 $(basename $ptx) $(sha256sum $ptx | cut -c1-16) — $(grep '^깃발' $W/out/$m.모듈.txt)"
  $G/bin/jaegi-step $M/$m.gguf $ptx $W/input32k.bin $L --깊이 1024,8192 --행 1,8 --번 10 --프롬프트 512,2048,8192 --프롬프트번 2 > $o 2>&1
  rc=$?
  echo $rc > $o.rc
  echo "  종료 $rc $(date -u '+%T') — 판 줄 $(grep -c '^  ' $o) · $(grep '^오류칸' $o)"
  [ $rc = 0 ] || tail -n 3 $o
done
echo "== 끝 $(date -u '+%F %T') UTC"
