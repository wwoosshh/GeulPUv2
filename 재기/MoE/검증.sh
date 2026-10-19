#!/bin/bash
# MoE 재기 — 흉내(LRU)가 llama.cpp 의 실제 캐시 적중률과 맞는지: llama-completion 으로 같은 설정 · 고정 글을 생성하고(끝날 때 통계를 적는다),
# 그 토큰열로 전문가를 기록해 같은 슬롯 수의 LRU 를 흉내 낸다.
#   bash 검증.sh <작업 폴더(E:/geulpu-work/moe)> <출력 토큰 수>
set -eu
D=$1; N=${2:-768}
H=$(cd "$(dirname "$0")" && pwd)
V=$D/valid
cd "$V"
"$D/llama-b11727/llama-completion.exe" -m "$D/models/Qwen3.6-35B-A3B-Q4_K_M.gguf" -f prompt.txt -n "$N" --temp 0 -no-cnv \
  --no-display-prompt --simple-io -c 8192 -fa on -ngl 99 -cmoe --fit off --load-mode none --moe-cache-mib 6400 \
  > out.txt 2> err.txt < /dev/null
grep -E "llama_moe_cache|MoE cache|slots|eval time" err.txt || true
python -X utf8 - "$D" <<'PY'
import json, subprocess, sys
D = sys.argv[1]
tok = f"{D}/llama-b11727/llama-tokenize.exe"
def ids(path):
    o = subprocess.run([tok, "-m", f"{D}/models/Qwen3.6-35B-A3B-Q4_K_M.gguf", "-f", path, "--ids", "--log-disable",
                        "--no-bos"], capture_output=True, text=True, encoding="utf-8").stdout
    return json.loads(o.strip().splitlines()[-1])
p = open("prompt.txt", encoding="utf-8").read()
o = open("out.txt", encoding="utf-8", errors="replace").read()
open("both.txt", "w", encoding="utf-8", newline="").write(p + o)
P, B = ids("prompt.txt"), ids("both.txt")
assert B[:len(P)] == P, "프롬프트 토큰이 앞에 그대로 있지 않다"
json.dump({"spans": [{"tokens": P, "kind": "prompt", "req": 0}, {"tokens": B[len(P):], "kind": "output", "req": 0}]},
          open("spans.json", "w", encoding="utf-8"))
print("프롬프트", len(P), "출력", len(B) - len(P))
PY
python -X utf8 "$H/전문가기록.py" "$D/models/Qwen3.6-35B-A3B-Q4_K_M.gguf" spans.json valid.npz --ub 2048 --ctx 8192 | tail -1
python -X utf8 "$H/캐시흉내.py" 3792 valid.npz | grep -E "LRU |걸음"
