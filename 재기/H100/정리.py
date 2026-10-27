"""서버 GPU 의 속도 측정(문서/H100.md)의 정리 — results.jsonl(태그마다 마지막 값)에서 엔진별 표.
  python -I -X utf8 재기/H100/정리.py <results.jsonl> [<results2.jsonl> ...]      (이 기록의 원자료: 재기/H100/결과.jsonl)
비율은 README 와 같은 규칙: GeulPU 의 시간 ÷ llama.cpp 의 시간(f16 · q8_0 KV 가운데 빠른 쪽), 낮을수록 GeulPU 가 빠르다.
"""
import json
import sys

rows = []
for p in sys.argv[1:]:
    rows += [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]
last = {}
for r in rows:                                    # 같은 설정 · 일을 다시 측정했으면 마지막 값
    k = (r["tag"], r.get("work"), r.get("tokens") if r.get("work") == "prefill" else r.get("depth") if r.get("work") == "decode" else r.get("concurrency"))
    last[k] = r
bw = next((r for r in rows if r["tag"] == "bandwidth"), None)
크기 = {"q3_8b": 8.71e9, "q25_7b": 4.683e9, "q3_32b": 34.82e9}
이름 = {"q3_8b": "Qwen3-8B Q8_0", "q25_7b": "Qwen2.5-7B-Instruct Q4_K_M", "q3_32b": "Qwen3-32B Q8_0"}
엔진 = [("geulpu_fast", "GeulPU 빠른"), ("geulpu_exact", "GeulPU 정밀"), ("llama_f16", "llama f16"), ("llama_q8", "llama q8_0"),
        ("vllm_bf16", "vLLM BF16"), ("vllm_fp8", "vLLM FP8")]


def 값(m, e, mode, work, x):
    r = last.get((f"{m}_{e}_{mode}" if not e.startswith("vllm") else f"{m}_{e}", work, x))
    return r


print(f"대역폭(복사, 읽기+쓰기): {bw['copy_GBps']:.0f} GB/s · {bw['device']} · SM {bw['sm']}" if bw else "대역폭 없음")
out = {}
for m in ("q3_8b", "q25_7b", "q3_32b"):
    print(f"\n===== {이름[m]} ({크기[m] / 1e9:.2f} GB)")
    floor = bw["copy_GBps"] * 1e9 / 크기[m] if bw else None
    print(f"  생성의 상한(대역폭 ÷ 파일 크기): {floor:.0f} 토큰/초" if floor else "")
    for work, xs, unit in (("prefill", (512, 2048, 8192, 30720), "토큰/초"), ("decode", (64, 8192, 30720), "토큰/초"), ("conc", (1, 4, 8, 32), "토큰/초(전체)")):
        print(f"  [{work}] {unit}")
        for x in xs:
            cells, vals = [], {}
            for e, en in 엔진:
                mode = "conc" if work == "conc" else "single"
                r = 값(m, e, mode, work, x)
                if r and r.get("tok_per_s"):
                    vals[e] = r["tok_per_s"]
                    note = ""
                    if work == "decode" and r.get("completion_tokens") and min(r["completion_tokens"]) != r["new"] + 1:
                        note = f"(생성 {r['completion_tokens']})"
                    cells.append(f"{en} {r['tok_per_s']:.1f}{note}")
            best_llama = max([vals[k] for k in ("llama_f16", "llama_q8") if k in vals], default=None)
            ratio = ""
            if best_llama:
                for e in ("geulpu_fast", "geulpu_exact"):
                    if e in vals:
                        ratio += f" · {e.split('_')[1]} 시간비 {best_llama / vals[e]:.2f}×"
                if work == "decode" and floor:
                    ratio += f" · 상한 대비 GeulPU빠른 {vals.get('geulpu_fast', 0) / floor * 100:.0f}% · llama {best_llama / floor * 100:.0f}%"
            print(f"    {x:>6}: " + " | ".join(cells) + ratio)
            out[(m, work, x)] = vals
