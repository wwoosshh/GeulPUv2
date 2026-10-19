#!/usr/bin/env python3
"""MoE 재기 — 바닥 셈과 판정(문서/MoE재기.md 의 미리 정한 기준).

  python 판정.py <모델크기.json> <대역폭.json> <흉내.json,...> <llama 깊이.jsonl> [--깊이 16384,32768]

대역폭.json: {"vram_peak": GB/s, "pcie_expert": GB/s, "cpu_expert": GB/s(llama.cpp 고정 배치에서 잰 CPU 전문가 처리량), "eff": 0.87}
토큰 하나의 바이트(깊이 d):
  VRAM = 비전문가(층 · 공유 전문가 · 출력) + DeltaNet 상태(읽기 + 쓰기) + KV(d) + 적중 × 고른 전문가
  놓침 = 출력 토큰당 놓친 전문가 × 전문가 하나
  가) GPU 차례: VRAM / vram_peak + 놓침 / pcie_expert
  나) 놓친 것을 CPU 가 함께: max(VRAM / vram_peak, 놓침 / cpu_expert)
  속도 = 1 / 시간 × eff(GeulPU 가 밀집 생성에서 낸 대역폭 효율 — 미리 정함)
판정: Belady(+채움)의 가) 나) 중 나은 것 ≥ 1.5 × llama.cpp 최선(잰 값)이 깊이 16k · 32k 둘 다에서.
"""
import json
import statistics
import sys

G = 1 << 30
크기 = json.load(open(sys.argv[1], encoding="utf-8"))
대 = json.load(open(sys.argv[2], encoding="utf-8"))
흉내들 = [json.load(open(f, encoding="utf-8")) for f in sys.argv[3].split(",")]
llama = [json.loads(l) for l in open(sys.argv[4], encoding="utf-8")]
인자 = sys.argv[5:]
깊이들 = [int(x) for x in (인자[인자.index("--깊이") + 1] if "--깊이" in 인자 else "16384,32768").split(",")]

meta = 크기["meta"]
arch = 크기["arch"]
L = 크기["n_layer"]
NE, K = 크기["n_expert"], 크기["n_expert_used"]
전문가하나 = 크기["전체"]["전문가"] // (L * NE)
비전문가 = 크기["전체"]["비전문가"] + 크기["전체"]["출력"]
간격 = meta.get(f"{arch}.full_attention_interval", 1)
어텐션층 = L // 간격
델타층 = L - 어텐션층
kv토큰 = 어텐션층 * 2 * meta[f"{arch}.attention.head_count_kv"] * meta[f"{arch}.attention.key_length"] * 2
v머리 = meta[f"{arch}.ssm.time_step_rank"]
상태 = 델타층 * v머리 * meta[f"{arch}.ssm.state_size"] * (meta[f"{arch}.ssm.inner_size"] // v머리) * 4 * 2
고른 = L * K * 전문가하나
print(f"전문가 하나 {전문가하나 / 2**20:.4f} MiB · 토큰당 고른 전문가 {고른 / G:.3f} GiB · 비전문가 {비전문가 / G:.3f} GiB · "
      f"상태 {상태 / 2**20:.0f} MiB · KV {kv토큰} B/토큰")


def 속도(적중, 놓침수, d):
    vram = 비전문가 + 상태 + kv토큰 * d + 적중 * 고른
    놓 = 놓침수 * 전문가하나
    가 = vram / (대["vram_peak"] * 1e9) + 놓 / (대["pcie_expert"] * 1e9)
    나 = max(vram / (대["vram_peak"] * 1e9), 놓 / (대["cpu_expert"] * 1e9))
    return 대["eff"] / 가, 대["eff"] / 나


# 흉내: 세션들을 출력 토큰 수로 가중 평균
정책들 = list(흉내들[0]["policies"])
합 = {}
for p in 정책들:
    w = [(s["output_tokens"], s["policies"][p]) for s in 흉내들 if p in s["policies"]]
    n = sum(a for a, _ in w)
    합[p] = {"hit_output": sum(a * b["hit_output"] for a, b in w) / n,
            "miss_per_output_token": sum(a * b["miss_per_output_token"] for a, b in w) / n}

잰 = {}
for c in llama:
    for r in c.get("requests", []):
        if not r["warm"] and r.get("predicted_per_second"):
            잰.setdefault((c["config"], r["name"]), []).append(r["predicted_per_second"])
print("\nllama.cpp 잰 값(생성 토큰/초):")
for (cfg, name), v in sorted(잰.items()):
    print(f"  {cfg:24s} {name:10s} {statistics.median(v):6.1f}")

결과 = {"policy": 합, "rows": []}
판정 = []
for d in 깊이들:
    이름 = f"d{d // 1024}k"
    best = max((statistics.median(v) for (cfg, name), v in 잰.items() if name == 이름), default=None)
    print(f"\n깊이 {d}: llama.cpp 최선 {best}")
    for p, v in 합.items():
        가, 나 = 속도(v["hit_output"], v["miss_per_output_token"], d)
        비 = max(가, 나) / best if best else None
        print(f"  {p:10s} 출력 적중 {100 * v['hit_output']:5.1f} %  놓침 {v['miss_per_output_token']:5.1f}/토큰  "
              f"바닥×eff 가) {가:6.1f}  나) {나:6.1f}  " + (f"→ llama.cpp 의 {비:.2f} 배" if 비 else ""))
        결과["rows"].append({"depth": d, "policy": p, "gpu_serial": 가, "cpu_parallel": 나, "llama_best": best, "ratio": 비})
        if p.startswith("Belady+") and 비 is not None:
            판정.append(비)
print("\n판정(Belady+채움, 깊이마다):", [round(x, 2) for x in 판정],
      "→ 구조의 몫이 있다" if 판정 and min(판정) >= 1.5 else "→ 1.5 배에 못 미친다")
if "--json" in 인자:
    json.dump(결과, open(인자[인자.index("--json") + 1], "w", encoding="utf-8"), ensure_ascii=False, indent=1)
