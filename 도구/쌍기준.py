#!/usr/bin/env python3
"""파일 쌍 재현성(앱의 벤치마크 — 앱/쌍검증.gl)의 기준 데이터를 만든다. 표준 라이브러리만.

    python -I -X utf8 도구/쌍기준.py requests TESTSET.jsonl RUNLOG.jsonl 교차검증/쌍/requests.json
        테스트 세트(messages 꼴 — 교차검증의 수학 문제 32 개)와, 그 세트를 앱으로 돌린 실행 로그(도구/debugger.py run-set — 응답의
        geulpu.prompt_tokens)에서 문제마다 프롬프트 토큰을 꺼낸다. 앱의 작업은 이 토큰을 /v1/completions 로 보낸다 — 채팅 틀의
        코드가 판마다 달라도 같은 프롬프트(Qwen2.5 의 채팅 틀 — 기본 시스템 프롬프트)

    python -I -X utf8 도구/쌍기준.py expected 결과폴더 교차검증/쌍/expected.json
        기준 PC 에서 기준 없이 돌린 작업의 결과 폴더(자료\\결과\\파일쌍-C-…)에서 파일마다 · 문제마다 스텝 수 · 끝 까닭 · 토큰과 스텝마다
        로짓 해시(FNV-1a 64 — 16 자씩 이은 글)를 기준으로 적는다. reference 칸은 그 결과의 environment(앱 판 · 커밋 · GPU · 드라이버)
"""
import hashlib
import json
import os
import sys


def 읽기줄들(p):
    with open(p, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def 쓰기(p, o, 들여=None):
    os.makedirs(os.path.dirname(os.path.abspath(p)), exist_ok=True)
    with open(p, "w", encoding="utf-8", newline="\n") as f:
        json.dump(o, f, ensure_ascii=False, indent=들여, separators=None if 들여 else (",", ":"))
        f.write("\n")


def 요청들(세트, 로그, 출력):
    문제들 = 읽기줄들(세트)
    줄들 = [x for x in 읽기줄들(로그) if "response" in x]
    줄들.sort(key=lambda x: x["i"])
    if len(줄들) != len(문제들):
        sys.exit(f"테스트 세트 {len(문제들)} 문제 · 로그 {len(줄들)} 요청 — 같아야 한다")
    out = []
    for k, (q, x) in enumerate(zip(문제들, 줄들)):
        if x["request"].get("messages") != q["messages"]:
            sys.exit(f"요청 {k} 의 messages 가 테스트 세트와 다르다")
        out.append({"id": k, "question": q["messages"][-1]["content"], "prompt_tokens": x["response"]["geulpu"]["prompt_tokens"]})
    쓰기(출력, out, 들여=None)
    print(f"{출력}: 문제 {len(out)} 개 · 프롬프트 {min(len(o['prompt_tokens']) for o in out)} ~ {max(len(o['prompt_tokens']) for o in out)} 토큰")


def 기준(폴더, 출력):
    with open(os.path.join(폴더, "summary.json"), encoding="utf-8") as f:
        s = json.load(f)
    env = s["environment"]
    items = {}
    for run in s["runs"]:
        라벨 = run["file"]
        if run.get("error"):
            sys.exit(f"{라벨}: 오류가 있는 실행 — {run['error']}")
        rows = []
        for it in run["items"]:
            k = it["id"]
            with open(os.path.join(폴더, 라벨, "responses", f"{k:02d}.json"), encoding="utf-8") as f:
                r = json.load(f)
            g = r["geulpu"]
            해시 = g["logit_hashes"]
            if any(len(h) != 16 for h in 해시) or len(해시) != len(g["tokens"]):
                sys.exit(f"{라벨} 문제 {k}: 해시 꼴이 틀리다")
            rows.append({"id": k, "n": len(해시), "finish": r["choices"][0].get("finish_reason"),
                         "tokens_sha256": hashlib.sha256(",".join(str(t) for t in g["tokens"]).encode()).hexdigest(),
                         "hashes": "".join(해시)})
        items[라벨] = rows
    o = {"reference": {"gpu": env.get("gpu"), "driver": env.get("driver"), "os": (env.get("os") or {}).get("caption"),
                       "geulpu": f"{env['app']['version']} (commit {env['app']['commit']})", "layer": env.get("gpu_layer"),
                       "contract_sha256": env["app"].get("contract_sha256"), "contract_reference_sha256": env["app"].get("contract_reference_sha256"),
                       "checked_at": env.get("checked_at"), "result": os.path.basename(os.path.normpath(폴더))},
         "settings": {"temperature": 0, "max_tokens": 512, "context": 1024, "precision": "exact",
                      "prompt": "requests.json 의 prompt_tokens(Qwen2.5 의 채팅 틀 — 기본 시스템 프롬프트)를 /v1/completions 로"},
         "items": items}
    쓰기(출력, o)
    print(f"{출력}: " + " · ".join(f"{k} {sum(r['n'] for r in v)} 스텝" for k, v in items.items()))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    if len(sys.argv) == 5 and sys.argv[1] == "requests":
        요청들(*sys.argv[2:])
    elif len(sys.argv) == 4 and sys.argv[1] == "expected":
        기준(*sys.argv[2:])
    else:
        sys.exit(__doc__)
