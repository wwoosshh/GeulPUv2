#!/usr/bin/env python3
"""1.0.0 측정 E — 에이전트 재생(문서/1.0.0.md): 기록한 코딩 에이전트의 요청(재기/MoE/세션.py 의 jsonl — 줄마다 "request")을 기록의
차례대로 OpenAI 호환 서버(GeulPU 서버 · llama-server)에 보내고 요청마다 시간을 적는다.

  python -I -X utf8 재기/에이전트재생.py <서버 url> <출력.json> <기록.jsonl> [<기록.jsonl> ...] [--최대 1024] [--이름 모델] [--geulpu] [--토큰]

- 요청: 기록의 messages · tools 그대로(메시지는 role · content · tool_calls · tool_call_id · name 만), temperature 0, max_tokens = --최대,
  stream(첫 토큰 시각을 재려고) · stream_options.include_usage. --geulpu 면 "geulpu": {} 를 더해 서버의 덧붙임(다시 쓴 토큰 · 첫 토큰 ·
  생성 시간 · 초안)을 받는다.
- 처음에 데우기 요청 하나(기록의 첫 요청과 같은 꼴 — 첫 메시지 앞에 '#' 을 더해 앞부분을 다시 쓰지 않는다, 답 16 토큰, 결과에 넣지
  않는다). 기록 사이에 서버를 다시 띄우지 않는다 — 슬롯 하나가 앞 기록의 앞부분을 그대로 가진 채 다음 기록으로 간다(실제 사용처럼).
- 요청마다 적는 것: 보낸 때부터 첫 토큰(내용 · 생각 · 도구 호출의 첫 조각)까지 ttft_s, 첫 토큰부터 끝까지 gen_s, 끝까지 wall_s, usage
  (prompt_tokens · completion_tokens), 끝 까닭, 서버의 값(GeulPU: geulpu 덧붙임, llama-server: timings).
- 요약: 기록마다 · 모두의 ttft 합(프롬프트 처리), 생성 합, 생성 토큰, 생성의 토큰당 ms, 끝까지 합.
"""
import json
import sys
import time
import urllib.request


def 인자():
    a = sys.argv[1:]
    if len(a) < 3:
        print(__doc__)
        sys.exit(2)
    opt = {"최대": 1024, "이름": "model", "geulpu": False, "토큰": False}
    pos = []
    i = 0
    while i < len(a):
        if a[i] == "--최대":
            opt["최대"] = int(a[i + 1]); i += 2
        elif a[i] == "--이름":
            opt["이름"] = a[i + 1]; i += 2
        elif a[i] == "--geulpu":
            opt["geulpu"] = True; i += 1
        elif a[i] == "--토큰":                # GeulPU 만 — 프롬프트 토큰과 답 토큰을 결과에 남긴다(초안 모델 받아들임 흉내에 쓴다)
            opt["토큰"] = True; opt["geulpu"] = True; i += 1
        else:
            pos.append(a[i]); i += 1
    return pos[0].rstrip("/"), pos[1], pos[2:], opt


def 메시지(ms):
    out = []
    for m in ms:
        x = {"role": m["role"]}
        for k in ("content", "tool_calls", "tool_call_id", "name"):
            if k in m and m[k] is not None:
                x[k] = m[k]
        if "content" not in x:
            x["content"] = ""
        out.append(x)
    return out


def 보내기(url, 몸, 시간=3600):
    rq = urllib.request.Request(url + "/v1/chat/completions", data=json.dumps(몸).encode("utf-8"), method="POST",
                                headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    첫 = None
    usage, 끝까닭, 덧, timings, 오류 = None, None, None, None, None
    조각 = 0
    with urllib.request.urlopen(rq, timeout=시간) as r:
        for 줄 in r:
            줄 = 줄.decode("utf-8", "replace").strip()
            if not 줄.startswith("data:"):
                continue
            d = 줄[5:].strip()
            if d == "[DONE]":
                break
            try:
                j = json.loads(d)
            except ValueError:
                continue
            if "error" in j:
                오류 = j["error"]
                continue
            for c in j.get("choices") or []:
                δ = c.get("delta") or {}
                if 첫 is None and (δ.get("content") or δ.get("reasoning_content") or δ.get("tool_calls")):
                    첫 = time.perf_counter()
                if δ.get("content") or δ.get("reasoning_content") or δ.get("tool_calls"):
                    조각 += 1
                if c.get("finish_reason"):
                    끝까닭 = c["finish_reason"]
            if j.get("usage"):
                usage = j["usage"]
            if j.get("geulpu"):
                덧 = j["geulpu"]
            if j.get("timings"):
                timings = j["timings"]
    t1 = time.perf_counter()
    if 첫 is None:
        첫 = t1
    return {"ttft_s": 첫 - t0, "gen_s": t1 - 첫, "wall_s": t1 - t0, "usage": usage, "finish": 끝까닭, "chunks": 조각,
            "geulpu": 덧, "timings": timings, "error": 오류}


def 줄이기(덧):
    # geulpu 덧붙임에서 시간 · 재사용 · 초안만(토큰 목록은 빼고)
    if not 덧:
        return None
    keep = {k: 덧[k] for k in ("prompt_tokens_cached", "slot", "first_token_ms", "generate_us", "speculative", "draft_proposed",
                               "draft_accepted", "draft_steps") if k in 덧}
    if isinstance(덧.get("store"), dict):
        keep["store"] = {k: 덧["store"][k] for k in ("slot_cached_tokens", "slot_copy_tokens", "restored_tokens", "reuse_end", "break")
                         if k in 덧["store"]}
    if isinstance(덧.get("tokens"), list):
        keep["n_tokens"] = len(덧["tokens"])
    return keep


def main():
    url, 출력, 기록들, opt = 인자()
    세션들 = []
    for p in 기록들:
        줄들 = [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]
        세션들.append((p, [x["request"] for x in 줄들 if "request" in x]))

    def 몸(rq, 최대):
        b = {"model": opt["이름"], "messages": 메시지(rq["messages"]), "temperature": 0, "max_tokens": 최대, "stream": True,
             "stream_options": {"include_usage": True}}
        if rq.get("tools"):
            b["tools"] = rq["tools"]
        if opt["geulpu"]:
            b["geulpu"] = {"prompt_tokens": True} if opt["토큰"] else {}
        return b

    결과 = {"url": url, "opt": opt, "files": 기록들, "warmup": None, "sessions": []}
    # 데우기 — 첫 요청과 길이 · 꼴은 같되 첫 메시지의 앞에 글자를 하나 더해 앞부분을 다시 쓰지 않게(첫 요청의 프롬프트 처리를 재도록)
    데움 = dict(세션들[0][1][0])
    데움["messages"] = [dict(데움["messages"][0], content="#" + str(데움["messages"][0].get("content") or ""))] + list(데움["messages"][1:])
    결과["warmup"] = 보내기(url, 몸(데움, 16))
    print(f"데우기 {결과['warmup']['wall_s']:.1f} s", flush=True)
    for p, rqs in 세션들:
        줄 = []
        for i, rq in enumerate(rqs):
            r = 보내기(url, 몸(rq, opt["최대"]))
            if opt["토큰"] and r["geulpu"]:
                r["prompt_ids"] = r["geulpu"].get("prompt_tokens")
                r["answer_ids"] = r["geulpu"].get("tokens")
            r["geulpu"] = 줄이기(r["geulpu"])
            r["i"] = i + 1
            줄.append(r)
            u = r["usage"] or {}
            print(f"{p.split('/')[-1].split(chr(92))[-1]} {i + 1:2d}  프롬프트 {u.get('prompt_tokens', '?'):>6}  첫 토큰 {r['ttft_s']:7.2f} s"
                  f"  생성 {u.get('completion_tokens', '?'):>5} 토큰 {r['gen_s']:6.2f} s  {r['finish']}"
                  + (f"  오류 {r['error']}" if r["error"] else ""), flush=True)
        결과["sessions"].append({"file": p, "requests": 줄})

    def 합(줄들):
        tt = sum(r["ttft_s"] for r in 줄들)
        gs = sum(r["gen_s"] for r in 줄들)
        gt = sum((r["usage"] or {}).get("completion_tokens", 0) or 0 for r in 줄들)
        pt = sum((r["usage"] or {}).get("prompt_tokens", 0) or 0 for r in 줄들)
        return {"n": len(줄들), "ttft_s": tt, "gen_s": gs, "gen_tokens": gt, "prompt_tokens": pt,
                "gen_ms_per_token": 1000 * gs / gt if gt else None, "wall_s": sum(r["wall_s"] for r in 줄들),
                "errors": sum(1 for r in 줄들 if r["error"])}

    결과["summary"] = {"sessions": [dict(file=s["file"], **합(s["requests"])) for s in 결과["sessions"]],
                       "all": 합([r for s in 결과["sessions"] for r in s["requests"]])}
    json.dump(결과, open(출력, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    a = 결과["summary"]["all"]
    print(f"모두: 요청 {a['n']} · 첫 토큰 합 {a['ttft_s']:.1f} s · 생성 {a['gen_tokens']} 토큰 {a['gen_s']:.1f} s"
          f" ({a['gen_ms_per_token'] or 0:.2f} ms/토큰) · 끝까지 {a['wall_s']:.1f} s · 오류 {a['errors']}")


if __name__ == "__main__":
    main()
