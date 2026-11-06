#!/usr/bin/env python3
"""GeulPU 0.21.0 cache+scheduling stress pilot (first pilot).

One model (qwen3-8b-q4_k_m), one fixed config (kit "default": precision=fast,
kv8_cache=true, fast_attention=true, slots=2, context=8192). 16 requests:
the 12 fixed kit requests (from requests.json) plus 4 pilot requests
(gen-1k, prefix-a, prefix-b, chat-2). Phases:

  P1 baseline   solo, cache_prompt=false, own fresh data dir -> fully computed reference
                (kit data dir untouched; cache_prompt=false still verified in --smoke)
  P2 batch      16 requests 8-at-once into 2 slots, shuffled order x3 (fixed shuffle seeds)
  P3 slots1     settings slots=1, solo pass, then slots back to 2
  P4 churn      gen-1k resident in background while other 15 run sequentially, x2
  P5 cache-warm prefix-a x2, prefix-b after prefix-a, long-1 repeat (default caching)
  P6 cache-restored  /api/server stop+start (same data dir), prefix-a + long-1 repeat
  P8 app-restart (only in --restored-check mode, run by runner after an app restart)

Pass criterion per request/phase: prompt_tokens sha, generated tokens sha and every
per-step logit hash identical to the P1 baseline (contract: batch/slot/cache/speculative
paths must not change the bits). Cache phases additionally require observed evidence in
the server log (캐시/저장소 counters > 0); if evidence is absent the phase is labelled
EVIDENCE-MISSING, not passed. Exit 0 all-same, 1 difference, 2 could not finish.

Stdlib only. --selftest runs offline unit tests of compare/log parsing and exits.
"""
import argparse, concurrent.futures, datetime, hashlib, json, os, platform, random, re, subprocess, sys, time, urllib.error, urllib.request

PATH_FIELDS = ("layer", "precision", "kv8", "fast_attention", "speculative", "yarn", "template")
LOG_RE = re.compile(r"\[(\d+)\] 슬롯 (\d+) · 프롬프트 (\d+) \(캐시 (\d+) · 저장소 (\d+) · 다른 슬롯 (\d+)\)")

class PilotError(Exception):
    pass

def api(base, path, body=None, timeout=600):
    data = None if body is None else json.dumps(body, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(base + path, data=data, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        raise PilotError(f"{path}: HTTP {e.code} {e.read()[:300].decode('utf-8','replace')}") from None

def state(base):
    return api(base, "/api/state", timeout=60)

def sha256_text(s):
    return hashlib.sha256(s.encode("utf-8")).hexdigest()

def sha256_ints(xs):
    return sha256_text(",".join(str(int(x)) for x in xs))

DATA_DIR = None  # set in main(): the app's own data dir (기록/서버-*.log lives there)

def _server_log_file():
    if not DATA_DIR:
        return None
    d = os.path.join(DATA_DIR, "기록")
    try:
        files = [os.path.join(d, f) for f in os.listdir(d) if f.startswith("서버-") and f.endswith(".log")]
    except OSError:
        return None
    return max(files, key=os.path.getmtime) if files else None

def server_log_lines(st):
    sv = st.get("server") or {}
    txt = (sv.get("log") or "") + "\n" + (sv.get("log_tail") or "")
    lines = [l for l in txt.splitlines() if LOG_RE.search(l)]
    f = _server_log_file()
    if f:
        try:
            with open(f, encoding="utf-8", errors="replace") as fh:
                lines += [l for l in fh.read().splitlines() if LOG_RE.search(l)]
        except OSError:
            pass
    return lines

def parse_counters(line):
    m = LOG_RE.search(line)
    if not m:
        return None
    return {"n": int(m.group(1)), "slot": int(m.group(2)), "prompt": int(m.group(3)),
            "cache": int(m.group(4)), "store": int(m.group(5)), "other_slot": int(m.group(6))}

def log_marker(base):
    """(current server-log file, request-log line count), so a phase can diff them."""
    try:
        return (_server_log_file(), len(server_log_lines(state(base))))
    except PilotError:
        return (None, -1)

def new_counters(base, before, after_hint=0):
    """Counters parsed from log lines added since marker `before`.
    If the server restarted since the marker (new log file), count the whole new file."""
    path, n = before
    if n < 0:
        return []
    lines = server_log_lines(state(base))
    new = lines[n:] if _server_log_file() == path else lines
    return [c for c in (parse_counters(l) for l in new) if c]

def load_model(base, mid, settings):
    st = state(base)
    if (st.get("server") or {}).get("state") not in (None, "stopped", "idle", "off", ""):
        api(base, "/api/server/stop", {})
        for _ in range(120):
            if (state(base).get("server") or {}).get("state") in (None, "stopped", "idle", "off", ""):
                break
            time.sleep(1)
    api(base, "/api/settings", settings)
    api(base, "/api/server/start", {"model": mid})
    for _ in range(900):
        sv = state(base).get("server") or {}
        s = sv.get("state")
        if s == "ready":
            return sv
        if s in ("failed", "error", "stopped"):
            raise PilotError(f"model did not load (server state {s}: {sv.get('reason') or sv.get('error') or ''})")
        time.sleep(2)
    raise PilotError("model did not load within 30 minutes")

def find_model(base, spec, dirs):
    for m in state(base).get("models") or []:
        if m.get("size") == spec["size"] or m.get("bytes") == spec["size"]:
            return m["id"]
    for d in dirs:
        p = os.path.join(os.path.expanduser(d), spec["file"])
        if os.path.isfile(p) and os.path.getsize(p) == spec["size"]:
            r = api(base, "/api/models/add", {"path": os.path.abspath(p).replace("\\", "/")})
            if r.get("id"):
                return r["id"]
            for m in state(base).get("models") or []:
                if os.path.normcase(os.path.abspath(m.get("path",""))) == os.path.normcase(os.path.abspath(p)):
                    return m["id"]
    raise PilotError(f"model {spec['name']} not found")

def model_hash_ok(base, mid, spec):
    for _ in range(1800):
        m = next((x for x in state(base).get("models") or [] if x.get("id") == mid), None)
        if m is None:
            raise PilotError("model disappeared from list")
        h = m.get("sha256")
        if h and not m.get("verifying"):
            if h != spec["sha256"]:
                raise PilotError(f"{spec['file']}: SHA-256 mismatch")
            return True
        time.sleep(2)
    raise PilotError(f"{spec['file']}: SHA-256 check did not finish")

# Canonical request schema for this pinned kit. Unknown top-level fields must
# fail before ANY endpoint call rather than silently changing prompt semantics.
PROMPT_FIELDS = frozenset(("messages", "chat_template_kwargs", "tools"))
GENERATION_FIELDS = frozenset(("max_tokens", "temperature", "seed", "top_p"))
REQUEST_SCHEMA_VERSION = "semantic-request-v2-tools"


def semantic_request(rq):
    unknown = set(rq) - PROMPT_FIELDS - GENERATION_FIELDS - {"id", "kind"}
    if unknown:
        raise PilotError(f"unsupported request fields: {sorted(unknown)}; endpoint semantics unverified")
    if "messages" not in rq:
        raise PilotError("request has no messages")
    # Insertion order and nested tool schemas are preserved, never normalized.
    return {k: v for k, v in rq.items() if k not in ("id", "kind")}


def tokenize_request(rq):
    body = semantic_request(rq)
    # Only generation controls are removed. These are passed to generation by
    # the pinned kit, not referenced by the installed GGUF prompt template.
    # /tokenize support for tools is UNVERIFIED until strict live id parity.
    return {k: v for k, v in body.items() if k in PROMPT_FIELDS}


def chat_request(rq, cache_prompt=None):
    body = semantic_request(rq)
    body.update({"model": "geulpu", "stream": False,
                 "geulpu": {"logit_hash": True, "prompt_tokens": True}})
    if cache_prompt is not None:
        body["cache_prompt"] = cache_prompt
    return body


def chat(base, rq, cache_prompt=None):
    body = chat_request(rq, cache_prompt)
    t0 = time.time()
    r = api(base, "/v1/chat/completions", body, timeout=1800)
    g = r.get("geulpu") or {}
    tokens, hashes, prompt = g.get("tokens"), g.get("logit_hashes"), g.get("prompt_tokens")
    if tokens is None or hashes is None or prompt is None:
        raise PilotError(f"{rq['id']}: response lacks geulpu.tokens/logit_hashes/prompt_tokens")
    ch = (r.get("choices") or [{}])[0]
    msg = ch.get("message") or {}
    text = (msg.get("reasoning_content") or "") + "\x1f" + (msg.get("content") or "") + "\x1f" + json.dumps(
        msg.get("tool_calls") or [], ensure_ascii=False, sort_keys=True)
    return {"id": rq["id"], "n_prompt": len(prompt), "prompt_tokens_sha256": sha256_ints(prompt),
            "prompt_ids": prompt,
            "n": len(tokens), "finish": ch.get("finish_reason"), "tokens_sha256": sha256_ints(tokens),
            "logit_hashes": [str(h) for h in hashes], "text_sha256": sha256_text(text),
            "seconds": round(time.time() - t0, 1)}

def compare(res, base_res):
    """Identity vs baseline: prompt tokens, generated tokens, every per-step logit hash."""
    if base_res is None:
        return {"same": False, "why": "no baseline"}
    if res["prompt_tokens_sha256"] != base_res["prompt_tokens_sha256"]:
        return {"same": False, "why": "prompt tokens differ"}
    a, b = res["logit_hashes"], base_res["logit_hashes"]
    for i in range(min(len(a), len(b))):
        if a[i] != b[i]:
            return {"same": False, "why": f"logits differ from step {i}", "first_diff_step": i}
    if res["tokens_sha256"] != base_res["tokens_sha256"] or len(a) != len(b) or res["n"] != base_res["n"]:
        return {"same": False, "why": "generated tokens differ"}
    return {"same": True}

PREFIX = (
    "The following is a technical operations log from a distributed inference cluster. "
    + " ".join(f"Entry {i}: node rack-{i%7} reported latency {(37*i)%89+11} ms, queue depth {(13*i)%41}, "
               f"temperature {(29*i)%23+51} C, status {'ok' if i%5 else 'degraded'}." for i in range(1, 220))
)
PILOT_REQUESTS = [
    {"id": "gen-1k", "kind": "long generation",
     "messages": [{"role": "user", "content": "Write a detailed numbered explanation of how a transformer neural network works: embeddings, attention, feed-forward layers, training, and inference. Aim for a thorough answer."}],
     "max_tokens": 1024, "temperature": 0, "seed": 7},
    {"id": "prefix-a", "kind": "shared prefix A",
     "messages": [{"role": "user", "content": PREFIX + "\n\nQuestion: which rack reported the highest temperature, and in which entry?"}],
     "max_tokens": 128, "temperature": 0, "seed": 7},
    {"id": "prefix-b", "kind": "shared prefix B",
     "messages": [{"role": "user", "content": PREFIX + "\n\nQuestion: how many entries reported status degraded? Answer with just the number."}],
     "max_tokens": 128, "temperature": 0, "seed": 7},
    {"id": "chat-2", "kind": "multi-turn fixed",
     "messages": [{"role": "user", "content": "Name three prime numbers."},
                  {"role": "assistant", "content": "2, 3, and 5."},
                  {"role": "user", "content": "Square each of them."},
                  {"role": "assistant", "content": "4, 9, and 25."},
                  {"role": "user", "content": "Now sum those squares and tell me the result."}],
     "max_tokens": 128, "temperature": 0, "seed": 7},
]

def save(result, out):
    try:
        json.dump(result, open(out, "w"), ensure_ascii=False, indent=1)
    except Exception:
        pass

def save(result, out):
    try:
        json.dump(result, open(out, "w"), ensure_ascii=False, indent=1)
    except Exception:
        pass

def run_phase(base, name, reqs, baseline, result, cache_prompt=None, parallel=1, order_seed=None):
    items = list(reqs)
    if order_seed is not None:
        rnd = random.Random(order_seed)
        rnd.shuffle(items)
    mark = log_marker(base)
    outs, spans = {}, {}
    def one(rq):
        t0 = time.time()
        r = chat(base, rq, cache_prompt=cache_prompt)
        spans[rq["id"]] = (t0, time.time())
        return rq["id"], r
    if parallel > 1:
        with concurrent.futures.ThreadPoolExecutor(max_workers=parallel) as ex:
            for rid, res in ex.map(one, items):
                outs[rid] = res
    else:
        for rq in items:
            rid, res = one(rq)
            outs[rid] = res
    counters = new_counters(base, mark)
    iv = sorted(spans.values())
    overlap = any(iv[i][1] > iv[i+1][0] for i in range(len(iv) - 1))
    agg = {"cache": sum(c["cache"] for c in counters), "store": sum(c["store"] for c in counters),
           "other_slot": sum(c["other_slot"] for c in counters), "lines": len(counters),
           "slots_used": sorted({c["slot"] for c in counters}),
           "client_overlap": overlap if parallel > 1 else None}
    for rq in reqs:
        res = outs[rq["id"]]
        v = compare(res, baseline.get(rq["id"]))
        result["runs"].append({"phase": name, "id": rq["id"], "n": res["n"], "finish": res["finish"],
                               "seconds": res["seconds"], **v})
    result.setdefault("evidence", {})[name] = agg
    return outs

def selftest():
    # compare(): identical, logit-diff, token-diff, prompt-diff
    b = {"id": "x", "prompt_tokens_sha256": "p", "tokens_sha256": "t", "logit_hashes": ["a", "b", "c"], "n": 3}
    assert compare(dict(b), b)["same"]
    d = dict(b, logit_hashes=["a", "X", "c"]); v = compare(d, b)
    assert not v["same"] and v["first_diff_step"] == 1
    assert not compare(dict(b, tokens_sha256="z"), b)["same"]
    assert not compare(dict(b, prompt_tokens_sha256="z"), b)["same"]
    # log parser: cold and warm lines
    cold = parse_counters("[11] 슬롯 0 · 프롬프트 6077 (캐시 0 · 저장소 0 · 다른 슬롯 0) · 생성 71 · stop · 첫 토큰 1351.9 ms · 9.68 ms/토큰")
    warm = parse_counters("[3] 슬롯 1 · 프롬프트 1500 (캐시 1408 · 저장소 64 · 다른 슬롯 0) · 생성 128 · length · 첫 토큰 20.0 ms · 9.1 ms/토큰")
    assert cold == {"n": 11, "slot": 0, "prompt": 6077, "cache": 0, "store": 0, "other_slot": 0}
    assert warm["cache"] == 1408 and warm["store"] == 64
    # shuffle determinism
    r1 = random.Random(7); a = list(range(16)); r1.shuffle(a)
    r2 = random.Random(7); bl = list(range(16)); r2.shuffle(bl)
    assert a == bl
    print("SELFTEST OK")
    return 0

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:44544")
    ap.add_argument("--models-dir", action="append", default=[])
    ap.add_argument("--data", default=os.environ.get("GEULPU_HOME"), help="app data dir (for 기록/서버-*.log request counters)")
    ap.add_argument("--requests", required=True)
    ap.add_argument("--expected", help="kit expected.json, for cross-checking the baseline of the 12 kit requests")
    ap.add_argument("--out", required=True)
    ap.add_argument("--smoke", action="store_true", help="2-request smoke gate only")
    ap.add_argument("--restored-check", action="store_true", help="only P8: verify SSD-store restore after a full app restart")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    global DATA_DIR
    DATA_DIR = a.data
    if a.selftest:
        return selftest()
    base = a.url.rstrip("/")
    kit = json.load(open(a.requests, encoding="utf-8"))
    spec = next(m for m in kit["models"] if m["key"] == "qwen3-8b-q4_k_m")
    settings = dict(kit["configs"]["default"])
    reqs = list(kit["requests"]) + PILOT_REQUESTS
    state(base)
    st = state(base)
    if (st.get("job") or {}).get("state") == "running":
        raise PilotError("app busy with another job")
    env = {"time_utc": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
           "geulpu_version": (st.get("app") or {}).get("version"), "geulpu_commit": (st.get("app") or {}).get("commit"),
           "device": (st.get("device") or {}).get("name"), "python": platform.python_version()}
    print(f"pilot on {env['device']}, GeulPU {env['geulpu_version']} ({env['geulpu_commit']})", flush=True)
    mid = find_model(base, spec, a.models_dir)
    print("checking model SHA-256 ...", flush=True)
    model_hash_ok(base, mid, spec)
    sv = load_model(base, mid, settings)
    path = {k: sv.get(k) for k in PATH_FIELDS}
    print("server path:", json.dumps(path), flush=True)
    result = {"pilot": "cache+scheduling-1", "environment": env, "settings": settings, "path": path,
              "runs": [], "evidence": {}, "notes": []}

    if a.smoke:
        rq = reqs[1]   # math-2, short greedy
        print("smoke: cache_prompt=false semantics check ...", flush=True)
        marks = []
        for _ in range(2):
            m = log_marker(base)
            res = chat(base, rq, cache_prompt=False)
            marks.append(new_counters(base, m))
            print(f"  cache_prompt=false run: {res['n']} tokens {res['seconds']} s, counters {marks[-1]}", flush=True)
        off_hits = sum(c["cache"] + c["store"] + c["other_slot"] for run in marks for c in run)
        m = log_marker(base)
        chat(base, rq)   # default caching: warm
        res = chat(base, rq)   # should hit
        hit = sum(c["cache"] + c["store"] + c["other_slot"] for c in new_counters(base, m))
        print(f"  default-caching repeat: counters sum {hit}", flush=True)
        ok_off = off_hits == 0 and all(marks[i] for i in range(2))
        ok_on = hit > 0
        if not ok_off:
            why = "no request-log lines observed (evidence source broken)" if not all(marks[i] for i in range(2)) else "cache_prompt=false still served/stored cache"
        elif not ok_on:
            why = "default-caching repeat showed no cache hit (semantics unverifiable)"
        else:
            why = ""
        result["summary"] = {"smoke": "ok" if (ok_off and ok_on) else "FAILED", "why": why,
                             "cache_prompt_false_hits": off_hits, "default_repeat_hits": hit}
        json.dump(result, open(a.out, "w"), ensure_ascii=False, indent=1)
        if not ok_off:
            print("smoke FAILED:", result["summary"]["why"], flush=True)
            return 2
        if not ok_on:
            print("smoke FAILED: repeated request with default caching showed no cache evidence", flush=True)
            return 2
        return 0

    baseline_file = a.out.replace(".json", "-baseline.json")
    if a.restored_check:
        try:
            baseline = {r["id"]: r for r in json.load(open(baseline_file))}
            outs = {}
            mark = log_marker(base)
            for rid in ("prefix-a", "long-1"):
                rq = next(r for r in reqs if r["id"] == rid)
                outs[rid] = chat(base, rq)   # default caching -> SSD store restore expected
            counters = new_counters(base, mark)
            ev = {"cache": sum(c["cache"] for c in counters), "store": sum(c["store"] for c in counters)}
            ok = all(compare(outs[rid], baseline[rid])["same"] for rid in outs)
            result["runs"] = [{"phase": "P8-app-restart", "id": rid, **compare(outs[rid], baseline[rid])} for rid in outs]
            result["evidence"]["P8-app-restart"] = ev
            result["summary"] = {"identical": sum(r["same"] for r in result["runs"]), "total": len(result["runs"]),
                                 "restore_evidence": "VERIFIED" if ev["store"] > 0 else "EVIDENCE-MISSING"}
            out2 = a.out[:-5] + "-restored.json" if a.out.endswith(".json") else a.out + "-restored.json"
            json.dump(result, open(out2, "w"), ensure_ascii=False, indent=1)
            print(json.dumps(result["summary"]), flush=True)
            return 0 if ok and ev["store"] > 0 else 1
        except PilotError:
            raise
        except Exception as e:
            result["summary"] = {"status": "INCOMPLETE", "error": repr(e)}
            save(result, a.out[:-5] + "-restored.json" if a.out.endswith(".json") else a.out)
            print(f"restored-check INCOMPLETE: {e!r}", flush=True)
            return 2
    # P1 baseline onward runs inside run_full; unexpected exceptions -> exit 2 with partial output
    try:
        return run_full(a, base, kit, reqs, mid, settings, baseline_file, result)
    except PilotError:
        raise
    except Exception as e:
        result["summary"] = {"status": "INCOMPLETE", "error": repr(e)}
        save(result, a.out)
        print(f"pilot INCOMPLETE: {e!r}", flush=True)
        return 2

def run_full(a, base, kit, reqs, mid, settings, baseline_file, result):
    print("P1 baseline (solo, cache_prompt=false) ...", flush=True)
    outs1 = run_phase(base, "P1-baseline", reqs, {}, result, cache_prompt=False)
    baseline = outs1
    json.dump([strip(r) for r in (dict(v, id=k) for k, v in baseline.items())], open(baseline_file, "w"), ensure_ascii=False, indent=1)
    # cross-check the 12 kit requests against expected.json (same config) when provided
    if a.expected:
        try:
            exp_items = json.load(open(a.expected))["items"]["qwen3-8b-q4_k_m/default"]
            need = {"id", "n_prompt", "prompt_tokens_sha256", "n", "tokens_sha256", "logit_hashes"}
            exp = {e["id"]: e for e in exp_items if need <= set(e)}
            n_ok = sum(compare(stripv(baseline[rid]), exp.get(rid))["same"] for rid in (e["id"] for e in kit["requests"]) if rid in exp)
            result["notes"].append(f"kit-reference cross-check: {n_ok}/{len(exp)} match expected.json (same default config)")
        except (KeyError, ValueError, TypeError) as e:
            result["notes"].append(f"kit-reference cross-check SKIPPED (schema/file issue: {e!r})")
        print(result["notes"][-1], flush=True)
    for rq in reqs:
        v = {"same": True}
        result["runs"].append({"phase": "P1-baseline-ref", "id": rq["id"], **v})
    result["runs"] = [r for r in result["runs"] if r["phase"] != "P1-baseline"]

    print("P2 batch shuffled x3 (8 concurrent into 2 slots) ...", flush=True)
    for i in range(3):
        run_phase(base, f"P2-batch-shuffle{i+1}", reqs, baseline, result, cache_prompt=False, parallel=8, order_seed=100 + i)
        save(result, a.out)
    print("P3 slots=1 solo ...", flush=True)
    sv = load_model(base, mid, dict(settings, slots=1))
    result["path_slots1"] = {k: sv.get(k) for k in PATH_FIELDS}
    run_phase(base, "P3-slots1", reqs, baseline, result, cache_prompt=False)
    save(result, a.out)
    sv = load_model(base, mid, settings)   # back to slots=2
    print("P4 churn x2 (gen-1k resident) ...", flush=True)
    for i in range(2):
        long_rq = next(r for r in reqs if r["id"] == "gen-1k")
        others = [r for r in reqs if r["id"] != "gen-1k"]
        random.Random(200 + i).shuffle(others)
        mark = log_marker(base)
        spans_long, spans_others = [], {}
        def timed(rq):
            t0 = time.time()
            r = chat(base, rq, False)
            return rq["id"], r, (t0, time.time())
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:
            fut_long = ex.submit(timed, long_rq)
            time.sleep(1.5)   # let the long generation become resident
            outs = {}
            for rq in others:
                rid, r, sp = timed(rq)
                outs[rid] = r
                spans_others[rid] = sp
            rid, r, sp = fut_long.result()
            outs["gen-1k"] = r
            spans_long.append(sp)
        counters = new_counters(base, mark)
        t0l, t1l = spans_long[0]
        overlap_n = sum(1 for rid, (t0, t1) in spans_others.items() if t0 < t1l and t1 > t0l)
        result["evidence"][f"P4-churn{i+1}"] = {"cache": sum(c["cache"] for c in counters),
                                                "store": sum(c["store"] for c in counters), "lines": len(counters),
                                                "slots_used": sorted({c["slot"] for c in counters}),
                                                "resident_long_seconds": round(t1l - t0l, 1),
                                                "requests_overlapping_resident": overlap_n,
                                                "resident_overlap": "VERIFIED" if overlap_n > 0 else "EVIDENCE-MISSING"}
        for rq in reqs:
            result["runs"].append({"phase": f"P4-churn{i+1}", "id": rq["id"], **compare(outs[rq["id"]], baseline[rq["id"]])})
        save(result, a.out)
    print("P5 cache warm (default caching) ...", flush=True)
    seq = ["prefix-a", "prefix-a", "prefix-b", "long-1", "long-1"]
    outs5 = {}
    mark = log_marker(base)
    for i, rid in enumerate(seq):
        rq = next(r for r in reqs if r["id"] == rid)
        outs5[f"{rid}#{i}"] = chat(base, rq)
    c5 = new_counters(base, mark)
    result["evidence"]["P5-cache-warm"] = {"per_line": c5, "cache": sum(c["cache"] for c in c5),
                                           "store": sum(c["store"] for c in c5)}
    for i, rid in enumerate(seq):
        result["runs"].append({"phase": "P5-cache-warm", "id": f"{rid}#{i}", **compare(outs5[f"{rid}#{i}"], baseline[rid])})
    save(result, a.out)
    print("P6 server restart -> store restore ...", flush=True)
    api(base, "/api/server/stop", {})
    for _ in range(120):
        if (state(base).get("server") or {}).get("state") in (None, "stopped", "idle", "off", ""):
            break
        time.sleep(1)
    api(base, "/api/server/start", {"model": mid})
    for _ in range(900):
        if (state(base).get("server") or {}).get("state") == "ready":
            break
        time.sleep(2)
    mark = log_marker(base)
    outs6 = {}
    for rid in ("prefix-a", "long-1"):
        rq = next(r for r in reqs if r["id"] == rid)
        outs6[rid] = chat(base, rq)
    c6 = new_counters(base, mark)
    result["evidence"]["P6-server-restart"] = {"per_line": c6, "store": sum(c["store"] for c in c6)}
    for rid in outs6:
        result["runs"].append({"phase": "P6-server-restart", "id": rid, **compare(outs6[rid], baseline[rid])})
    save(result, a.out)
    # verdicts: evidence labels for cache phases
    ev5, ev6 = result["evidence"].get("P5-cache-warm", {}), result["evidence"].get("P6-server-restart", {})
    diffs0 = [r for r in result["runs"] if not r.get("same")]
    def cls(d):
        w = d.get("why", "")
        return "prompt-tokens-differ(input changed; triage, not path nondeterminism)" if "prompt" in w else                "logits-differ(candidate divergence; triage needed)" if "logits" in w else                "tokens-differ(candidate divergence; triage needed)"
    result["summary"] = {
        "identical": sum(1 for r in result["runs"] if r.get("same")), "total": len(result["runs"]),
        "candidate_divergences": [{"phase": d["phase"], "id": d["id"], "class": cls(d)} for d in diffs0],
        "P5_cache_evidence": "VERIFIED" if (ev5.get("cache", 0) + ev5.get("store", 0)) > 0 else "EVIDENCE-MISSING",
        "P6_restore_evidence": "VERIFIED" if ev6.get("store", 0) > 0 else "EVIDENCE-MISSING"}
    save(result, a.out)
    print(json.dumps(result["summary"]), flush=True)
    diffs = [r for r in result["runs"] if not r.get("same")]
    for d in diffs[:10]:
        print("DIFF:", json.dumps(d)[:200], flush=True)
    return 1 if diffs else 0

def strip(r):
    return {k: v for k, v in r.items() if k != "seconds"}

def stripv(r):
    return r

if __name__ == "__main__":
    try:
        sys.exit(main())
    except PilotError as e:
        print(f"pilot stopped: {e}", flush=True)
        sys.exit(2)
