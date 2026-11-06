#!/usr/bin/env python3
"""GeulPU core stress test, round 2 (modules M2-M5 of the approved core design).

Reuses the reviewed v6 machinery from stress_pilot.py (imported). Runs on the same
three boxes with the same safety posture: own data dir, model/wheel hash gates in
the runner, smoke gate before this full run.

Modules (GPU):
  B0  cold solo references (cache_prompt=false) for every request used below.
      The two context-boundary requests double as M2 (solo cold boundary runs):
        ctx-long: prompt <=5500 tokens + 2048 max gen  (total <=7548 <= 8192)
        ctx-max:  prompt ~7700 tokens  + 250  max gen  (total <=8120, reserve stated)
      Prompt-token bounds are asserted from ACTUAL tokenization at runtime
      (review criterion 3); a violation aborts with INCOMPLETE, never a pass.
  M3  sustained churn: resident gen-4k in background + 15 staggered arrivals
      (1 s submission stagger, concurrent in flight). Per-request wall spans and
      completion index recorded; overlap with the resident is MEASURED and
      reported as a count (>=14/15 is a coverage statement, not a fudge).
  M4  KV eviction: three distinct ~7.5-7.7k-token prompts A,B,C sequentially at
      slots=2 (KV capacity 2x8192=16384; A+B+C ~22.8k forces eviction), then
      re-request A. Per-request log-line counters attribute cold/restore/hit;
      eviction evidence is classified and labeled (EVIDENCE-MISSING if the
      re-request lands fully in memory). Bit-identity vs cold B0 references.
  M5  slots=4 burst: settings slots=4 applied and VERIFIED via /api/state; if
      the app does not accept it, the module reports FAILED-UNSUPPORTED (no
      assumed fallback). 16 requests, 4-way parallel submission, fixed seed.
      Settings restored to the original afterwards.

M1 (cross-box baseline formalization), M6 (cross-GPU comparison) and M7
(performance) are zero-GPU analyses over these artifacts; they run offline
(core_analysis.py), not here.

Exit codes: 0 = all comparisons identical AND every module executed with its
evidence; 1 = candidate divergences (triage classes); 2 = INCOMPLETE
(exception, watchdog, or a module that could not execute).
"""
import argparse, concurrent.futures, datetime, hashlib, json, os, platform, random, re, sys, threading, time

import stress_pilot as sp
from stress_pilot import PilotError, api, state, chat, compare, log_marker, new_counters, \
    find_model, model_hash_ok, load_model, save

CTX = 8192
RESERVE_MIN = 72            # slack demanded beyond prompt+gen at the near-limit case
BOUNDS = {"ctx-long": (4500, 5500, 2048), "ctx-max": (7400, 7870, 250)}
EVICT_BOUNDS = (7300, 7870)               # evict-a/b/c prompt-token bounds
SEED = 7


def long_prompt(tag, n_entries, question, num_off=0):
    """Deterministic long prompt, same generator family as pilot-1's PREFIX
    (219 entries ~ 6882 prompt tokens on all three boxes). The tag appears in
    every entry and a per-tag numeric offset shifts the values, so distinct
    tags share no exploitable prefix (M4 needs genuinely distinct prompts)."""
    intro = ("The following is a technical operations log from a distributed inference "
             f"cluster, series {tag}. ")
    body = " ".join(
        f"Entry {i} [{tag}]: node rack-{i % 7} reported latency {(37 * i) % 89 + 11 + num_off} ms, "
        f"queue depth {(13 * i) % 41}, temperature {(29 * i) % 23 + 51 + num_off} C, "
        f"status {'ok' if i % 5 else 'degraded'}."
        for i in range(1, n_entries + 1))
    return intro + body + "\n\n" + question


CORE_SPECS = [
    {"id": "gen-4k", "kind": "resident long generation", "sized": False,
     "messages": [{"role": "user", "content":
                   "Write a detailed numbered explanation of how a transformer neural network works: "
                   "embeddings, attention, feed-forward layers, training, and inference. Aim for a "
                   "thorough and extensive answer."}],
     "max_tokens": 4096, "temperature": 0, "seed": SEED},
    {"id": "ctx-long", "kind": "context boundary: long generation", "sized": True,
     "tag": "L", "start": 160, "question": "Question: which rack reported the highest temperature, and in which entry?",
     "num_off": 0, "max_tokens": 2048, "temperature": 0, "seed": SEED},
    {"id": "ctx-max", "kind": "context boundary: near-limit prompt", "sized": True,
     "tag": "M", "start": 225, "question": "Question: how many entries reported status degraded? Answer with just the number.",
     "num_off": 0, "max_tokens": 250, "temperature": 0, "seed": SEED},
    {"id": "evict-a", "kind": "eviction prompt A", "sized": True,
     "tag": "EA", "start": 225, "question": "Question: which entry had the lowest latency?", "num_off": 0,
     "max_tokens": 64, "temperature": 0, "seed": SEED},
    {"id": "evict-b", "kind": "eviction prompt B", "sized": True,
     "tag": "EB", "start": 218, "question": "Question: which entry had the highest queue depth?", "num_off": 4,
     "max_tokens": 64, "temperature": 0, "seed": SEED},
    {"id": "evict-c", "kind": "eviction prompt C", "sized": True,
     "tag": "EC", "start": 221, "question": "Question: how many entries were on rack-3?", "num_off": 9,
     "max_tokens": 64, "temperature": 0, "seed": SEED},
]
SIZED_BOUNDS = {"ctx-long": BOUNDS["ctx-long"][:2], "ctx-max": BOUNDS["ctx-max"][:2],
                "evict-a": EVICT_BOUNDS, "evict-b": EVICT_BOUNDS, "evict-c": EVICT_BOUNDS}


def fingerprint(rq, settings, ident=None):
    """Content+config+identity fingerprint of a request. References may be
    reused on --resume only when this matches exactly (no id-only reuse).
    Identity = GeulPU version+commit, model id, resolved chat template kwargs:
    a wheel, model or template change between launches invalidates reuse (B3)."""
    ident = ident or {}
    return sp.sha256_text(json.dumps(
        {"request": sp.semantic_request(rq),
         "schema": sp.REQUEST_SCHEMA_VERSION,
         "settings": {k: settings.get(k) for k in
                      ("precision", "kv8_cache", "fast_attention", "slots", "context")},
         "geulpu_version": ident.get("version"), "geulpu_commit": ident.get("commit"),
         "model": ident.get("model")},
        sort_keys=False, ensure_ascii=False))


def auto_size(count_tokens, spec):
    """Size a long prompt by MEASURED tokenization (never estimates): probe the
    per-entry slope with two actual tokenize counts, jump to the bound midpoint,
    then converge with measured counts. Returns (request, entries, measured)."""
    lo, hi = SIZED_BOUNDS[spec["id"]]
    def mk(n):
        return [{"role": "user", "content": long_prompt(spec["tag"], n, spec["question"], spec["num_off"])}]
    c10, c60 = count_tokens(mk(10)), count_tokens(mk(60))
    slope = (c60 - c10) / 50.0
    if not slope > 0:
        raise ContextSizeError(f"{spec['id']}: tokenize probe showed no growth ({c10}->{c60}); cannot size")
    intercept = c10 - slope * 10
    target = (lo + hi) // 2
    n = max(20, round((target - intercept) / slope))
    for _ in range(8):
        c = count_tokens(mk(n))
        if lo <= c <= hi:
            rq = {"id": spec["id"], "kind": spec["kind"], "messages": mk(n),
                  "max_tokens": spec["max_tokens"], "temperature": 0, "seed": SEED}
            return rq, n, c
        n = max(20, n + round((target - c) / slope))
    raise ContextSizeError(f"{spec['id']}: could not size into [{lo},{hi}] after 8 measured iterations")


class ContextSizeError(Exception):
    """A constructed prompt exceeded the server context (measured, not estimated)."""


class ModuleTimeout(Exception):
    """A module exceeded its per-module watchdog; module = INCOMPLETE-TIMEOUT."""


# Per-module watchdogs (seconds), generous vs estimates derived from measured
# speeds (slowest test machine ~2x estimate). Checked after each request; a single
# in-flight request is bounded by the 1800 s chat timeout, and the runner's
# absolute hard stop bounds everything. These are coverage watchdogs, not a
# user GPU-time budget.
MODULE_DEADLINE = {"B0": 3600, "M3": 2400, "M4": 1500, "M5": 2400}


class Recorder:
    """Per-request instrumentation: wall span, completion index, counters."""
    def __init__(self, base):
        self.base = base
        self.recs = {}
        self._idx = 0
        self._lock = threading.Lock()
        self.mark = log_marker(base)

    def work(self, rq, cache_prompt=None):
        t0 = time.time()
        r = chat(self.base, rq, cache_prompt)
        t1 = time.time()
        with self._lock:
            idx = self._idx
            self._idx += 1
        self.recs[rq["id"]] = {"res": r, "t0": t0, "t1": t1, "idx": idx}

    def run(self, items, parallel=1, stagger=0.0, cache_prompt=None, order_seed=None):
        items = list(items)
        if order_seed is not None:
            random.Random(order_seed).shuffle(items)
        if parallel == 1 and stagger == 0.0:
            for rq in items:
                self.work(rq, cache_prompt)
        else:
            with concurrent.futures.ThreadPoolExecutor(max_workers=parallel) as ex:
                futs = []
                for rq in items:
                    futs.append(ex.submit(self.work, rq, cache_prompt))
                    if stagger:
                        time.sleep(stagger)
                for f in futs:
                    f.result()
        self.counters = new_counters(self.base, self.mark)

    def record(self, result, phase, baseline):
        instr = result["instrumentation"].setdefault(phase, {})
        outs = result["outputs"].setdefault(phase, {})
        for rid, rec in self.recs.items():
            outs[rid] = sp.strip(rec["res"])
            instr[rid] = {"t0": round(rec["t0"], 2), "t1": round(rec["t1"], 2),
                          "idx": rec["idx"], "seconds": rec["res"]["seconds"],
                          "n_prompt": rec["res"]["n_prompt"], "n": rec["res"]["n"],
                          "finish": rec["res"]["finish"]}
            result["runs"].append({"phase": phase, "id": rid,
                                   **compare(rec["res"], baseline.get(rid))})
        result["evidence"][phase] = {"lines": len(self.counters),
                                     "cache": sum(c["cache"] for c in self.counters),
                                     "store": sum(c["store"] for c in self.counters),
                                     "other_slot": sum(c["other_slot"] for c in self.counters),
                                     "per_line": self.counters}


def check_bounds(res, lo, hi, max_tokens, rid):
    n = res["n_prompt"]
    if not (lo <= n <= hi):
        raise PilotError(f"{rid}: actual prompt tokens {n} outside design bounds [{lo},{hi}]")
    if n + max_tokens > CTX - RESERVE_MIN and rid == "ctx-max":
        raise PilotError(f"{rid}: prompt {n} + gen {max_tokens} leaves <{RESERVE_MIN} tokens of reserve in {CTX}")
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:44544")
    ap.add_argument("--models-dir", action="append", default=[])
    ap.add_argument("--data", default=os.environ.get("GEULPU_HOME"))
    ap.add_argument("--requests")
    ap.add_argument("--expected")
    ap.add_argument("--out")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--resume", action="store_true",
                    help="skip B0 requests and whole modules already present in --out")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    sp.DATA_DIR = a.data
    if a.selftest:
        return selftest()
    if not a.requests or not a.out:
        ap.error("--requests and --out are required (except with --selftest)")
    base = a.url.rstrip("/")
    kit = json.load(open(a.requests, encoding="utf-8"))
    spec = next(m for m in kit["models"] if m["key"] == "qwen3-8b-q4_k_m")
    settings = dict(kit["configs"]["default"])
    # Kit manifest requests are preserved VERBATIM; constructed requests are
    # copied so calibration-time kwargs attachment never mutates shared specs.
    reqs16 = list(kit["requests"]) + [dict(r) for r in sp.PILOT_REQUESTS]
    # Manifest identity binds FULL constructed-spec content (not just ids): a
    # semantics change (e.g. template-kwargs attachment) invalidates ALL prior
    # constructed references/modules through the existing B3 identity gate.
    manifest_sha = hashlib.sha256(
        open(a.requests, "rb").read() + sp.REQUEST_SCHEMA_VERSION.encode()
        + json.dumps(CORE_SPECS, sort_keys=True, ensure_ascii=False).encode()
        + json.dumps(sp.PILOT_REQUESTS, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    st = state(base)
    if (st.get("job") or {}).get("state") == "running":
        print("core pilot INCOMPLETE: app busy with another job", flush=True)
        return 2
    env = {"time_utc": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
           "geulpu_version": (st.get("app") or {}).get("version"),
           "geulpu_commit": (st.get("app") or {}).get("commit"),
           "device": (st.get("device") or {}).get("name"),
           "python": platform.python_version()}
    print(f"core pilot on {env['device']}, GeulPU {env['geulpu_version']} ({env['geulpu_commit']})", flush=True)
    mid = find_model(base, spec, a.models_dir)
    print("checking model SHA-256 ...", flush=True)
    model_hash_ok(base, mid, spec)
    sv = load_model(base, mid, settings)
    result = {"pilot": "core-stress-2", "environment": env, "settings": settings,
              "path": {k: sv.get(k) for k in sp.PATH_FIELDS},
              "runs": [], "evidence": {}, "instrumentation": {}, "outputs": {}, "modules": {}, "notes": [],
              "manifest_sha256": manifest_sha, "model_id": mid}

    if a.smoke:
        return smoke(a, base, reqs16, result)

    result["b0_fingerprints"] = {}   # stamped per request at execution time only (never pre-stamped)

    refs_file = a.out.replace(".json", "-references.json")
    core_reqs = []   # built inside phase_b0 after the parity probe (list mutated in place)
    old = None
    if a.resume and os.path.exists(a.out):
        try:
            old = json.load(open(a.out))
        except Exception as e:
            print(f"resume: could not parse existing output, starting fresh ({e!r})", flush=True)
            old = None
    if old is not None:
        # identity gates (B3): a version/commit/model/manifest change between
        # launches invalidates ALL resume reuse (request skips + module carry)
        old_env = old.get("environment") or {}
        idm = [k for k in ("geulpu_version", "geulpu_commit") if old_env.get(k) != env.get(k)]
        if old.get("model_id") != mid:
            idm.append("model")
        if old.get("manifest_sha256") != manifest_sha:
            idm.append("manifest")
        if idm:
            result["notes"].append(f"resume: run identity changed ({', '.join(idm)}); "
                                   f"all resume skips and module carry invalidated, full re-run")
            old = None
    try:
        refs = {}
        if os.path.exists(refs_file):
            refs = {r["id"]: r for r in json.load(open(refs_file))}
        try:
            refs, done_mods = phase_b0(a, base, kit, reqs16, core_reqs, refs, refs_file, result, old)
        except ContextSizeError as e:
            result["modules"]["B0"] = {"status": "INCOMPLETE-OVERSIZE", "why": str(e)}
            result["summary"] = {"status": "INCOMPLETE", "error": str(e)}
            save(result, a.out)
            print(f"B0 INCOMPLETE-OVERSIZE: {e}", flush=True)
            return 2
        except ModuleTimeout as e:
            result["modules"]["B0"] = {"status": "INCOMPLETE-TIMEOUT", "why": str(e)}
            result["summary"] = {"status": "INCOMPLETE",
                                 "error": f"B0 watchdog: {e}; rerun with --resume to continue from saved references"}
            save(result, a.out)
            print(f"B0 INCOMPLETE-TIMEOUT: {e}", flush=True)
            return 2
        for m in sorted(done_mods):
            carry_module(result, old, m)
        if result["modules"].get("B0", {}).get("status") != "DONE":
            result["notes"].append("B0 not DONE: M3/M4/M5 not attempted this launch")
        for m, fn in (("M3", phase_m3), ("M4", phase_m4), ("M5", phase_m5)):
            if result["modules"].get("B0", {}).get("status") != "DONE":
                break
            if m in done_mods:
                print(f"{m}: already DONE (resume)", flush=True)
                continue
            try:
                fn(a, base, reqs16, core_reqs, refs, mid, settings, result)
                mark_module_done(result, m)   # never overwrite a phase-recorded terminal status (B5)
            except ModuleTimeout as e:
                result["modules"][m] = {"status": "INCOMPLETE-TIMEOUT", "why": str(e)}
                print(f"{m} INCOMPLETE-TIMEOUT: {e}", flush=True)
            save(result, a.out)
        finalize(result)
        save(result, a.out)
        print(json.dumps(result["summary"]), flush=True)
        diffs = [r for r in result["runs"] if not r.get("same")]
        for d in diffs[:10]:
            print("DIFF:", json.dumps(d)[:200], flush=True)
        if result["summary"]["coverage_incomplete"]:
            return 2
        return 1 if diffs else 0
    except PilotError as e:
        result["summary"] = {"status": "INCOMPLETE", "error": repr(e),
                             "modules": {m: v.get("status") for m, v in result["modules"].items()}}
        save(result, a.out)
        print(f"core pilot INCOMPLETE: {e!r}", flush=True)
        return 2
    except Exception as e:
        result["summary"] = {"status": "INCOMPLETE", "error": repr(e),
                             "modules": {m: v.get("status") for m, v in result["modules"].items()}}
        save(result, a.out)
        print(f"core pilot INCOMPLETE: {e!r}", flush=True)
        return 2


def resume_selection(old, current_fps=None):
    """What a --resume may skip. Repair rule: a DONE module whose coverage
    evidence failed its gate (M3 overlap NOT-MET, M4 eviction EVIDENCE-MISSING)
    is NOT skipped - it re-runs. A B0 request is skipped only if its recorded
    evidence was cold=True (no vacuous or warm-line carryover)."""
    old_mods = old.get("modules") or {}
    old_ev = old.get("evidence") or {}
    def module_clean(m):
        if (old_mods.get(m) or {}).get("status") != "DONE":
            return False
        if m == "M3":
            return (old_ev.get("M3-churn") or {}).get("overlap_target_14_of_15") == "MET"
        if m == "M4":
            return not str((old_ev.get("M4-eviction") or {}).get("re_request_classification", "")
                           ).startswith("EVIDENCE-MISSING")
        return True
    old_fps = old.get("b0_fingerprints") or {}
    done_mods = {m for m in ("M3", "M4", "M5") if module_clean(m)}
    done_reqs = set()
    for rid in (old.get("b0_completed") or []):
        if (old_ev.get(f"B0:{rid}") or {}).get("cold") is not True:
            continue
        if current_fps is not None and old_fps.get(rid) != current_fps.get(rid):
            continue   # content/config changed -> reference invalid, re-run
        done_reqs.add(rid)
    return done_mods, done_reqs


def resume_reqs(old, current_fps):
    """B0 requests a --resume may skip: recorded cold=True AND fingerprint match
    against the freshly built prompts. math-2 is NEVER skipped (live parity)."""
    if not old:
        return set()
    old_ev = old.get("evidence") or {}
    old_fps = old.get("b0_fingerprints") or {}
    out = set()
    for rid in (old.get("b0_completed") or []):
        if rid == "math-2":
            continue
        if (old_ev.get(f"B0:{rid}") or {}).get("cold") is True and old_fps.get(rid) == current_fps.get(rid):
            out.add(rid)
    return out


CARRY_PHASES = {"M3": "M3-churn", "M4": "M4-eviction", "M5": "M5-slots4"}


def carry_module(result, old, m):
    """Carry a coverage-clean skipped module into the new result WITH its
    evidence, outputs, instrumentation and runs (B1): a carried module stays
    fully visible to finalize's coverage gates, to M6/M7 analysis and to the
    exit-1 divergence path. Everything carried is labeled carried_from."""
    provenance = "prior launch (coverage gate passed, identity verified)"
    result["modules"][m] = dict((old.get("modules") or {}).get(m) or {}, carried_from=provenance)
    phase = CARRY_PHASES[m]
    old_ev = (old.get("evidence") or {}).get(phase)
    if old_ev is not None:
        result["evidence"][phase] = dict(old_ev, carried_from=provenance)
    for bucket in ("outputs", "instrumentation"):
        old_b = (old.get(bucket) or {}).get(phase)
        if old_b is not None:
            result.setdefault(bucket, {})[phase] = old_b
    for run in (old.get("runs") or []):
        if run.get("phase") == phase:
            result["runs"].append(dict(run, carried_from=provenance))


def mark_module_done(result, m):
    """DONE only when the phase recorded no status of its own (B5): never
    overwrite a phase-recorded terminal status like FAILED-UNSUPPORTED (M5
    slots rejection) or INCOMPLETE-*."""
    if result["modules"].get(m, {}).get("status") is None:
        result["modules"].setdefault(m, {})["status"] = "DONE"


PARITY_CANDIDATES = [None, {"enable_thinking": True}, {"enable_thinking": False}]


def smoke(a, base, reqs16, result):
    rq = reqs16[1]   # math-2, short greedy (same smoke request as pilot 1)
    print("smoke: cache_prompt=false semantics check ...", flush=True)
    marks = []
    for _ in range(2):
        m = log_marker(base)
        res = chat(base, rq, cache_prompt=False)
        marks.append(new_counters(base, m))
        print(f"  cache_prompt=false run: {res['n']} tokens {res['seconds']} s, counters {marks[-1]}", flush=True)
    off_hits = sum(c["cache"] + c["store"] + c["other_slot"] for run in marks for c in run)
    m = log_marker(base)
    chat(base, rq)
    chat(base, rq)
    hit = sum(c["cache"] + c["store"] + c["other_slot"] for c in new_counters(base, m))
    print(f"  default-caching repeat: counters sum {hit}", flush=True)
    ok_off = off_hits == 0 and all(marks[i] for i in range(2))
    ok_on = hit > 0
    why = "" if (ok_off and ok_on) else (
        "no request-log lines observed (evidence source broken)" if not all(marks[i] for i in range(2))
        else "cache_prompt=false still served/stored cache" if not ok_off
        else "default-caching repeat showed no cache hit (semantics unverifiable)")
    result["summary"] = {"smoke": "ok" if (ok_off and ok_on) else "FAILED", "why": why,
                         "cache_prompt_false_hits": off_hits, "default_repeat_hits": hit}
    json.dump(result, open(a.out, "w"), ensure_ascii=False, indent=1)
    if not ok_off or not ok_on:
        print("smoke FAILED:", why, flush=True)
        return 2
    return 0


def phase_b0(a, base, kit, reqs16, core_reqs, refs, refs_file, result, old=None):
    """Cold solo references (cache_prompt=false) for all core requests; the two
    context-boundary requests double as M2 with runtime tokenization asserts.
    Order: math-2 live (never skipped) -> template parity CALIBRATION (token-id
    sha preferred; a count match never overrides a sha mismatch) -> resolved-
    template resume gate -> measured sizing -> 1-token long-shape probe (B2) ->
    fingerprinted resume skip selection -> execution (watchdog per request) ->
    multi-shape parity verification over everything executed -> evidence-
    completeness gate (A2). Returns (refs, done_mods); module carry is decided
    here, AFTER parity, so a template change invalidates module reuse too."""
    print("B0 references (solo, cache_prompt=false) ...", flush=True)
    seen, all16 = set(), []
    for rq in reqs16:   # dedupe by id (defensive); math-2 leads
        if rq["id"] not in seen:
            seen.add(rq["id"])
            all16.append(rq)
    all16.sort(key=lambda r: r["id"] != "math-2")
    t_mod = time.time()
    ident = {"version": result["environment"]["geulpu_version"],
             "commit": result["environment"]["geulpu_commit"],
             "model": result.get("model_id"), "kwargs": None}
    executed = []   # ids executed live THIS launch (multi-shape parity below)

    def stamp_and_save(rq, rec):
        result.setdefault("b0_completed", []).append(rq["id"])
        r = rec.recs[rq["id"]]["res"]
        result["outputs"].setdefault("B0", {})[rq["id"]] = sp.strip(r)
        result["instrumentation"].setdefault("B0", {})[rq["id"]] = {
            "t0": round(rec.recs[rq["id"]]["t0"], 2), "t1": round(rec.recs[rq["id"]]["t1"], 2),
            "idx": rec.recs[rq["id"]]["idx"], "seconds": r["seconds"],
            "n_prompt": r["n_prompt"], "n": r["n"], "finish": r["finish"]}
        result["b0_fingerprints"][rq["id"]] = fingerprint(rq, result["settings"], ident)
        result["evidence"][f"B0:{rq['id']}"] = {"per_line": rec.counters,
            "cold": bool(rec.counters) and not any(
                c["cache"] or c["store"] or c["other_slot"] for c in rec.counters)}
        save(result, a.out)
        json.dump([dict(sp.strip(dict(v, id=k)), fingerprint=result["b0_fingerprints"].get(k))
                   for k, v in refs.items()], open(refs_file, "w"), ensure_ascii=False, indent=1)
        print(f"  {rq['id']}: {r['n']} tokens in {r['seconds']} s (prompt {r['n_prompt']})", flush=True)

    def run_one(rq):
        # A1: the B0 watchdog is checked before EVERY request, 16-set and sized
        if time.time() - t_mod > MODULE_DEADLINE["B0"]:
            raise ModuleTimeout(f"B0 exceeded {MODULE_DEADLINE['B0']} s after "
                                f"{len(result.get('b0_completed', []))} requests; partial references saved for --resume")
        rec = Recorder(base)
        try:
            rec.run([rq], cache_prompt=False)
        except PilotError as e:
            m = re.search(r"(\d+) tokens", str(e))
            if "context_length_exceeded" in str(e) or "문맥" in str(e):
                result["evidence"][f"B0:{rq['id']}"] = {
                    "context_length_exceeded": True,
                    "measured_prompt_tokens": m.group(1) if m else "?",
                    "note": "server context guard rejected the constructed prompt (real boundary datum)"}
                save(result, a.out)
                raise ContextSizeError(
                    f"{rq['id']}: prompt rejected by the server context guard ({e}); "
                    f"recorded as boundary datum, run INCOMPLETE") from e
            raise
        refs[rq["id"]] = rec.recs[rq["id"]]["res"]
        executed.append(rq["id"])
        stamp_and_save(rq, rec)

    def tokenize(rq):
        body = sp.tokenize_request(rq)
        r = api(base, "/tokenize", body, timeout=120)
        if not isinstance(r, dict) or not isinstance(r.get("count"), int):
            raise ContextSizeError(f"/tokenize returned no integer count: {str(r)[:200]}")
        tids = r.get("tokens") if isinstance(r.get("tokens"), list) else None
        tsha = sp.sha256_ints(tids) if tids is not None else None
        if tids is None or len(tids) != r["count"]:
            raise ContextSizeError("/tokenize must return token ids with count == len(ids); INCOMPLETE")
        return r["count"], tsha, tids

    # Fail unknown manifest fields before any generation, never silently drop.
    for rq in all16:
        sp.semantic_request(rq)

    # 1) math-2 always live (never skipped): the parity anchor
    anchor_probe = chat(base, dict(all16[0], max_tokens=1), cache_prompt=False)

    # 2) parity: /tokenize with the anchor request's OWN chat_template_kwargs must
    #    EXACTLY mirror its chat call. Token ids decide when the endpoint returns
    #    them; a count match NEVER overrides a token-sha mismatch. The winner is
    #    ALWAYS the anchor's own kwargs - no candidate search may credit a global
    #    default (B2 root cause, 2026-11-05). Other candidates are diagnostics.
    anchor_kw = all16[0].get("chat_template_kwargs")
    actual, actual_sha = anchor_probe["n_prompt"], anchor_probe["prompt_tokens_sha256"]
    win, probes = None, []
    for cand in [anchor_kw] + [c for c in PARITY_CANDIDATES if c != anchor_kw]:
        cnt, tsha, _t = tokenize(dict(all16[0], chat_template_kwargs=cand))
        probes.append({"kwargs": cand, "count": cnt, "token_ids": tsha is not None,
                       "count_match": cnt == actual,
                       "sha_match": (tsha == actual_sha) if tsha is not None else None})
        if cand == anchor_kw and ((tsha is not None and tsha == actual_sha)
                                  or (tsha is None and cnt == actual)):
            win = {"kwargs": cand, "count": cnt, "by_token_ids": tsha is not None}
    result["evidence"]["preflight"] = {"parity_request": "math-2", "parity_actual_n_prompt": actual,
                                       "probes": probes}
    if win is None:
        save(result, a.out)
        raise ContextSizeError(
            f"tokenize/chat parity FAILED for the anchor's OWN kwargs {anchor_kw!r} "
            f"(probes {probes}) vs actual n_prompt {actual}; "
            f"preflight counts are not trustworthy, run INCOMPLETE")
    result["evidence"]["preflight"]["parity"] = "EXACT"
    result["evidence"]["preflight"]["winning"] = win
    ident["kwargs"] = win["kwargs"]
    # Explicit per-request kwargs ALWAYS win; the calibrated winner fills ONLY
    # constructed requests MISSING kwargs. Kit manifest dicts stay untouched.
    kit_ids = {r["id"] for r in kit["requests"]}
    filled = []
    if win["kwargs"] is not None:
        for r in all16:
            if r["id"] not in kit_ids and "chat_template_kwargs" not in r:
                r["chat_template_kwargs"] = dict(win["kwargs"])
                filled.append(r["id"])
    if filled:
        result["notes"].append(
            f"template kwargs filled from the calibrated winner {win['kwargs']} into constructed "
            f"requests {sorted(filled)}; kit requests verbatim")
    result["b0_fingerprints"]["math-2"] = fingerprint(all16[0], result["settings"], ident)
    print(f"preflight parity EXACT: kwargs={win['kwargs']} -> {win['count']} == chat n_prompt {actual}"
          + (" (token-id match)" if win["by_token_ids"] else " (count only: endpoint returned no token ids)"),
          flush=True)

    # Cheap shape preflight BEFORE expensive B0 references. Every manifest/pilot
    # request is represented, including tools, system and multi-turn messages.
    # It is not a baseline: max_tokens=1, but all prompt fields remain identical.
    shape_ev = {}
    for rq in all16:
        cnt, tsha, tids = tokenize(rq)
        probe = chat(base, dict(rq, max_tokens=1), cache_prompt=False)
        pids = probe.get("prompt_ids")
        good = cnt == probe["n_prompt"] and tsha is not None and tsha == probe["prompt_tokens_sha256"]
        ev = {"tokenize_body": sp.tokenize_request(rq),
              "chat_body": sp.chat_request(dict(rq, max_tokens=1), False),
              "len_tokenize": cnt, "len_chat": probe["n_prompt"],
              "tokenize_sha256": tsha, "chat_sha256": probe["prompt_tokens_sha256"],
              "parity": "EXACT" if good else "FAILED"}
        if tids is not None and pids is not None:
            fd = next((i for i in range(min(len(tids), len(pids))) if tids[i] != pids[i]),
                      -1 if len(tids) == len(pids) else min(len(tids), len(pids)))
            lo = max(0, fd - 10)
            ev.update({"first_diff_index": fd, "window_tokenize": tids[lo:fd + 10] if fd >= 0 else [],
                       "window_chat": pids[lo:fd + 10] if fd >= 0 else []})
            if not good:
                ev.update({"tokenize_ids_full": tids, "chat_ids_full": pids})
        shape_ev[rq["id"]] = ev
        result["evidence"]["preflight"]["shape_preflight"] = shape_ev
        if not good:
            save(result, a.out)
            raise ContextSizeError(f"shape preflight FAILED on {rq['id']}: tokenize {cnt} vs chat {probe['n_prompt']}; INCOMPLETE before expensive B0")
    print(f"preflight: all {len(shape_ev)} manifest/pilot shapes EXACT (1-token chat)", flush=True)

    # 3) resolved-template gate: a template change invalidates ALL resume reuse
    if old is not None:
        old_win = (((old.get("evidence") or {}).get("preflight") or {}).get("winning") or {})
        if old_win.get("kwargs", "ABSENT") != win["kwargs"]:
            result["notes"].append(
                f"resume: resolved chat template kwargs changed ({old_win.get('kwargs')} -> {win['kwargs']}); "
                f"all resume skips and module carry invalidated")
            old = None

    # 4) module-level carry decision (post-parity, identity-gated)
    done_mods = resume_selection(old)[0] if old else set()
    if done_mods:
        result["notes"].append(f"resumed: modules skipped {sorted(done_mods)} (clean only, identity-verified)")

    # 5) measured sizing with the winning invocation (before any long generation)
    def specd_prompt(specd):
        return {k: v for k, v in specd.items() if k in sp.PROMPT_FIELDS}

    def count_tokens_for(specd):
        kws = specd.get("chat_template_kwargs", win["kwargs"])
        return lambda messages: tokenize(dict(specd_prompt(specd), messages=messages, chat_template_kwargs=kws))[0]
    for specd in CORE_SPECS:
        if specd["sized"]:
            rq2, n, c = auto_size(count_tokens_for(specd), specd)
            rq2.update(specd_prompt(specd))
            if win["kwargs"] is not None:
                rq2.setdefault("chat_template_kwargs", dict(win["kwargs"]))
            result["evidence"]["preflight"][specd["id"]] = {
                "entries": n, "measured_prompt_tokens": c, "bounds": list(SIZED_BOUNDS[specd["id"]])}
            print(f"preflight: {specd['id']} sized to {n} entries = {c} tokens (measured)", flush=True)
            core_reqs.append(rq2)
        else:
            rq3 = {k: specd[k] for k in ("id", "kind", "messages", "max_tokens", "temperature", "seed")}
            if "chat_template_kwargs" in specd:
                rq3["chat_template_kwargs"] = specd["chat_template_kwargs"]
            elif win["kwargs"] is not None:
                rq3["chat_template_kwargs"] = dict(win["kwargs"])
            core_reqs.append(rq3)
    ev_sum = sum(result["evidence"]["preflight"][r2]["measured_prompt_tokens"]
                 for r2 in ("evict-a", "evict-b", "evict-c"))
    result["evidence"]["preflight"]["evict_measured_sum"] = ev_sum
    result["evidence"]["preflight"]["kv_capacity_assumed"] = 2 * CTX
    if ev_sum <= 2 * CTX:
        save(result, a.out)
        raise ContextSizeError(f"eviction math broken: measured evict sum {ev_sum} <= KV capacity {2 * CTX}")
    print(f"preflight: evict sum {ev_sum} > KV capacity {2 * CTX}", flush=True)

    # 6) B2: long-shape probe - 1-token chat on the sized ctx-long candidate,
    #    cache_prompt=false. Actual n_prompt (+ token ids) must equal tokenize
    #    EXACTLY before any full-length generation. The probe is NOT a reference.
    ctx_long = next(r for r in core_reqs if r["id"] == "ctx-long")
    ckw = ctx_long.get("chat_template_kwargs")
    cnt, tsha, tids = tokenize(ctx_long)
    probe = chat(base, dict(ctx_long, max_tokens=1), cache_prompt=False)   # id stays "ctx-long": 1-token probe, NOT a reference
    probe_sha = probe["prompt_tokens_sha256"]
    pids = probe.get("prompt_ids")
    ok = cnt == probe["n_prompt"] and (tsha is None or tsha == probe_sha)
    pev = {"tokenize_count": cnt, "chat_n_prompt": probe["n_prompt"], "max_tokens": 1,
           "kwargs": ckw, "token_ids": tsha is not None and pids is not None,
           "sha_match": (tsha == probe_sha) if tsha is not None else None,
           "len_tokenize": cnt, "len_chat": probe["n_prompt"]}
    if tids is not None and pids is not None:
        fd = next((i for i in range(min(len(tids), len(pids))) if tids[i] != pids[i]),
                  -1 if len(tids) == len(pids) else min(len(tids), len(pids)))
        pev["first_diff_index"] = fd
        if fd >= 0:
            lo = max(0, fd - 10)
            pev["window_tokenize"] = tids[lo:fd + 10]
            pev["window_chat"] = pids[lo:fd + 10]
            pev["tokenize_ids_full"] = tids   # full sequences kept on divergence
            pev["chat_ids_full"] = pids
    pev["parity"] = "EXACT" if ok else "FAILED"
    result["evidence"]["preflight"]["ctx_long_probe"] = pev
    if not ok:
        save(result, a.out)
        raise ContextSizeError(
            f"long-shape parity probe FAILED: tokenize {cnt} vs chat n_prompt {probe['n_prompt']} "
            f"(sha_match {pev['sha_match']}, first_diff {pev.get('first_diff_index')}); "
            f"run INCOMPLETE before any long generation")
    print(f"preflight: long-shape probe EXACT (ctx-long {probe['n_prompt']} tokens, 1-token generation)",
          flush=True)

    # 7) request-level resume skip: fingerprints now carry the resolved identity
    fps = {r["id"]: fingerprint(r, result["settings"], ident) for r in all16[1:] + core_reqs}
    skip = {rid for rid in resume_reqs(old, fps) if rid in refs}
    if skip:
        result["notes"].append(f"resume: skipping {len(skip)} cold-verified fingerprint-matched B0 requests")
        old_ev = (old or {}).get("evidence") or {}
        old_fps = (old or {}).get("b0_fingerprints") or {}
        old_out = ((old or {}).get("outputs") or {}).get("B0") or {}
        old_in = ((old or {}).get("instrumentation") or {}).get("B0") or {}
        for rid in sorted(skip):
            result["b0_fingerprints"][rid] = old_fps[rid]
            result.setdefault("b0_completed", []).append(rid)
            result["evidence"][f"B0:{rid}"] = dict(old_ev.get(f"B0:{rid}") or {},
                                                   carried_from="prior launch (fingerprint match, cold=True)")
            if rid in old_out:   # B1: carried B0 requests keep outputs/instrumentation too
                result["outputs"].setdefault("B0", {})[rid] = old_out[rid]
            if rid in old_in:
                result["instrumentation"].setdefault("B0", {})[rid] = old_in[rid]

    # 8) Only after ALL cheap prompt-shape gates may full baseline generation start.
    # math-2 always live, never resumed; the earlier anchor was only 1 token.
    run_one(all16[0])
    for rq in all16[1:] + core_reqs:
        if rq["id"] in skip:
            continue
        run_one(rq)

    # 9) multi-shape parity: tokenize must mirror the chat path for EVERY
    #    request executed live (short and long shapes), sha-level when possible
    ms = {}
    for rid in executed:
        rq = next(r for r in all16 + core_reqs if r["id"] == rid)
        rkw = rq.get("chat_template_kwargs")
        cnt, tsha, tids = tokenize(rq)
        good = cnt == refs[rid]["n_prompt"] and tsha is not None and tsha == refs[rid]["prompt_tokens_sha256"]
        ms[rid] = {"count": cnt, "n_prompt": refs[rid]["n_prompt"], "kwargs": rkw,
                   "tokenize_sha256": tsha, "chat_sha256": refs[rid]["prompt_tokens_sha256"],
                   "token_ids": tsha is not None, "parity": "EXACT" if good else "FAILED"}
        if not good:
            pidsr = refs[rid].get("prompt_ids")
            if tids is not None and pidsr is not None:
                fd = next((i for i in range(min(len(tids), len(pidsr))) if tids[i] != pidsr[i]),
                          -1 if len(tids) == len(pidsr) else min(len(tids), len(pidsr)))
                lo = max(0, fd - 10)
                ms[rid].update({"first_diff_index": fd, "len_tokenize": len(tids),
                                "len_chat": len(pidsr),
                                "window_tokenize": tids[lo:fd + 10], "window_chat": pidsr[lo:fd + 10]})
            result["evidence"]["preflight"]["multishape"] = ms
            save(result, a.out)
            raise ContextSizeError(
                f"tokenize/chat parity FAILED on executed request {rid}: tokenize {cnt} vs chat "
                f"n_prompt {refs[rid]['n_prompt']}; run INCOMPLETE")
    result["evidence"]["preflight"]["multishape"] = ms
    print(f"preflight: multi-shape parity EXACT across {len(ms)} executed requests", flush=True)

    # 10) A2: explicit evidence-completeness gate over every expected B0 id
    expected = sorted({r["id"] for r in all16} | {r["id"] for r in core_reqs})
    result["notes"].append(f"B0 expected request ids: {len(expected)}")
    missing = [rid for rid in expected
               if result["evidence"].get(f"B0:{rid}", {}).get("cold") is not True
               or rid not in refs or rid not in (result["outputs"].get("B0") or {})]
    if missing:
        result["modules"]["B0"] = {"status": "INCOMPLETE-EVIDENCE", "missing": missing}
        save(result, a.out)
        return refs, done_mods

    boundary = {}
    for rid in ("ctx-long", "ctx-max"):
        lo, hi, mt = BOUNDS[rid]
        boundary[rid] = check_bounds(refs[rid], lo, hi, mt, rid)
    for rid in ("evict-a", "evict-b", "evict-c"):
        check_bounds(refs[rid], EVICT_BOUNDS[0], EVICT_BOUNDS[1], 64, rid)
    # M2 verdict from the B0 boundary runs
    ev_ok = all(result["evidence"][f"B0:{rid}"]["cold"] for rid in ("ctx-long", "ctx-max"))
    result["modules"]["M2"] = {"status": "DONE",
        "ctx-long": {"n_prompt": boundary["ctx-long"], "gen": 2048,
                     "total": boundary["ctx-long"] + 2048, "reserve": CTX - boundary["ctx-long"] - 2048},
        "ctx-max": {"n_prompt": boundary["ctx-max"], "gen": 250,
                    "total": boundary["ctx-max"] + 250, "reserve": CTX - boundary["ctx-max"] - 250},
        "cold_counters": "VERIFIED" if ev_ok else "EVIDENCE-MISSING"}
    # kit cross-check of the 12 kit requests (same default config)
    if a.expected:
        try:
            exp_items = json.load(open(a.expected))["items"]["qwen3-8b-q4_k_m/default"]
            need = {"id", "n_prompt", "prompt_tokens_sha256", "n", "tokens_sha256", "logit_hashes"}
            exp = {e["id"]: e for e in exp_items if need <= set(e)}
            n_ok = sum(compare(refs[rid], exp.get(rid))["same"]
                       for rid in (e["id"] for e in kit["requests"]) if rid in exp)
            result["notes"].append(f"kit-reference cross-check: {n_ok}/{len(exp)} match expected.json")
        except (KeyError, ValueError, TypeError) as e:
            result["notes"].append(f"kit-reference cross-check SKIPPED ({e!r})")
        print(result["notes"][-1], flush=True)
    result["modules"]["B0"] = {"status": "DONE", "references": len(refs)}
    save(result, a.out)
    return refs, done_mods


def phase_m3(a, base, reqs16, core_reqs, refs, mid, settings, result):
    """Sustained churn: resident gen-4k + 15 staggered arrivals (1 s, concurrent)."""
    print("M3 sustained churn (gen-4k resident + 15 staggered) ...", flush=True)
    resident = next(r for r in core_reqs if r["id"] == "gen-4k")
    others = [r for r in reqs16 if r["id"] != "gen-1k"]
    random.Random(300).shuffle(others)
    rec = Recorder(base)
    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as ex:
        fut_res = ex.submit(rec.work, resident, None)
        time.sleep(1.5)   # let the resident settle in before arrivals
        futs = []
        for rq in others:
            futs.append(ex.submit(rec.work, rq, None))
            time.sleep(1.0)
        fut_res.result()
        for f in futs:
            f.result()
    rec.counters = new_counters(base, rec.mark)
    rec.record(result, "M3-churn", refs)
    rt0, rt1 = rec.recs["gen-4k"]["t0"], rec.recs["gen-4k"]["t1"]
    overlaps = {rid: (r["t0"] < rt1 and r["t1"] > rt0) for rid, r in rec.recs.items() if rid != "gen-4k"}
    n_ov = sum(overlaps.values())
    result["evidence"]["M3-churn"].update({
        "resident_seconds": round(rt1 - rt0, 1),
        "overlap_count": n_ov, "overlap_total": len(overlaps),
        "overlap_target_14_of_15": "MET" if n_ov >= 14 else f"NOT-MET (measured {n_ov}/{len(overlaps)})",
        "per_request_overlap": overlaps})
    if rec.recs["gen-4k"]["t1"] - rec.recs[min(rec.recs, key=lambda k: rec.recs[k]["t0"])]["t0"] > MODULE_DEADLINE["M3"]:
        raise ModuleTimeout(f"M3 exceeded {MODULE_DEADLINE['M3']} s wall time")
    result["modules"]["M3"] = {"status": "DONE", "overlap": f"{n_ov}/{len(overlaps)}"}
    print(f"  resident {round(rt1-rt0,1)} s, overlap {n_ov}/{len(overlaps)}", flush=True)


def phase_m4(a, base, reqs16, core_reqs, refs, mid, settings, result):
    """KV eviction: A,B,C distinct ~7.5-7.7k prompts (sum ~22.8k > 2x8192 KV), then A again.
    Default caching on: the module tests the cache/eviction path itself."""
    print("M4 KV eviction (A B C A') ...", flush=True)
    by_id = {r["id"]: r for r in core_reqs}
    mark = log_marker(base)
    recs = {}
    for label, rid in (("evict-a#1", "evict-a"), ("evict-b", "evict-b"),
                       ("evict-c", "evict-c"), ("evict-a#2", "evict-a")):
        t0 = time.time()
        r = chat(base, by_id[rid], None)
        t1 = time.time()
        recs[label] = {"res": r, "t0": t0, "t1": t1, "rid": rid}
    counters = new_counters(base, mark)
    instr = result["instrumentation"].setdefault("M4-eviction", {})
    outs = result["outputs"].setdefault("M4-eviction", {})
    for idx, (label, rec) in enumerate(recs.items()):
        outs[label] = sp.strip(rec["res"])
        instr[label] = {"t0": round(rec["t0"], 2), "t1": round(rec["t1"], 2), "idx": idx,
                        "seconds": rec["res"]["seconds"], "n_prompt": rec["res"]["n_prompt"],
                        "n": rec["res"]["n"], "finish": rec["res"]["finish"]}
        result["runs"].append({"phase": "M4-eviction", "id": label,
                               **compare(rec["res"], refs.get(rec["rid"]))})
    # classify the A' re-request from the LAST log line matching its prompt size
    # (prompt,gen fingerprints of evict-a#1 and evict-a#2 are identical by design;
    # sequential execution makes completion order the discriminator)
    fp = recs["evict-a#2"]["res"]["n_prompt"]
    lines = [c for c in counters if c["prompt"] == fp]
    last = lines[-1] if lines else None
    if last is None:
        label = "EVIDENCE-MISSING (no attributable log line)"
    elif last["cache"] + last["other_slot"] >= fp - 64:
        label = "EVIDENCE-MISSING (re-request served from memory; eviction did not happen or KV larger than assumed)"
    elif last["store"] > 0:
        label = "RESTORED-FROM-STORE (evicted from KV, restored from SSD store)"
    else:
        label = "RECOMPUTED-COLD (evicted, no store reuse)"
    result["evidence"]["M4-eviction"] = {"per_line": counters,
        "kv_capacity_assumed": 2 * CTX,
        "prompt_sum_abc": sum(recs[r]["res"]["n_prompt"] for r in ("evict-a#1", "evict-b", "evict-c")),
        "re_request_line": last,
        "re_request_classification": label,
        "A_prime_seconds": recs["evict-a#2"]["res"]["seconds"],
        "A_cold_seconds": recs["evict-a#1"]["res"]["seconds"]}
    if recs["evict-a#2"]["t1"] - recs["evict-a#1"]["t0"] > MODULE_DEADLINE["M4"]:
        raise ModuleTimeout(f"M4 exceeded {MODULE_DEADLINE['M4']} s wall time")
    result["modules"]["M4"] = {"status": "DONE", "classification": label}
    print(f"  A' classification: {label}", flush=True)


def phase_m5(a, base, reqs16, core_reqs, refs, mid, settings, result):
    """slots=4 burst: apply + verify slots=4, run the 16-set 4-way parallel, restore."""
    print("M5 slot scaling (slots=4, 16 requests) ...", flush=True)
    load_model(base, mid, dict(settings, slots=4))
    eff = (state(base).get("settings") or {}).get("slots")
    if eff != 4:
        result["modules"]["M5"] = {"status": "FAILED-UNSUPPORTED",
                                   "why": f"settings slots requested 4, effective {eff}"}
        result["evidence"]["M5-slots4"] = {"slots_effective": eff}
        load_model(base, mid, settings)
        print(f"  slots=4 unsupported (effective {eff}); module FAILED-UNSUPPORTED", flush=True)
        return
    try:
        rec = Recorder(base)
        rec.run(reqs16, parallel=4, cache_prompt=False, order_seed=400)
        rec.record(result, "M5-slots4", refs)
        span = max(r["t1"] for r in rec.recs.values()) - min(r["t0"] for r in rec.recs.values())
        if span > MODULE_DEADLINE["M5"]:
            raise ModuleTimeout(f"M5 exceeded {MODULE_DEADLINE['M5']} s wall time")
        result["modules"]["M5"] = {"status": "DONE", "slots_effective": eff}
    finally:
        load_model(base, mid, settings)   # restore slots=2 even on failure/timeout
    print(f"  slots=4 burst done ({len(reqs16)} requests, 4-way)", flush=True)


def finalize(result):
    def cls(d):
        w = d.get("why", "")
        return ("prompt-tokens-differ(input changed; triage, not path nondeterminism)" if "prompt" in w else
                "logits-differ(candidate divergence; triage needed)" if "logits" in w else
                "tokens-differ(candidate divergence; triage needed)")
    diffs = [r for r in result["runs"] if not r.get("same")]
    incomplete = [m for m, v in result["modules"].items() if v.get("status") != "DONE"]
    for m in ("B0", "M2", "M3", "M4", "M5"):
        if m not in result["modules"]:
            incomplete.append(f"module-not-executed:{m}")
    missing_ev = [p for p, ev in result["evidence"].items()
                  if p.startswith("B0:") and not ev.get("cold")]
    # coverage gates: a weak measured overlap or an unproven eviction is unmet
    # coverage -> INCOMPLETE, never a label-only pass
    m3ev = result["evidence"].get("M3-churn", {})
    if result["modules"].get("M3", {}).get("status") == "DONE" and \
            m3ev.get("overlap_target_14_of_15") != "MET":
        incomplete.append(f"M3-overlap-not-met({m3ev.get('overlap_count')}/{m3ev.get('overlap_total')})")
    m4ev = result["evidence"].get("M4-eviction", {})
    if result["modules"].get("M4", {}).get("status") == "DONE" and \
            str(m4ev.get("re_request_classification", "")).startswith("EVIDENCE-MISSING"):
        incomplete.append("M4-eviction-unproven")
    result["summary"] = {
        "identical": sum(1 for r in result["runs"] if r.get("same")), "total": len(result["runs"]),
        "candidate_divergences": [{"phase": d["phase"], "id": d["id"], "class": cls(d)} for d in diffs],
        "modules": {m: v.get("status") for m, v in result["modules"].items()},
        "M3_overlap": result["evidence"].get("M3-churn", {}).get("overlap_target_14_of_15"),
        "M4_classification": result["evidence"].get("M4-eviction", {}).get("re_request_classification"),
        "coverage_incomplete": sorted(incomplete) + (["B0-warm-lines"] if missing_ev else [])}


def selftest():
    # determinism of the long-prompt generator (same output on every call/box)
    p1 = long_prompt("EA", 225, "q", 0)
    p2 = long_prompt("EA", 225, "q", 0)
    assert p1 == p2
    assert long_prompt("EA", 225, "q", 0) != long_prompt("EB", 225, "q", 0)
    assert long_prompt("EA", 225, "q", 0) != long_prompt("EA", 225, "q", 4)
    # no shared prefix between evict prompts beyond the fixed intro words
    pa, pb = long_prompt("EA", 225, "q", 0), long_prompt("EB", 218, "q", 4)
    common = 0
    for x, y in zip(pa, pb):
        if x != y:
            break
        common += 1
    assert common < 120, f"evict prompts share {common} chars of prefix"
    # bounds arithmetic
    assert BOUNDS["ctx-long"][1] + 2048 <= CTX
    assert BOUNDS["ctx-max"][1] + 250 <= CTX - 0
    # check_bounds behavior
    fake = {"n_prompt": 5000}
    assert check_bounds(fake, *BOUNDS["ctx-long"], "ctx-long") == 5000
    try:
        check_bounds({"n_prompt": 8100}, *BOUNDS["ctx-max"], "ctx-max")
        raise AssertionError("bounds check should have raised")
    except PilotError:
        pass
    # Recorder.record schema
    class FakeRec:
        recs = {"x": {"res": {"id": "x", "n_prompt": 3, "n": 2, "finish": "stop",
                              "prompt_tokens_sha256": "p", "tokens_sha256": "t",
                              "logit_hashes": ["a"], "seconds": 1.0},
                      "t0": 1.0, "t1": 2.0, "idx": 0}}
        counters = [{"slot": 0, "prompt": 3, "cache": 0, "store": 0, "other_slot": 0}]
    res = {"runs": [], "evidence": {}, "instrumentation": {}, "outputs": {}}
    Recorder.record(FakeRec(), res, "T", {"x": FakeRec.recs["x"]["res"]})
    assert res["runs"][0]["same"] and res["instrumentation"]["T"]["x"]["idx"] == 0
    assert MODULE_DEADLINE["B0"] >= 1800 and all(v >= 600 for v in MODULE_DEADLINE.values())
    # --- negative tests: gates must force coverage INCOMPLETE (exit-2 path) ---
    def fake_result(m3_target="MET", m4_label="RESTORED-FROM-STORE (x)",
                    m3_status="DONE", m4_status="DONE", b0_cold=True):
        r = {"runs": [{"phase": "M3-churn", "id": "x", "same": True}],
             "evidence": {"M3-churn": {"overlap_target_14_of_15": m3_target,
                                       "overlap_count": 9, "overlap_total": 15},
                          "M4-eviction": {"re_request_classification": m4_label},
                          "B0:x": {"cold": b0_cold}},
             "instrumentation": {}, "outputs": {},
             "modules": {"B0": {"status": "DONE"}, "M2": {"status": "DONE"},
                         "M3": {"status": m3_status}, "M4": {"status": m4_status},
                         "M5": {"status": "DONE"}}}
        finalize(r)
        return r["summary"]["coverage_incomplete"]
    assert fake_result() == []
    assert any(s.startswith("M3-overlap-not-met") for s in fake_result(m3_target="NOT-MET (measured 9/15)"))
    assert any(s == "M4-eviction-unproven" for s in fake_result(m4_label="EVIDENCE-MISSING (served from memory)"))
    both = fake_result(m3_target="NOT-MET", m4_label="EVIDENCE-MISSING")
    assert any("M3" in s for s in both) and any("M4" in s for s in both)
    assert "B0-warm-lines" in fake_result(b0_cold=False)
    r = fake_result()
    # missing module = unmet coverage
    rr = {"runs": [], "evidence": {}, "instrumentation": {}, "outputs": {},
          "modules": {"B0": {"status": "DONE"}}}
    finalize(rr)
    assert any(s.startswith("module-not-executed:") for s in rr["summary"]["coverage_incomplete"])
    # empty counters must not read as cold (vacuous pass guard)
    assert not (bool([]) and not any([]))
    # --- resume repair: DONE-but-coverage-missing re-runs; clean DONE skips ---
    old = {"modules": {"M3": {"status": "DONE"}, "M4": {"status": "DONE"}, "M5": {"status": "DONE"}},
           "evidence": {"M3-churn": {"overlap_target_14_of_15": "NOT-MET (measured 9/15)"},
                        "M4-eviction": {"re_request_classification": "EVIDENCE-MISSING (memory hit)"},
                        "B0:a": {"cold": True}, "B0:b": {"cold": False}, "B0:c": {}},
           "b0_completed": ["a", "b", "c", "d"]}
    dm, dr = resume_selection(old)
    assert dm == {"M5"}, dm                     # M3 weak overlap and M4 unproven re-run
    assert dr == {"a"}, dr                      # only cold-verified B0 requests skip
    old2 = {"modules": {"M3": {"status": "DONE"}, "M4": {"status": "DONE"}},
            "evidence": {"M3-churn": {"overlap_target_14_of_15": "MET"},
                         "M4-eviction": {"re_request_classification": "RECOMPUTED-COLD (x)"},
                         "B0:a": {"cold": True}},
            "b0_completed": ["a"]}
    dm2, dr2 = resume_selection(old2)
    assert dm2 == {"M3", "M4"} and dr2 == {"a"}
    assert issubclass(ContextSizeError, Exception)
    # --- auto_size converges on MEASURED counts (mock tokenizer, linear growth) ---
    def mock_count(messages):
        return 120 + 33 * messages[0]["content"].count("[")   # ~intercept + per-entry slope
    for sid, want in (("ctx-long", (4500, 5500)), ("ctx-max", (7400, 7870)), ("evict-b", (7300, 7870))):
        spec = next(s for s in CORE_SPECS if s["id"] == sid)
        rq, n, c = auto_size(mock_count, spec)
        assert want[0] <= c <= want[1], (sid, n, c)
        assert rq["max_tokens"] == spec["max_tokens"] and rq["messages"][0]["content"].count("[") == n
    # auto_size refuses a broken tokenizer (no growth)
    try:
        auto_size(lambda m: 100, next(s for s in CORE_SPECS if s["id"] == "ctx-max"))
        raise AssertionError("flat tokenizer must fail")
    except ContextSizeError:
        pass
    # --- fingerprint: identical request+settings match; any content/config change mismatches ---
    settings = {"precision": "fast", "kv8_cache": True, "fast_attention": True, "slots": 2, "context": 8192}
    rq1 = {"messages": [{"role": "user", "content": "abc"}], "max_tokens": 64, "temperature": 0, "seed": 7}
    rq2 = {"messages": [{"role": "user", "content": "abd"}], "max_tokens": 64, "temperature": 0, "seed": 7}
    assert fingerprint(rq1, settings) == fingerprint(rq1, settings)
    assert fingerprint(rq1, settings) != fingerprint(rq2, settings)
    assert fingerprint(rq1, settings) != fingerprint(rq1, dict(settings, slots=4))
    # --- resume: fingerprint mismatch re-runs despite cold=True ---
    old3 = {"modules": {"M3": {"status": "DONE"}},
            "evidence": {"M3-churn": {"overlap_target_14_of_15": "MET"}, "B0:a": {"cold": True}, "B0:b": {"cold": True}},
            "b0_completed": ["a", "b"],
            "b0_fingerprints": {"a": "fp-old-a", "b": "fp-old-b"}}
    dm3, dr3 = resume_selection(old3, {"a": "fp-old-a", "b": "fp-NEW-b"})
    assert dr3 == {"a"}, dr3          # b re-runs: content/config changed
    dm4, dr4 = resume_selection(old3, {"a": "fp-old-a", "b": "fp-old-b"})
    assert dr4 == {"a", "b"}, dr4     # fingerprints match -> reuse
    # live-parity rule: resume_reqs NEVER skips math-2 (it re-runs live every launch)
    old4 = {"evidence": {"B0:math-2": {"cold": True}, "B0:a": {"cold": True}},
            "b0_completed": ["math-2", "a"],
            "b0_fingerprints": {"math-2": "fp", "a": "fp"}}
    assert resume_reqs(old4, {"math-2": "fp", "a": "fp"}) == {"a"}
    # B3: fingerprint identity - resolved kwargs, version, commit, model all matter
    id1 = {"version": "0.21.0", "commit": "abc", "model": "m1", "kwargs": {"enable_thinking": True}}
    assert fingerprint(rq1, settings, id1) != fingerprint(rq1, settings)
    # the REQUEST's OWN effective kwargs bind (per-request B2 fix); ident kwargs do not
    rq1k = dict(rq1, chat_template_kwargs={"enable_thinking": True})
    assert fingerprint(rq1k, settings, id1) != fingerprint(rq1, settings, id1)
    assert fingerprint(rq1k, settings, id1) != fingerprint(dict(rq1, chat_template_kwargs=None), settings, id1)
    assert fingerprint(rq1, settings, id1) != fingerprint(rq1, settings, dict(id1, version="0.21.1"))
    assert fingerprint(rq1, settings, id1) != fingerprint(rq1, settings, dict(id1, commit="def"))
    assert fingerprint(rq1, settings, id1) != fingerprint(rq1, settings, dict(id1, model="m2"))
    assert fingerprint(rq1, settings, id1) == fingerprint(rq1, settings, dict(id1))
    # B5: DONE is never stamped over a phase-recorded terminal status
    r5 = {"modules": {"M5": {"status": "FAILED-UNSUPPORTED"}}}
    mark_module_done(r5, "M5")
    assert r5["modules"]["M5"]["status"] == "FAILED-UNSUPPORTED"
    mark_module_done(r5, "M3")
    assert r5["modules"]["M3"]["status"] == "DONE"
    # B1: carrying a module merges evidence/outputs/instrumentation/runs with provenance
    oldc = {"modules": {"M3": {"status": "DONE", "overlap": "15/15"}},
            "evidence": {"M3-churn": {"overlap_target_14_of_15": "MET"}},
            "outputs": {"M3-churn": {"gen-4k": {"n": 1}}},
            "instrumentation": {"M3-churn": {"gen-4k": {"idx": 0}}},
            "runs": [{"phase": "M3-churn", "id": "gen-4k", "same": True},
                     {"phase": "M5-slots4", "id": "math-1", "same": True}]}
    newc = {"modules": {}, "evidence": {}, "outputs": {}, "instrumentation": {}, "runs": []}
    carry_module(newc, oldc, "M3")
    assert newc["modules"]["M3"]["overlap"] == "15/15" and "carried_from" in newc["modules"]["M3"]
    assert newc["evidence"]["M3-churn"]["overlap_target_14_of_15"] == "MET"
    assert "carried_from" in newc["evidence"]["M3-churn"]
    assert newc["outputs"]["M3-churn"]["gen-4k"] == {"n": 1}
    assert newc["instrumentation"]["M3-churn"]["gen-4k"]["idx"] == 0
    assert len(newc["runs"]) == 1 and newc["runs"][0].get("carried_from")
    # B1 regression: carried M3 evidence satisfies the coverage gate (no None/None exit-2 loop)
    rc_ = {"runs": list(newc["runs"]),
           "evidence": {**newc["evidence"], "M4-eviction": {"re_request_classification": "RECOMPUTED-COLD (x)"},
                        "B0:x": {"cold": True}},
           "instrumentation": {}, "outputs": {},
           "modules": {"B0": {"status": "DONE"}, "M2": {"status": "DONE"},
                       "M3": newc["modules"]["M3"], "M4": {"status": "DONE"}, "M5": {"status": "DONE"}}}
    finalize(rc_)
    assert rc_["summary"]["coverage_incomplete"] == [], rc_["summary"]
    print("SELFTEST OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
