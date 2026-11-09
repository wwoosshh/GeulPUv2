#!/usr/bin/env python3
"""GeulPU reproducibility kit — does this machine give the same bits as the reference PC?

Connects to a running GeulPU 0.23.0 app (start it with `geulpu`; default address http://127.0.0.1:44544), loads each kit model
under each settings set of requests.json, sends the 12 fixed chat requests one at a time and then all at once, and compares the
prompt tokens, the generated tokens and every step's logit hash with expected.json (made on the reference PC). Standard library
only (Python 3.8+).

    python run_kit.py --models-dir ~/models                 # all models and settings
    python run_kit.py --models-dir ~/models --model qwen3-8b-q8_0 --config exact
    python run_kit.py --builtin-check                       # also run the app's own reproducibility check (R1, 32 problems)
    python run_kit.py --url http://127.0.0.1:44544 --out my-result.json

Models are taken from the app's model list when a file with the same size is there, otherwise added from --models-dir (the
app checks each file's SHA-256 against requests.json). The kit changes the app's settings while it runs and puts them back at the
end. Exit code: 0 everything identical to the reference, 1 a difference, 2 the kit could not finish.
Maintainers: --record FILE writes the results in the expected.json format (the reference is made on the reference PC with it).
"""
import argparse
import concurrent.futures
import datetime
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
KIT = "1"
# Server-state fields that decide the computation path; the kit compares them with the reference's.
PATH_FIELDS = ("layer", "precision", "kv8", "fast_attention", "speculative", "yarn", "template")


class KitError(Exception):
    pass


def api(base, path, body=None, timeout=600):
    data = None if body is None else json.dumps(body, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(base + path, data=data, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        raise KitError(f"{path}: HTTP {e.code} {e.read()[:300].decode('utf-8', 'replace')}") from None
    except urllib.error.URLError as e:
        raise KitError(f"cannot reach GeulPU at {base} ({e.reason}) — start it with `geulpu` first, or pass --url") from None


def state(base):
    return api(base, "/api/state", timeout=60)


def sha256_text(s):
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def sha256_ints(xs):
    return sha256_text(",".join(str(int(x)) for x in xs))


def wait_job(base, what, limit=7200):
    t0 = time.time()
    while time.time() - t0 < limit:
        j = state(base).get("job")
        if not j or j.get("state") != "running":
            return j or {}
        time.sleep(3)
    raise KitError(f"{what}: still running after {limit} s")


def wait_device(base):
    for _ in range(180):
        if (state(base).get("device") or {}).get("checked"):
            return
        time.sleep(1)
    raise KitError("the app did not finish checking the GPU")


def environment(st):
    app = st.get("app") or {}
    dev = st.get("device") or {}
    env = {
        "kit": KIT,
        "time_utc": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "geulpu_version": app.get("version"),
        "geulpu_commit": app.get("commit"),
        "geulpu_installer": app.get("installer"),
        "device": dev.get("name"),
        "device_layer": dev.get("layer"),
        "os": platform.platform(),
        "python": platform.python_version(),
    }
    smi = shutil.which("nvidia-smi")
    if smi:
        try:
            out = subprocess.run([smi, "--query-gpu=name,driver_version,compute_cap,memory.total", "--format=csv,noheader"],
                                 capture_output=True, text=True, timeout=30).stdout.strip()
            env["nvidia_smi"] = [line.strip() for line in out.splitlines() if line.strip()]
        except (OSError, subprocess.SubprocessError):
            pass
    return env


def find_model(base, spec, dirs):
    for m in state(base).get("models") or []:
        if m.get("size") == spec["size"] or (m.get("bytes") == spec["size"]):
            return m["id"]
    for d in dirs:
        p = os.path.join(os.path.expanduser(d), spec["file"])
        if os.path.isfile(p) and os.path.getsize(p) == spec["size"]:
            r = api(base, "/api/models/add", {"path": os.path.abspath(p).replace("\\", "/")})
            if r.get("id"):
                return r["id"]
            for m in state(base).get("models") or []:
                if os.path.normcase(os.path.abspath(m.get("path", ""))) == os.path.normcase(os.path.abspath(p)):
                    return m["id"]
    raise KitError(f"model {spec['name']} not found — download {spec['repo']}/{spec['file']} ({spec['size']:,} bytes) into a folder "
                   f"and pass it with --models-dir, or run: geulpu --hf {spec['repo']}/{spec['file']}")


def model_hash(base, mid, spec):
    """Wait until the app has hashed the file, then compare with requests.json."""
    for _ in range(1800):
        m = next((x for x in state(base).get("models") or [] if x.get("id") == mid), None)
        if m is None:
            raise KitError(f"model {mid} disappeared from the app's list")
        h = m.get("sha256")
        if h and not m.get("verifying"):
            if h != spec["sha256"]:
                raise KitError(f"{spec['file']}: SHA-256 {h[:16]}… differs from the kit's {spec['sha256'][:16]}… — download it again")
            return h
        time.sleep(2)
    raise KitError(f"{spec['file']}: the app did not finish its SHA-256 check")


def load(base, mid, settings):
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
            raise KitError(f"the model did not load (server state {s}: {sv.get('reason') or sv.get('error') or ''})")
        time.sleep(2)
    raise KitError("the model did not load within 30 minutes")


def chat(base, rq):
    body = {k: v for k, v in rq.items() if k not in ("id", "kind")}
    body.update({"model": "geulpu", "stream": False, "geulpu": {"logit_hash": True, "prompt_tokens": True}})
    t0 = time.time()
    r = api(base, "/v1/chat/completions", body, timeout=1800)
    g = r.get("geulpu") or {}
    tokens, hashes, prompt = g.get("tokens"), g.get("logit_hashes"), g.get("prompt_tokens")
    if tokens is None or hashes is None or prompt is None:
        raise KitError(f"{rq['id']}: the response has no geulpu.tokens / logit_hashes / prompt_tokens — is this GeulPU 0.23.0?")
    ch = (r.get("choices") or [{}])[0]
    msg = ch.get("message") or {}
    text = (msg.get("reasoning_content") or "") + "\x1f" + (msg.get("content") or "") + "\x1f" + json.dumps(
        msg.get("tool_calls") or [], ensure_ascii=False, sort_keys=True)
    return {
        "id": rq["id"],
        "n_prompt": len(prompt),
        "prompt_tokens_sha256": sha256_ints(prompt),
        "n": len(tokens),
        "finish": ch.get("finish_reason"),
        "tokens_sha256": sha256_ints(tokens),
        "logit_hashes": [str(h) for h in hashes],
        "text_sha256": sha256_text(text),
        "seconds": round(time.time() - t0, 2),
    }


def compare(got, exp):
    if exp is None:
        return {"same": False, "why": "no reference for this request"}
    if got["prompt_tokens_sha256"] != exp["prompt_tokens_sha256"]:
        return {"same": False, "why": "prompt tokens differ (chat template or tokenizer)", "n_prompt": got["n_prompt"],
                "n_prompt_expected": exp["n_prompt"]}
    a, b = got["logit_hashes"], exp["logit_hashes"]
    for i in range(min(len(a), len(b))):
        if a[i] != b[i]:
            return {"same": False, "why": f"logits differ from step {i}", "first_diff_step": i}
    if got["tokens_sha256"] != exp["tokens_sha256"] or len(a) != len(b) or got["n"] != exp["n"]:
        return {"same": False, "why": "generated tokens differ", "n": got["n"], "n_expected": exp["n"]}
    return {"same": True}


def strip(res):
    return {k: v for k, v in res.items() if k != "seconds"}


def main():
    ap = argparse.ArgumentParser(description="GeulPU reproducibility kit (GeulPU 0.23.0)")
    ap.add_argument("--url", default="http://127.0.0.1:44544", help="address of the running GeulPU app")
    ap.add_argument("--models-dir", action="append", default=[], help="folder that holds the model files (repeatable)")
    ap.add_argument("--model", action="append", help="only this model key (repeatable)")
    ap.add_argument("--config", action="append", help="only this settings set: default or exact (repeatable)")
    ap.add_argument("--concurrency", type=int, default=8, help="requests sent at once in the batch pass (default 8)")
    ap.add_argument("--requests", default=os.path.join(HERE, "requests.json"))
    ap.add_argument("--expected", default=os.path.join(HERE, "expected.json"))
    ap.add_argument("--out", help="result file (default result-<device>-<date>.json in the current folder)")
    ap.add_argument("--builtin-check", action="store_true", help="also run the app's reproducibility check (R1, 32 problems)")
    ap.add_argument("--record", help="maintainers: write the results in the expected.json format to this file")
    ap.add_argument("--any-version", action="store_true", help="run on a GeulPU version other than the kit's")
    a = ap.parse_args()
    base = a.url.rstrip("/")

    kit = json.load(open(a.requests, encoding="utf-8"))
    expected = None if a.record else json.load(open(a.expected, encoding="utf-8"))
    models = [m for m in kit["models"] if not a.model or m["key"] in a.model]
    configs = {k: v for k, v in kit["configs"].items() if not a.config or k in a.config}
    if not models or not configs:
        sys.exit("no model or settings set selected — see requests.json for the keys")

    state(base)                 # fails early with a clear message when the app is not running
    wait_device(base)
    st = state(base)
    env = environment(st)
    print(f"GeulPU {env['geulpu_version']} ({env['geulpu_commit']}) on {env['device']} [{env['device_layer']}] — kit {KIT}, "
          f"reference GeulPU {kit['geulpu']}", flush=True)
    if env["geulpu_version"] != kit["geulpu"] and not a.any_version:
        sys.exit(f"this kit is for GeulPU {kit['geulpu']} (install it with: pip install geulpu=={kit['geulpu']}) — "
                 f"the running app is {env['geulpu_version']}; pass --any-version to run anyway")
    if (state(base).get("job") or {}).get("state") == "running":
        sys.exit("the app is busy with another job — wait until it ends and run the kit again")
    saved = dict(st.get("settings") or {})

    result = {"kit": KIT, "reference_geulpu": kit["geulpu"], "environment": env, "runs": [], "builtin_check": None}
    recorded = {"kit": KIT, "geulpu": kit["geulpu"], "reference": env, "configs": configs, "items": {}}
    total = same = 0
    exit_code = 0
    try:
        for spec in models:
            mid = find_model(base, spec, a.models_dir)
            print(f"\n{spec['name']}: checking SHA-256 …", flush=True)
            model_hash(base, mid, spec)
            for cname, settings in configs.items():
                key = f"{spec['key']}/{cname}"
                print(f"{key}: loading ({', '.join(f'{k}={v}' for k, v in settings.items())}) …", flush=True)
                sv = load(base, mid, settings)
                path = {k: sv.get(k) for k in PATH_FIELDS}
                exp_items = {} if expected is None else {e["id"]: e for e in expected["items"].get(key, [])}
                if expected is not None and not exp_items:
                    print(f"  no reference for {key} in {os.path.basename(a.expected)} — skipped", flush=True)
                    continue
                if expected is not None:
                    ref_path = (expected.get("servers") or {}).get(key) or {}
                    diff = {k: (path.get(k), ref_path.get(k)) for k in PATH_FIELDS if k in ref_path and path.get(k) != ref_path.get(k)}
                    if diff:
                        print(f"  note: the server's computation path differs from the reference's (here, reference): {diff}", flush=True)
                        result.setdefault("path_differences", {})[key] = diff
                seq = []
                for rq in kit["requests"]:
                    res = chat(base, rq)
                    seq.append(res)
                    v = compare(res, exp_items.get(rq["id"])) if expected is not None else {"same": True}
                    print(f"  {rq['id']:<10} alone  {res['n']:>4} tokens {res['seconds']:>6.1f} s  "
                          f"{'same' if v['same'] else 'DIFFERENT — ' + v['why']}", flush=True)
                    result["runs"].append({"key": key, "pass": "alone", "id": rq["id"], "n": res["n"], "finish": res["finish"],
                                           "seconds": res["seconds"], **v,
                                           **({} if v["same"] else {"logit_hashes": res["logit_hashes"],
                                                                     "tokens_sha256": res["tokens_sha256"]})})
                    total += 1
                    same += v["same"]
                with concurrent.futures.ThreadPoolExecutor(max_workers=a.concurrency) as ex:
                    batch = list(ex.map(lambda rq: chat(base, rq), kit["requests"]))
                by_id = {r["id"]: r for r in seq}
                for res in batch:
                    ref = exp_items.get(res["id"]) if expected is not None else by_id[res["id"]]
                    v = compare(res, ref)
                    print(f"  {res['id']:<10} batch  {res['n']:>4} tokens {res['seconds']:>6.1f} s  "
                          f"{'same' if v['same'] else 'DIFFERENT — ' + v['why']}", flush=True)
                    result["runs"].append({"key": key, "pass": f"batch of {a.concurrency}", "id": res["id"], "n": res["n"],
                                           "finish": res["finish"], "seconds": res["seconds"], **v,
                                           **({} if v["same"] else {"logit_hashes": res["logit_hashes"],
                                                                     "tokens_sha256": res["tokens_sha256"]})})
                    total += 1
                    same += v["same"]
                result.setdefault("servers", {})[key] = {k: sv.get(k) for k in sorted(sv) if k not in ("log", "log_tail")}
                if a.record:
                    if not all(compare(b, by_id[b["id"]])["same"] for b in batch):
                        raise KitError(f"{key}: the batch pass differs from the one-at-a-time pass — not recording")
                    recorded["items"][key] = [strip(r) for r in seq]
                    recorded.setdefault("servers", {})[key] = path
        if a.builtin_check:
            print("\nbuilt-in reproducibility check (R1, 32 problems) …", flush=True)
            r1 = kit["builtin_check"]
            find_model(base, r1, a.models_dir)   # the app's check finds the file by its size and checks its SHA-256 itself
            api(base, "/api/server/stop", {})
            api(base, "/api/bench/crosscheck", {"modules": "both"})
            j = wait_job(base, "reproducibility check")
            result["builtin_check"] = {k: j.get(k) for k in ("state", "summary", "reason", "title") if k in j}
            passed = j.get("state") == "done" and "32/32" in str(j.get("summary"))
            print(f"  {'all 32 problems identical to the reference PC (32/32)' if passed else 'NOT identical or not finished'}"
                  f" — app: {j.get('state')} · {j.get('summary') or ''} {j.get('reason') or ''}", flush=True)
            if not passed:
                exit_code = max(exit_code, 1)
    except KitError as e:
        print(f"\nstopped: {e}", flush=True)
        result["error"] = str(e)
        exit_code = 2
    finally:
        try:
            api(base, "/api/server/stop", {})
            if saved:
                api(base, "/api/settings", {k: saved[k] for k in ("precision", "kv8_cache", "fast_attention", "slots", "context")
                                            if k in saved})
        except KitError:
            pass

    result["summary"] = {"identical": same, "total": total}
    if exit_code == 0 and same != total:
        exit_code = 1
    if a.record and exit_code == 0:
        with open(a.record, "w", encoding="utf-8", newline="\n") as f:
            json.dump(recorded, f, ensure_ascii=False, indent=1)
            f.write("\n")
        print(f"\nrecorded {sum(len(v) for v in recorded['items'].values())} requests -> {a.record}", flush=True)
    dev = "".join(c if c.isalnum() else "-" for c in (env.get("device") or "gpu")).strip("-")
    out = a.out or f"result-{dev}-{time.strftime('%Y%m%d-%H%M%S')}.json"
    with open(out, "w", encoding="utf-8", newline="\n") as f:
        json.dump(result, f, ensure_ascii=False, indent=1)
        f.write("\n")
    verdict = {0: "IDENTICAL to the reference", 1: "DIFFERENT from the reference", 2: "NOT FINISHED"}[exit_code]
    print(f"\n{same}/{total} requests identical — {verdict}. Result file: {out}", flush=True)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
