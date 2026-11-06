#!/usr/bin/env python3
"""GeulPU llama.cpp replay kit — does this GPU recompute llama.cpp's runs bit for bit?

The six receipts in receipts/ were recorded from llama-server (llama.cpp b12008, official Windows CUDA build) on an NVIDIA
RTX 4070 Ti. Each holds the request, the prompt tokens, every generated token and the top-20 log-probabilities of every step.
This kit starts a private GeulPU instance (its own port and a temporary data folder — your own GeulPU app and data are not
touched), gives it the receipts, and asks it to recompute each run with its replay kernels on this machine's GPU (Vulkan by
default) and compare the tokens and log-probabilities bit for bit. Standard library only (Python 3.8+).

    python run_replay_kit.py --models-dir D:\\models                          # GeulPU from PATH (pip install geulpu)
    python run_replay_kit.py --models-dir D:\\models --app C:\\GeulPU\\GeulPU.exe  # the Windows zip
    python run_replay_kit.py --models-dir D:\\models --download                # fetch missing models from Hugging Face
    python run_replay_kit.py --models-dir D:\\models --layer cuda              # NVIDIA: the CUDA layer instead of Vulkan

Exit code: 0 every receipt recomputed bit for bit, 1 a difference or a refusal, 2 the kit could not finish.
The result file (result-<gpu>-<time>.json) is what to send back.
"""
import argparse
import datetime
import hashlib
import json
import os
import platform
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
KIT = "llama-replay-1"


class KitError(Exception):
    pass


def api(base, path, body=None, timeout=600):
    data = None if body is None else json.dumps(body, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(base + path, data=data, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode("utf-8") or "null")
    except urllib.error.HTTPError as e:
        raise KitError(f"{path}: HTTP {e.code} {e.read()[:300].decode('utf-8', 'replace')}") from None


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(1 << 22)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def download(url, dest):
    tmp = dest + ".part"
    print(f"  downloading {url}", flush=True)
    with urllib.request.urlopen(url, timeout=600) as r, open(tmp, "wb") as f:
        total = int(r.headers.get("Content-Length") or 0)
        got, last = 0, 0.0
        while True:
            b = r.read(1 << 22)
            if not b:
                break
            f.write(b)
            got += len(b)
            if total and time.time() - last > 10:
                print(f"    {got / total:.0%} of {total / 1e9:.1f} GB", flush=True)
                last = time.time()
    os.replace(tmp, dest)


def find_app(arg):
    """the app executable itself (not the pip launcher, so that stopping it stops the app)"""
    if arg:
        if not os.path.exists(arg):
            raise KitError(f"--app {arg} does not exist")
        return [arg]
    try:
        from geulpu import app_dir                     # pip install geulpu — the bundled app
        exe = os.path.join(app_dir(), "GeulPU.exe" if os.name == "nt" else "GeulPU")
        if os.path.isfile(exe):
            return [exe]
    except ImportError:
        pass
    for name in ("GeulPU.exe", "GeulPU"):
        p = shutil.which(name)
        if p:
            return [p]
    raise KitError("GeulPU not found — install it (pip install geulpu==0.22.0) or unpack the Windows zip and pass --app <GeulPU.exe>")


def free_port(preferred):
    for p in [preferred] + list(range(preferred + 1, preferred + 50)):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(("127.0.0.1", p)) != 0:
                return p
    raise KitError("no free port near %d" % preferred)


def main():
    ap = argparse.ArgumentParser(description="GeulPU llama.cpp replay kit")
    ap.add_argument("--models-dir", required=True, action="append",
                    help="folder with Qwen3.5-9B-Q4_K_M.gguf and Qwen3-8B-Q4_K_M.gguf (repeatable; downloads go to the first)")
    ap.add_argument("--app", help="GeulPU executable (default: geulpu on PATH)")
    ap.add_argument("--layer", choices=["vulkan", "cuda"], default="vulkan")
    ap.add_argument("--device", default="", help="part of the Vulkan device name, when there are several GPUs")
    ap.add_argument("--port", type=int, default=44671)
    ap.add_argument("--only", action="append", help="only this receipt (repeatable)")
    ap.add_argument("--download", action="store_true", help="download missing model files from Hugging Face")
    ap.add_argument("--keep", action="store_true", help="keep the temporary data folder")
    ap.add_argument("--out", help="result file (default result-<gpu>-<time>.json in the current folder)")
    a = ap.parse_args()
    exp = json.load(open(os.path.join(HERE, "expected.json"), encoding="utf-8"))
    names = [n for n in exp["receipts"] if not a.only or n in a.only]
    if not names:
        print("no receipt selected")
        return 2
    os.makedirs(a.models_dir[0], exist_ok=True)
    data = tempfile.mkdtemp(prefix="geulpu-replay-kit-")
    proc, log = None, None
    results, env = {}, {}
    try:
        # 1. model files
        need = sorted({exp["receipts"][n]["model"] for n in names})
        paths = {}
        for m in need:
            spec = exp["models"][m]
            p = next((os.path.join(d, m) for d in a.models_dir if os.path.exists(os.path.join(d, m))), os.path.join(a.models_dir[0], m))
            if not os.path.exists(p):
                if not a.download:
                    raise KitError(f"{p} is missing — download {spec['url']} (or pass --download)")
                download(spec["url"], p)
            if os.path.getsize(p) != spec["size"]:
                raise KitError(f"{m}: size {os.path.getsize(p)} differs from {spec['size']} — not the kit's file")
            print(f"checking {m} (SHA-256) …", flush=True)
            h = sha256_file(p)
            if h != spec["sha256"]:
                raise KitError(f"{m}: SHA-256 {h[:12]}… differs from {spec['sha256'][:12]}… — not the kit's file")
            paths[m] = os.path.abspath(p)
        # 2. receipts into the private data folder, pointing at the local model files
        rd = os.path.join(data, "영수증")
        os.makedirs(rd, exist_ok=True)
        for n in names:
            r = json.load(open(os.path.join(HERE, "receipts", n + ".json"), encoding="utf-8"))
            r["model"]["path"] = paths[exp["receipts"][n]["model"]]
            json.dump(r, open(os.path.join(rd, n + ".json"), "w", encoding="utf-8"), ensure_ascii=False)
        # 3. private GeulPU instance
        cmd = find_app(a.app)
        port = free_port(a.port)
        base = f"http://127.0.0.1:{port}"
        log = open(os.path.join(data, "kit-app.log"), "wb")
        proc = subprocess.Popen(cmd + ["--port", str(port), "--data", data, "--no-open"], stdout=log, stderr=subprocess.STDOUT,
                                stdin=subprocess.DEVNULL)
        print(f"started GeulPU on port {port} with data folder {data}", flush=True)
        for _ in range(240):
            if proc.poll() is not None:
                raise KitError("GeulPU exited at start — see " + os.path.join(data, "kit-app.log"))
            try:
                api(base, "/api/ping", timeout=5)
                break
            except (urllib.error.URLError, ConnectionError, OSError, KitError):
                time.sleep(0.5)
        st = {}
        for _ in range(900):
            st = api(base, "/api/state", timeout=60)
            if (st.get("device") or {}).get("checked") and (st.get("job") or {}).get("state") != "running":
                break
            time.sleep(1)
        env = {"app": st.get("version"), "device": st.get("device"), "os": platform.platform(), "python": platform.python_version(),
               "layer": a.layer, "vulkan_device": a.device}
        print(f"GeulPU {st.get('version')} — device check done", flush=True)
        # 4. verify each receipt
        for n in names:
            e = exp["receipts"][n]
            print(f"{n}: recomputing {e['prompt_tokens']} prompt tokens + {e['steps']} steps on {a.layer} …", flush=True)
            t0 = time.time()
            j = api(base, "/api/llama/verify", {"receipt": n, "layer": a.layer, "device": a.device})
            job = j.get("job")
            fin = {}
            while time.time() - t0 < 4 * 3600:
                time.sleep(2)
                fin = api(base, "/api/state", timeout=60).get("job") or {}
                if fin.get("id") == job and fin.get("state") != "running":
                    break
                if proc.poll() is not None:
                    raise KitError("GeulPU exited during the check — see " + os.path.join(data, "kit-app.log"))
            sm = {}
            if fin.get("result"):
                p = os.path.join(data, "결과", fin["result"], "summary.json")
                if os.path.exists(p):
                    sm = json.load(open(p, encoding="utf-8"))
            ok = bool(sm.get("same")) and sm.get("values") == e["values"] and sm.get("steps") == e["steps"]
            results[n] = {"same_as_reference": ok, "state": fin.get("state"), "summary": fin.get("summary"), "reason": fin.get("reason"),
                          "steps": sm.get("steps"), "values": sm.get("values"), "mismatches": sm.get("mismatches"),
                          "first_divergence": sm.get("first_divergence"), "seconds": round(time.time() - t0, 1)}
            print(f"  {'SAME' if ok else 'DIFFERENT'} — {fin.get('summary') or fin.get('reason')}", flush=True)
    except KitError as ex:
        print("kit stopped:", ex)
        results["_error"] = str(ex)
    finally:
        if proc is not None and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(30)
            except subprocess.TimeoutExpired:
                proc.kill()
        if log is not None:
            log.close()
    dev = ((env.get("device") or {}).get("name") or (env.get("device") or {}).get("gpu") or "gpu").replace(" ", "-")
    out = a.out or f"result-{''.join(c for c in dev if c.isalnum() or c in '-_')[:40]}-{datetime.datetime.now():%Y%m%d-%H%M%S}.json"
    rep = {"kit": KIT, "time": datetime.datetime.now().isoformat(timespec="seconds"), "environment": env, "results": results}
    json.dump(rep, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    if a.keep or "_error" in results:
        print("data folder kept:", data)
    else:
        shutil.rmtree(data, ignore_errors=True)
    done = [n for n in names if n in results]
    same = sum(1 for n in done if results[n]["same_as_reference"])
    print(f"result: {same} of {len(names)} receipts recomputed bit for bit — written to {out}")
    if "_error" in results or len(done) < len(names):
        return 2
    return 0 if same == len(names) else 1


if __name__ == "__main__":
    sys.exit(main())
