#!/usr/bin/env python3
"""Same bits? — does a server give the same answer when the same request is sent alone and inside a batch of concurrent requests?

Sends one request (temperature 0, fixed seed, `cache_prompt: false`) to an OpenAI-compatible `/v1/chat/completions` endpoint —
GeulPU, llama.cpp, vLLM, SGLang, Ollama … — first alone, then concurrently with other prompts, several rounds, and compares every
answer with the first one token by token: the first token where they fork and the largest difference in the per-token
log-probabilities (when the server returns them). Standard library only.

    python 도구/same_bits.py --url http://127.0.0.1:8080/v1 [--label llama.cpp] [--rounds 5] [--batch 8] [--max-tokens 160]
    python 도구/same_bits.py --url … --prompt "…" --fillers FILE      # FILE: one filler prompt per line (the other batch members)
    python 도구/same_bits.py --url … --strict                          # exit 1 when any answer differs

Output (one line per answer) and a summary: "<label>: alone N/N identical · in a batch of B M/N identical · first fork at token k".
What it shows: with most engines the answer alone repeats, but the same request inside a batch forks (the kernels sum in another order
once other sequences are in the batch, a close top-2 flips, and the rest of the answer changes). GeulPU's answers and log-probabilities are
the same bits in both cases — its precision contract makes the summation order irrelevant.
"""
import argparse
import json
import sys
import threading
import time
import urllib.error
import urllib.request

DEFAULT_PROMPT = ("A bakery sells muffins for $2.50 and scones for $3.25. On Monday it sold 48 muffins and 27 scones; on Tuesday it sold "
                  "a third fewer muffins and 50% more scones than on Monday. How much more or less money did it make on Tuesday than on "
                  "Monday? Show your reasoning step by step, then give the final amount.")
DEFAULT_FILLERS = [
    "Write a haiku about a lighthouse in winter.",
    "Explain in two paragraphs why the sky is blue, for a ten-year-old.",
    "List five common causes of a slow SQL query and one fix for each.",
    "Translate into French: 'The train leaves at seven, so we should be at the station by half past six.'",
    "What is the difference between a process and a thread? Answer in about 120 words.",
    "Give me a 7-day beginner running plan with distances in kilometres.",
    "Summarize the plot of Romeo and Juliet in exactly four sentences.",
    "Write a Python function that returns the n-th Fibonacci number iteratively, with a short docstring.",
    "Name three differences between TCP and UDP and when you would use each.",
    "Draft a polite email asking a colleague to review a document by Friday.",
    "Explain what a hash table is and why lookups are fast on average.",
    "What are the main ingredients of a classic margherita pizza, and in what order are they added?",
]


def chat(url, prompt, max_tokens, seed, timeout):
    body = {"model": "x", "messages": [{"role": "user", "content": prompt}], "temperature": 0, "seed": seed, "max_tokens": max_tokens,
            "logprobs": True, "top_logprobs": 1, "stream": False, "cache_prompt": False}
    req = urllib.request.Request(url.rstrip("/") + "/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json", "Authorization": "Bearer same-bits"})
    t0 = time.time()
    while True:
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                j = json.loads(r.read())
            break
        except urllib.error.HTTPError as e:
            text = e.read()[:200].decode("utf-8", "replace")
            if e.code == 503 and time.time() - t0 < timeout:      # llama.cpp answers 503 "Loading model" until the model is in — wait
                time.sleep(2)
                continue
            return {"error": f"HTTP {e.code}: {text}", "ms": (time.time() - t0) * 1000}
        except (urllib.error.URLError, OSError, ValueError) as e:
            return {"error": str(e), "ms": (time.time() - t0) * 1000}
    ch = (j.get("choices") or [{}])[0]
    text = ((ch.get("message") or {}).get("content")) or ""
    content = ((ch.get("logprobs") or {}).get("content")) or []
    tokens = [c.get("token", "") for c in content]
    lps = [c.get("logprob") for c in content]
    return {"text": text, "tokens": tokens, "logprobs": lps, "finish": ch.get("finish_reason"), "ms": (time.time() - t0) * 1000}


def compare(ref, ans):
    """(same, first_fork_index, max_abs_logprob_diff, pair) — tokens when the server gives them, else the text."""
    if "error" in ans or "error" in ref:
        return False, None, None, None
    if ref["tokens"] and ans["tokens"]:
        a, b = ref["tokens"], ans["tokens"]
        n = min(len(a), len(b))
        fork = next((i for i in range(n) if a[i] != b[i]), None)
        if fork is None and len(a) != len(b):
            fork = n
        common = n if fork is None else fork
        dmax = 0.0
        for i in range(common):
            la, lb = ref["logprobs"][i], ans["logprobs"][i]
            if isinstance(la, (int, float)) and isinstance(lb, (int, float)):
                dmax = max(dmax, abs(la - lb))
        same = fork is None and dmax == 0.0
        pair = None if fork is None else (a[fork] if fork < len(a) else "<end>", b[fork] if fork < len(b) else "<end>")
        return same, fork, dmax, pair
    same = ref["text"] == ans["text"]
    fork = None
    if not same:
        fork = next((i for i, (x, y) in enumerate(zip(ref["text"], ans["text"])) if x != y), min(len(ref["text"]), len(ans["text"])))
    return same, fork, None, None


def show(label, same, fork, dmax, pair, ans, unit):
    if "error" in ans:
        return f"  {label:<14} error — {ans['error']}"
    if same:
        return f"  {label:<14} identical ({len(ans['tokens']) or len(ans['text'])} {unit}, {ans['ms']:.0f} ms)"
    d = f", max |Δ log-prob| {dmax:.3g} before the fork" if dmax is not None and fork else ""
    at = "" if fork is None else f" — forks at {unit[:-1]} {fork}" + (f": {pair[0]!r} vs {pair[1]!r}" if pair else "")
    if fork is None and dmax:
        at = f" — same tokens, log-probabilities differ (max |Δ| {dmax:.3g})"
        d = ""
    return f"  {label:<14} DIFFERENT{at}{d}"


def main():
    p = argparse.ArgumentParser(description="Same request alone vs inside a batch — are the answers the same bits?")
    p.add_argument("--url", required=True, help="OpenAI-compatible base URL, e.g. http://127.0.0.1:8080/v1")
    p.add_argument("--label", default=None, help="name shown in the output (default: the URL)")
    p.add_argument("--rounds", type=int, default=5, help="rounds (each: once alone, once in a batch)")
    p.add_argument("--batch", type=int, default=8, help="concurrent requests in the batch, the probe included")
    p.add_argument("--max-tokens", type=int, default=160)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--prompt", default=DEFAULT_PROMPT)
    p.add_argument("--fillers", default=None, help="file with one filler prompt per line (default: built-in prompts)")
    p.add_argument("--timeout", type=float, default=600)
    p.add_argument("--strict", action="store_true", help="exit 1 when any answer differs from the first")
    a = p.parse_args()
    label = a.label or a.url
    fillers = DEFAULT_FILLERS
    if a.fillers:
        with open(a.fillers, encoding="utf-8") as f:
            fillers = [ln.rstrip("\n") for ln in f if ln.strip()]
    if a.batch < 2:
        p.error("--batch must be at least 2")
    print(f"{label} — {a.url}")
    print(f"probe: {a.prompt[:72]!r}… · temperature 0 · seed {a.seed} · max_tokens {a.max_tokens} · cache_prompt false")
    ref = chat(a.url, a.prompt, a.max_tokens, a.seed, a.timeout)
    if "error" in ref:
        print(f"  first request failed — {ref['error']}")
        return 2
    unit = "tokens" if ref["tokens"] else "chars"
    lp = "with per-token log-probabilities" if ref["tokens"] and any(isinstance(x, (int, float)) for x in ref["logprobs"]) else "text only (no log-probabilities returned)"
    print(f"  reference      {len(ref['tokens']) or len(ref['text'])} {unit}, finish {ref['finish']}, {ref['ms']:.0f} ms — compared {lp}")
    print(f"  answer starts: {ref['text'][:100]!r}")
    alone_same = batch_same = 0
    forks = []
    for r in range(1, a.rounds + 1):
        ans = chat(a.url, a.prompt, a.max_tokens, a.seed, a.timeout)
        same, fork, dmax, pair = compare(ref, ans)
        alone_same += same
        if not same and fork is not None:
            forks.append(fork)
        print(show(f"round {r} alone", same, fork, dmax, pair, ans, unit))
        results = [None] * a.batch

        def worker(i, prompt):
            results[i] = chat(a.url, prompt, a.max_tokens, a.seed + i, a.timeout)

        others = [fillers[(r * (a.batch - 1) + k) % len(fillers)] for k in range(a.batch - 1)]
        threads = [threading.Thread(target=worker, args=(0, a.prompt))] + [threading.Thread(target=worker, args=(i + 1, q)) for i, q in enumerate(others)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        same, fork, dmax, pair = compare(ref, results[0])
        batch_same += same
        if not same and fork is not None:
            forks.append(fork)
        errs = sum(1 for x in results[1:] if x and "error" in x)
        print(show(f"round {r} batch{a.batch}", same, fork, dmax, pair, results[0], unit) + (f"  ({errs} filler errors)" if errs else ""))
    first = f" · first fork at {unit[:-1]} {min(forks)}" if forks else ""
    verdict = "same bits every time" if alone_same == a.rounds and batch_same == a.rounds else "the answer changes"
    print(f"{label}: alone {alone_same}/{a.rounds} identical · in a batch of {a.batch} {batch_same}/{a.rounds} identical{first} — {verdict}")
    return 1 if a.strict and verdict != "same bits every time" else 0


if __name__ == "__main__":
    sys.exit(main())
