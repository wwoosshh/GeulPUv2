"""GeulPU from Python — drive a GeulPU app: load a GGUF model, generate with every step's logit hash, score text, branch at a
token, send a request again and compare runs bit for bit. Python standard library only (3.9+).

    import geulpu

    with geulpu.start() as g:                          # a private GeulPU without a window; geulpu.connect() uses a running one
        g.load("~/models/Qwen3-8B-Q4_K_M.gguf", precision="exact")
        a = g.chat([{"role": "user", "content": "Name one planet."}], max_tokens=32, seed=1)
        b = g.rerun(a)                                 # the same request with the same seed
        print(a.text, geulpu.compare(a, b))            # prompt tokens, every step's logit hash, the tokens
        s = g.score(a)                                 # each answer token's log-probability, recomputed
        f = g.branch(a, step=3)                        # take the runner-up token at step 3 and continue
        print(g.info())                                # GeulPU version, GPU, driver, model SHA-256, computation path

Everything goes through the app's HTTP API — the one its research panel and debugger use (docs/research-guide.md section 5,
docs/debugger-guide.md section 8) — so every value is the engine's own; this module computes nothing itself. docs/python-api.md
describes each function.
"""
from __future__ import annotations

import atexit
import copy
import http.client
import json
import os
import platform
import re
import socket
import subprocess
import threading
import time
import urllib.error
import urllib.request
import warnings
import weakref
from dataclasses import dataclass, field
from typing import List, Optional

__all__ = ["GeulPU", "GeulPUError", "Generation", "Branch", "Score", "Comparison", "connect", "start", "compare"]

DEFAULT_URL = "http://127.0.0.1:44544"
CHAT = "/v1/chat/completions"
COMPLETIONS = "/v1/completions"
# Request fields a token branch carries over from the original request (the debugger's rule — docs/debugger-guide.md section 5).
SAMPLING_KEYS = ("temperature", "top_p", "top_k", "min_p", "stop", "ignore_eos")
# App settings load() may change; close() puts back the values it found.
SETTING_KEYS = ("precision", "precision_unquantized", "kv8_cache", "fast_attention", "speculative", "context", "slots", "gpu_layer")
STOPPED = (None, "", "off", "failed")


class GeulPUError(Exception):
    """An error from the GeulPU app (`status` and `body` are set when it answered with an HTTP error) or from this module."""

    def __init__(self, message, status=None, body=None):
        super().__init__(message)
        self.status = status
        self.body = body


@dataclass(repr=False)
class Generation:
    """One answer from /v1/chat/completions or /v1/completions.

    `tokens` and `logit_hashes` include the step that sampled the end token. `logprobs`, `token_texts`, `entropy`, `top_ids` and
    `top_logprobs` cover the answer's tokens only, as in OpenAI — one shorter when the answer ends with the end token (score()
    gives that step too) — and are None unless the request asked for logprobs. A logit hash is a 16-digit hex string: a 64-bit
    hash of the step's whole logit row. `request` is the body that was sent and `response` the app's JSON as it came."""

    text: str
    tokens: List[int]
    logit_hashes: List[str]
    prompt_tokens: List[int]
    seed: Optional[int] = None
    finish_reason: Optional[str] = None
    reasoning: Optional[str] = None
    tool_calls: Optional[list] = None
    logprobs: Optional[list] = None
    token_texts: Optional[list] = None
    entropy: Optional[list] = None
    top_ids: Optional[list] = None
    top_logprobs: Optional[list] = None
    model: Optional[str] = None
    usage: Optional[dict] = None
    path: str = CHAT
    request: dict = field(default_factory=dict)
    response: dict = field(default_factory=dict)
    seconds: float = 0.0

    def __repr__(self):
        t = self.text if len(self.text) <= 60 else self.text[:57] + "..."
        return f"{type(self).__name__}({len(self.tokens)} tokens, finish={self.finish_reason!r}, seed={self.seed}, text={t!r})"


@dataclass(repr=False)
class Branch(Generation):
    """A token branch: the original answer up to `step`, then `token` in place of `original_token`, then what the model generates
    after it with the same seed and the original's random numbers. `tokens`, `logit_hashes` and the log-probability lists cover
    the whole branched answer, so compare(original, branch) names the first step that differs; `new_tokens` are the tokens
    generated after the forced one. `prefix_same` says whether the branch recomputed the original's logit hashes of steps
    0 … `step` (None when the original had none). `text` is the branched answer decoded with special tokens as text."""

    step: int = 0
    token: int = 0
    original_token: Optional[int] = None
    new_tokens: List[int] = field(default_factory=list)
    prefix_same: Optional[bool] = None
    max_tokens_total: Optional[int] = None      # the original request's max_tokens — a branch asks for what is left of it

    def __repr__(self):
        return (f"Branch(step {self.step}: {self.original_token} -> {self.token}, {len(self.tokens)} tokens, "
                f"finish={self.finish_reason!r}, prefix_same={self.prefix_same})")


@dataclass(repr=False)
class Score:
    """Each token's log-probability after a prompt, recomputed without generating (echo). The values are the model's own
    distribution (temperature 1, no filtering — precision contract 5.2). With the same model and computation mode the logit
    hashes equal those of the generation that produced the tokens."""

    tokens: List[int]
    prompt_tokens: List[int]
    logprobs: list
    entropy: list
    top_ids: list
    top_logprobs: list
    token_texts: list
    logit_hashes: List[str]
    request: dict = field(default_factory=dict)
    response: dict = field(default_factory=dict)

    @property
    def total(self):
        """The sum of the tokens' log-probabilities (nats)."""
        return sum(x for x in self.logprobs if x is not None)

    def __repr__(self):
        return f"Score({len(self.tokens)} tokens, total log-prob {self.total:.4f})"


@dataclass
class Comparison:
    """The result of compare(a, b) — true when the prompt tokens, every step's logit hash and the tokens are all identical."""

    same: bool
    why: str = "identical"
    first_diff_step: Optional[int] = None

    def __bool__(self):
        return self.same


def compare(a, b):
    """Compare two results (Generation, Branch or Score) bit for bit: the prompt tokens, then each step's logit hash, then the
    tokens. Returns a Comparison (true when identical; `why` and `first_diff_step` say where they part)."""
    pa, pb = getattr(a, "prompt_tokens", None), getattr(b, "prompt_tokens", None)
    if pa is not None and pb is not None and list(pa) != list(pb):
        return Comparison(False, "prompt tokens differ")
    ha, hb = list(getattr(a, "logit_hashes", None) or []), list(getattr(b, "logit_hashes", None) or [])
    ta, tb = list(a.tokens), list(b.tokens)
    for i in range(max(len(ta), len(tb))):
        if i < len(ha) and i < len(hb) and ha[i] != hb[i]:
            return Comparison(False, f"logits differ from step {i}", i)
        if i >= len(ta) or i >= len(tb):
            return Comparison(False, f"one answer ends at step {i}, the other goes on", i)
        if ta[i] != tb[i]:
            return Comparison(False, f"tokens differ from step {i} (same logits there)", i)
    if ha and hb and len(ha) != len(hb):
        return Comparison(False, "different numbers of logit hashes", min(len(ha), len(hb)))
    return Comparison(True)


class GeulPU:
    """A GeulPU app at `url` — connect() returns one for an app that is running, start() launches a private one. The methods are
    safe to call from several threads (the app's server runs as many requests at once as it has slots)."""

    def __init__(self, url=DEFAULT_URL, timeout=3600.0):
        self.url = url.rstrip("/")
        self.timeout = timeout                  # seconds per request (a long answer can take minutes)
        self._proc = None                       # the app process, when start() launched it
        self._console_log = None
        self._saved = None                      # the app's settings before this client first changed them
        self._loaded = None                     # (model id, settings) of the last load() — reloading is skipped when equal
        self._lock = threading.Lock()
        self._closed = False

    def __repr__(self):
        return f"GeulPU({self.url!r}{', started here' if self._proc is not None else ''})"

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    # ---- HTTP ----
    def request(self, method, path, body=None, timeout=None):
        """Send one request to the app and return its JSON (None for an empty answer). HTTP errors raise GeulPUError."""
        data = None if body is None else json.dumps(body, ensure_ascii=False).encode("utf-8")
        # X-GeulPU-Source: the app records an event for every call that computes or changes records, with its source (/api/events)
        rq = urllib.request.Request(self.url + path, data=data, method=method,
                                    headers={"Content-Type": "application/json", "X-GeulPU-Source": "python"})
        t = self.timeout if timeout is None else timeout
        try:
            with urllib.request.urlopen(rq, timeout=t) as r:
                raw = r.read()
        except urllib.error.HTTPError as e:
            raw = e.read()
            parsed = _loads(raw)
            raise GeulPUError(f"{path}: HTTP {e.code} — {_message(parsed)}", status=e.code, body=parsed) from None
        except urllib.error.URLError as e:
            if isinstance(e.reason, (socket.timeout, TimeoutError)):
                raise GeulPUError(f"{path}: no answer from GeulPU within {t:g} s") from None
            raise GeulPUError(f"cannot reach GeulPU at {self.url} ({e.reason}) — start it with `geulpu --no-open` "
                              f"or use geulpu.start()") from None
        except (socket.timeout, TimeoutError):
            raise GeulPUError(f"{path}: no answer from GeulPU within {t:g} s") from None
        except (ConnectionError, http.client.HTTPException) as e:
            raise GeulPUError(f"{path}: the connection to GeulPU broke ({e!r})") from None
        return _loads(raw) if raw else None

    def get(self, path, timeout=None):
        return self.request("GET", path, None, timeout)

    def post(self, path, body=None, timeout=None):
        return self.request("POST", path, {} if body is None else body, timeout)

    # ---- the app ----
    def ping(self):
        """{"app": "geulpu", "version": …} when a GeulPU app answers at this address."""
        return self.get("/api/ping", timeout=10)

    def state(self):
        """The app's /api/state as it is — app, settings, device, server, job and models."""
        return self.get("/api/state", timeout=60)

    def models(self):
        """The app's model list (id, name, path, size, sha256, shape …)."""
        return list(self.state().get("models") or [])

    def wait_device(self, timeout=180.0):
        """Wait until the app has checked the GPU (it does so when it starts) and return its device entry."""
        end = time.time() + timeout
        while True:
            d = self.state().get("device") or {}
            if d.get("checked"):
                return d
            if time.time() > end:
                raise GeulPUError(f"the app did not finish checking the GPU within {timeout:g} s")
            time.sleep(0.2)

    def info(self, provenance=False):
        """What a methods section needs: the GeulPU version and commit, the GPU and its driver, the loaded model file with its
        SHA-256, and the server's computation path (precision, 8-bit KV cache, fast attention, speculation, context, YaRN).
        provenance=True adds /api/provenance — the SHA-256 of the server binaries and of the loaded kernel module."""
        st = self.state()
        app, dev, sv = st.get("app") or {}, st.get("device") or {}, st.get("server") or {}
        m = _by_id(st, sv.get("model")) if sv.get("model") else None
        shape = (m or {}).get("shape") or {}
        out = {
            "geulpu": {k: app.get(k) for k in ("version", "commit", "built", "installer")},
            "url": self.url,
            "data_dir": app.get("data_dir"),
            "device": {k: dev.get(k) for k in ("name", "vendor", "driver", "memory", "layer", "variant")},
            "server": {k: sv.get(k) for k in ("state", "layer", "precision", "kv8", "fast_attention", "speculative", "template",
                                              "slots", "context", "yarn")},
            "model": None if m is None else {
                "id": m.get("id"), "file": os.path.basename(m.get("path") or ""), "path": m.get("path"), "size": m.get("size"),
                "sha256": m.get("sha256") or None, "arch": shape.get("arch"), "quant": shape.get("quant"),
                "file_type": shape.get("file_type")},
            "python": platform.python_version(),
            "os": platform.platform(),
        }
        if provenance:
            out["provenance"] = self.get("/api/provenance", timeout=600)
        return out

    def add_model(self, path, verify=True, timeout=1800.0):
        """Add a local .gguf file to the app's model list (or find it there) and, with verify=True, wait until the app has its
        SHA-256. Returns the app's model entry."""
        p = os.path.abspath(os.path.expanduser(str(path)))
        if not os.path.isfile(p):
            raise GeulPUError(f"no such file: {p}")
        m = _by_path(self.state(), p)
        if m is None:
            mid = None
            try:
                mid = (self.post("/api/models/add", {"path": p}) or {}).get("id")
            except GeulPUError as e:            # 409: already in the list under another spelling of the path — the message ends with its id
                if e.status != 409:
                    raise
                hit = re.search(r":\s*(\S+)\s*$", _message(e.body))
                mid = hit.group(1) if hit else None
            st = self.state()
            m = (_by_id(st, mid) if mid else None) or _by_path(st, p)
            if m is None:
                raise GeulPUError(f"the app did not list {p} after adding it")
        return self._verified(m, timeout) if verify else m

    def load(self, model, *, precision=None, kv8=None, fast_attention=None, speculative=None, context=None, slots=None,
             layer=None, timeout=1800.0):
        """Load `model` — a .gguf path (added to the app when needed), a model id from models() or 8+ hex digits of its SHA-256 —
        and wait until the server is ready. The keyword arguments change the app's settings first (they apply when a model loads):
        precision "exact" or "fast", kv8 (8-bit KV cache), fast_attention, speculative (bit-identical speculative decoding),
        context, slots (requests at once, 1–8) and layer ("cuda", "vulkan" or "auto"); leave one out to keep the app's setting.
        Returns the server state and warns when a setting did not take effect: the app decides per model and device — the 8-bit
        KV cache needs fast precision, CUDA and a model that takes it (the model's shape.kv8_ok), fast attention runs on top of
        it. close() puts the settings back."""
        changes = {}
        if precision is not None:
            if precision not in ("exact", "fast"):
                raise ValueError("precision is 'exact' or 'fast'")
            changes["precision"] = changes["precision_unquantized"] = precision
        for key, v in (("kv8_cache", kv8), ("fast_attention", fast_attention), ("speculative", speculative)):
            if v is not None:
                changes[key] = bool(v)
        for key, v in (("context", context), ("slots", slots)):
            if v is not None:
                changes[key] = int(v)
        if layer is not None:
            if layer not in ("cuda", "vulkan", "auto"):
                raise ValueError("layer is 'cuda', 'vulkan' or 'auto'")
            changes["gpu_layer"] = layer
        m = self._resolve(model, timeout)
        with self._lock:
            st = self.state()
            if changes:
                if self._saved is None:
                    s = st.get("settings") or {}
                    self._saved = {k: s[k] for k in SETTING_KEYS if k in s}
                self.post("/api/settings", changes)
            if layer in ("cuda", "vulkan") and (st.get("device") or {}).get("layer") != layer:
                self._device_check(layer, timeout)
            st = self.state()
            s = st.get("settings") or {}
            key = (m["id"], json.dumps({k: s.get(k) for k in SETTING_KEYS}, sort_keys=True))
            sv = st.get("server") or {}
            if self._loaded == key and sv.get("state") == "ready" and sv.get("model") == m["id"]:
                return sv
            self._stop_server(timeout)
            self.post("/api/server/start", {"model": m["id"]})
            sv = self._wait_ready(m["id"], timeout)
            self._loaded = key
        _warn_unapplied(changes, sv, m.get("shape") or {})
        return sv

    def unload(self):
        """Stop the app's server (the model leaves the GPU)."""
        with self._lock:
            self._stop_server()

    # ---- generating ----
    def chat(self, messages, **params):
        """POST /v1/chat/completions without streaming and return a Generation. `params` are the request's fields — max_tokens,
        temperature (the API's default is 0.8), seed, top_p, top_k, min_p, stop, tools, tool_choice, logprobs, top_logprobs (0–20)
        and so on; the request always asks for the prompt tokens and every step's logit hash, and geulpu={…} adds other
        extension fields (docs/user-guide.md). Leave `model` out: the loaded model answers and `model` in the answer names it."""
        body = dict(params)
        body["messages"] = list(messages)
        return self._generate(CHAT, body)

    def complete(self, prompt, **params):
        """POST /v1/completions without streaming — `prompt` is text (no chat template) or a list of token ids; `logprobs=k` asks
        for each token's log-probability and the top k. Returns a Generation."""
        body = dict(params)
        body["prompt"] = prompt if isinstance(prompt, str) else [int(t) for t in prompt]
        return self._generate(COMPLETIONS, body)

    def rerun(self, gen, **changes):
        """Send `gen`'s request again with its seed — a replay, identical when nothing changed. Keyword arguments replace request
        fields first (a request branch, e.g. messages=… or temperature=…). A Branch is sent again as the same branch."""
        if isinstance(gen, Branch):
            if changes:
                raise ValueError("rerun() of a Branch takes no changes — call branch() on the original answer instead")
            return self._run_branch(copy.deepcopy(gen.request), gen.prompt_tokens, gen.tokens[:gen.step], gen.token,
                                    gen.original_token, gen.step, gen.logit_hashes[:gen.step + 1] or None, gen.max_tokens_total)
        body = copy.deepcopy(gen.request)
        if gen.seed is not None and "seed" not in changes:
            body["seed"] = gen.seed
        body.update(copy.deepcopy(changes))
        return self._generate(gen.path, body)

    def score(self, prompt, answer=None, *, top=2):
        """Each token's log-probability after a prompt, computed without generating (echo on /v1/completions); returns a Score.
        score(gen) scores a Generation's own tokens after its prompt — with the same model and computation mode the logit hashes
        equal the generation's. Otherwise `prompt` is a list of token ids, text (tokenized without a chat template) or chat
        messages (with the model's chat template, up to the answer), and `answer` is a list of token ids, a Generation (its
        tokens) or, with a text prompt, text (the tokens of prompt + answer after those of the prompt). top = alternatives kept
        per step."""
        if isinstance(prompt, Generation) and answer is None:
            P, A = list(prompt.prompt_tokens), list(prompt.tokens)
        else:
            P = self._prompt_ids(prompt)
            A = self._answer_ids(prompt, P, answer)
        if not P:
            raise GeulPUError("score() needs at least one prompt token (the first token of a sequence has no probability)")
        body = {"prompt": P + A, "echo": True, "max_tokens": 0, "logprobs": max(1, int(top)), "temperature": 0, "stream": False,
                "geulpu": {"echo_from": len(P), "logit_hash": True, "special": True}}
        r = self.post(COMPLETIONS, body)
        if not isinstance(r, dict):
            raise GeulPUError(f"{COMPLETIONS}: the app's answer is not a JSON object")
        k, n = len(P), len(A)
        lp = ((r.get("choices") or [{}])[0]).get("logprobs") or {}
        g = r.get("geulpu") or {}

        def part(xs):                       # the echo arrays have one entry per prompt position (null before echo_from)
            return list((xs or [])[k:k + n])

        return Score(tokens=A, prompt_tokens=P, logprobs=part(lp.get("token_logprobs")), entropy=part(g.get("prompt_entropy")),
                     top_ids=part(g.get("prompt_top_ids")), top_logprobs=part(g.get("prompt_top_logprobs")),
                     token_texts=part(lp.get("tokens")), logit_hashes=list((g.get("prompt_logit_hashes") or [])[:n]),
                     request=body, response=r)

    def branch(self, gen, step, token=None, *, top=2):
        """Continue `gen` from `step` with another token and return a Branch — the debugger's token branch (docs/debugger-guide.md
        section 5). token=None takes the highest-ranked token other than the original, token="same" the original (a self-check:
        the rest of the answer must come back), or pass a token id. The branch reuses the prompt tokens, the answer before `step`,
        the seed, the sampling fields and what is left of max_tokens, and continues with the original's random numbers. It runs
        on /v1/completions, so tool calls stay plain text and it ends only at the end token, max_tokens or stop."""
        if not isinstance(gen, Generation):
            raise TypeError("branch() takes a Generation (from chat, complete, rerun or branch)")
        P, T = list(gen.prompt_tokens), list(gen.tokens)
        k = int(step)
        if not 0 <= k < len(T):
            raise GeulPUError(f"step {k} is outside the answer (steps 0 … {len(T) - 1})")
        if gen.seed is None:
            raise GeulPUError("the answer has no seed — branch() needs the seed the app reports")
        orig = int(T[k])
        if isinstance(token, str):
            if token != "same":
                raise ValueError("token is None, 'same' or a token id")
            alt = orig
        elif token is None:
            alt = self._runner_up(gen, k)
        else:
            alt = int(token)
        src = gen.request or {}
        total = gen.max_tokens_total if isinstance(gen, Branch) else src.get("max_tokens", src.get("max_completion_tokens"))
        body = {"prompt": P + T[:k] + [alt], "echo": True, "logprobs": max(1, int(top)), "stream": False, "seed": gen.seed,
                "geulpu": {"echo_from": len(P), "logit_hash": True, "prompt_tokens": True, "step_offset": k + 1, "special": True}}
        for key in SAMPLING_KEYS:
            if key in src:
                body[key] = copy.deepcopy(src[key])
        if total is not None:
            body["max_tokens"] = max(0, int(total) - (k + 1))
        return self._run_branch(body, P, T[:k], alt, orig, k, gen.logit_hashes[:k + 1] or None, total)

    def tokenize(self, text_or_messages):
        """Token ids of text (the model's tokenizer without a chat template — a BOS token first for Llama 3 and Mistral, as in
        /v1/completions) or of chat messages (with the model's chat template, up to the answer)."""
        x = text_or_messages
        body = {"prompt": x} if isinstance(x, str) else {"messages": list(x)}
        return [int(t) for t in self.post("/tokenize", body)["tokens"]]

    def detokenize(self, tokens):
        """The text of token ids (special tokens as their text)."""
        return (self.post("/detokenize", {"tokens": [int(t) for t in tokens]}) or {}).get("prompt") or ""

    # ---- ending ----
    def close(self):
        """Put back the app settings this client changed; for an app launched by start(), also stop its server and the app."""
        if self._closed:
            return
        self._closed = True
        alive = self._proc is None or self._proc.poll() is None
        try:
            if self._proc is not None and alive:
                try:
                    self._stop_server(60)
                except GeulPUError:
                    pass
            if self._saved and alive:
                try:
                    self.post("/api/settings", self._saved, timeout=30)
                except GeulPUError:
                    pass
        finally:
            if self._proc is not None:
                _end_process(self._proc)

    # ---- inside ----
    def _generate(self, path, body):
        body = copy.deepcopy(body)
        body["stream"] = False
        body.pop("stream_options", None)
        if path == CHAT and body.get("top_logprobs") and "logprobs" not in body:
            body["logprobs"] = True
        gp = dict(body.get("geulpu") or {})
        gp.update({"logit_hash": True, "prompt_tokens": True})
        body["geulpu"] = gp
        t0 = time.time()
        r = self.post(path, body)
        return _generation(path, body, r, time.time() - t0)

    def _run_branch(self, body, P, prefix, alt, orig, k, ref, total):
        t0 = time.time()
        r = self.post(COMPLETIONS, body)
        dt = time.time() - t0
        if not isinstance(r, dict):
            raise GeulPUError(f"{COMPLETIONS}: the app's answer is not a JSON object")
        g = r.get("geulpu") or {}
        ch = (r.get("choices") or [{}])[0]
        lp = ch.get("logprobs") or {}
        n = len(P)
        new = [int(t) for t in g.get("tokens") or []]
        pre = list(g.get("prompt_logit_hashes") or [])          # the echo rows of steps 0 … k (they score the prefix and the token)
        tokens = list(prefix) + [alt] + new
        return Branch(text=self.detokenize(tokens), tokens=tokens, logit_hashes=pre + list(g.get("logit_hashes") or []),
                      prompt_tokens=list(P), seed=body.get("seed"), finish_reason=ch.get("finish_reason"),
                      logprobs=list((lp.get("token_logprobs") or [])[n:]), token_texts=list((lp.get("tokens") or [])[n:]),
                      entropy=list((g.get("prompt_entropy") or [])[n:]) + list(g.get("entropy") or []),
                      top_ids=list((g.get("prompt_top_ids") or [])[n:]) + list(g.get("top_ids") or []),
                      top_logprobs=list((g.get("prompt_top_logprobs") or [])[n:]) + list(g.get("top_logprobs") or []),
                      model=r.get("model"), usage=r.get("usage"), path=COMPLETIONS, request=body, response=r, seconds=round(dt, 3),
                      step=k, token=alt, original_token=orig, new_tokens=new,
                      prefix_same=None if ref is None else pre == list(ref), max_tokens_total=total)

    def _runner_up(self, gen, k):
        orig = int(gen.tokens[k])
        ids = gen.top_ids[k] if gen.top_ids and k < len(gen.top_ids) else None
        alt = next((int(t) for t in ids or [] if int(t) != orig), None)
        if alt is None:                         # not asked for, only one kept, or the end token's step — score the step
            ids = self.score(gen.prompt_tokens, gen.tokens[:k + 1], top=2).top_ids[k]
            alt = next((int(t) for t in ids or [] if int(t) != orig), None)
        if alt is None:
            raise GeulPUError(f"the top list of step {k} has no token other than the original — pass token=<id>")
        return alt

    def _prompt_ids(self, prompt):
        if isinstance(prompt, str):
            return self.tokenize(prompt)
        xs = list(prompt)
        if xs and isinstance(xs[0], dict):
            return self.tokenize(xs)
        return [int(t) for t in xs]

    def _answer_ids(self, prompt, P, answer):
        if answer is None:
            raise TypeError("score(prompt, answer) needs an answer (or a Generation alone)")
        if isinstance(answer, Generation):
            return list(answer.tokens)
        if isinstance(answer, str):
            if not isinstance(prompt, str):
                raise GeulPUError("a text answer needs a text prompt — pass the answer as token ids or a Generation")
            full = self.tokenize(prompt + answer)
            if full[:len(P)] != P:
                raise GeulPUError("the prompt's tokens change when the answer is appended (a token spans the join) — "
                                  "pass the answer as token ids")
            return full[len(P):]
        return [int(t) for t in answer]

    def _resolve(self, model, timeout):
        if isinstance(model, dict):
            model = model.get("id")
        s = str(model)
        if os.path.isfile(os.path.expanduser(s)):
            return self.add_model(s, True, timeout)
        ms = self.models()
        hit = [m for m in ms if m.get("id") == s]
        if not hit and len(s) >= 8 and re.fullmatch(r"[0-9a-fA-F]+", s):
            hit = [m for m in ms if str(m.get("sha256") or "").startswith(s.lower())]
        if len(hit) != 1:
            names = ", ".join(str(m.get("id")) for m in ms) or "none"
            raise GeulPUError(f"{s!r} matches {len(hit)} of the app's models (give a .gguf path, a model id or 8+ hex digits of its "
                              f"SHA-256) — the app has: {names}")
        return self._verified(hit[0], timeout)

    def _verified(self, m, timeout):
        """Wait for the model's SHA-256 — the app computes it on its own only for files it knows by size, so ask for the rest."""
        mid, asked, end = m.get("id"), False, time.time() + timeout
        while True:
            m = _by_id(self.state(), mid)
            if m is None:
                raise GeulPUError(f"model {mid} is no longer in the app's list")
            if m.get("sha256") and not m.get("verifying"):
                return m
            if not m.get("verifying") and not asked:
                try:
                    self.post("/api/models/verify", {"id": mid})
                except GeulPUError as e:
                    if e.status != 409:         # 409: a check is already running
                        raise
                asked = True
            if time.time() > end:
                raise GeulPUError(f"{os.path.basename(m.get('path') or str(mid))}: no SHA-256 after {timeout:g} s")
            time.sleep(0.2)

    def _stop_server(self, timeout=120.0):
        if (self.state().get("server") or {}).get("state") in STOPPED:
            self._loaded = None
            return
        self.post("/api/server/stop", {})
        end = time.time() + timeout
        while (self.state().get("server") or {}).get("state") not in STOPPED:
            if time.time() > end:
                raise GeulPUError(f"the server did not stop within {timeout:g} s")
            time.sleep(0.3)
        self._loaded = None

    def _wait_ready(self, mid, timeout):
        end = time.time() + timeout
        while True:
            st = self.state()
            sv, j = st.get("server") or {}, st.get("job") or {}
            if sv.get("state") == "ready" and sv.get("model") == mid:
                return sv
            if sv.get("state") == "failed" or (j.get("kind") == "server" and j.get("state") == "failed"):
                why = " ".join(str(x) for x in (j.get("reason"), sv.get("reason"), (sv.get("log_tail") or "")[-400:]) if x)
                raise GeulPUError(f"the model did not load: {why}")
            if time.time() > end:
                raise GeulPUError(f"the model did not load within {timeout:g} s (server state {sv.get('state')})")
            time.sleep(0.2)

    def _device_check(self, layer, timeout):
        job = (self.post("/api/device/check", {}) or {}).get("job")
        end = time.time() + timeout
        while True:
            time.sleep(0.2)
            st = self.state()
            j = st.get("job") or {}
            if j.get("state") != "running" and (job is None or j.get("id") == job):
                break
            if time.time() > end:
                raise GeulPUError(f"the app's hardware check did not end within {timeout:g} s")
        d = st.get("device") or {}
        if d.get("layer") != layer:
            raise GeulPUError(f"the app could not switch to {layer} (it uses {d.get('layer')}): {d.get('reason') or ''}")


# ---- starting and connecting ----
_started = weakref.WeakSet()


def connect(url=None, timeout=3600.0):
    """A client for the GeulPU app at `url` (default: $GEULPU_URL, else http://127.0.0.1:44544 — start one with
    `geulpu --no-open`). `timeout` is the limit per request in seconds."""
    g = GeulPU(url or os.environ.get("GEULPU_URL") or DEFAULT_URL, timeout)
    p = g.ping()
    if not (isinstance(p, dict) and p.get("app") == "geulpu"):
        raise GeulPUError(f"{g.url} is not a GeulPU app (it answered {p!r})")
    return g


def start(data=None, port=None, *, exe=None, timeout=180.0, wait_device=True, request_timeout=3600.0):
    """Launch a private GeulPU app without a window and return a client for it; close() — or the end of a `with` block, or of
    the Python process — stops it. data = its data folder (default ~/GeulPU-api, apart from the folder of the app you use,
    ~/GeulPU — two apps must not share one); port = its port (default: a free one); exe = the GeulPU executable (default: the
    one in this package). The app's console output goes to geulpu-console.log in the data folder."""
    exe = os.path.abspath(os.path.expanduser(exe)) if exe else _packaged_app()
    if not os.path.isfile(exe):
        raise GeulPUError(f"no GeulPU app at {exe} — reinstall the package (pip install --force-reinstall geulpu) or pass exe=")
    folder = os.path.abspath(os.path.expanduser(data or os.path.join("~", "GeulPU-api")))
    os.makedirs(folder, exist_ok=True)
    port = int(port) if port else _free_port()
    log_path = os.path.join(folder, "geulpu-console.log")
    flags = 0x08000000 if os.name == "nt" else 0              # CREATE_NO_WINDOW — no console window
    with open(log_path, "wb") as log:
        proc = subprocess.Popen([exe, "--port", str(port), "--data", folder, "--no-open"], stdin=subprocess.DEVNULL,
                                stdout=log, stderr=subprocess.STDOUT, creationflags=flags)
    g = GeulPU(f"http://127.0.0.1:{port}", request_timeout)
    g._proc, g._console_log = proc, log_path
    _started.add(g)
    try:
        end = time.time() + timeout
        while True:
            if proc.poll() is not None:
                raise GeulPUError(f"GeulPU exited (code {proc.returncode}) before it answered — {_tail(log_path)}")
            try:
                p = g.get("/api/ping", timeout=3)
                if isinstance(p, dict) and p.get("app") == "geulpu":
                    break
            except GeulPUError:
                pass
            moved = _console_port(log_path)     # the app moves to a nearby port when the asked one is taken
            if moved and moved != port:
                port = moved
                g.url = f"http://127.0.0.1:{port}"
            if time.time() > end:
                raise GeulPUError(f"GeulPU did not answer on port {port} within {timeout:g} s — {_tail(log_path)}")
            time.sleep(0.3)
        got = (g.state().get("app") or {}).get("data_dir") or ""
        if _norm_dir(got) != _norm_dir(folder):
            raise GeulPUError(f"another GeulPU answered on port {port} (its data folder is {got})")
        if wait_device:
            g.wait_device(timeout)
    except BaseException:
        g.close()
        raise
    return g


@atexit.register
def _close_started():
    for g in list(_started):
        try:
            g.close()
        except Exception:
            pass


# ---- helpers ----
def _generation(path, body, r, seconds):
    if not isinstance(r, dict):
        raise GeulPUError(f"{path}: the app's answer is not a JSON object")
    g = r.get("geulpu") or {}
    tokens, hashes, prompt = g.get("tokens"), g.get("logit_hashes"), g.get("prompt_tokens")
    if tokens is None or hashes is None or prompt is None:
        raise GeulPUError(f"{path}: the answer has no geulpu.tokens, logit_hashes and prompt_tokens — is this a GeulPU app?")
    ch = (r.get("choices") or [{}])[0]
    lp = ch.get("logprobs") or {}
    if path == CHAT:
        msg = ch.get("message") or {}
        text, reasoning, calls = msg.get("content") or "", msg.get("reasoning_content"), msg.get("tool_calls")
        content = lp.get("content")
        logprobs = None if content is None else [c.get("logprob") for c in content]
        texts = None if content is None else [c.get("token") for c in content]
    else:
        text, reasoning, calls = ch.get("text") or "", None, None
        logprobs, texts = lp.get("token_logprobs"), lp.get("tokens")
    return Generation(text=text, tokens=[int(t) for t in tokens], logit_hashes=list(hashes), prompt_tokens=[int(t) for t in prompt],
                      seed=g.get("seed"), finish_reason=ch.get("finish_reason"), reasoning=reasoning, tool_calls=calls,
                      logprobs=logprobs, token_texts=texts, entropy=g.get("entropy"), top_ids=g.get("top_ids"),
                      top_logprobs=g.get("top_logprobs"), model=r.get("model"), usage=r.get("usage"), path=path, request=body,
                      response=r, seconds=round(seconds, 3))


def _warn_unapplied(changes, sv, shape):
    """Warn about settings the server did not take — the app decides them per model and device."""
    missed = []
    for setting, name in (("precision", "precision"), ("kv8_cache", "kv8"), ("fast_attention", "fast_attention"),
                          ("speculative", "speculative"), ("gpu_layer", "layer")):
        if setting not in changes or changes[setting] == "auto":
            continue
        want, got = changes[setting], sv.get(name)
        if (bool(got) if isinstance(want, bool) else got) == want:
            continue
        why = ""
        if setting in ("kv8_cache", "fast_attention") and want:
            if sv.get("precision") != "fast":
                why = "it needs precision='fast'"
            elif sv.get("layer") != "cuda":
                why = "it runs on CUDA only"
            elif setting == "kv8_cache" and shape.get("kv8_ok") is False:
                why = "this model does not take it (shape.kv8_ok is false)"
            elif setting == "fast_attention" and not sv.get("kv8"):
                why = "it runs only on top of the 8-bit KV cache"
            elif setting == "fast_attention" and shape.get("fast_attention_ok") is False:
                why = "this model does not take it (shape.fast_attention_ok is false)"
        elif setting == "speculative" and want and shape.get("speculative_ok") is False:
            why = "this model does not take it (shape.speculative_ok is false)"
        missed.append(f"{setting}={want!r} (the server has {got!r}" + (f" — {why})" if why else ")"))
    if missed:
        warnings.warn("not in effect for this model and device: " + "; ".join(missed), stacklevel=3)


def _by_id(st, mid):
    return next((m for m in st.get("models") or [] if mid is not None and m.get("id") == mid), None)


def _by_path(st, p):
    want = _norm_file(p)
    return next((m for m in st.get("models") or [] if m.get("path") and _norm_file(m["path"]) == want), None)


def _norm_file(p):
    return os.path.normcase(os.path.normpath(os.path.abspath(str(p))))


def _norm_dir(p):
    return os.path.normcase(os.path.normpath(str(p).replace("\\", "/"))).rstrip("\\/")


def _loads(raw):
    try:
        return json.loads(raw.decode("utf-8"))
    except ValueError:
        return raw.decode("utf-8", "replace")


def _message(body):
    if isinstance(body, dict):
        e = body.get("error", body)
        return str(e.get("message") or e) if isinstance(e, dict) else str(e)
    return str(body or "")[:300]


def _app_folder():
    try:
        from . import app_dir                   # inside the geulpu package
    except ImportError:                         # this file saved next to a script (GeulPU 0.21.0) — the installed package's app
        try:
            from geulpu import app_dir
        except ImportError:
            raise GeulPUError("no GeulPU app found — install it (pip install geulpu) or pass exe=") from None
    return app_dir()


def _packaged_app():
    folder = _app_folder()
    exe = os.path.join(folder, "GeulPU.exe" if os.name == "nt" else "GeulPU")
    if os.name != "nt" and os.path.isfile(exe) and not os.access(exe, os.X_OK):
        try:                                    # an installer that dropped the exec bit (as the `geulpu` command does)
            from .__main__ import 실행권한주기
        except ImportError:
            from geulpu.__main__ import 실행권한주기
        실행권한주기(folder)
    return exe


def _free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _console_port(path):
    try:
        with open(path, "rb") as f:
            found = re.findall(rb"http://127\.0\.0\.1:(\d+)/", f.read())
    except OSError:
        return None
    return int(found[-1]) if found else None


def _tail(path, n=600):
    try:
        with open(path, "rb") as f:
            text = f.read().decode("utf-8", "replace")
    except OSError:
        return "no console log"
    text = " ".join(text.split())
    return ("…" + text[-n:]) if len(text) > n else (text or "the console log is empty")


def _end_process(proc):
    if proc.poll() is not None:
        return
    proc.terminate()                # Windows: the app's job object ends its server too; Linux: SIGTERM, which the app passes on
    try:
        proc.wait(30)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(30)
