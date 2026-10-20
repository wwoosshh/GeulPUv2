#!/usr/bin/env python3
"""GeulPU 디버거 v1(문서/디버거.md) — 실행 기록 · 테스트 세트 · 재생(replay) · 점수(echo 스코어링) · 중단점 · 분기 · 비교를 스크립트로.
앱(GeulPU)의 API 와 /v1 으로 묻는다. 표준 라이브러리만.

    python 도구/debugger.py [--app http://127.0.0.1:44544] status
    python 도구/debugger.py record-start [NAME]       # 앱의 /v1 을 지나는 생성 요청을 실행 로그에(자료\\실행기록\\<NAME>.jsonl)
    python 도구/debugger.py record-stop
    python 도구/debugger.py get NAME FILE.jsonl         # 실행 로그를 받아 파일로
    python 도구/debugger.py delete NAME
    python 도구/debugger.py load MODEL [--precision exact|fast] [--kv8 on|off] [--layer cuda|vulkan]
                                                       # 앱의 모델을 불러온다(모델 아이디 · sha256 앞 8 자 이상 · run:실행)
    python 도구/debugger.py run-set TESTSET.jsonl NAME [--seed 1000]
    python 도구/debugger.py replay RUN [--out REPORT.json] [--requests 3,7] [--ignore-header]
    python 도구/debugger.py score RUN [--requests 3,7] [--check]
    python 도구/debugger.py breakpoints RUN [--margin 0.05] [--prob 0.05] [--requests 3,7] [--out FILE.json]
    python 도구/debugger.py branch RUN --request I --step K [--token ID|same] [--alt N] [--repeat 2] [--no-save]
    python 도구/debugger.py branch RUN --request I --set PATH=VALUE [--set-file PATH=FILE] [--repeat 2] [--no-save]
    python 도구/debugger.py compare RUN_A RUN_B [--out REPORT.json]

RUN 은 실행 로그 파일(.jsonl) 또는 앱에 있는 실행 이름이다.

실행 로그(run log, JSONL): 첫 줄은 헤더 {"format": "geulpu-run-log/1", "app", "commit", "model": {id, file, sha256, …}, "server":
{layer, precision, context, yarn, …}, "build": {server — 그 백엔드의 서버 실행 파일 sha256, module, module_sha256}}, 그 뒤 요청마다
{"i", "at", "ms", "path", "status", "model", "server", "request", "response"} — server 는 그 요청을 계산한 서버(모델 sha256 · 백엔드 ·
계산 방식 · 문맥 · YaRN 배), response 는 서버의 응답 JSON(스트리밍이면 마지막 이벤트)이고 그 "geulpu" 에 시드 · prompt_tokens ·
tokens · logit_hashes 가 있다. 첫 기록기의 헤더({"geulrun": 1, …})도 읽는다. 이 도구가 덧붙이는 분석 줄은 맨 위에 "type" 이 있다
(앱의 실행이면 /api/debug/run/append 로, 파일이면 그 파일 끝에):
  {"type": "score", "of": i, "model", "server", "k": 2, "logprobs", "entropy", "top_ids", "top_logprobs", "token_text",
   "logit_hashes", "hash_check"}  — 요청 i 의 답을 그 서버로 echo 스코어링한 값(스텝마다)
  {"type": "branch", "b", "of": i, "kind": "token" | "request", "path", "request", "response", "checks", …}  — 분기

테스트 세트(test set, JSONL): 줄마다 채팅 요청 하나({"messages": …, "max_tokens": …, "temperature": …}) 또는 {"path":
"/v1/completions", "request": {…}}. 시드가 없으면 --seed + 차례를 넣는다(같은 테스트 세트면 같은 실행). 스트리밍은 끈다.

재생: 기록한 요청(과 분기)을 그대로(시드는 기록한 값으로 고정해) 다시 보내 프롬프트 토큰 · 나온 토큰 · 스텝마다 로짓 해시 · 끝난
까닭(finish_reason)을 비교한다. 오류로 끝난 요청(문맥 초과 등)은 같은 오류(HTTP 상태 · 코드)가 나오는지 본다. 요청을 계산한 서버의
모델 sha256 · 계산 방식(정밀 · 빠른) · YaRN 배(빠른 모드면 백엔드도)가 지금과 다르면 다른 계산이라 먼저 알리고 멈춘다(--ignore-header
면 그대로). GeulPU 버전 · 서버 실행 파일 · 모듈(헤더의 build)이 다르면 알리기만 하고 돌린다 — 새 빌드가 비트를 바꿨나를 재생이 바로 본다.

점수(score): 요청마다 답(나온 토큰)을 지금 불러온 모델로 echo 스코어링한다 — 스텝마다 그 토큰의 로그확률 · 엔트로피 · 상위 2(번호 ·
로그확률) · 토큰의 글 · 로짓 해시. 분포는 모델의 것(온도 1 · 거르기 없음 — 정밀도 계약 5.2)이다. 지금 서버가 그 요청을 생성한 서버와
같으면(모델 sha256 · 계산 방식 · YaRN 배, 빠른 모드면 백엔드) 로짓 해시가 생성 때와 같아야 한다(hash_check "same" — 점수가 생성과 같은
계산이라는 확인). 다른 모델이면 "other_model" — 비교(compare)의 'A 의 답을 B 로 점수'가 이것이다. --check 는 덧붙이지 않고 이미 있는
점수와 비트까지 같은지 본다.

중단점(breakpoints): 그 요청을 생성한 모델의 점수에서 — 마진(1위와 2위의 로그확률 차, nat) < --margin(마진이 작은 스텝 · low_margin),
고른 토큰의 확률 < --prob(저확률 샘플 · low_prob), 도구 호출의 시작(토큰의 글이 <tool_call> · <|python_tag|> — Llama 3 의 JSON
꼴 호출은 스텝 0 · tool_call). 점수가 없고 지금 서버가 그 요청을 생성한 서버면 먼저 점수를 매겨 덧붙인다.

분기(branch): (1) 토큰 분기 — 요청 I 의 스텝 K 에서 다른 토큰(--token 번호, --alt N = 상위 N 위, 기본은 원래와 다른 가장 높은
토큰, --token same = 원래 토큰 — 원래 답이 그대로 나와야 하는 자기 확인)을 골라, /v1/completions 에 프롬프트 토큰 + 앞 K 토큰 + 고른
토큰을 넣고 같은 시드 · 같은 샘플링 설정 · step_offset K + 1(스텝 K + 1 부터 원래와 같은 난수)로 잇는다. echo 로 앞 K + 1 스텝의 로짓
해시를 함께 받아 원래와 같은 계산인지 본다. (2) 요청 분기 — 요청을 고쳐(--set messages.2.content=… · --set temperature=0 — 값은 JSON
이면 JSON, 아니면 글. --set-file 은 파일의 글) 같은 시드로 다시 생성해 원래와 비교한다(프롬프트 토큰이 처음 달라진 자리 · 답이 처음
갈라진 스텝). 둘 다 --repeat 번 돌려 같은 비트인지 본다. 토큰 분기는 /v1/completions 라 서버가 도구 호출을 풀지 않고(글 그대로),
parallel_tool_calls 가 거짓(Llama 3 채팅 템플릿)이어도 /v1/chat/completions 처럼 첫 호출에서 끝내지 않는다 — 분기는 EOS 토큰 ·
max_tokens · stop 에서만 끝난다.

비교(compare): 실행 A 와 B(같은 요청열 — 다른 모델 파일 · 양자화 · 설정 · GeulPU 버전)를 요청 차례(i)마다 — 같은 요청 · 같은 프롬프트
토큰인가, 같은 답인가 · 처음 갈라진 스텝, 그리고 A 의 답을 A 의 모델과 B 의 모델로 점수 매긴 로그확률의 차(스텝마다 B − A — 합 · 평균 ·
가장 나빠진 스텝), B 의 1위 토큰이 A 의 토큰과 다른 스텝 수. 점수는 실행 A 의 점수 줄을 쓴다(모델 B 를 불러온 뒤 score A).
"""
import argparse
import copy
import json
import math
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

생성길 = ("/v1/chat/completions", "/v1/completions")
도구호출표시 = ("<tool_call>", "<|python_tag|>")       # 도구 호출을 여는 토큰의 글(Qwen2.5 · Qwen3 · Llama 3.1)
덧붙임한도 = 8 * 1024 * 1024                          # 앱의 요청 본문 한도(16 MB) 안에서 나눠 보낸다
점수k = 2                                              # 점수 줄의 상위 k — 마진 · 1위면 둘로 된다(상위 10 은 화면이 그때 다시 계산한다)


def 묻기(앱, 방법, 길, 몸=None, 시간=3600, 날것=False):
    r = urllib.request.Request(앱.rstrip("/") + 길, data=None if 몸 is None else json.dumps(몸, ensure_ascii=False).encode("utf-8"),
                               method=방법, headers={"Content-Type": "application/json"})
    try:
        d = urllib.request.urlopen(r, timeout=시간).read()
    except urllib.error.HTTPError as e:
        본 = e.read().decode("utf-8", "replace")
        raise SystemExit(f"{길}: HTTP {e.code} — {본[:300]}")
    except urllib.error.URLError as e:
        raise SystemExit(f"앱({앱})에 연결하지 못했다 — {e.reason} (앱이 켜져 있나 · --app 또는 GEULPU_APP)")
    if 날것:
        return d
    return json.loads(d) if d else None


def 묻기상태(앱, 길, 몸, 시간=3600):
    """POST — (HTTP 상태, JSON 또는 None). 오류 응답도 돌려준다"""
    r = urllib.request.Request(앱.rstrip("/") + 길, data=json.dumps(몸, ensure_ascii=False).encode("utf-8"), method="POST",
                               headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(r, timeout=시간) as f:
            상태, d = f.status, f.read()
    except urllib.error.HTTPError as e:
        상태, d = e.code, e.read()
    except urllib.error.URLError as e:
        raise SystemExit(f"앱({앱})에 연결하지 못했다 — {e.reason} (앱이 켜져 있나 · --app 또는 GEULPU_APP)")
    try:
        return 상태, json.loads(d)
    except ValueError:
        return 상태, None


def 헤더인가(x):
    return isinstance(x, dict) and (str(x.get("format", "")).startswith("geulpu-run-log/") or x.get("geulrun") == 1)


def 줄짓기(x):
    """실행 로그의 한 줄 — 빈칸 없는 JSON(같은 값이면 같은 글)"""
    return json.dumps(x, ensure_ascii=False, separators=(",", ":"))


def 시각():
    return time.strftime("%Y-%m-%d %H:%M:%S")


def 보임(s, 최대=None):
    """토큰의 글을 한 줄로 — 줄바꿈 · 탭 · 제어 글자는 \\n 꼴로"""
    if s is None:
        return "∅"
    t = "".join(c if c >= " " and c != "\x7f" else {"\n": "\\n", "\t": "\\t", "\r": "\\r"}.get(c, f"\\x{ord(c):02x}") for c in s)
    return t if 최대 is None or len(t) <= 최대 else t[:최대] + "…"


class 실행:
    """실행 로그 하나 — 헤더 · 요청 줄(type 없음) · 분석 줄(score · branch), 그리고 있는 곳(파일 또는 앱의 실행 이름)"""

    def __init__(self, 헤더, 줄들, 파일=None, 이름=None):
        self.헤더 = 헤더
        self.요청들 = [x for x in 줄들 if "type" not in x]
        self.점수들 = [x for x in 줄들 if x.get("type") == "score"]
        self.분기들 = [x for x in 줄들 if x.get("type") == "branch"]
        self.파일, self.이름 = 파일, 이름

    def 요청(self, i):
        x = next((x for x in self.요청들 if x.get("i") == i), None)
        if x is None:
            raise SystemExit(f"요청 {i} 가 없다(요청 {len(self.요청들)} 개)")
        return x

    def 보임이름(self):
        return self.파일 or self.이름


def 실행나누기(본, 이름):
    줄 = [json.loads(l) for l in 본.splitlines() if l.strip()]
    if not 줄 or not 헤더인가(줄[0]):
        raise SystemExit(f"GeulPU 실행 로그가 아니다: {이름}")
    return 줄[0], 줄[1:]


def 실행불러오기(앱, run):
    if os.path.exists(run):
        헤더, 줄들 = 실행나누기(open(run, encoding="utf-8").read(), run)
        return 실행(헤더, 줄들, 파일=run)
    본 = 묻기(앱, "GET", "/api/debug/run?name=" + urllib.parse.quote(run), 날것=True).decode("utf-8")   # 앱에 있는 실행 이름이면 받아서
    헤더, 줄들 = 실행나누기(본, run)
    return 실행(헤더, 줄들, 이름=run)


def 덧붙이기(앱, 실, 줄들):
    """분석 줄을 실행 로그 끝에 — 파일이면 그 파일에, 앱의 실행이면 /api/debug/run/append 로(한 번에 덧붙임한도까지 나눠)"""
    if not 줄들:
        return
    글들 = [줄짓기(x) for x in 줄들]
    if 실.파일:
        with open(실.파일, "rb") as f:
            f.seek(0, 2)
            끝줄바꿈 = f.tell() == 0 or (f.seek(-1, 2) is not None and f.read(1) == b"\n")
        with open(실.파일, "ab") as f:
            f.write(("" if 끝줄바꿈 else "\n").encode("utf-8") + "".join(g + "\n" for g in 글들).encode("utf-8"))
        return
    묶음, 크기 = [], 0
    for g in 글들 + [None]:
        if g is None or (묶음 and 크기 + len(g.encode("utf-8")) + 1 > 덧붙임한도):
            묻기(앱, "POST", "/api/debug/run/append", {"name": 실.이름, "lines": "\n".join(묶음) + "\n"})
            묶음, 크기 = [], 0
        if g is not None:
            묶음.append(g)
            크기 += len(g.encode("utf-8")) + 1


def 기록중이면멈춤(앱):
    """앱이 기록 중이면 이 도구가 보내는 요청(재생 · 점수 · 분기)도 그 실행 로그에 들어간다 — 멈춘다"""
    d = 묻기(앱, "GET", "/api/debug")
    if d.get("recording"):
        raise SystemExit(f"앱이 지금 기록 중이다({d.get('name')}) — 이 명령이 보내는 요청도 그 실행 로그에 들어간다. 먼저 record-stop")


def 지금헤더(앱):
    """지금 앱 버전과 서버 — 요청 줄의 "server" 와 같은 꼴(YaRN 배는 엔진의 규칙대로)"""
    st = 묻기(앱, "GET", "/api/state")
    sv = st["server"]
    m = next((x for x in st.get("models", []) if x.get("id") == sv.get("model")), None)
    s = (m or {}).get("shape") or {}
    얀 = 0
    if s.get("yarn_orig", 0) > 0 and sv.get("context", 0) > s.get("context_native", 0):
        얀 = 2 if sv["context"] <= 2 * s["yarn_orig"] else 4
    빌드 = {}
    try:                                            # 서버 실행 파일 · 불러온 모듈의 해시(출처 API — 모듈은 파일을 계산하므로 조금 걸린다)
        p = 묻기(앱, "GET", "/api/provenance")
        빌드 = {"bins": {x.get("file"): x.get("sha256") for x in p.get("binaries") or []},
                "module_sha256": (p.get("module") or {}).get("sha256")}
    except SystemExit:
        pass
    return {"app": (st.get("app") or {}).get("version"), "state": sv.get("state"), "model": sv.get("model"), "build": 빌드,
            "model_info": {"id": (m or {}).get("id"), "file": os.path.basename((m or {}).get("path") or ""), "sha256": (m or {}).get("sha256"),
                           "vocab": s.get("vocab"), "template": (m or {}).get("template")},
            "server": {"sha256": (m or {}).get("sha256"), "layer": sv.get("layer"), "precision": sv.get("precision"),
                       "kv8": bool(sv.get("kv8")), "context": sv.get("context"), "yarn": 얀}}


def 준비된서버(앱):
    지금 = 지금헤더(앱)
    if 지금["state"] != "ready":
        raise SystemExit(f"서버가 준비되지 않았다(state {지금['state']}) — 모델을 불러온 뒤(load) 돌린다")
    return 지금


def 짧게(v):
    return v[:12] if isinstance(v, str) and len(v) > 12 else v


def 빌드다름(헤더, 지금):
    """알림만 — GeulPU 버전 · 서버 실행 파일 · 모듈이 기록한 때와 다른가(같은 백엔드 · 계산 방식일 때만 비교한다)"""
    다름 = []
    if 헤더.get("app") != 지금["app"]:
        다름.append(f"GeulPU 버전 {헤더.get('app')} → {지금['app']}")
    b = 헤더.get("build") or {}
    sv = 헤더.get("server") or {}
    if b and sv.get("layer") == 지금["server"]["layer"]:
        파일 = {"cuda": "bin\\server-cuda.exe", "vulkan": "bin\\server-vulkan.exe"}.get(sv.get("layer"))
        지금서버 = (지금.get("build") or {}).get("bins", {}).get(파일)
        if b.get("server") and 지금서버 and b["server"] != 지금서버:
            다름.append(f"서버 빌드 {짧게(b['server'])} → {짧게(지금서버)}")
        지금모듈 = (지금.get("build") or {}).get("module_sha256")
        if (sv.get("precision") == 지금["server"]["precision"] and bool(sv.get("kv8")) == bool(지금["server"].get("kv8"))
                and b.get("module_sha256") and 지금모듈 and b["module_sha256"] != 지금모듈):
            다름.append(f"모듈 {짧게(b['module_sha256'])} → {짧게(지금모듈)}")
    return 다름


def 기록서버(헤더, x):
    """요청을 계산한 서버 — 요청 줄의 "server"(없으면 헤더에서)"""
    if isinstance(x.get("server"), dict):
        return x["server"]
    sv = 헤더.get("server") or {}
    return {"sha256": (헤더.get("model") or {}).get("sha256"), "layer": sv.get("layer"), "precision": sv.get("precision"),
            "kv8": bool(sv.get("kv8")), "context": sv.get("context"), "yarn": sv.get("yarn")}


def 서버다름(기록, 지금):
    """비트가 같으려면 같아야 하는 것 — 모델 sha256 · 계산 방식 · KV 캐시(8비트 — 없으면 16비트) · YaRN 배, 빠른 모드면 백엔드도
    (CUDA ↔ Vulkan 은 정밀 모드만 같다)"""
    다름 = []
    for k, 이름 in (("sha256", "모델 sha256"), ("precision", "계산 방식"), ("yarn", "YaRN 배")):
        if 기록.get(k) != 지금.get(k):
            다름.append(f"{이름} {짧게(기록.get(k))} → {짧게(지금.get(k))}")
    if bool(기록.get("kv8")) != bool(지금.get("kv8")):          # GeulPUv2 ③b-2
        다름.append(f"KV 캐시 {'8비트' if 기록.get('kv8') else '16비트'} → {'8비트' if 지금.get('kv8') else '16비트'}")
    if 기록.get("precision") == "fast" == 지금.get("precision") and 기록.get("layer") != 지금.get("layer"):
        다름.append(f"백엔드 {기록.get('layer')} → {지금.get('layer')}(빠른 모드)")
    return 다름


def 고른차례(s):
    return None if not s else {int(v) for v in s.split(",") if v.strip()}


def 응답덧(x):
    return ((x.get("response") or {}).get("geulpu")) or {}


def 끝까닭(r):
    return ((r.get("choices") or [{}])[0]).get("finish_reason") if isinstance(r, dict) else None


def 처음다른(a, b):
    """두 목록이 처음 다른 자리 — 같으면 None(한쪽이 짧으면 짧은 쪽의 길이)"""
    for k in range(min(len(a), len(b))):
        if a[k] != b[k]:
            return k
    return None if len(a) == len(b) else min(len(a), len(b))


def 점수가맞나(점수, 서버):
    """점수 줄이 이 서버(모델 sha256 · 계산 방식 · YaRN 배 · 빠른 모드면 백엔드)로 계산한 것인가"""
    return not 서버다름(점수.get("server") or {}, 서버)


def 점수찾기(실, i, 서버):
    return next((s for s in 실.점수들 if s.get("of") == i and 점수가맞나(s, 서버)), None)


# ---- 점수(echo 스코어링) ----
def 점수요청(프롬프트, 토큰, k=점수k):
    """프롬프트 뒤의 토큰을 다시 계산하는 /v1/completions — 생성 없이(max_tokens 0) 스텝마다 로그확률 · 상위 k · 엔트로피 · 로짓 해시"""
    return {"prompt": list(프롬프트) + list(토큰), "echo": True, "max_tokens": 0, "logprobs": k, "temperature": 0, "stream": False,
            "geulpu": {"echo_from": len(프롬프트), "logit_hash": True, "special": True}}


def 점수풀기(r, P, n):
    """echo 응답에서 프롬프트 뒤 n 토큰의 스텝마다 값 — 점수 줄의 칸. text 는 답의 글(토큰을 바이트로 이은 것), text_offset 은 토큰마다
    그 글 안의 바이트 자리 — 한 글자가 토큰 둘에 걸쳐도 화면이 글자 단위로 묶는다(token_text 는 토큰 하나의 글이라 '�' 일 수 있다)"""
    c = r["choices"][0]
    lp = c.get("logprobs") or {}
    g = r.get("geulpu") or {}
    본 = (c.get("text") or "").encode("utf-8")
    오프 = lp.get("text_offset") or []
    a0 = 오프[P] if len(오프) > P else len(본)
    return {"logprobs": lp["token_logprobs"][P:P + n], "entropy": g["prompt_entropy"][P:P + n], "top_ids": g["prompt_top_ids"][P:P + n],
            "top_logprobs": g["prompt_top_logprobs"][P:P + n], "token_text": lp["tokens"][P:P + n],
            "text": 본[a0:].decode("utf-8", "replace"), "text_offset": [o - a0 for o in 오프[P:P + n]],
            "logit_hashes": (g.get("prompt_logit_hashes") or [])[:n]}


def 점수계산(앱, 프롬프트, 토큰):
    P, n = len(프롬프트), len(토큰)
    return 점수풀기(묻기(앱, "POST", "/v1/completions", 점수요청(프롬프트, 토큰)), P, n)


def 해시검사(기록, 지금서버, 생성해시, 점수해시):
    """점수의 로짓 해시가 생성 때와 같은가 — 같은 서버일 때만 뜻이 있다"""
    if 서버다름(기록, 지금서버):
        return "other_model"
    if 생성해시 is None:
        return "no_hashes"
    return "same" if 생성해시 == 점수해시 else "different"


def 점수줄짓기(x, 헤더, 지금, s):
    g = 응답덧(x)
    return dict({"type": "score", "of": x["i"], "at": 시각(), "model": {k: 지금["model_info"][k] for k in ("id", "file", "sha256")},
                 "server": 지금["server"], "k": 점수k}, **s,
                hash_check=해시검사(기록서버(헤더, x), 지금["server"], g.get("logit_hashes"), s["logit_hashes"]))


def 점수매길요청(실, 고른):
    """점수를 매길 수 있는 요청 — 성공했고 프롬프트 토큰 · 나온 토큰이 있는 것"""
    for x in 실.요청들:
        if 고른 is not None and x.get("i") not in 고른:
            continue
        g = 응답덧(x)
        if x.get("status") == 200 and g.get("prompt_tokens") and g.get("tokens"):
            yield x


def 점수(a):
    실 = 실행불러오기(a.app, a.run)
    기록중이면멈춤(a.app)
    지금 = 준비된서버(a.app)
    고른 = 고른차례(a.requests)
    새줄, 셈 = [], {"same": 0, "different": 0, "other_model": 0, "no_hashes": 0}
    확인같음 = 확인다름 = 건너뜀 = 0
    print(f"점수: {실.보임이름()} · 지금 모델 {지금['model']} · {지금['server']['precision']} · {지금['server']['layer']}", flush=True)
    for x in 점수매길요청(실, 고른):
        g = 응답덧(x)
        있던 = 점수찾기(실, x["i"], 지금["server"])
        if 있던 and not a.check:
            건너뜀 += 1
            continue
        if a.check and not 있던:
            print(f"{x['i']:4d} 비교할 점수가 없다(이 서버의 점수 줄 없음)", flush=True)
            continue
        t0 = time.time()
        s = 점수계산(a.app, g["prompt_tokens"], g["tokens"])
        줄 = 점수줄짓기(x, 실.헤더, 지금, s)
        if a.check:
            틀린 = [k for k in ("logprobs", "entropy", "top_ids", "top_logprobs", "token_text", "logit_hashes") if 있던.get(k) != 줄[k]]
            확인같음 += not 틀린
            확인다름 += bool(틀린)
            print(f"{x['i']:4d} 스텝 {len(g['tokens'])} · {'같음' if not 틀린 else '다름 — ' + ', '.join(틀린)} · {time.time() - t0:.2f} 초", flush=True)
            continue
        셈[줄["hash_check"]] += 1
        새줄.append(줄)
        print(f"{x['i']:4d} 스텝 {len(g['tokens'])} · 해시 {줄['hash_check']} · {time.time() - t0:.2f} 초", flush=True)
    if a.check:
        print(f"점수 다시 계산: 같음 {확인같음} · 다름 {확인다름}")
        sys.exit(0 if 확인다름 == 0 else 1)
    덧붙이기(a.app, 실, 새줄)
    print(f"점수: 덧붙임 {len(새줄)}" + (f" · 이미 있음 {건너뜀}" if 건너뜀 else "") + " — 해시 " +
          " · ".join(f"{k} {v}" for k, v in 셈.items() if v))
    sys.exit(0 if 셈["different"] == 0 else 1)


# ---- 중단점 ----
def 중단점찾기(토큰, s, 마진기준, 확률기준, 도구호출=False):
    """점수 줄 s(그 요청을 생성한 모델의 것)에서 스텝마다의 중단점 — 종류: low_margin(1위와 2위의 로그확률 차 < 마진기준),
    low_prob(고른 토큰의 확률 < 확률기준), tool_call(도구 호출의 시작). 도구호출 = 응답에 tool_calls 가 있는가(표시 토큰이 없는 Llama 3
    의 JSON 꼴 호출은 스텝 0)"""
    결과 = []
    for j, t in enumerate(토큰):
        lp = s["logprobs"][j]
        ids = s["top_ids"][j] or []
        tl = s["top_logprobs"][j] or []
        글 = s["token_text"][j]
        마진 = tl[0] - tl[1] if len(tl) >= 2 else None
        확률 = math.exp(lp) if lp is not None else None
        종류 = []
        if 마진 is not None and 마진 < 마진기준:
            종류.append("low_margin")
        if 확률 is not None and 확률 < 확률기준:
            종류.append("low_prob")
        if 글 is not None and any(m in 글 for m in 도구호출표시):
            종류.append("tool_call")
        if 종류:
            결과.append({"step": j, "kinds": 종류, "token": t, "text": 글, "prob": 확률, "margin": 마진,
                         "top_ids": ids, "top_probs": [math.exp(v) for v in tl]})
    if 도구호출 and 토큰 and not any("tool_call" in b["kinds"] for b in 결과):
        첫 = next((b for b in 결과 if b["step"] == 0), None)
        if 첫:
            첫["kinds"].append("tool_call")
        else:
            lp, tl = s["logprobs"][0], s["top_logprobs"][0] or []
            결과.insert(0, {"step": 0, "kinds": ["tool_call"], "token": 토큰[0], "text": s["token_text"][0],
                            "prob": math.exp(lp) if lp is not None else None, "margin": tl[0] - tl[1] if len(tl) >= 2 else None,
                            "top_ids": s["top_ids"][0] or [], "top_probs": [math.exp(v) for v in tl]})
    return 결과


def 도구호출있나(x):
    r = x.get("response") or {}
    m = ((r.get("choices") or [{}])[0]).get("message") or {}
    return bool(m.get("tool_calls")) or 끝까닭(r) == "tool_calls"


class 글찾기:
    """토큰 번호 → 글(/detokenize — 한 번 물은 것은 둔다)"""

    def __init__(self, 앱):
        self.앱, self.표 = 앱, {}

    def __call__(self, t):
        if t not in self.표:
            self.표[t] = 묻기(self.앱, "POST", "/detokenize", {"tokens": [t]}).get("prompt")
        return self.표[t]


def 중단점(a):
    실 = 실행불러오기(a.app, a.run)
    고른 = 고른차례(a.requests)
    요청들 = list(점수매길요청(실, 고른))
    없는 = [x for x in 요청들 if not 점수찾기(실, x["i"], 기록서버(실.헤더, x))]
    지금 = None
    if 없는:                                         # 생성한 모델의 점수가 없으면 — 지금 서버가 그 모델이면 매겨서 덧붙인다
        기록중이면멈춤(a.app)
        지금 = 준비된서버(a.app)
        다른 = [x for x in 없는 if 서버다름(기록서버(실.헤더, x), 지금["server"])]
        if 다른:
            raise SystemExit(f"요청 {len(다른)} 개는 그 요청을 생성한 모델의 점수가 없고 지금 서버도 다르다 — "
                             f"load run:{실.보임이름()} 뒤 score {실.보임이름()}")
        print(f"점수 매기는 중: 요청 {len(없는)}", flush=True)
        새줄 = []
        for x in 없는:
            g = 응답덧(x)
            새줄.append(점수줄짓기(x, 실.헤더, 지금, 점수계산(a.app, g["prompt_tokens"], g["tokens"])))
        덧붙이기(a.app, 실, 새줄)
        실.점수들 += 새줄
    찾기 = None                                     # 대안 토큰의 글 — 지금 서버가 같은 모델(같은 어휘)일 때만
    try:
        지금2 = 지금 or 지금헤더(a.app)
        if 요청들 and 지금2["state"] == "ready" and 지금2["server"]["sha256"] == 기록서버(실.헤더, 요청들[0]).get("sha256"):
            찾기 = 글찾기(a.app)
    except (SystemExit, OSError):
        pass
    보고, 셈 = [], {"low_margin": 0, "low_prob": 0, "tool_call": 0}
    해시틀림 = []
    for x in 요청들:
        s = 점수찾기(실, x["i"], 기록서버(실.헤더, x))
        if s.get("hash_check") == "different":
            해시틀림.append(x["i"])
        bs = 중단점찾기(응답덧(x)["tokens"], s, a.margin, a.prob, 도구호출있나(x))
        for b in bs:
            for k in b["kinds"]:
                셈[k] += 1
        보고.append({"i": x["i"], "steps": len(응답덧(x)["tokens"]), "breakpoints": bs})
        if bs:
            보인 = []
            for b in bs[:a.show]:
                대안 = ""
                if b["top_ids"]:
                    다른번호 = next((t for t in b["top_ids"] if t != b["token"]), None)
                    if 다른번호 is not None and 찾기:
                        p2 = b["top_probs"][b["top_ids"].index(다른번호)]
                        대안 = f" ↔ '{보임(찾기(다른번호), 12)}' {p2:.2f}"
                값 = f"{b['prob']:.2f}" if b["prob"] is not None else "?"
                마 = f" · 마진 {b['margin']:.3f}" if b["margin"] is not None else ""
                보인.append(f"{b['step']}[{','.join(b['kinds'])}] '{보임(b['text'], 12)}' {값}{대안}{마}")
            print(f"{x['i']:4d} 스텝 {len(응답덧(x)['tokens'])} · 중단점 {len(bs)}: " + " | ".join(보인) + (" …" if len(bs) > a.show else ""),
                  flush=True)
    print(f"중단점: 요청 {len(보고)} · 마진이 작은 스텝 {셈['low_margin']} · 저확률 샘플 {셈['low_prob']} · 도구 호출 시작 {셈['tool_call']}"
          f" (마진 < {a.margin} nat · 확률 < {a.prob})")
    if 해시틀림:
        print(f"주의: 요청 {해시틀림} 의 점수 로짓 해시가 생성 때와 다르다 — 점수가 생성과 다른 계산이다")
    if a.out:
        json.dump({"run": 실.보임이름(), "margin": a.margin, "prob": a.prob, "counts": 셈, "requests": 보고},
                  open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)


# ---- 분기 ----
def 분기요청(x, k, alt):
    """토큰 분기의 요청 — 원래 요청 x 의 스텝 k 에서 alt 를 골라 잇는다: 프롬프트 토큰 + 앞 k 토큰 + alt 를 /v1/completions 에, 같은
    시드 · 같은 샘플링 설정(원래 요청에 있던 것만 — 없으면 서버의 기본값이 같다), step_offset k + 1. echo(echo_from = 프롬프트 길이)로 앞
    k + 1 스텝의 로짓 해시를 함께 받는다. max_tokens 가 있었으면 남은 몫(k + 1 을 뺀 것)"""
    g = 응답덧(x)
    원 = x.get("request") or {}
    프롬프트, 토큰 = g["prompt_tokens"], g["tokens"]
    if not 0 <= k < len(토큰):
        raise SystemExit(f"스텝 {k} 가 답 밖이다(스텝 {len(토큰)})")
    몸 = {"prompt": list(프롬프트) + list(토큰[:k]) + [alt], "echo": True, "logprobs": 점수k, "stream": False, "seed": g["seed"],
          "geulpu": {"echo_from": len(프롬프트), "logit_hash": True, "prompt_tokens": True, "step_offset": k + 1, "special": True}}
    for 열쇠 in ("temperature", "top_p", "top_k", "min_p", "stop", "ignore_eos"):
        if 열쇠 in 원:
            몸[열쇠] = 원[열쇠]
    최대 = 원.get("max_tokens", 원.get("max_completion_tokens"))
    if 최대 is not None:                              # 없으면 원래처럼 서버의 기본값(문맥 끝까지)
        몸["max_tokens"] = max(0, 최대 - (k + 1))
    return 몸


def 경로바꾸기(값, 경로, 새값):
    """'messages.2.content' 꼴의 경로(점으로 나눔 — 배열은 번호, 음수는 뒤에서)에 새값을 넣은 사본"""
    루트 = copy.deepcopy(값)
    조각 = 경로.split(".")
    지금 = 루트
    for n, p in enumerate(조각):
        끝 = n == len(조각) - 1
        if isinstance(지금, list):
            if not p.lstrip("-").isdigit():
                raise ValueError(f"{경로}: '{p}' 는 배열의 번호여야 한다")
            q = int(p)
            if not -len(지금) <= q < len(지금):
                raise ValueError(f"{경로}: 번호 {q} 가 배열 밖이다(길이 {len(지금)})")
            if 끝:
                지금[q] = 새값
            else:
                지금 = 지금[q]
        elif isinstance(지금, dict):
            if 끝:
                지금[p] = 새값
            elif p not in 지금:
                raise ValueError(f"{경로}: '{p}' 가 없다")
            else:
                지금 = 지금[p]
        else:
            raise ValueError(f"{경로}: '{p}' 앞이 객체나 배열이 아니다")
    return 루트


def 바꿀값(s, 파일=False):
    """--set PATH=VALUE(값은 JSON 이면 JSON, 아니면 글) · --set-file PATH=FILE(파일의 글) — (경로, 값)"""
    if "=" not in s:
        raise SystemExit(f"'{s}' — PATH=VALUE 꼴이어야 한다")
    경로, v = s.split("=", 1)
    if 파일:
        return 경로, open(v, encoding="utf-8").read()
    try:
        return 경로, json.loads(v)
    except ValueError:
        return 경로, v


def 요청분기요청(x, 바꿈들):
    """요청 분기의 요청 — 원래 요청을 고치고(경로마다) 같은 시드로 · 스트리밍 끔 · 로짓 해시 · 프롬프트 토큰"""
    g = 응답덧(x)
    몸 = copy.deepcopy(x["request"])
    몸["stream"] = False
    몸.pop("stream_options", None)
    if "seed" in g:
        몸["seed"] = g["seed"]
    for 경로, v in 바꿈들:
        몸 = 경로바꾸기(몸, 경로, v)
    gp = 몸.get("geulpu") if isinstance(몸.get("geulpu"), dict) else {}
    gp.update({"logit_hash": True, "prompt_tokens": True})
    몸["geulpu"] = gp
    return 몸


def 다른토큰(x, s, k, alt, 순위):
    """분기할 토큰 — --token 번호 · same(원래 토큰) · --alt N(상위 N 위) · 기본(원래와 다른 가장 높은 것)"""
    원토큰 = 응답덧(x)["tokens"][k]
    if alt == "same":
        return 원토큰
    if alt is not None:
        return int(alt)
    ids = s["top_ids"][k] or []
    if 순위 is not None:
        if not 1 <= 순위 <= len(ids):
            raise SystemExit(f"--alt {순위} — 점수 줄에는 상위 {len(ids)} 까지만 있다")
        return ids[순위 - 1]
    t = next((t for t in ids if t != 원토큰), None)
    if t is None:
        raise SystemExit(f"스텝 {k} 의 상위에 원래와 다른 토큰이 없다 — --token 으로 정한다")
    return t


def 분기(a):
    실 = 실행불러오기(a.app, a.run)
    기록중이면멈춤(a.app)
    지금 = 준비된서버(a.app)
    x = 실.요청(a.request)
    g = 응답덧(x)
    if x.get("status") != 200 or not g.get("tokens") or not g.get("prompt_tokens") or "seed" not in g:
        raise SystemExit(f"요청 {a.request} 는 분기할 수 없다 — 성공한 생성이고 응답의 geulpu 에 시드 · prompt_tokens · tokens 가 있어야 한다")
    다름 = 서버다름(기록서버(실.헤더, x), 지금["server"])
    if 다름 and not a.ignore_header:
        raise SystemExit("그 요청을 생성한 서버와 지금 서버가 다르다 — " + " · ".join(다름) + " (load run:… 또는 --ignore-header)")
    바꿈들 = [바꿀값(s) for s in a.set or []] + [바꿀값(s, 파일=True) for s in a.set_file or []]
    if (a.step is None) == (not 바꿈들):
        raise SystemExit("--step(토큰 분기) 또는 --set · --set-file(요청 분기) 가운데 하나를 준다")
    토큰, 해시 = g["tokens"], g.get("logit_hashes")
    줄 = {"type": "branch", "b": len(실.분기들), "of": x["i"], "at": None, "model": {k: 지금["model_info"][k] for k in ("id", "file", "sha256")},
          "server": 지금["server"]}
    if a.step is not None:
        k = a.step
        s = 점수찾기(실, x["i"], 기록서버(실.헤더, x))
        if s is None and a.token is None:              # 대안을 고르려면 그 스텝의 상위가 든다 — 앞 k + 1 토큰만 점수
            s = 점수계산(a.app, g["prompt_tokens"], 토큰[:k + 1])
        alt = 다른토큰(x, s, k, a.token, a.alt)
        몸 = 분기요청(x, k, alt)
        길 = "/v1/completions"
        줄.update({"kind": "token", "step": k, "token": alt, "original_token": 토큰[k]})
    else:
        try:
            몸 = 요청분기요청(x, 바꿈들)
        except ValueError as e:
            raise SystemExit(str(e))
        길 = x["path"]
        줄.update({"kind": "request", "set": [{"path": p, "value": v} for p, v in 바꿈들]})
    응답들, 걸림 = [], []
    for _ in range(max(1, a.repeat)):
        t0 = time.time()
        상태, r = 묻기상태(a.app, 길, 몸)
        걸림.append(time.time() - t0)
        if 상태 != 200:
            raise SystemExit(f"분기 요청이 실패했다: HTTP {상태} — {json.dumps(r, ensure_ascii=False)[:300]}")
        응답들.append(r)
    r = 응답들[0]
    g2 = r.get("geulpu") or {}
    같음 = all((q.get("geulpu") or {}).get(f) == g2.get(f) for q in 응답들[1:] for f in ("tokens", "logit_hashes", "prompt_logit_hashes", "prompt_tokens"))
    검사 = {"repeat": len(응답들), "repeat_same": 같음}
    print(f"분기 b{줄['b']} — 실행 {실.보임이름()} · 요청 {x['i']}", flush=True)
    if 줄["kind"] == "token":
        k, alt = 줄["step"], 줄["token"]
        P = len(g["prompt_tokens"])
        앞해시 = g2.get("prompt_logit_hashes")
        검사["prefix_same"] = None if 해시 is None else 앞해시 == 해시[:k + 1]
        새토큰 = g2.get("tokens") or []
        lpt = (((r.get("choices") or [{}])[0]).get("logprobs") or {}).get("tokens") or []
        alt글 = lpt[P + k] if len(lpt) > P + k else None
        원글 = 묻기(a.app, "POST", "/detokenize", {"tokens": [토큰[k]]}).get("prompt")
        print(f"  스텝 {k}: 원래 '{보임(원글, 16)}' → '{보임(alt글, 16)}'" + (" (원래 토큰 — 자기 확인)" if alt == 토큰[k] else ""))
        if alt == 토큰[k]:                            # 원래 토큰이면 원래 답의 뒤가 그대로 나와야 한다
            n = len(토큰) - k - 1
            검사["null_branch_same"] = 새토큰[:n] == 토큰[k + 1:] and (해시 is None or (g2.get("logit_hashes") or [])[:n] == 해시[k + 1:])
            if len(새토큰) > n:
                검사["null_branch_longer"] = len(새토큰) - n
        print(f"  앞 {k + 1} 스텝의 로짓 해시: " + {True: "원래와 같음", False: "원래와 다름", None: "원래에 해시 없음"}[검사["prefix_same"]])
        원뒤 = 묻기(a.app, "POST", "/detokenize", {"tokens": 토큰[k + 1:]}).get("prompt") if k + 1 < len(토큰) else ""
        새뒤 = 묻기(a.app, "POST", "/detokenize", {"tokens": 새토큰}).get("prompt") if 새토큰 else ""   # 바이트로 이어 붙인 글(한 글자가 토큰 둘에 걸쳐도)
        print(f"  원래 뒤 {len(토큰) - k - 1} 토큰({끝까닭(x.get('response'))}): {보임(원뒤, 160)}")
        print(f"  분기 뒤 {len(새토큰)} 토큰({끝까닭(r)}): {보임(새뒤, 160)}")
        if "null_branch_same" in 검사:
            print("  원래 토큰으로 이은 답: " + ("원래와 같음" if 검사["null_branch_same"] else "원래와 다름")
                  + (f"(원래보다 {검사['null_branch_longer']} 토큰 김 — /v1/chat/completions 의 끝 규칙)" if 검사.get("null_branch_longer") else ""))
    else:
        p1, p2 = g["prompt_tokens"], g2.get("prompt_tokens") or []
        검사["prompt_diff_at"] = 처음다른(p1, p2)
        검사["first_divergence"] = 처음다른(토큰, g2.get("tokens") or [])
        print("  바꾼 것: " + " · ".join(f"{p}={보임(json.dumps(v, ensure_ascii=False), 40)}" for p, v in 바꿈들))
        print(f"  프롬프트 토큰 {len(p1)} → {len(p2)} · 처음 달라진 자리 {검사['prompt_diff_at']}")
        print(f"  답 {len(토큰)} → {len(g2.get('tokens') or [])} 토큰 · 처음 갈라진 스텝 {검사['first_divergence']} · 끝 {끝까닭(x.get('response'))} → {끝까닭(r)}")
        if 검사["first_divergence"] is not None:      # 갈라진 스텝부터의 글(토큰을 바이트로 이어 — 도구 호출의 표시도 그대로)
            k = 검사["first_divergence"]
            원뒤 = 묻기(a.app, "POST", "/detokenize", {"tokens": 토큰[k:]}).get("prompt") if k < len(토큰) else ""
            새토큰 = g2.get("tokens") or []
            새뒤 = 묻기(a.app, "POST", "/detokenize", {"tokens": 새토큰[k:]}).get("prompt") if k < len(새토큰) else ""
            print(f"  원래 답(스텝 {k}부터): {보임(원뒤, 160)}")
            print(f"  분기 답(스텝 {k}부터): {보임(새뒤, 160)}")
    print(f"  {검사['repeat']} 번 돌려 " + ("같은 비트" if 같음 else "다름") + f" · {sum(걸림) / len(걸림):.2f} 초")
    줄.update({"at": 시각(), "ms": round(걸림[0] * 1000), "path": 길, "request": 몸, "response": r, "checks": 검사})
    if not a.no_save:
        덧붙이기(a.app, 실, [줄])
        print(f"  실행 로그에 덧붙임(b{줄['b']})")
    ok = 같음 and 검사.get("prefix_same") is not False and 검사.get("null_branch_same") is not False
    sys.exit(0 if ok else 1)


# ---- 비교 ----
def 서버글(sv):
    kv = " · KV 8비트" if (sv or {}).get("kv8") else ""
    return f"{짧게((sv or {}).get('sha256'))} · {(sv or {}).get('precision')}{kv} · {(sv or {}).get('layer')}"


def 요청같나(x, y):
    a, b = dict(x.get("request") or {}), dict(y.get("request") or {})
    for d in (a, b):
        d.pop("model", None)
    return a == b


def 비교계산(A, B):
    """실행 A · B 의 요청 차례(i)마다 같은 요청 · 같은 프롬프트 · 같은 답 · 처음 갈라진 스텝, A 의 답을 A 와 B 의 서버로 점수 매긴
    로그확률(스텝마다 B − A). 점수는 A 의 점수 줄에서 — A 의 요청을 생성한 서버의 것과 B 의 요청을 생성한 서버의 것.
    같은 입력이면 같은 보고(차례대로 더한다)"""
    B요청 = {y.get("i"): y for y in B.요청들}
    행들 = []
    for x in sorted(A.요청들, key=lambda v: v.get("i")):
        i = x.get("i")
        y = B요청.get(i)
        gA = 응답덧(x)
        if y is None or x.get("status") != 200 or y.get("status") != 200 or gA.get("tokens") is None or 응답덧(y).get("tokens") is None:
            행들.append({"i": i, "compared": False, "reason": "B 에 없음" if y is None else "오류 응답이거나 토큰 없음"})
            continue
        gB = 응답덧(y)
        tA, tB = gA["tokens"], gB["tokens"]
        행 = {"i": i, "compared": True, "request_same": 요청같나(x, y), "prompt_same": gA.get("prompt_tokens") == gB.get("prompt_tokens"),
              "same": tA == tB, "first_divergence": 처음다른(tA, tB), "steps_a": len(tA), "steps_b": len(tB),
              "finish_a": 끝까닭(x.get("response")), "finish_b": 끝까닭(y.get("response"))}
        sAA = 점수찾기(A, i, 기록서버(A.헤더, x))
        sAB = 점수찾기(A, i, 기록서버(B.헤더, y))
        if sAA and sAB and len(sAA["logprobs"]) == len(tA) == len(sAB["logprobs"]):
            차 = [b - a for a, b in zip(sAA["logprobs"], sAB["logprobs"])]
            합 = 0.0
            for d in 차:
                합 += d
            합A = 0.0
            for v in sAA["logprobs"]:
                합A += v
            합B = 0.0
            for v in sAB["logprobs"]:
                합B += v
            나쁜 = sorted(range(len(차)), key=lambda j: (차[j], j))[:3]
            행.update({"scored": True, "logprob_a": 합A, "logprob_b": 합B, "delta_sum": 합, "delta_mean": 합 / len(차) if 차 else 0.0,
                       "b_top1_differs": sum(1 for j, t in enumerate(tA) if (sAB["top_ids"][j] or [None])[0] != t),
                       "a_top1_differs": sum(1 for j, t in enumerate(tA) if (sAA["top_ids"][j] or [None])[0] != t),
                       "worst": [{"step": j, "delta": 차[j], "text": sAA["token_text"][j]} for j in 나쁜]})
        else:
            행.update({"scored": False, "missing": [n for n, s in (("a_by_a", sAA), ("a_by_b", sAB)) if not s]})
        행들.append(행)
    본 = [r for r in 행들 if r["compared"]]
    다름 = [r for r in 본 if not r["same"]]
    갈림 = sorted(r["first_divergence"] for r in 다름 if r["first_divergence"] is not None)
    통 = {"0": 0, "1-4": 0, "5-16": 0, "17-64": 0, "65+": 0}
    for k in 갈림:
        통["0" if k == 0 else "1-4" if k <= 4 else "5-16" if k <= 16 else "17-64" if k <= 64 else "65+"] += 1
    점 = [r for r in 본 if r.get("scored")]
    합, 토큰수, b1, a1, 합A, 합B = 0.0, 0, 0, 0, 0.0, 0.0
    for r in 점:
        합 += r["delta_sum"]
        합A += r["logprob_a"]
        합B += r["logprob_b"]
        토큰수 += r["steps_a"]
        b1 += r["b_top1_differs"]
        a1 += r["a_top1_differs"]
    요약 = {"requests": len(본), "not_compared": len(행들) - len(본), "same": len(본) - len(다름), "different": len(다름),
            "request_different": sum(1 for r in 본 if not r["request_same"]), "prompt_different": sum(1 for r in 본 if not r["prompt_same"]),
            "first_divergence": {"min": 갈림[0] if 갈림 else None, "median": 갈림[(len(갈림) - 1) // 2] if 갈림 else None,
                                 "max": 갈림[-1] if 갈림 else None, "histogram": 통},
            "scored": len(점), "tokens": 토큰수, "logprob_a": 합A, "logprob_b": 합B, "delta_sum": 합,
            "delta_mean_per_token": 합 / 토큰수 if 토큰수 else None, "b_top1_differs": b1, "a_top1_differs": a1}
    return {"a": {"run": A.보임이름(), "model": (A.헤더.get("model") or {}).get("id"), "server": 기록서버(A.헤더, A.요청들[0]) if A.요청들 else None},
            "b": {"run": B.보임이름(), "model": (B.헤더.get("model") or {}).get("id"), "server": 기록서버(B.헤더, B.요청들[0]) if B.요청들 else None},
            "summary": 요약, "requests": 행들}


def 비교(a):
    A, B = 실행불러오기(a.app, a.run_a), 실행불러오기(a.app, a.run_b)
    보고 = 비교계산(A, B)
    print(f"비교: A {보고['a']['run']} ({보고['a']['model']} · {서버글(보고['a']['server'])}) ↔ B {보고['b']['run']} ({보고['b']['model']} · "
          f"{서버글(보고['b']['server'])})")
    print("   i  답    처음 갈라진 스텝  스텝 A/B   로그확률 차(B − A) 합 · 평균   B 의 1위 ≠ A 토큰")
    for r in 보고["requests"]:
        if not r["compared"]:
            print(f"{r['i']:4d}  비교 안 함 — {r['reason']}")
            continue
        점수글 = (f"{r['delta_sum']:9.3f} · {r['delta_mean']:7.4f}        {r['b_top1_differs']}" if r.get("scored")
                 else "점수 없음(" + " · ".join(r["missing"]) + ")")
        print(f"{r['i']:4d}  {'같음' if r['same'] else '다름'}  {str(r['first_divergence'] if r['first_divergence'] is not None else '-'):>8}"
              f"         {r['steps_a']:3d}/{r['steps_b']:<3d}   {점수글}" + ("" if r["prompt_same"] else "  (프롬프트 다름)"))
    s = 보고["summary"]
    fd = s["first_divergence"]
    print(f"요약: 요청 {s['requests']} · 같은 답 {s['same']} · 다른 답 {s['different']}"
          + (f" · 처음 갈라진 스텝 가운데값 {fd['median']}(최소 {fd['min']} · 최대 {fd['max']})" if fd["median"] is not None else "")
          + (f" · 프롬프트 다름 {s['prompt_different']}" if s["prompt_different"] else ""))
    if s["scored"]:
        print(f"      점수 {s['scored']} 요청 · 토큰 {s['tokens']:,} — 로그확률 A {s['logprob_a']:.3f} · B {s['logprob_b']:.3f} · 차 합 "
              f"{s['delta_sum']:.3f}(토큰당 {s['delta_mean_per_token']:.5f}) · B 의 1위가 A 토큰과 다른 스텝 {s['b_top1_differs']:,}")
    없음 = {m for r in 보고["requests"] if r.get("compared") and not r.get("scored") for m in r.get("missing", [])}
    if 없음:
        print("점수가 없는 요청이 있다 — " + " · ".join(
            {"a_by_a": f"A 의 모델로: load run:{A.보임이름()} 뒤 score {A.보임이름()}",
             "a_by_b": f"B 의 모델로: load run:{B.보임이름()} 뒤 score {A.보임이름()}"}[m] for m in sorted(없음)))
    if a.out:
        open(a.out, "w", encoding="utf-8", newline="\n").write(json.dumps(보고, ensure_ascii=False, indent=1) + "\n")


# ---- 기록 · 테스트 세트 · 모델 ----
def 상태(a):
    d = 묻기(a.app, "GET", "/api/debug")
    print(json.dumps(d, ensure_ascii=False, indent=1))


def 기록시작(a):
    d = 묻기(a.app, "POST", "/api/debug/record/start", {"name": a.name or ""})
    print(f"기록 시작: {d['name']} — {d['dir']}")


def 기록멈춤(a):
    d = 묻기(a.app, "POST", "/api/debug/record/stop", {})
    print(f"기록 멈춤: {d.get('name')} · 요청 {d.get('count')}")


def 받기(a):
    d = 묻기(a.app, "GET", "/api/debug/run?name=" + urllib.parse.quote(a.name), 날것=True)
    open(a.file, "wb").write(d)
    print(f"받음: {a.file} ({len(d):,} 바이트)")


def 지우기(a):
    묻기(a.app, "POST", "/api/debug/run/delete", {"name": a.name})
    print(f"지움: {a.name}")


def 불러오기(a):
    """앱의 모델을 불러온다 — 모델 아이디 · sha256 앞 8 자 이상 · run:실행(그 실행의 첫 요청을 생성한 모델). 설정(계산 방식 · KV 캐시 ·
    문맥 · 슬롯)은 앱의 설정 그대로(--precision · --kv8 · --layer 를 주면 바꾼다)"""
    st = 묻기(a.app, "GET", "/api/state")
    모델들 = st.get("models") or []
    대상 = a.model
    if 대상.startswith("run:"):
        실 = 실행불러오기(a.app, 대상[4:])
        sha = 기록서버(실.헤더, 실.요청들[0]).get("sha256") if 실.요청들 else (실.헤더.get("model") or {}).get("sha256")
        후보 = [m for m in 모델들 if sha and m.get("sha256") == sha]
    else:
        후보 = [m for m in 모델들 if m.get("id") == 대상]
        if not 후보 and len(대상) >= 8:
            후보 = [m for m in 모델들 if isinstance(m.get("sha256"), str) and m["sha256"].startswith(대상.lower())]
    if len(후보) != 1:
        raise SystemExit(f"'{대상}' 에 맞는 모델이 {len(후보)} 개다 — 앱의 모델: " + " · ".join(m.get("id", "?") for m in 모델들))
    m = 후보[0]
    바꿈 = {k: v for k, v in (("precision", a.precision), ("gpu_layer", a.layer)) if v}
    if a.kv8:                                        # GeulPUv2 ③b-2 — KV 캐시 8비트(빠름 · CUDA · k-양자화 모델에서만 쓴다)
        바꿈["kv8_cache"] = a.kv8 == "on"
    if 바꿈:                                         # 앱의 설정을 바꾼다(다음에 불러올 때 적용 — 아래에서 다시 불러온다)
        묻기(a.app, "POST", "/api/settings", 바꿈)
    if a.layer and (st.get("device") or {}).get("layer") != a.layer:   # 앱이 쓰는 백엔드는 하드웨어 확인이 정한다 — 다시 확인
        일 = 묻기(a.app, "POST", "/api/device/check", {}) or {}
        t1 = time.time()
        while True:
            time.sleep(1)
            st = 묻기(a.app, "GET", "/api/state")
            j = st.get("job") or {}
            if j.get("id") == 일.get("job") and j.get("state") != "running":
                break
            if time.time() - t1 > a.timeout:
                raise SystemExit(f"{a.timeout} 초 안에 하드웨어 확인이 끝나지 않았다")
        if (st.get("device") or {}).get("layer") != a.layer:
            raise SystemExit(f"하드웨어 확인 뒤에도 백엔드가 {(st.get('device') or {}).get('layer')} 다 — {(st.get('device') or {}).get('reason') or ''}")
    sv = st["server"]

    def 맞나(sv):
        return (sv.get("state") == "ready" and sv.get("model") == m["id"] and (not a.precision or sv.get("precision") == a.precision)
                and (not a.layer or sv.get("layer") == a.layer) and (not a.kv8 or bool(sv.get("kv8")) == (a.kv8 == "on")))
    if 맞나(sv):
        print(f"이미 불러옴: {m['id']} · {sv.get('precision')} · {sv.get('layer')} · 문맥 {sv.get('context')}")
        return
    t0 = time.time()
    if sv.get("state") == "ready" and sv.get("model") == m["id"]:   # 같은 모델에 설정만 다르면 멈췄다가(앱은 같은 모델을 다시 불러오지 않는다)
        묻기(a.app, "POST", "/api/server/stop", {})
        while 묻기(a.app, "GET", "/api/state")["server"].get("state") not in ("off", "failed"):
            if time.time() - t0 > a.timeout:
                raise SystemExit(f"{a.timeout} 초 안에 서버가 멈추지 않았다")
            time.sleep(0.5)
    일 = 묻기(a.app, "POST", "/api/server/start", {"model": m["id"]}) or {}
    while True:
        time.sleep(1)
        st = 묻기(a.app, "GET", "/api/state")
        sv, j = st["server"], st.get("job") or {}
        if 맞나(sv):
            break
        if sv.get("state") == "failed" or (j.get("kind") == "server" and j.get("state") == "failed"):
            raise SystemExit(f"불러오지 못했다: {j.get('reason') or ''} {(sv.get('reason') or '')[-300:]}")
        if j.get("kind") == "server" and j.get("id") == 일.get("job") and j.get("state") == "done" and sv.get("state") == "ready":
            raise SystemExit(f"불러왔지만 바란 설정과 다르다: {sv.get('model')} · {sv.get('precision')} · {sv.get('layer')}")
        if time.time() - t0 > a.timeout:
            raise SystemExit(f"{a.timeout} 초 안에 준비되지 않았다(state {sv.get('state')})")
    print(f"불러옴: {m['id']} · {sv.get('precision')} · {sv.get('layer')} · 문맥 {sv.get('context')} · {time.time() - t0:.0f} 초")


def 테스트세트(a):
    세트 = [json.loads(l) for l in open(a.testset, encoding="utf-8") if l.strip()]
    d = 묻기(a.app, "POST", "/api/debug/record/start", {"name": a.name})
    print(f"기록 시작: {d['name']} · 요청 {len(세트)}", flush=True)
    try:
        for i, x in enumerate(세트):
            길 = x.get("path", "/v1/chat/completions") if "request" in x else "/v1/chat/completions"
            몸 = copy.deepcopy(x["request"] if "request" in x else x)
            몸["stream"] = False
            몸.setdefault("seed", a.seed + i)
            t0 = time.time()
            r = 묻기(a.app, "POST", 길, 몸)
            g = r.get("geulpu") or {}
            print(f"{i:3d} {길} · 프롬프트 {(r.get('usage') or {}).get('prompt_tokens')} · 생성 {len(g.get('tokens') or [])} · "
                  f"{(r['choices'][0].get('finish_reason'))} · {time.time() - t0:.1f} 초", flush=True)
    finally:
        d = 묻기(a.app, "POST", "/api/debug/record/stop", {})
        print(f"기록 멈춤: {d.get('name')} · 요청 {d.get('count')}", flush=True)


# ---- 재생 ----
def 재생(a):
    실 = 실행불러오기(a.app, a.run)
    기록중이면멈춤(a.app)
    헤더 = 실.헤더
    지금 = 준비된서버(a.app)
    알림 = 빌드다름(헤더, 지금)
    if 알림:
        print("기록한 빌드와 지금 빌드가 다르다 — " + " · ".join(알림) + " (커널이 바뀐 빌드면 비트가 다를 수 있다 — 재생으로 확인한다)")
    고른 = 고른차례(a.requests)
    기록 = [x for x in 실.요청들 if 고른 is None or x.get("i") in 고른] + [x for x in 실.분기들 if 고른 is None or x.get("of") in 고른]
    다름 = {}
    for x in 기록:
        for d in 서버다름(기록서버(헤더, x), 지금["server"]):
            다름[d] = 다름.get(d, 0) + 1
    if 다름:
        print("기록한 서버와 지금 서버가 다르다 — " + " · ".join(f"{d}(요청 {n})" for d, n in 다름.items()))
        if not a.ignore_header:
            raise SystemExit("재생을 멈춘다(--ignore-header 로 그대로 돌릴 수 있다 — 비트가 다를 수 있다)")
    보고 = []
    같음수 = 0
    for x in 기록:
        표 = f"b{x.get('b')}" if x.get("type") == "branch" else f"{x.get('i')}"
        열 = {"b": x.get("b"), "of": x.get("of")} if x.get("type") == "branch" else {"i": x.get("i")}
        resp = x.get("response") or {}
        g = resp.get("geulpu") or {}
        if x.get("path") not in 생성길 or "request" not in x:
            보고.append(dict(열, result="skipped", reason="not a generation path, or the request is not JSON"))
            continue
        몸 = copy.deepcopy(x["request"])
        몸["stream"] = False
        몸.pop("stream_options", None)
        if x.get("status", 200) != 200:              # 오류 응답이면 같은 오류(상태 · 코드)가 나오나
            if not isinstance(resp.get("error"), dict):
                보고.append(dict(열, result="skipped", reason=f"status {x.get('status')} without an error response"))
                continue
            t0 = time.time()
            상태2, r = 묻기상태(a.app, x["path"], 몸)
            걸림 = time.time() - t0
            코드1, 코드2 = resp["error"].get("code"), ((r or {}).get("error") or {}).get("code")
            틀린 = [] if 상태2 == x["status"] and 코드1 == 코드2 else [f"error {x['status']} {코드1} → {상태2} {코드2}"]
            같음수 += not 틀린
            보고.append(dict(열, result="same" if not 틀린 else "different", differences=틀린, error=x["status"], seconds=round(걸림, 2)))
            print(f"{표:>4} {x['path']} · 오류 {x['status']} {코드1} · {'같음' if not 틀린 else '다름 — ' + 틀린[0]} · {걸림:.1f} 초", flush=True)
            continue
        if g.get("tokens") is None:
            보고.append(dict(열, result="skipped", reason="no tokens in the response (cut stream or no geulpu object)"))
            continue
        if "seed" in g:
            몸["seed"] = g["seed"]
        gp = 몸.get("geulpu") if isinstance(몸.get("geulpu"), dict) else {}
        gp.update({"logit_hash": True, "prompt_tokens": True})
        몸["geulpu"] = gp
        t0 = time.time()
        r = 묻기(a.app, "POST", x["path"], 몸)
        걸림 = time.time() - t0
        g2 = r.get("geulpu") or {}
        끝1, 끝2 = 끝까닭(resp), 끝까닭(r)
        틀린 = []
        if g.get("prompt_tokens") is not None and g.get("prompt_tokens") != g2.get("prompt_tokens"):
            틀린.append("prompt tokens")
        t1, t2 = g.get("tokens") or [], g2.get("tokens") or []
        h1, h2 = g.get("logit_hashes"), g2.get("logit_hashes")
        첫다른 = 처음다른(t1, t2)
        if 첫다른 is not None:
            틀린.append(f"tokens (step {첫다른})")
        if h1 is not None and h1 != h2:
            틀린.append(f"logit hashes (step {처음다른(h1, h2 or [])})")
        ph1, ph2 = g.get("prompt_logit_hashes"), g2.get("prompt_logit_hashes")
        if ph1 is not None and ph1 != ph2:
            틀린.append(f"echo logit hashes (position {처음다른(ph1, ph2 or [])})")
        if 끝1 is not None and 끝1 != 끝2:
            틀린.append(f"finish_reason {끝1} → {끝2}")
        같음수 += not 틀린
        보고.append(dict(열, result="same" if not 틀린 else "different", differences=틀린, steps=len(t1), hashes=h1 is not None,
                         seconds=round(걸림, 2)))
        print(f"{표:>4} {x['path']} · 스텝 {len(t1)}{'' if h1 is not None else ' (해시 없음 — 토큰만)'} · {'같음' if not 틀린 else '다름'}"
              + (f" — {', '.join(틀린)}" if 틀린 else "") + f" · {걸림:.1f} 초", flush=True)
    돌린 = sum(1 for x in 보고 if x["result"] != "skipped")
    분기수 = sum(1 for x in 기록 if x.get("type") == "branch")
    print(f"재생: 요청 {돌린} 중 같음 {같음수}" + (f"(분기 {분기수} 포함)" if 분기수 else "") + (f" · 건너뜀 {len(보고) - 돌린}" if len(보고) > 돌린 else "")
          + (f" · 점수 줄 {len(실.점수들)} 은 재생하지 않음(score --check)" if 실.점수들 else ""))
    if a.out:
        json.dump({"header": 헤더, "current": 지금, "build_differences": 알림, "server_differences": 다름, "results": 보고},
                  open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    sys.exit(0 if 같음수 == 돌린 else 1)


def main():
    p = argparse.ArgumentParser(description="GeulPU 디버거 v1 — 실행 기록 · 테스트 세트 · 재생 · 점수 · 중단점 · 분기 · 비교")
    p.add_argument("--app", default=os.environ.get("GEULPU_APP", "http://127.0.0.1:44544"), help="GeulPU 앱의 주소(환경 변수 GEULPU_APP)")
    sp = p.add_subparsers(dest="command", required=True)
    sp.add_parser("status", help="기록 상태와 실행 목록")
    s = sp.add_parser("record-start", help="앱의 /v1 을 지나는 생성 요청을 실행 로그로 기록하기 시작")
    s.add_argument("name", nargs="?")
    sp.add_parser("record-stop", help="기록 멈춤")
    s = sp.add_parser("get", help="실행 로그를 받아 파일로")
    s.add_argument("name")
    s.add_argument("file")
    s = sp.add_parser("delete", help="실행 로그 지우기")
    s.add_argument("name")
    s = sp.add_parser("load", help="앱의 모델 불러오기(모델 아이디 · sha256 앞 8 자 이상 · run:실행)")
    s.add_argument("model")
    s.add_argument("--precision", choices=("exact", "fast"), help="계산 방식(앱의 설정을 바꾼다)")
    s.add_argument("--kv8", choices=("on", "off"), help="KV 캐시 8비트(앱의 설정을 바꾼다 — 빠름 · CUDA · k-양자화 모델에서만 쓴다)")
    s.add_argument("--layer", choices=("cuda", "vulkan"), help="백엔드(앱의 설정을 바꾼다)")
    s.add_argument("--timeout", type=int, default=900, help="기다릴 초")
    s = sp.add_parser("run-set", help="테스트 세트를 실행 하나로 기록")
    s.add_argument("testset")
    s.add_argument("name")
    s.add_argument("--seed", type=int, default=1000, help="시드가 없는 요청에 넣을 시드의 시작(+ 차례)")
    s = sp.add_parser("replay", help="실행 로그(요청 · 분기)를 다시 돌려 비트까지 비교")
    s.add_argument("run", metavar="RUN")
    s.add_argument("--out", help="보고서 JSON")
    s.add_argument("--requests", help="이 차례(i)만 — 쉼표로(분기는 그 요청의 것)")
    s.add_argument("--ignore-header", action="store_true", help="기록한 서버와 달라도 그대로 돌린다")
    s = sp.add_parser("score", help="요청마다 답을 지금 모델로 echo 스코어링해 점수 줄을 덧붙인다")
    s.add_argument("run", metavar="RUN")
    s.add_argument("--requests", help="이 차례(i)만 — 쉼표로")
    s.add_argument("--check", action="store_true", help="덧붙이지 않고 이미 있는 점수와 비트까지 같은지 본다")
    s = sp.add_parser("breakpoints", help="중단점 — 마진이 작은 스텝 · 저확률 샘플 · 도구 호출의 시작")
    s.add_argument("run", metavar="RUN")
    s.add_argument("--margin", type=float, default=0.05, help="마진(1위와 2위의 로그확률 차, nat) 기준")
    s.add_argument("--prob", type=float, default=0.05, help="고른 토큰의 확률 기준")
    s.add_argument("--requests", help="이 차례(i)만 — 쉼표로")
    s.add_argument("--show", type=int, default=6, help="요청마다 보일 중단점 수")
    s.add_argument("--out", help="중단점 JSON")
    s = sp.add_parser("branch", help="분기 — 스텝 K 에서 다른 토큰으로 잇거나(--step) 요청을 고쳐 다시(--set)")
    s.add_argument("run", metavar="RUN")
    s.add_argument("--request", type=int, required=True, help="요청 차례(i)")
    s.add_argument("--step", type=int, help="토큰 분기의 스텝(0 = 첫 생성 토큰)")
    s.add_argument("--token", help="고를 토큰 번호, 또는 same(원래 토큰 — 원래 답이 그대로 나와야 한다)")
    s.add_argument("--alt", type=int, help="점수 줄의 상위 N 위 토큰(기본: 원래와 다른 가장 높은 토큰)")
    s.add_argument("--set", action="append", metavar="PATH=VALUE", help="요청 분기 — 요청의 경로(messages.2.content)를 이 값으로(JSON 이면 JSON)")
    s.add_argument("--set-file", action="append", metavar="PATH=FILE", help="요청 분기 — 값을 파일의 글로")
    s.add_argument("--repeat", type=int, default=2, help="같은 분기를 몇 번 돌려 비트를 비교할지")
    s.add_argument("--no-save", action="store_true", help="실행 로그에 덧붙이지 않는다")
    s.add_argument("--ignore-header", action="store_true", help="그 요청을 생성한 서버와 달라도 돌린다")
    s = sp.add_parser("compare", help="실행 A · B 비교 — 같은 답 · 처음 갈라진 스텝 · A 의 답을 B 로 점수")
    s.add_argument("run_a", metavar="RUN_A")
    s.add_argument("run_b", metavar="RUN_B")
    s.add_argument("--out", help="보고서 JSON")
    a = p.parse_args()
    {"status": 상태, "record-start": 기록시작, "record-stop": 기록멈춤, "get": 받기, "delete": 지우기, "load": 불러오기, "run-set": 테스트세트,
     "replay": 재생, "score": 점수, "breakpoints": 중단점, "branch": 분기, "compare": 비교}[a.command](a)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
