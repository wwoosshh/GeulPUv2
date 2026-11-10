#!/usr/bin/env python3
"""바깥 에이전트가 안내 한 쪽(docs/cli.md)만으로 한 시험(2026-11-10)에서 막힌 곳을 빌드한 앱으로 본다 — 앱을 임시 자료 폴더와 빈 포트로
띄우고 CLI 로 부른다(모델 · GPU 계산 없이).

  1. experiment update 의 판정 기준 — text 가 없는 항목은 이미 있는 그 id 의 판정(result · note)만 바꾸고 나머지 판정 기준은 그대로다.
     없는 id · id 없는 항목 · 글을 바꾸는 항목은 400 과 CLI 의 영어 next 를 낸다. 항목마다 text 가 있는 요청은 전과 같다(목록 전체를 정하고,
     첫 실행 뒤에는 id · text 가 같아야 하며 다르면 409 와 영어 next).
  2. --wait 로 기다린 작업의 답은 {"job": {...}} 이고(안내가 그렇게 적는다) 종료 코드는 job.state 를 따른다. --select job.state 는 그 값만.

    python -I -X utf8 시험/실험수정점검.py <빌드한 GeulPU 실행 파일>
"""
import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
결과 = []


def 본다(무엇, 맞음, 덧=""):
    결과.append(bool(맞음))
    print(f"{'맞음' if 맞음 else '틀림'} {무엇}" + (f" — {덧}" if 덧 else ""), flush=True)


def 빈포트():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def 꾸리기(tmp, exe):
    """실행 파일과 그 옆의 data/ — 앱\\꾸리기.ps1 · 꾸리기.sh 가 넣는 것 가운데 실험과 CLI 가 읽는 것만."""
    새exe = os.path.join(tmp, os.path.basename(exe))
    shutil.copy2(exe, 새exe)
    data = os.path.join(tmp, "data")
    os.makedirs(data)
    shutil.copy2(os.path.join(ROOT, "앱", "data", "operations.json"), data)
    shutil.copy2(os.path.join(ROOT, "docs", "cli.md"), os.path.join(data, "cli-guide.md"))
    shutil.copytree(os.path.join(ROOT, "experiments", "_template"), os.path.join(data, "experiment-template"))
    shutil.copytree(os.path.join(ROOT, "앱", "data", "experiment-template-ko"), os.path.join(data, "experiment-template-ko"))
    shutil.copy2(os.path.join(ROOT, "experiments", "experiment.schema.json"), data)
    shutil.copy2(os.path.join(ROOT, "도구", "experiment_check.py"), data)
    return 새exe


def 앱기다리기(port, proc, 초=90):
    끝 = time.time() + 초
    while time.time() < 끝:
        if proc.poll() is not None:
            return False
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/ping", timeout=3) as r:
                if b"geulpu" in r.read():
                    return True
        except OSError:
            time.sleep(0.3)
    return False


def main():
    if len(sys.argv) != 2 or not os.path.isfile(sys.argv[1]):
        print("쓰는 법: python -I -X utf8 시험/실험수정점검.py <빌드한 GeulPU 실행 파일>")
        return 2
    t0 = time.time()
    tmp = tempfile.mkdtemp(prefix="geulpu_update_check_")      # 영어 이름 — 본문 파일의 경로를 CLI 인자로 넘긴다
    proc = None
    try:
        exe = 꾸리기(tmp, sys.argv[1])
        port = 빈포트()
        자료 = os.path.join(tmp, "appdata")
        os.makedirs(자료)
        환경 = {k: v for k, v in os.environ.items() if k not in ("GEULPU_EXPERIMENT", "GEULPU_SESSION")}
        환경["GEULPU_PORT"] = str(port)
        with open(os.path.join(tmp, "app.log"), "wb") as log:
            proc = subprocess.Popen([exe, "--port", str(port), "--data", 자료, "--no-open"], stdin=subprocess.DEVNULL, stdout=log,
                                    stderr=subprocess.STDOUT, creationflags=0x08000000 if os.name == "nt" else 0)
        if not 앱기다리기(port, proc):
            print("틀림 앱이 뜨지 않았다 — " + open(os.path.join(tmp, "app.log"), encoding="utf-8", errors="replace").read()[-400:])
            return 1

        def cli(*args):
            p = subprocess.run([exe] + list(args), capture_output=True, stdin=subprocess.DEVNULL, timeout=180, env=환경)
            글 = p.stdout.decode("utf-8", "replace").strip()
            try:
                return p.returncode, json.loads(글)
            except ValueError:
                return p.returncode, {"_raw": 글[:300]}

        def 고치기(몸):
            길 = os.path.join(tmp, "body.json")
            with open(길, "w", encoding="utf-8", newline="\n") as f:
                json.dump(몸, f, ensure_ascii=False)
            return cli("experiment", "update", "--json", "@" + 길)

        def 기준들(out):
            return (out.get("experiment") or {}).get("criteria") if isinstance(out, dict) else None

        def 틀린답(rc, out, 상태, 다음에):
            e = out.get("error") if isinstance(out, dict) else None
            return rc == 1 and isinstance(e, dict) and e.get("status") == 상태 and 다음에 in (e.get("next") or "")

        rc, out = cli("experiment", "create", "--name", "verdicts", "--title", "Verdict entries", "--question", "Do verdict entries work?")
        길 = out.get("path") if isinstance(out, dict) else None
        본다("실험 만들기", rc == 0 and bool(길), f"종료 {rc} {str(out)[:160]}")
        if not 길:
            return 1

        # 실행 전 — 항목마다 text: 목록 전체를 정한다(전과 같다)
        rc, out = 고치기({"path": 길, "plan": "# plan\n", "criteria": [{"id": "C1", "text": "first"}, {"id": "C2", "text": "second"}]})
        본다("실행 전 — text 가 든 목록이 판정 기준 전체를 정한다",
           rc == 0 and 기준들(out) == [{"id": "C1", "text": "first", "result": "pending"}, {"id": "C2", "text": "second", "result": "pending"}],
           str(기준들(out))[:200])

        # text 가 없는 항목 — 그 id 의 판정만
        rc, out = 고치기({"path": 길, "criteria": [{"id": "C2", "result": "pass", "note": "seen in r1"}]})
        본다("text 가 없는 항목은 그 id 의 판정만 바꾼다(나머지는 그대로)",
           rc == 0 and 기준들(out) == [{"id": "C1", "text": "first", "result": "pending"},
                                    {"id": "C2", "text": "second", "result": "pass", "note": "seen in r1"}], str(기준들(out))[:200])
        rc, out = 고치기({"path": 길, "criteria": [{"id": "C2", "note": ""}, {"id": "C1", "note": "open"}]})
        본다("주지 않은 result 는 그대로, 빈 note 는 뺀다",
           rc == 0 and 기준들(out) == [{"id": "C1", "text": "first", "result": "pending", "note": "open"},
                                    {"id": "C2", "text": "second", "result": "pass"}], str(기준들(out))[:200])
        rc, out = 고치기({"path": 길, "criteria": [{"id": "C1", "text": "first", "result": "fail"}, {"id": "C2", "result": "fail"}]})
        본다("같은 text 를 준 항목이 섞여도 판정만 바꾼다",
           rc == 0 and [c.get("result") for c in 기준들(out) or []] == ["fail", "fail"] and [c.get("text") for c in 기준들(out)] == ["first", "second"],
           str(기준들(out))[:200])

        # 틀린 요청 — 400 과 영어 next
        for 무엇, 항목들 in (("없는 id", [{"id": "C9", "result": "pass"}]),
                          ("id 가 없는 항목", [{"result": "pass", "note": "x"}]),
                          ("판정만 적는 요청에서 text 를 바꿈", [{"id": "C1", "text": "changed"}, {"id": "C2", "result": "pass"}]),
                          ("모르는 result", [{"id": "C1", "result": "yes"}])):
            rc, out = 고치기({"path": 길, "criteria": 항목들})
            본다(f"{무엇} → 400 과 영어 next", 틀린답(rc, out, 400, "to record a verdict"), f"종료 {rc} {str(out)[:200]}")
        rc, out = cli("experiment", "show", "--path", 길)
        본다("틀린 요청은 아무것도 바꾸지 않는다", [c.get("result") for c in 기준들(out) or []] == ["fail", "fail"], str(기준들(out))[:200])

        # --wait — 작업의 답은 {"job": {...}}, 종료 코드는 job.state 를 따른다
        rc, out = cli("experiment", "check", "--path", 길, "--wait")
        job = out.get("job") if isinstance(out, dict) else None
        본다('--wait 의 답은 {"job": {...}}',
           isinstance(job, dict) and list(out) == ["job"] and all(k in job for k in ("id", "state", "summary", "reason"))
           and job["state"] in ("done", "failed", "cancelled") and (rc == 0) == (job["state"] == "done"), f"종료 {rc} {str(out)[:200]}")
        rc2, out2 = cli("experiment", "check", "--path", 길, "--wait", "--select", "job.state")
        본다("--select job.state 는 그 값만", isinstance(job, dict) and out2 == job.get("state"), f"종료 {rc2} {str(out2)[:80]}")

        # 실행 전 — text 가 든 목록은 여전히 목록 전체를 바꾼다
        rc, out = 고치기({"path": 길, "criteria": [{"id": "C1", "text": "first"}, {"id": "C3", "text": "third", "result": "pass"}]})
        본다("실행 전 — text 가 든 목록은 목록 전체를 바꾼다(전과 같다)",
           rc == 0 and 기준들(out) == [{"id": "C1", "text": "first", "result": "pending"}, {"id": "C3", "text": "third", "result": "pass"}],
           str(기준들(out))[:200])

        # 첫 실행 뒤(실행 기록 한 줄을 손으로 더해 흉내 낸다 — 진짜 실행에는 모델이 든다): id · text 는 고정, 판정은 그대로 적힌다
        제이슨길 = os.path.join(길, "experiment.json")
        e = json.load(open(제이슨길, encoding="utf-8"))
        e["runs"] = [{"id": "r1", "kind": "generate", "operation": "run.complete", "record": "records/runs/r1.json"}]
        with open(제이슨길, "w", encoding="utf-8", newline="\n") as f:
            json.dump(e, f, ensure_ascii=False, indent=1)
        rc, out = 고치기({"path": 길, "criteria": [{"id": "C1", "text": "first, reworded"}, {"id": "C3", "text": "third"}]})
        본다("첫 실행 뒤 — text 를 바꾸면 409 와 영어 next", 틀린답(rc, out, 409, "after the first run"), f"종료 {rc} {str(out)[:200]}")
        rc, out = 고치기({"path": 길, "criteria": [{"id": "C3", "result": "fail", "note": "r1 step 40"}], "status": "done", "results": "# results\n"})
        본다("첫 실행 뒤 — text 가 없는 항목으로 판정을 적는다",
           rc == 0 and 기준들(out) == [{"id": "C1", "text": "first", "result": "pending"},
                                    {"id": "C3", "text": "third", "result": "fail", "note": "r1 step 40"}]
           and (out.get("experiment") or {}).get("status") == "done", f"종료 {rc} {str(out)[:240]}")
        rc, out = 고치기({"path": 길, "criteria": [{"id": "C1", "text": "first", "result": "pass"}, {"id": "C3", "text": "third", "result": "fail"}]})
        본다("첫 실행 뒤 — id · text 가 같은 목록 전체도 전과 같이 받는다",
           rc == 0 and 기준들(out) == [{"id": "C1", "text": "first", "result": "pass"}, {"id": "C3", "text": "third", "result": "fail"}],
           f"종료 {rc} {str(기준들(out))[:200]}")
    finally:
        if proc is not None and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(30)
            except subprocess.TimeoutExpired:
                proc.kill()
        shutil.rmtree(tmp, ignore_errors=True)
    print(f"실험 수정 점검: {sum(결과)}/{len(결과)} 맞음 ({time.time() - t0:.1f} 초)")
    return 0 if 결과 and all(결과) else 1


if __name__ == "__main__":
    sys.exit(main())
