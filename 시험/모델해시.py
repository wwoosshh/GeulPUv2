#!/usr/bin/env python3
"""GeulPUv2 — 앱이 모델 파일의 SHA-256 과 함께 그때의 크기 · 수정 시각을 두고, 같은 경로의 파일이 바뀌면 옛 해시를 버리는가(#19)를
GPU 없이 본다(빠른 점검 4h — 윈도만). 옛 해시가 프롬프트 캐시의 시드로 남으면 옛 파일의 청크가 새 파일의 KV 로 복원된다.

  python -I 시험/모델해시.py <GeulPU.exe> <빈 자료 폴더>

앱을 임시 자료 폴더 · 빈 포트로 띄우고(장치 · 모델 없이 — 모델 파일은 1 MiB 의 정한 바이트), API 와 모델목록.json 으로:
  1 확인(/api/models/verify) → 파일의 실제 SHA-256, 목록에 sha256_size · sha256_mtime
  2 다시 켬(파일 그대로) → 해시가 남는다
  3 끄고 같은 크기로 한 바이트를 바꿈 → 다시 켬 → 해시와 확인 표시를 버린다
  4 0.10.1 까지의 목록(지문 없음) → 해시는 화면에 그대로, 지문은 없음(프롬프트 캐시에 쓰기 전에 다시 센다) → 확인하면 지문이 붙는다
불러오기(프롬프트 캐시 켬 · 끔)의 길은 GPU 가 있어야 해 PR 의 수동 시험이 맡는다.
마지막 줄 '모델 해시 지문: 모두 통과 (N 가지)'(종료 0) 또는 '… 틀림 N'(종료 1)."""
import hashlib
import json
import os
import shutil
import socket
import subprocess
import sys
import time
import urllib.request

FILETIME_1970 = 116444736000000000


def 빈포트():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class 앱:
    def __init__(self, exe, 자료):
        self.exe, self.자료, self.p, self.주소 = exe, 자료, None, None
        self.기록 = open(os.path.join(os.path.dirname(자료.rstrip("\\/")), "모델해시_앱.txt"), "ab")

    def 묻기(self, 길, 몸=None):
        r = urllib.request.Request(self.주소 + 길, data=None if 몸 is None else json.dumps(몸).encode("utf-8"),
                                   method="GET" if 몸 is None else "POST", headers={"Content-Type": "application/json"})
        return json.loads(urllib.request.urlopen(r, timeout=30).read() or b"null")

    def 켜기(self):
        포트 = 빈포트()
        self.주소 = f"http://127.0.0.1:{포트}"
        self.p = subprocess.Popen([self.exe, "--포트", str(포트), "--자료", self.자료, "--열지않음"], stdout=self.기록,
                                  stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
        for _ in range(300):
            try:
                self.묻기("/api/state")
                return
            except Exception:
                if self.p.poll() is not None:
                    break
                time.sleep(0.1)
        raise SystemExit(f"앱이 뜨지 않았다(종료 코드 {self.p.poll()})")

    def 끄기(self):
        if self.p:
            self.p.terminate()
            self.p.wait(30)
            self.p = None


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    exe, 자료 = os.path.abspath(sys.argv[1]), os.path.abspath(sys.argv[2])
    shutil.rmtree(자료, ignore_errors=True)
    os.makedirs(os.path.join(자료, "모델"))
    모델 = os.path.join(자료, "모델", "t.gguf")
    목록 = os.path.join(자료, "모델목록.json")
    with open(모델, "wb") as f:
        f.write(bytes(range(256)) * 4096)
    본것, 틀림 = [], []

    def 봄(이름, 맞음, 덧=""):
        print(f"{'통과' if 맞음 else '틀림'}  {이름}" + (f" — {덧}" if 덧 else ""), flush=True)
        본것.append(이름)
        if not 맞음:
            틀림.append(이름)

    def 실제해시():
        return hashlib.sha256(open(모델, "rb").read()).hexdigest()

    def 시각():
        return os.stat(모델).st_mtime_ns // 100 + FILETIME_1970

    def 칸():
        return next(m for m in a.묻기("/api/state")["models"] if os.path.normcase(m["path"]) == os.path.normcase(모델))

    def 목록칸():
        return next(e for e in json.load(open(목록, encoding="utf-8"))["models"] if os.path.normcase(e["path"]) == os.path.normcase(모델))

    def 확인():
        a.묻기("/api/models/verify", {"id": 칸()["id"]})
        for _ in range(600):
            m = 칸()
            if not m.get("verifying"):
                return m
            time.sleep(0.1)
        raise SystemExit("확인이 끝나지 않았다")

    a = 앱(exe, 자료)
    try:
        a.켜기()
        m = 확인()
        h0, e = 실제해시(), 목록칸()
        봄("1 확인이 실제 SHA-256 을 센다", m.get("sha256") == h0, (m.get("sha256") or "없음")[:12])
        봄("1 목록에 크기 · 수정 시각", e.get("sha256_size") == os.path.getsize(모델) and e.get("sha256_mtime") == 시각(),
           f"{e.get('sha256_size')} · {e.get('sha256_mtime')} (파일 {os.path.getsize(모델)} · {시각()})")

        a.끄기(); a.켜기()
        봄("2 파일이 그대로면 다시 켜도 해시가 남는다", 칸().get("sha256") == h0)

        a.끄기()
        time.sleep(2.1)                       # FAT · exFAT 의 수정 시각은 2 초 단위 — 실제 교체처럼 다음 칸에서 바꾼다
        with open(모델, "r+b") as f:
            f.seek(-1, os.SEEK_END)
            f.write(b"\x00")                  # 마지막 바이트 255 → 0, 크기는 같다
        h1 = 실제해시()
        a.켜기()
        m = 칸()
        봄("3 같은 크기로 바뀐 파일은 해시와 확인 표시를 버린다", m.get("sha256") is None and m.get("verified") is False,
           f"앱 {(m.get('sha256') or '없음')[:12]} · 확인 {m.get('verified')} · 파일 {h1[:12]}")

        a.끄기()
        j = json.load(open(목록, encoding="utf-8"))
        for e in j["models"]:
            if os.path.normcase(e["path"]) == os.path.normcase(모델):
                e.update(sha256=h1, size=os.path.getsize(모델), verified=True)
                e.pop("sha256_size", None)
                e.pop("sha256_mtime", None)
        with open(목록, "w", encoding="utf-8", newline="\n") as f:
            json.dump(j, f)
        a.켜기()
        m, e = 칸(), 목록칸()
        봄("4 옛 목록의 해시는 화면에 그대로", m.get("sha256") == h1 and m.get("verified") is True)
        봄("4 옛 목록의 해시에는 지문을 달지 않는다(쓰기 전에 다시 센다)", not (e.get("sha256_mtime") or 0) > 0, f"{e.get('sha256_mtime')}")
        m = 확인()
        e = 목록칸()
        봄("4 다시 세면 지문이 붙는다", m.get("sha256") == h1 and e.get("sha256_mtime") == 시각(), f"{e.get('sha256_mtime')}")
    finally:
        a.끄기()
    print(f"모델 해시 지문: 모두 통과 ({len(본것)} 가지)" if not 틀림 else f"모델 해시 지문: 틀림 {len(틀림)} — {'; '.join(틀림)}")
    sys.exit(1 if 틀림 else 0)


if __name__ == "__main__":
    main()
