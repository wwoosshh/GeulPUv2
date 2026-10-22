#!/usr/bin/env python3
"""GeulPUv2 — 오래 켜 둔 앱의 메모리가 늘지 않는가(#22)를 GPU 없이 본다(빠른 점검 4j — 윈도만). 글(geul) 런타임에는 가비지 컬렉터가
없어 놓지 않은 할당은 쌓인다.

  python -I 시험/앱누수.py <GeulPU.exe> <빈 작업 폴더> <받을 모델 목록(앱/data/models.json)>

앱을 작업 폴더에 받을 모델 목록과 함께 두고(화면이 1 초마다 부르는 /api/state 가 목록을 쓴다) 빈 포트로 띄운 뒤, 데우고 나서 앱의
private 메모리(PrivateUsage)를 재고 세 가지를 되풀이한다 — 되풀이마다 늘어난 바이트가 한도 아래인가:
  1 /api/state 2,000 번(목록의 항목마다 글 열둘을 복사만 하고 놓지 않았다)
  2 글 1,000 개가 든 올바른 JSON 본문 500 번(파서가 글마다 바이트열의 머리를 남겼다)
  3 끝이 잘린 JSON 본문 500 번(파서가 오류에서 만들던 나무를 통째로 남겼다)
마지막 줄 '앱 누수: 모두 통과 (3 가지)'(종료 0) 또는 '… 틀림 N'(종료 1)."""
import ctypes
import ctypes.wintypes as w
import json
import os
import shutil
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request


class 메모리셈(ctypes.Structure):
    _fields_ = [("cb", w.DWORD), ("PageFaultCount", w.DWORD), ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t), ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t), ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t), ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t), ("PrivateUsage", ctypes.c_size_t)]


def 사적메모리(pid):
    k = ctypes.WinDLL("kernel32", use_last_error=True)
    k.OpenProcess.restype = w.HANDLE
    h = k.OpenProcess(0x1000 | 0x0010, False, pid)            # PROCESS_QUERY_LIMITED_INFORMATION | PROCESS_VM_READ
    try:
        m = 메모리셈()
        m.cb = ctypes.sizeof(m)
        if not k.K32GetProcessMemoryInfo(h, ctypes.byref(m), m.cb):
            raise OSError(ctypes.get_last_error())
        return m.PrivateUsage
    finally:
        k.CloseHandle(h)


def 빈포트():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def 보내기(주소, 길, 몸=None):
    r = urllib.request.Request(주소 + 길, data=몸, method="GET" if 몸 is None else "POST", headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(r, timeout=30) as f:
            f.read()
    except urllib.error.HTTPError as e:
        e.read()


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    원exe, 작업, 목록 = os.path.abspath(sys.argv[1]), os.path.abspath(sys.argv[2]), os.path.abspath(sys.argv[3])
    shutil.rmtree(작업, ignore_errors=True)
    os.makedirs(os.path.join(작업, "data"))
    exe = os.path.join(작업, "GeulPU.exe")
    shutil.copyfile(원exe, exe)
    shutil.copyfile(목록, os.path.join(작업, "data", "models.json"))
    자료 = os.path.join(작업, "자료")
    포트 = 빈포트()
    주소 = f"http://127.0.0.1:{포트}"
    기록 = open(os.path.join(작업, "앱.txt"), "ab")
    p = subprocess.Popen([exe, "--포트", str(포트), "--자료", 자료, "--열지않음"], stdout=기록, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
    틀림 = []
    try:
        for _ in range(300):
            try:
                보내기(주소, "/api/state")
                break
            except Exception:
                if p.poll() is not None:
                    raise SystemExit(f"앱이 뜨지 않았다(종료 코드 {p.poll()})")
                time.sleep(0.1)
        올바른 = json.dumps({"id": "없는-모델", "pad": [f"글{i}" for i in range(1000)]}).encode("utf-8")
        잘린 = 올바른[:-200]
        표 = [("/api/state 2,000 번", 2000, lambda: 보내기(주소, "/api/state"), 1 << 20),
              ("글 1,000 개의 본문 500 번", 500, lambda: 보내기(주소, "/api/models/verify", 올바른), 1 << 20),
              ("잘린 본문 500 번", 500, lambda: 보내기(주소, "/api/models/verify", 잘린), 1 << 20)]
        for 이름, n, 하나, 한도 in 표:
            for _ in range(n):                             # 데우기 — 힙이 자리를 잡게(고친 앱도 /api/state 처음 수천 번은 수백 KB 늘고 그 뒤 0)
                하나()
            m0 = 사적메모리(p.pid)
            for _ in range(n):
                하나()
            m1 = 사적메모리(p.pid)
            늘어남 = m1 - m0
            맞음 = 늘어남 < 한도
            print(f"{'통과' if 맞음 else '틀림'}  {이름} — private {m0 / 1e6:.1f} → {m1 / 1e6:.1f} MB (늘어남 {늘어남 / 1e3:,.0f} KB · 한도 "
                  f"{한도 / 1e3:,.0f} KB)", flush=True)
            if not 맞음:
                틀림.append(이름)
    finally:
        p.terminate()
        p.wait(30)
    print(f"앱 누수: 모두 통과 ({len(표)} 가지)" if not 틀림 else f"앱 누수: 틀림 {len(틀림)} — {'; '.join(틀림)}")
    sys.exit(1 if 틀림 else 0)


if __name__ == "__main__":
    main()
