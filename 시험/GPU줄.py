#!/usr/bin/env python3
"""GPU 줄(대기열) — 이 PC 의 GPU 를 한 번에 한 일만 쓰게 한다. 여러 세션 · 작업 트리가 GPU 하나를 나눠 쓸 때 측정이 서로 섞이거나 그래픽
메모리가 모자라지 않게(문서/개발.md 의 '여러 세션으로 개발하기').

  python -I -X utf8 시험/GPU줄.py 실행 [--이름 "<가지 — 할 일>"] -- <명령> [인자 …]
  python -I -X utf8 시험/GPU줄.py 상태
  python -I -X utf8 시험/GPU줄.py 자체시험

실행: 줄을 서서(먼저 선 차례대로) GPU 를 잡고, 명령을 돌리고, 끝나면 놓는다 — 명령의 종료 코드를 그대로 낸다. 명령이 띄운 프로세스도 모두
명령과 함께 끝난다(윈도: 작업 개체 — 이 프로그램이 죽어도 · 리눅스: 프로세스 그룹 — 이 프로그램이 끝날 때). 잠금은 운영체제의 파일 잠금이라
잡은 프로세스가 어떻게 끝나든 풀리고, 줄의 자리(기다림/)는 그 프로세스가 아직 살아 있는지로 치운다.
상태: 잡은 일 · 기다리는 일, 그리고 nvidia-smi 의 그래픽 메모리 · GPU 를 쓰는 프로세스(줄 밖에서 쓰는 것 — 사용자의 앱 등 — 도 보인다).
자체시험: GPU 없이 — 짧은 명령 셋을 거의 함께 넣어 겹치지 않고 선 차례대로 도는지, 죽은 대기자의 자리를 치우는지(빠른 점검).

폴더: 환경 변수 GEULPU_GPU_LOCK, 없으면 윈도 %LOCALAPPDATA%/GeulPU-dev/gpu · 리눅스 ~/.cache/geulpu-dev/gpu — 작업 트리 밖이라 모든 세션이
같은 줄을 본다.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time

윈도 = os.name == "nt"
if 윈도:
    import ctypes
    import msvcrt
    from ctypes import wintypes

    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    k32.OpenProcess.restype = wintypes.HANDLE
    k32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    k32.GetExitCodeProcess.restype = wintypes.BOOL
    k32.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
    k32.GetProcessTimes.restype = wintypes.BOOL
    k32.GetProcessTimes.argtypes = [wintypes.HANDLE] + [ctypes.POINTER(wintypes.FILETIME)] * 4
    k32.CloseHandle.argtypes = [wintypes.HANDLE]
    k32.GetCurrentProcess.restype = wintypes.HANDLE
    k32.CreateJobObjectW.restype = wintypes.HANDLE
    k32.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
    k32.SetInformationJobObject.restype = wintypes.BOOL
    k32.SetInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
    k32.AssignProcessToJobObject.restype = wintypes.BOOL
    k32.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
else:
    import fcntl
    import signal


def 폴더():
    d = os.environ.get("GEULPU_GPU_LOCK")
    if not d:
        if 윈도:
            d = os.path.join(os.environ.get("LOCALAPPDATA") or os.path.expanduser("~"), "GeulPU-dev", "gpu")
        else:
            d = os.path.join(os.path.expanduser("~"), ".cache", "geulpu-dev", "gpu")
    os.makedirs(os.path.join(d, "기다림"), exist_ok=True)
    return d


def 시작값(pid):
    """프로세스가 생긴 때 — pid 를 다시 쓴 다른 프로세스와 가르려고(없으면 None)"""
    if 윈도:
        h = k32.OpenProcess(0x1000, False, pid)            # PROCESS_QUERY_LIMITED_INFORMATION
        if not h:
            return None
        try:
            c = wintypes.DWORD()
            if not k32.GetExitCodeProcess(h, ctypes.byref(c)) or c.value != 259:     # STILL_ACTIVE
                return None
            t = [wintypes.FILETIME() for _ in range(4)]
            if not k32.GetProcessTimes(h, *[ctypes.byref(x) for x in t]):
                return None
            return (t[0].dwHighDateTime << 32) | t[0].dwLowDateTime
        finally:
            k32.CloseHandle(h)
    try:
        with open(f"/proc/{pid}/stat", encoding="ascii") as f:
            return int(f.read().rsplit(")", 1)[1].split()[19])
    except (OSError, ValueError, IndexError):
        return None


def 살아있나(정보):
    return 정보.get("pid") is not None and 시작값(정보["pid"]) == 정보.get("시작값")


def 읽기(경로):
    try:
        with open(경로, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def 쓰기(경로, 값):
    임시 = 경로 + f".{os.getpid()}.tmp"
    with open(임시, "w", encoding="utf-8", newline="\n") as f:
        json.dump(값, f, ensure_ascii=False)
    os.replace(임시, 경로)


class 잠금:
    """운영체제의 파일 잠금(윈도 msvcrt.locking · 리눅스 flock) — 잡은 프로세스가 끝나면 운영체제가 푼다"""

    def __init__(self, 경로):
        open(경로, "ab").close()
        self.f = open(경로, "r+b")

    def 시도(self):
        try:
            if 윈도:
                self.f.seek(0)
                msvcrt.locking(self.f.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                fcntl.flock(self.f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            return True
        except OSError:
            return False

    def 놓기(self):
        try:
            if 윈도:
                self.f.seek(0)
                msvcrt.locking(self.f.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(self.f.fileno(), fcntl.LOCK_UN)
        finally:
            self.f.close()


def 대기자들(d):
    """선 차례대로 (파일 이름, 정보) — 죽은 프로세스의 자리는 치운다"""
    곳 = os.path.join(d, "기다림")
    산 = []
    for 이름 in sorted(os.listdir(곳)):
        if not 이름.endswith(".json"):
            continue
        정보 = 읽기(os.path.join(곳, 이름))
        if 정보 is None:
            continue                                   # 쓰는 중
        if 살아있나(정보):
            산.append((이름, 정보))
        else:
            try:
                os.remove(os.path.join(곳, 이름))
            except OSError:
                pass
    return 산


def 작업개체로():
    """윈도 — 이 프로세스를 '닫히면 모두 끝내는' 작업 개체에: 명령과 그 자손이 들어가 이 프로그램이 죽어도 함께 끝난다"""

    class 기본(ctypes.Structure):
        _fields_ = [("PerProcessUserTimeLimit", ctypes.c_int64), ("PerJobUserTimeLimit", ctypes.c_int64),
                    ("LimitFlags", wintypes.DWORD), ("MinimumWorkingSetSize", ctypes.c_size_t),
                    ("MaximumWorkingSetSize", ctypes.c_size_t), ("ActiveProcessLimit", wintypes.DWORD),
                    ("Affinity", ctypes.c_size_t), ("PriorityClass", wintypes.DWORD), ("SchedulingClass", wintypes.DWORD)]

    class 확장(ctypes.Structure):
        _fields_ = [("BasicLimitInformation", 기본), ("IoInfo", ctypes.c_uint64 * 6), ("ProcessMemoryLimit", ctypes.c_size_t),
                    ("JobMemoryLimit", ctypes.c_size_t), ("PeakProcessMemoryUsed", ctypes.c_size_t),
                    ("PeakJobMemoryUsed", ctypes.c_size_t)]

    job = k32.CreateJobObjectW(None, None)
    정보 = 확장()
    정보.BasicLimitInformation.LimitFlags = 0x2000          # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
    if not job or not k32.SetInformationJobObject(job, 9, ctypes.byref(정보), ctypes.sizeof(정보)) \
            or not k32.AssignProcessToJobObject(job, k32.GetCurrentProcess()):
        print(f"GPU 줄: 작업 개체를 만들지 못했다(오류 {ctypes.get_last_error()}) — 이 프로그램이 죽으면 명령이 남을 수 있다",
              file=sys.stderr, flush=True)
    return job                                          # 닫지 않는다 — 이 프로세스가 끝날 때 운영체제가 닫는다


def 줄서기(d, 나, 조용히=False):
    곳 = os.path.join(d, "기다림")
    내이름 = f"{time.time_ns():020d}_{os.getpid()}.json"
    쓰기(os.path.join(곳, 내이름), 나)
    잠 = 잠금(os.path.join(d, "잠금"))
    알린 = None
    잡음 = False
    try:
        while True:
            if not os.path.exists(os.path.join(곳, 내이름)):
                쓰기(os.path.join(곳, 내이름), 나)       # 누가 지웠으면 같은 차례로 다시
            앞 = 대기자들(d)
            if 앞 and 앞[0][0] == 내이름 and 잠.시도():
                잡음 = True
                break
            if not 조용히:
                쥔 = 읽기(os.path.join(d, "쥔것.json"))
                차례 = next((i for i, (이름, _) in enumerate(앞) if 이름 == 내이름), 0)
                글 = f"GPU 줄: 기다림 — 앞에 {차례} 개" + (f", 쥔 일 '{쥔['이름']}'" if 쥔 and 살아있나(쥔) else "")
                if 글 != 알린:
                    print(글, file=sys.stderr, flush=True)
                    알린 = 글
            time.sleep(0.25)
    finally:
        try:
            os.remove(os.path.join(곳, 내이름))
        except OSError:
            pass
        if not 잡음:
            잠.f.close()
    쓰기(os.path.join(d, "쥔것.json"), dict(나, 잡은때=time.time()))
    return 잠


def 놓기(d, 잠):
    쥔 = 읽기(os.path.join(d, "쥔것.json"))
    if 쥔 and 쥔.get("pid") == os.getpid():
        try:
            os.remove(os.path.join(d, "쥔것.json"))
        except OSError:
            pass
    잠.놓기()


def 기본이름():
    try:
        가지 = subprocess.run(["git", "rev-parse", "--abbrev-ref", "HEAD"], capture_output=True, text=True, timeout=5).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        가지 = ""
    return f"{os.path.basename(os.getcwd())}" + (f" [{가지}]" if 가지 else "")


def 실행(a):
    명령 = a.명령[1:] if a.명령[:1] == ["--"] else a.명령
    if not 명령:
        print("GPU 줄: 돌릴 명령이 없다 — 실행 [--이름 …] -- <명령> …", file=sys.stderr)
        return 2
    d = 폴더()
    나 = {"pid": os.getpid(), "시작값": 시작값(os.getpid()), "이름": a.이름 or 기본이름(), "명령": " ".join(명령)[:300],
         "폴더": os.getcwd(), "선때": time.time()}
    t0 = time.time()
    잠 = 줄서기(d, 나, a.조용히)
    기다린 = time.time() - t0
    if not a.조용히:
        print(f"GPU 줄: 잡음 ({기다린:.0f} 초 기다림) — {나['이름']}", file=sys.stderr, flush=True)
    rc = 1
    try:
        if 윈도:
            작업개체로()
            p = subprocess.Popen(명령)
        else:
            p = subprocess.Popen(명령, start_new_session=True)
        try:
            rc = p.wait()
        except KeyboardInterrupt:
            p.terminate()
            rc = p.wait()
        finally:
            if not 윈도:
                try:
                    os.killpg(p.pid, signal.SIGTERM)
                except (ProcessLookupError, PermissionError):
                    pass
    finally:
        놓기(d, 잠)
    if not a.조용히:
        print(f"GPU 줄: 놓음 — {time.time() - t0 - 기다린:.0f} 초, 종료 {rc}", file=sys.stderr, flush=True)
    return rc


def 상태(a):
    d = 폴더()
    쥔 = 읽기(os.path.join(d, "쥔것.json"))
    지금 = time.time()
    if 쥔 and 살아있나(쥔):
        print(f"쥔 일: {쥔['이름']} — {(지금 - 쥔['잡은때']) / 60:.1f} 분째, pid {쥔['pid']}\n  명령: {쥔['명령']}\n  폴더: {쥔['폴더']}")
    else:
        print("쥔 일: 없음" + (f"(마지막: {쥔['이름']} — 끝남)" if 쥔 else ""))
    앞 = 대기자들(d)
    print(f"기다리는 일: {len(앞)} 개")
    for i, (_, 정보) in enumerate(앞, 1):
        print(f"  {i}. {정보['이름']} — {(지금 - 정보['선때']) / 60:.1f} 분째, pid {정보['pid']}: {정보['명령'][:120]}")
    smi = shutil.which("nvidia-smi")
    if smi:
        r = subprocess.run([smi, "--query-gpu=name,memory.used,memory.total,utilization.gpu", "--format=csv,noheader"],
                           capture_output=True, text=True)
        for 줄 in r.stdout.strip().splitlines():
            print(f"GPU: {줄}")
        r = subprocess.run([smi, "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader"],
                           capture_output=True, text=True)
        앱 = [x for x in r.stdout.strip().splitlines() if x.strip()]
        print(f"GPU 를 쓰는 프로세스: {len(앱)} 개" + "".join(f"\n  {x}" for x in 앱))
    print(f"폴더: {d}")
    return 0


def 자체시험(a):
    """GPU 없이 — 셋을 거의 함께 줄 세워 겹침 · 차례를 보고, 죽은 대기자의 자리를 앞에 두어 치우는지 본다"""
    임시 = tempfile.mkdtemp(prefix="gpu줄_")
    환경 = dict(os.environ, GEULPU_GPU_LOCK=os.path.join(임시, "gpu"))
    기록 = os.path.join(임시, "기록.txt")
    d = os.path.join(임시, "gpu")
    os.makedirs(os.path.join(d, "기다림"))
    끝난 = subprocess.Popen([sys.executable, "-c", "pass"])
    끝난.wait()
    with open(os.path.join(d, "기다림", f"{0:020d}_{끝난.pid}.json"), "w", encoding="utf-8") as f:
        json.dump({"pid": 끝난.pid, "시작값": 1, "이름": "죽은 대기자", "명령": "-", "폴더": "-", "선때": 0}, f)
    일 = ("import sys, time\n"
          "open(sys.argv[1], 'a', encoding='utf-8').write('시작 ' + sys.argv[2] + '\\n')\n"
          "time.sleep(float(sys.argv[3]))\n"
          "open(sys.argv[1], 'a', encoding='utf-8').write('끝 ' + sys.argv[2] + '\\n')\n")
    ps = []
    t0 = time.time()
    try:
        # 0 이 도는 동안 1 · 2 가 차례로 선다 — 0 이 놓을 때 둘 다 기다리고 있어 선 차례(1 먼저)를 본다
        for i, 길이 in enumerate((2.0, 0.3, 0.3)):
            ps.append(subprocess.Popen([sys.executable, "-I", "-X", "utf8", os.path.abspath(__file__), "실행", "--조용히",
                                        "--이름", f"시험 {i}", "--", sys.executable, "-X", "utf8", "-c", 일, 기록, str(i),
                                        str(길이)], env=환경))
            time.sleep(0.6)
        rcs = [p.wait(timeout=60) for p in ps]
        줄들 = open(기록, encoding="utf-8").read().split("\n")[:-1]
        기대 = [f"{w} {i}" for i in range(3) for w in ("시작", "끝")]
        남은 = os.listdir(os.path.join(d, "기다림"))
        된 = rcs == [0, 0, 0] and 줄들 == 기대 and not 남은 and not os.path.exists(os.path.join(d, "쥔것.json"))
        print(f"GPU 줄 자체시험: {'통과' if 된 else '못 미침'} — 종료 {rcs}, 차례 {줄들}, 남은 자리 {남은} ({time.time() - t0:.1f} 초)")
        return 0 if 된 else 1
    finally:
        for p in ps:
            if p.poll() is None:
                p.kill()
        shutil.rmtree(임시, ignore_errors=True)


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
    p = argparse.ArgumentParser(description="GPU 줄 — 이 PC 의 GPU 를 한 번에 한 일만")
    s = p.add_subparsers(dest="일", required=True)
    q = s.add_parser("실행", help="줄을 서서 GPU 를 잡고 명령을 돌린다")
    q.add_argument("--이름", default="", help="상태에 보일 이름 — '<가지> — <할 일>' (기본: 폴더 [가지])")
    q.add_argument("--조용히", action="store_true", help="기다림 · 잡음 · 놓음을 찍지 않는다")
    q.add_argument("명령", nargs=argparse.REMAINDER)
    s.add_parser("상태", help="잡은 일 · 기다리는 일 · GPU 메모리")
    s.add_parser("자체시험", help="GPU 없이 — 줄의 겹침 · 차례 · 죽은 자리 치우기")
    a = p.parse_args()
    return {"실행": 실행, "상태": 상태, "자체시험": 자체시험}[a.일](a)


if __name__ == "__main__":
    sys.exit(main())
