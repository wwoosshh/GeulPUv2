"""`geulpu` 명령(그리고 `python -m geulpu`) — 꾸러미에 든 GeulPU.exe(리눅스는 GeulPU)를 받은 인자 그대로 실행한다.

  geulpu                      # 앱을 띄우고 브라우저에 http://127.0.0.1:44544 를 연다(리눅스는 주소를 알린다)
  geulpu --포트 44600         # 다른 포트
  geulpu --자료 D:\\GeulPU    # 자료 폴더(기본 %USERPROFILE%\\GeulPU · 리눅스 ~/GeulPU — 환경 변수 GEULPU_HOME 으로도)
  geulpu --열지않음           # 브라우저를 열지 않는다
  geulpu --검사 쌍 --빠름     # 화면 없이 검사 하나를 돌리고 끝낸다(--check pair --quick)

앱은 이 콘솔을 그대로 쓴다(창을 닫거나 Ctrl+C 면 앱이 끝난다). 실행기는 Ctrl+C 를 앱에 맡기고 앱이 끝나기를 기다려 그 종료 코드를
돌려주며, 앱이 바꾼 콘솔의 코드 페이지를 되돌린다. 환경 변수 GEULPU_INSTALL(pip · pipx · uv — 이 꾸러미가 깔린 자리로 가림)을 넘겨
앱이 새 버전 알림에 그 설치 도구의 업데이트 명령을 보이게 한다.
"""
import os
import platform
import signal
import subprocess
import sys

from . import __version__, app_dir


def 설치방식():
    """이 꾸러미를 깐 도구 — pipx(…\\pipx\\…\\venvs\\…) · uv(…\\uv\\tools\\…) · 그 밖은 pip"""
    조각 = os.path.normcase(os.path.abspath(sys.prefix)).replace("\\", "/").split("/")
    if "pipx" in 조각 and "venvs" in 조각:
        return "pipx"
    if "uv" in 조각 and "tools" in 조각:
        return "uv"
    return "pip"


def 코드페이지():
    """지금 콘솔의 (입력, 출력) 코드 페이지 — 콘솔이 없으면 None"""
    try:
        import ctypes
        k = ctypes.windll.kernel32
        cp = (k.GetConsoleCP(), k.GetConsoleOutputCP())
        return cp if all(cp) else None
    except Exception:
        return None


def 코드페이지되돌리기(cp):
    if not cp:
        return
    try:
        import ctypes
        k = ctypes.windll.kernel32
        k.SetConsoleCP(cp[0])
        k.SetConsoleOutputCP(cp[1])
    except Exception:
        pass


def 실행권한주기(앱):
    """리눅스 — 설치 도구가 휠 항목의 실행 권한을 잃었으면 앱 실행 파일 · bin/* 에 다시 준다(내 파일일 때만 된다)"""
    bin폴더 = os.path.join(앱, "bin")
    for p in [os.path.join(앱, "GeulPU")] + [os.path.join(bin폴더, n) for n in (sorted(os.listdir(bin폴더)) if os.path.isdir(bin폴더) else [])]:
        if os.path.isfile(p) and not os.access(p, os.X_OK):
            try:
                os.chmod(p, os.stat(p).st_mode | 0o111)
            except OSError:
                pass


def main(argv=None):
    args = list(sys.argv[1:] if argv is None else argv)
    if args[:1] in (["--version"], ["-V"]):
        print(f"geulpu {__version__}")
        return 0
    기계 = platform.machine().lower()
    if sys.platform == "win32" and 기계 in ("amd64", "x86_64"):
        exe = os.path.join(app_dir(), "GeulPU.exe")
    elif sys.platform.startswith("linux") and 기계 in ("x86_64", "amd64"):
        exe = os.path.join(app_dir(), "GeulPU")
    else:
        sys.stderr.write("GeulPU 의 PyPI 꾸러미는 지금 Windows x64 · 리눅스 x86-64 만 담습니다 — 다른 환경은 소스에서 빌드하세요"
                         "(https://github.com/wwoosshh/GeulPUv2#소스에서-빌드하기).\n"
                         "The GeulPU package ships Windows x64 and Linux x86-64 only; build from source elsewhere.\n")
        return 2
    if not os.path.isfile(exe):
        if os.name == "nt":
            sys.stderr.write(f"GeulPU.exe 가 없습니다: {exe}\n백신이 격리했다면 되돌린 뒤, 아니면 꾸러미를 다시 설치하세요"
                             "(pip install --force-reinstall geulpu).\n")
        else:
            sys.stderr.write(f"GeulPU 가 없습니다: {exe}\n이 플랫폼의 휠이 아닌 꾸러미가 깔렸습니다 — 꾸러미를 다시 설치하세요"
                             "(pip install --force-reinstall geulpu).\n")
        return 2
    if os.name != "nt" and not os.access(exe, os.X_OK):
        실행권한주기(app_dir())
    env = dict(os.environ)
    env["GEULPU_INSTALL"] = 설치방식()
    # Ctrl+C 는 같은 콘솔의 앱도 받는다 — 실행기는 무시하고 앱이 스스로 끝나기를 기다린다(앱을 억지로 죽여 서버가 남지 않게)
    try:
        signal.signal(signal.SIGINT, signal.SIG_IGN)
    except (ValueError, OSError):
        pass
    cp = 코드페이지()
    try:
        # 절대 경로 · 셸 없이(현재 폴더의 다른 GeulPU.exe 가 잡히지 않게). 작업 폴더는 그대로 — 앱은 자기 폴더를 실행 파일의 자리로 찾고,
        # --자료 의 상대 경로는 셸 기준이다
        p = subprocess.Popen([exe] + args, env=env, shell=False)
    except OSError as e:
        sys.stderr.write(f"{os.path.basename(exe)} 를 실행하지 못했습니다: {e}\n")
        return 2
    try:
        return p.wait()
    finally:
        코드페이지되돌리기(cp)


if __name__ == "__main__":
    sys.exit(main())
