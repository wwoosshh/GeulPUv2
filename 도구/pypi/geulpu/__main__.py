"""`geulpu` 명령(그리고 `python -m geulpu`) — 꾸러미에 든 GeulPU.exe 를 받은 인자 그대로 실행한다.

  geulpu                      # 앱을 띄우고 브라우저에 http://127.0.0.1:44544 를 연다
  geulpu --포트 44600         # 다른 포트
  geulpu --자료 D:\\GeulPU    # 자료 폴더(기본 %USERPROFILE%\\GeulPU — 환경 변수 GEULPU_HOME 으로도)
  geulpu --열지않음           # 브라우저를 열지 않는다

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


def main(argv=None):
    args = list(sys.argv[1:] if argv is None else argv)
    if args[:1] in (["--version"], ["-V"]):
        print(f"geulpu {__version__}")
        return 0
    if sys.platform != "win32" or platform.machine().lower() not in ("amd64", "x86_64"):
        sys.stderr.write("GeulPU 의 PyPI 꾸러미는 지금 Windows x64 만 담습니다 — 다른 환경은 소스에서 빌드하세요"
                         "(https://github.com/wwoosshh/GeulPUv2#소스에서-빌드하기).\n"
                         "The GeulPU package currently ships Windows x64 only; build from source elsewhere.\n")
        return 2
    exe = os.path.join(app_dir(), "GeulPU.exe")
    if not os.path.isfile(exe):
        sys.stderr.write(f"GeulPU.exe 가 없습니다: {exe}\n백신이 격리했다면 되돌린 뒤, 아니면 꾸러미를 다시 설치하세요"
                         "(pip install --force-reinstall geulpu).\n")
        return 2
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
        sys.stderr.write(f"GeulPU.exe 를 실행하지 못했습니다: {e}\n")
        return 2
    try:
        return p.wait()
    finally:
        코드페이지되돌리기(cp)


if __name__ == "__main__":
    sys.exit(main())
