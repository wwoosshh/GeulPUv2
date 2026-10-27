"""GeulPU — 내 PC 의 그래픽카드로 돌리는 AI 채팅(글(geul) 언어로 쓴 엔진). PyPI 꾸러미의 실행기.

이 꾸러미는 GitHub 릴리스의 앱(GeulPU-<판>-windows-x64.zip · GeulPU-<판>-linux-x64.tar.gz 의 폴더)을 `geulpu/app/` 에 그대로 담고,
`geulpu` 명령이 그 GeulPU.exe(리눅스는 GeulPU)를 실행한다. 모델 · 이 GPU 용 커널 · 결과 · 설정은 앱이 %USERPROFILE%\\GeulPU(리눅스
~/GeulPU, 또는 GEULPU_HOME)에 둔다 — 설치 폴더에는 쓰지 않는다.
"""
import os

__version__ = "0.0.0"  # 휠을 만들 때 앱의 판으로 바꿔 쓴다(도구/pypi/휠만들기.py)


def app_dir():
    """꾸러미에 든 앱 폴더(GeulPU.exe 또는 GeulPU · bin · kernels · web · data)."""
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "app")
