"""시험의 GPU 드라이버 — CUDA(시험/쿠다드라이버.py). Vulkan 흉내(GEULPU_VULKAN)는 GeulPU 연구 저장소의 탐침/vulkan 에 남겼다."""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def 드라이버():
    if os.environ.get("GEULPU_VULKAN"):
        raise SystemExit("GeulPUv2 의 시험은 CUDA 만 — Vulkan 흉내는 GeulPU 연구 저장소의 시험장치를 쓴다")
    sys.path.insert(0, HERE)
    import 쿠다드라이버 as G
    return G.드라이버()


def 한도넘음(dr, 이름):
    """Vulkan 장치의 한도로 못 돌리는 커널이면 까닭 — CUDA 드라이버는 늘 None."""
    return None
