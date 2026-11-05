#!/usr/bin/env python3
"""판실행(엔진/판실행몸.gl)의 파이썬 쪽 — 커널 모듈의 판을 CUDA(판실행 · PTX) 또는 Vulkan(판실행V · .spvs)에서 같은 명령으로 띄운다.
llama.cpp 실행의 비트 재현 3단계(문서/라마재현.md): 시험이 같은 판을 같은 입력으로 두 GPU 층에서 띄워 비트를 견준다.
자료는 작업 폴더의 파일로 주고받는다(경로에 빈칸이 없어야 한다). Vulkan 장치는 장치= 의 글이 이름에 든 첫 장치(GEULPU_VULKAN).
"""
import os
import subprocess

import numpy as np


class 판실행:
    def __init__(self, exe, 모듈, 폴더, 장치=None):
        if " " in 폴더:
            raise ValueError("작업 폴더 경로에 빈칸이 있다")
        os.makedirs(폴더, exist_ok=True)
        환경 = dict(os.environ)
        if 장치:
            환경["GEULPU_VULKAN"] = 장치
        self.폴더, self.번호 = 폴더, 0
        self.오류파일 = open(os.path.join(폴더, f"판실행_{os.getpid()}_{id(self)}.err"), "w+", encoding="utf-8")
        self.p = subprocess.Popen([exe, 모듈], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.오류파일,
                                  text=True, encoding="utf-8", env=환경)
        첫 = self.p.stdout.readline()
        if not 첫.startswith("장치 "):
            raise RuntimeError(f"판실행이 시작하지 못했다: {첫!r} {self._오류()}")
        self.장치 = 첫[3:].strip()

    def _오류(self):
        self.오류파일.flush()
        self.오류파일.seek(0)
        return self.오류파일.read()[-2000:]

    def _명령(self, 글):
        self.p.stdin.write(글 + "\n")
        self.p.stdin.flush()
        답 = self.p.stdout.readline()
        if not 답:
            raise RuntimeError(f"판실행이 끝났다 ({글[:80]}): {self._오류()}")
        if not 답.startswith("됨"):
            raise RuntimeError(f"{글[:80]}: {답.strip()}")
        return int(답.split()[1])

    def _파일(self):
        self.번호 += 1
        return os.path.join(self.폴더, f"판_{os.getpid()}_{self.번호 % 4}.bin")

    def 할당(self, 바이트):
        return self._명령(f"할당 {max(int(바이트), 4)}")

    def 올리기(self, 주소, arr):
        arr = np.ascontiguousarray(arr)
        경로 = self._파일()
        arr.tofile(경로)
        self._명령(f"올림 {주소} {arr.nbytes} {경로}")

    def 내리기(self, 주소, arr):
        경로 = self._파일()
        self._명령(f"내림 {주소} {arr.nbytes} {경로}")
        arr[...] = np.fromfile(경로, dtype=arr.dtype).reshape(arr.shape)
        return arr

    def 띄우기(self, 이름, grid, block, 값들):
        """grid · block: (x, y, z) 또는 정수. 값들: 매개변수마다 정수(주소 · 정수 · 짧은실수의 비트)."""
        g = tuple(grid) + (1,) * (3 - len(grid)) if isinstance(grid, (tuple, list)) else (grid, 1, 1)
        b = tuple(block) + (1,) * (3 - len(block)) if isinstance(block, (tuple, list)) else (block, 1, 1)
        값글 = " ".join(str(int(v) - (1 << 64) if int(v) >= (1 << 63) else int(v)) for v in 값들)
        self._명령(f"띄움 {이름} {g[0]} {g[1]} {g[2]} {b[0]} {b[1]} {b[2]} {값글}".rstrip())

    def 맞추기(self):
        """기다린 뒤 모듈의 오류 칸"""
        return self._명령("맞춤")

    def 놓기(self):
        self._명령("놓기")

    def 닫기(self):
        if self.p.poll() is None:
            self.p.stdin.write("끝\n")
            self.p.stdin.flush()
            self.p.wait(timeout=60)
        self.오류파일.close()
