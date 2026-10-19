"""파일을 OS 파일 캐시에서 뺀다(관리자 권한 · 재부팅 없이) — 캐시 없이(FILE_FLAG_NO_BUFFERING) 한 번 열고 닫으면 NTFS 가 그 파일의 캐시를
비운다(다른 프로세스가 사상해 두지 않았을 때). 문맥 저장소의 '차가운 SSD' 재기에 쓴다(문서/문맥저장소.md 4단계).

    python 재기/캐시비우기.py <파일이나 폴더 …>            # 폴더는 그 안의 파일 모두(하위 폴더까지)
    python 재기/캐시비우기.py --재기 <파일> [MB]           # 확인: 읽기(데움) → 다시 읽기(캐시) → 비우기 → 다시 읽기(디스크)
"""
import ctypes
import os
import sys
import time
from ctypes import wintypes

k = ctypes.WinDLL("kernel32", use_last_error=True)
k.CreateFileW.restype = wintypes.HANDLE
k.CreateFileW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD, ctypes.c_void_p, wintypes.DWORD, wintypes.DWORD,
                          wintypes.HANDLE]
k.CloseHandle.argtypes = [wintypes.HANDLE]
없음 = wintypes.HANDLE(-1).value


def 비우기(경로):
    h = k.CreateFileW(경로, 0x80000000, 0x1 | 0x2, None, 3, 0x20000000, None)     # GENERIC_READ · 공유 읽기 쓰기 · OPEN_EXISTING · NO_BUFFERING
    if h is None or h == 없음:
        return False
    k.CloseHandle(h)
    return True


def 파일들(경로들):
    for p in 경로들:
        if os.path.isdir(p):
            for 뿌리, _, 이름들 in os.walk(p):
                for n in 이름들:
                    yield os.path.join(뿌리, n)
        else:
            yield p


def 읽기(경로, 바이트):
    칸 = bytearray(8 << 20)
    t0 = time.perf_counter()
    남음 = 바이트
    with open(경로, "rb", buffering=0) as f:
        while 남음 > 0:
            n = f.readinto(memoryview(칸)[:min(남음, len(칸))])
            if not n:
                break
            남음 -= n
    return time.perf_counter() - t0


if __name__ == "__main__":
    if sys.argv[1] == "--재기":
        p = sys.argv[2]
        mb = int(sys.argv[3]) if len(sys.argv) > 3 else 1024
        n = min(mb << 20, os.path.getsize(p))
        for 이름 in ("처음", "캐시"):
            t = 읽기(p, n)
            print(f"{이름}: {n / t / 1e9:.2f} GB/s ({t:.3f} 초)")
        비우기(p)
        t = 읽기(p, n)
        print(f"비운 뒤: {n / t / 1e9:.2f} GB/s ({t:.3f} 초)")
    else:
        수 = sum(1 for p in 파일들(sys.argv[1:]) if 비우기(p))
        print(f"캐시에서 뺀 파일 {수} 개")
