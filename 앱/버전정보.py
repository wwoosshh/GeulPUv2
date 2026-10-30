#!/usr/bin/env python3
"""Windows 실행 파일에 버전 정보(VS_VERSIONINFO 리소스)를 넣는다 — 꾸리기(앱/꾸리기.ps1)가 GeulPU.exe 를 만든 뒤 부른다. 표준 라이브러리만(ctypes).

    python -X utf8 앱/버전정보.py <실행 파일> --판 X.Y.Z [--제품 GeulPU] [--설명 "…"] [--회사 "…"] [--저작권 "…"]

글(geul) 컴파일러는 리소스를 넣지 않아 탐색기의 속성 창에 제품 이름 · 버전이 비어 있었다. 코드 서명(SignPath — README 의 Code signing policy)은
서명할 파일의 제품 이름이 프로젝트 이름과 같고 제품 버전이 빌드마다 같아야 한다. 리소스는 Windows 의 UpdateResource 로 넣는다(파일을 다시
쓴다 — bin/ 의 서버 · 생성기 · 뒷단에는 넣지 않는다: 교차검증이 그 바이트를 기준과 견준다).
마지막 줄 '버전 정보: <파일> <판>'(종료 0), 실패하면 1.
"""
import argparse
import ctypes
import struct
import sys
from ctypes import wintypes


def _wz(s):
    return s.encode("utf-16-le") + b"\0\0"


def _pad(b):
    return b + b"\0" * ((4 - len(b) % 4) % 4)


def _block(key, value, binary, children=b""):
    """VS_VERSIONINFO 의 블록 하나: wLength · wValueLength · wType · szKey · 채움 · Value · 채움 · Children"""
    head = _pad(struct.pack("<HHH", 0, 0, 0 if binary else 1) + _wz(key))
    body = head + _pad(value) + children
    nval = len(value) if binary else len(value) // 2      # 글 값은 WCHAR 수(끝 0 포함)
    return struct.pack("<HHH", len(body), nval, 0 if binary else 1) + body[6:]


def version_info(file_ver, product_ver, strings):
    def nums(v):
        p = [int(x) for x in v.split(".")[:4]] + [0, 0, 0, 0]
        return p[0], p[1], p[2], p[3]
    f = nums(file_ver)
    p = nums(product_ver)
    fixed = struct.pack("<13I", 0xFEEF04BD, 0x00010000, (f[0] << 16) | f[1], (f[2] << 16) | f[3], (p[0] << 16) | p[1], (p[2] << 16) | p[3],
                        0x3F, 0, 0x40004, 1, 0, 0, 0)
    table = b"".join(_block(k, _wz(v), False) for k, v in strings.items())
    sfi = _block("StringFileInfo", b"", False, _block("040904B0", b"", False, table))
    vfi = _block("VarFileInfo", b"", False, _block("Translation", struct.pack("<HH", 0x0409, 0x04B0), True))
    return _block("VS_VERSION_INFO", fixed, True, sfi + vfi)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("exe")
    p.add_argument("--판", required=True, help="X.Y.Z — 제품 버전 · 파일 버전")
    p.add_argument("--제품", default="GeulPU")
    p.add_argument("--설명", default="GeulPU — a deterministic local LLM engine and debugger")
    p.add_argument("--회사", default="GeulPU (Sunghyeun Woo)")
    p.add_argument("--저작권", default="MIT License")
    a = p.parse_args()
    if not sys.platform.startswith("win"):
        print("버전 정보: 윈도에서만(ELF 에는 리소스가 없다) — 건너뜀")
        return 0
    numeric = ".".join((a.판.split("-")[0].split("+")[0].split(".") + ["0", "0", "0"])[:4])
    data = version_info(numeric, numeric, {
        "CompanyName": a.회사, "FileDescription": a.설명, "FileVersion": a.판, "InternalName": a.제품,
        "LegalCopyright": a.저작권, "OriginalFilename": a.제품 + ".exe", "ProductName": a.제품, "ProductVersion": a.판,
    })
    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    k32.BeginUpdateResourceW.restype = wintypes.HANDLE
    k32.BeginUpdateResourceW.argtypes = [wintypes.LPCWSTR, wintypes.BOOL]
    k32.UpdateResourceW.restype = wintypes.BOOL
    k32.UpdateResourceW.argtypes = [wintypes.HANDLE, wintypes.LPCWSTR, wintypes.LPCWSTR, wintypes.WORD, wintypes.LPVOID, wintypes.DWORD]
    k32.EndUpdateResourceW.restype = wintypes.BOOL
    k32.EndUpdateResourceW.argtypes = [wintypes.HANDLE, wintypes.BOOL]
    h = k32.BeginUpdateResourceW(a.exe, False)
    if not h:
        print(f"버전 정보: BeginUpdateResource 실패 (오류 {ctypes.get_last_error()}) — {a.exe}")
        return 1
    buf = ctypes.create_string_buffer(data, len(data))
    rt_version = ctypes.cast(ctypes.c_void_p(16), wintypes.LPCWSTR)      # RT_VERSION
    res_id = ctypes.cast(ctypes.c_void_p(1), wintypes.LPCWSTR)           # 리소스 번호 1
    ok = k32.UpdateResourceW(h, rt_version, res_id, 0x0409, buf, len(data))
    if not ok:
        err = ctypes.get_last_error()
        k32.EndUpdateResourceW(h, True)
        print(f"버전 정보: UpdateResource 실패 (오류 {err}) — {a.exe}")
        return 1
    if not k32.EndUpdateResourceW(h, False):
        print(f"버전 정보: EndUpdateResource 실패 (오류 {ctypes.get_last_error()}) — {a.exe}")
        return 1
    print(f"버전 정보: {a.exe} {a.판}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
