#!/usr/bin/env python3
"""실행 파일의 가져오기 표 — 라이브러리마다 가져오는 함수 수. 표준 라이브러리만(18a · 22c 의 조건: kernel32 · shell32 · nvcuda 뿐인가).
PE(윈도) 와 ELF(리눅스 — 29단계, 글 D-50) 둘 다 읽는다. ELF 는 DT_NEEDED 의 라이브러리마다, 정의 없는 기호의 수를 첫 라이브러리에 둔다
(ELF 의 기호는 라이브러리에 묶이지 않는다).

  python -I 시험/가져오기표.py <실행 파일>
"""
import struct
import sys


def 가져오기(경로):
    b = open(경로, "rb").read()
    if b[:4] == b"\x7fELF":
        return ELF가져오기(b)
    pe = struct.unpack_from("<I", b, 0x3C)[0]
    assert b[pe:pe + 4] == b"PE\0\0"
    n절 = struct.unpack_from("<H", b, pe + 6)[0]
    선택크기 = struct.unpack_from("<H", b, pe + 20)[0]
    선택 = pe + 24
    매직 = struct.unpack_from("<H", b, 선택)[0]
    자료목록 = 선택 + (112 if 매직 == 0x20B else 96)
    imp_rva = struct.unpack_from("<I", b, 자료목록 + 8)[0]
    절들 = []
    for i in range(n절):
        o = 선택 + 선택크기 + 40 * i
        vsz, va, rawsz, raw = struct.unpack_from("<IIII", b, o + 8)      # VirtualSize · VirtualAddress · SizeOfRawData · PointerToRawData
        절들.append((va, max(vsz, rawsz), raw))

    def 위치(rva):
        for va, sz, raw in 절들:
            if va <= rva < va + sz:
                return raw + rva - va
        raise ValueError(hex(rva))

    def 글(rva):
        o = 위치(rva)
        return b[o:b.index(b"\0", o)].decode("ascii")
    out = []
    o = 위치(imp_rva)
    while True:
        oft, _, _, name, ft = struct.unpack_from("<IIIII", b, o)
        if name == 0:
            break
        thunk = 위치(oft or ft)
        n = 0
        while struct.unpack_from("<Q" if 매직 == 0x20B else "<I", b, thunk)[0]:
            n += 1
            thunk += 8 if 매직 == 0x20B else 4
        out.append((글(name), n))
        o += 20
    return out


def ELF가져오기(b):
    """ELF64: 프로그램 머리의 PT_DYNAMIC → DT_NEEDED · DT_STRTAB · DT_SYMTAB · DT_HASH(기호 수)."""
    phoff, = struct.unpack_from("<Q", b, 32)
    phnum, = struct.unpack_from("<H", b, 56)
    올림들, 동적 = [], None
    for k in range(phnum):
        종류, _, off, va, _, filesz, _, _ = struct.unpack_from("<IIQQQQQQ", b, phoff + 56 * k)
        if 종류 == 1:
            올림들.append((va, off, filesz))
        elif 종류 == 2:
            동적 = (off, filesz)

    def 위치(va):
        for v, o, n in 올림들:
            if v <= va < v + n:
                return o + (va - v)
        raise ValueError(hex(va))
    표 = [struct.unpack_from("<qQ", b, 동적[0] + 16 * k) for k in range(동적[1] // 16)]
    값 = {}
    for 꼬리표, v in 표:
        값.setdefault(꼬리표, v)
    문자 = 위치(값[5])
    필요 = [b[문자 + v:b.index(b"\0", 문자 + v)].decode("ascii") for 꼬리표, v in 표 if 꼬리표 == 1]
    기호수 = struct.unpack_from("<I", b, 위치(값[4]) + 4)[0] - 1 if 4 in 값 else 0
    return [(이름, 기호수 if i == 0 else 0) for i, 이름 in enumerate(필요)]


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    for dll, n in 가져오기(sys.argv[1]):
        print(f"{dll} {n}")
