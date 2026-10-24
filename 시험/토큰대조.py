#!/usr/bin/env python3
"""0.12.0 12-3h-2 — 엔진의 토크나이저(엔진/시험_토큰.gl)를 llama.cpp 의 llama_tokenize 와 글마다 견준다(GPU 없이 — llama.cpp 는 어휘만 읽는다,
시험/품질.py 의 라마어휘). 두 꼴: 어휘 시험 꼴(BOS 없이 · 제어 토큰의 글을 풀지 않고 — add_special · parse_special 거짓)과 서버 꼴(시험_토큰
--특수 — 제어 토큰의 글도 풀고 BOS, 둘 다 참). 엔진 쪽은 --되풀이 로 토큰을 다시 이은 글(잇기)이 원문과 바이트까지 같은지도 본다.

  python -I 시험/토큰대조.py <시험_토큰 실행 파일> <gguf> <llama.cpp 폴더> [글 파일 …]

글: 아래의 짧은 글 열과 채팅 꼴 넷(특수 토큰이 든 글 — 서버 꼴에서 특수 토큰으로 떼어진다), 그리고 준 파일들(예: 사용안내.md · README.md ·
연구자안내.md). 끝 줄 '토큰 대조: 모두 같음' 이면 종료 0.
"""
import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

구분 = b"\n__ggml_vocab_test__\n"
짧은글 = [
    "안녕하세요, 반갑습니다! 오늘은 2026년 10월 23일입니다.",
    "I'm sure we'll see they've done it; don't you think? It's 3:45pm.",
    "1234567 12 3 0.5 1,000,000 3.14159 007 x2y34z567",
    "  앞 공백   여러 칸\n\n\n  줄 앞 공백\t탭\t\t끝  ",
    "def f(x):\n    return x ** 2  # 제곱\n\nprint(f(12345))\n",
    "😀 emoji 👍🏽 and 中文字符 与 日本語のテキスト、カタカナ。",
    "abc123def 456ghi  789\r\n윈도 줄바꿈\r\n",
    "https://example.com/path?q=1&b=%20x — em dash · middle dot … ellipsis",
    " 123  456   789\n 7",
    "한국어 문장이 길게 이어질 때: 엔진은 같은 바이트면 같은 토큰을 내야 하고, 숫자 2,048 과 8,192 와 32k 같은 표기도 흔하다. 또 '따옴표' "
    "\"큰따옴표\" (괄호) [대괄호] {중괄호} <꺾쇠> 도 섞인다.",
]
채팅꼴 = [                                                           # Gemma 4 의 채팅 틀이 그리는 꼴(<|turn> · <turn|> 는 제어, <|"|> 는 사용자 정의)
    "<bos><|turn>user\nWhat is the capital of France?<turn|>\n<|turn>model\n",
    "<|turn>system\n너는 짧게 답한다.<turn|>\n<|turn>user\n대한민국의 수도는?<turn|>\n<|turn>model\n서울입니다.<turn|>\n",
    "<|turn>model\n<|tool_call>call:get_weather{city:<|\"|>Seoul<|\"|>}<tool_call|><|tool_response>response:get_weather{temp:21}<tool_response|>",
    "글 중간의 <eos> 와 </s> 와 <unk> 그리고 <mask> — 줄바꿈\n\n\n\n과 탭\t\t도",
]


def 엔진(exe, gguf, 글들, 특수):
    with tempfile.TemporaryDirectory() as d:
        i, o = os.path.join(d, "입력.inp"), os.path.join(d, "출력.txt")
        open(i, "wb").write(b"".join(g + 구분 for g in 글들))
        r = subprocess.run([exe, gguf, i, o, "--되풀이"] + (["--특수"] if 특수 else []), capture_output=True)
        알림 = (r.stdout + r.stderr).decode("utf-8", "replace")
        if r.returncode != 0:
            raise SystemExit(f"시험_토큰 실패: {알림[-400:]}")
        줄들 = open(o, encoding="utf-8").read().split("\n")
    다름 = [int(x.split()[-1]) for x in 알림.splitlines() if x.startswith("되풀이 다름")]
    return [[int(t) for t in 줄.split()] for 줄 in 줄들[:len(글들)]], 다름


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    exe, gguf, 판 = sys.argv[1:4]
    파일들 = sys.argv[4:]
    글들 = [g.encode("utf-8") for g in 짧은글 + 채팅꼴] + [open(f, "rb").read() for f in 파일들]
    이름들 = [f"짧은 글 {k}" for k in range(len(짧은글))] + [f"채팅 꼴 {k}" for k in range(len(채팅꼴))] + [os.path.basename(f) for f in 파일들]
    import 품질 as QQ                                                 # noqa: E402 — llama.cpp 의 어휘만(vocab_only)
    import ctypes
    어휘 = QQ.라마어휘(판, gguf)
    틀림 = 0
    for 특수 in (False, True):
        얻음, 되풀이다름 = 엔진(exe, gguf, 글들, 특수)
        꼴 = "서버 꼴(특수 토큰 · BOS)" if 특수 else "어휘 시험 꼴"
        토큰수 = 다른글 = 0
        for k, (이름, b) in enumerate(zip(이름들, 글들)):
            buf = (ctypes.c_int32 * (len(b) + 16))()
            n = 어휘.L.llama_tokenize(어휘.vocab, b, len(b), buf, len(b) + 16, 특수, 특수)
            assert n >= 0, (이름, n)
            기대 = [int(x) for x in buf[:n]]
            토큰수 += n
            if 기대 != 얻음[k]:
                틀림 += 1
                다른글 += 1
                p = next((i for i in range(min(len(기대), len(얻음[k]))) if 기대[i] != 얻음[k][i]), min(len(기대), len(얻음[k])))
                print(f"  다름 {꼴} · {이름}: llama.cpp {n} 토큰 · 엔진 {len(얻음[k])} 토큰, 처음 다른 자리 {p} — llama.cpp {기대[p:p + 6]} · "
                      f"엔진 {얻음[k][p:p + 6]}")
        되풀이 = [이름들[k] for k in 되풀이다름]
        if 되풀이:
            틀림 += 1
        print(f"{꼴}: 글 {len(글들)} · 토큰 {토큰수:,} — llama.cpp 와 다른 글 {다른글} · 되풀이(잇기 = 원문) {len(글들) - len(되풀이)}/{len(글들)}"
              + (f" — 다름: {되풀이}" if 되풀이 else ""))
    print("토큰 대조: 모두 같음" if 틀림 == 0 else f"토큰 대조: 다름 {틀림}")
    sys.exit(0 if 틀림 == 0 else 1)


if __name__ == "__main__":
    main()
