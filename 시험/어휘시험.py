#!/usr/bin/env python3
"""0.12.0 12-G2 — 토크나이저를 llama.cpp 의 어휘 시험(models/ggml-vocab-*.gguf · .inp · .out — tests/test-tokenizer-0)과 견준다(GPU 없이 —
엔진/시험_토큰.gl). .out 은 Hugging Face 토크나이저가 낸 번호(llama.cpp 의 convert_hf_to_gguf_update.py)다.

  python -I 시험/어휘시험.py <시험_토큰 실행 파일> <llama.cpp 의 models 폴더> [이름 …]   (이름 없으면 폴더의 .inp 가 있는 어휘 모두)

같음 = 경우마다 번호 목록이 같다. 끝 줄 '어휘 시험: 모두 같음' 이면 종료 0.
"""
import os
import subprocess
import sys
import tempfile

sys.stdout.reconfigure(encoding="utf-8")


def main():
    exe, 폴더 = sys.argv[1:3]
    이름들 = sys.argv[3:] or sorted(f[len("ggml-vocab-"):-len(".gguf.inp")] for f in os.listdir(폴더)
                                  if f.startswith("ggml-vocab-") and f.endswith(".gguf.inp"))
    틀림 = 0
    for 이름 in 이름들:
        g = os.path.join(폴더, f"ggml-vocab-{이름}.gguf")
        기대 = [[int(x) for x in 줄.split()] for 줄 in open(g + ".out", encoding="utf-8").read().split("\n")]
        while 기대 and 기대[-1] == [] and len(기대) > 0:
            기대.pop()                                     # 끝의 빈 줄(파일 끝 줄바꿈)
        with tempfile.TemporaryDirectory() as d:
            o = os.path.join(d, "o.txt")
            r = subprocess.run([exe, g, g + ".inp", o], capture_output=True)
            알림 = (r.stdout + r.stderr).decode("utf-8", "replace").strip()
            if r.returncode != 0:
                print(f"실행 실패 {이름}: {알림[-300:]}")
                틀림 += 1
                continue
            얻음 = [[int(x) for x in 줄.split()] for 줄 in open(o, encoding="utf-8").read().split("\n")]
        while 얻음 and 얻음[-1] == [] and len(얻음) > len(기대):
            얻음.pop()
        다름 = [k for k in range(max(len(기대), len(얻음))) if (기대[k] if k < len(기대) else None) != (얻음[k] if k < len(얻음) else None)]
        for k in 다름[:3]:
            print(f"  다름 {이름} 경우 {k}: 기대 {기대[k] if k < len(기대) else None}\n{' ' * 10}얻음 {얻음[k] if k < len(얻음) else None}")
        print(f"{'같음' if not 다름 else '다름'} {이름}: {len(기대) - len(다름)}/{len(기대)}")
        틀림 += len(다름) > 0
    print("어휘 시험: 모두 같음" if 틀림 == 0 else f"어휘 시험: 다름 {틀림}")
    sys.exit(0 if 틀림 == 0 else 1)


if __name__ == "__main__":
    main()
