"""RoPE 표 시험(GPU 없이) — 엔진/시험_로프표(로프.gl)가 만든 표를 CPU 참조(계약/정수계약.rope표)와 바이트까지 견준다: GeulPUv2 ③a 의
YaRN(계약 4.8 — 배 2 · 4)과 늘이지 않은 표(배 0 — 지금 그대로인가). 머리 128(반 64) · θ 10⁶ · 원래 32,768(Qwen2.5 · Qwen3). 0.13.0 — 넓은 표
둘(Gemma 4: 윈도 층 반 128 · θ 10⁴, 전체 층 반 256 · θ 10⁶ · rope_freqs 1 이 64 칸 · 10³⁰ 이 192 칸 — 작은 각의 길)도. 빠른점검이 부른다.

    python -I 시험/로프표.py <시험_로프표 실행 파일> [--길이 0:32768,2:65536,4:131072] [--넓은길이 4096]
"""
import argparse
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.join(ROOT, "계약"))
import 정수계약 as C        # noqa: E402

반, 밑, 원래 = 64, 1e6, 32768


def main():
    p = argparse.ArgumentParser()
    p.add_argument("exe")
    p.add_argument("--길이", default="0:32768,2:65536,4:131072")
    p.add_argument("--넓은길이", type=int, default=4096)
    a = p.parse_args()
    경우 = [tuple(int(v) for v in x.split(":")) for x in a.길이.split(",")]
    틀림 = []
    뿌리 = tempfile.mkdtemp(prefix="로프표_")
    try:
        for 배, n in 경우:
            out = os.path.join(뿌리, f"표_{배}.bin")
            r = subprocess.run([os.path.abspath(a.exe), str(n), repr(밑), str(반), str(배), str(원래), out], capture_output=True, timeout=600)
            줄 = r.stdout.decode("utf-8", "replace").strip()
            if r.returncode != 0:
                틀림.append(f"배 {배}: 종료 {r.returncode} {줄[-200:]} {r.stderr.decode('utf-8', 'replace')[-200:]}")
                continue
            b = open(out, "rb").read()
            바이트 = n * 반 * 4
            if len(b) != 2 * 바이트 + 8 * 반 + 8:
                틀림.append(f"배 {배}: 출력 크기 {len(b)}")
                continue
            ec = np.frombuffer(b[:바이트], np.float32).reshape(n, 반)
            es = np.frombuffer(b[바이트:2 * 바이트], np.float32).reshape(n, 반)
            eg = np.frombuffer(b[2 * 바이트:2 * 바이트 + 8 * 반], np.float64)
            em = np.frombuffer(b[2 * 바이트 + 8 * 반:], np.float64)[0]
            얀 = (배, 원래) if 배 else None
            rc, rs = C.rope표(n, 밑, hd=2 * 반, 얀=얀)
            if 얀:
                rg, rm = C.얀인자(2 * 반, 밑, 배, 원래)
            else:
                rg, rm = np.ones(반), 1.0
            gc = int((ec.view(np.uint32) != rc.view(np.uint32)).sum())
            gs = int((es.view(np.uint32) != rs.view(np.uint32)).sum())
            gg = int((eg.view(np.uint64) != np.asarray(rg, np.float64).view(np.uint64)).sum())
            gm = np.float64(em).view(np.uint64) != np.float64(rm).view(np.uint64)
            요약 = f"배 {배} · 위치 {n:,} — cos 다름 {gc} · sin 다름 {gs} · g 다름 {gg} · m {'다름' if gm else '같음'}({em!r})"
            if 얀:
                바뀐 = [i for i in range(반) if 0.0 < 1.0 - rg[i] < 1.0 - 1.0 / 배]
                요약 += f" · 섞인 짝 {바뀐[0]}…{바뀐[-1]}" if 바뀐 else ""
            print(요약 + f" · {줄.split(' · ')[-1]}", flush=True)
            if gc or gs or gg or gm:
                틀림.append(요약)
        # 0.13.0 — 넓은 표(로프표짓기 — 칸 수에 묶이지 않는다): Gemma 4 의 윈도 층 · 전체 층(rope_freqs 10³⁰ — 작은 각은 바로 정한다)
        n = a.넓은길이
        rf = np.concatenate([np.ones(64, np.float32), np.full(192, 1e30, np.float32)])
        for 이름, θ, 반w, 인자 in (("윈도(반 128 · θ 10⁴)", 1e4, 128, None), ("전체(반 256 · θ 10⁶ · rope_freqs)", 1e6, 256, rf)):
            out = os.path.join(뿌리, f"넓은_{반w}.bin")
            rff = "-"
            if 인자 is not None:
                rff = os.path.join(뿌리, f"rf_{반w}.bin")
                인자.tofile(rff)
            r = subprocess.run([os.path.abspath(a.exe), str(n), repr(θ), str(반w), "0", str(원래), out, rff], capture_output=True, timeout=600)
            줄 = r.stdout.decode("utf-8", "replace").strip()
            if r.returncode != 0:
                틀림.append(f"{이름}: 종료 {r.returncode} {줄[-200:]} {r.stderr.decode('utf-8', 'replace')[-200:]}")
                continue
            b = open(out, "rb").read()
            바이트 = n * 반w * 4
            ec = np.frombuffer(b[:바이트], np.float32).reshape(n, 반w)
            es = np.frombuffer(b[바이트:2 * 바이트], np.float32).reshape(n, 반w)
            rc, rs = C.rope표(n, θ, hd=2 * 반w, 인자=인자)
            gc = int((ec.view(np.uint32) != rc.view(np.uint32)).sum())
            gs = int((es.view(np.uint32) != rs.view(np.uint32)).sum())
            요약 = f"{이름} · 위치 {n:,} — cos 다름 {gc} · sin 다름 {gs}"
            print(요약 + f" · {줄.split(' · ')[-2]} · {줄.split(' · ')[-1]}", flush=True)
            if gc or gs:
                틀림.append(요약)
    finally:
        shutil.rmtree(뿌리, ignore_errors=True)
    if 틀림:
        print("RoPE 표(GPU 없이): 못 미침 — " + " / ".join(틀림))
        return 1
    print("RoPE 표(GPU 없이): 통과 — 엔진의 표(늘이지 않음 · YaRN 배 2 · 4 · 넓은 표 둘)가 CPU 참조와 바이트까지 같다")
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.exit(main())
