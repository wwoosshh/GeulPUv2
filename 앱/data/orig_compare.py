"""GeulPU 모델 평가 2-3(문서/모델평가.md 의 '2-3 의 판정') — 기준 모델(GGUF)의 기준값을 원래 모델(transformers)과 견준다. 앱이 'GGUF 만들기'의
가상 환경(torch · transformers)으로 돌린다:

    python -E -s -X utf8 orig_compare.py --hf <원래 모델 폴더> --ref <기준값 폴더> --out <결과 JSON> [--dtype float32|bfloat16]

기준값 폴더의 reference.json(대화마다 프롬프트 · 답의 토큰 번호)과 topk.bin(답의 위치마다 GGUF 기준 모델의 상위 K 번호 · 로그확률 · 답 토큰의
로그확률 — 8 바이트 정수 2K + 1 개)을 읽고, 같은 토큰을 원래 모델에 넣어(teacher forcing) 위치마다 견준다:

- KL divergence — GGUF 의 상위 K 와 꼬리(1 − Σp)를 기준으로 원래 모델의 같은 번호의 확률(앱의 지표와 같은 식 · 같은 방향, 바닥 10⁻¹²). 지수 ·
  로그는 파이썬의 것이라 앱의 결과 해시와는 상관없다.
- top-1 — 원래 모델이 가장 높게 본 토큰(모든 어휘)이 GGUF 의 첫째와 같은가.
- 답 토큰의 로그확률 차의 절댓값 평균.

원래 모델은 CPU 에서 dtype 으로 계산한다(float32 가 기본 — 가중치가 bfloat16 이어도 정확히 늘려 계산). 결과는 --out 의 JSON 하나.
"""
import argparse
import json
import math
import os
import struct
import sys
import time

바닥 = 1e-12


def 비트(x):
    return struct.unpack("<d", struct.pack("<q", x))[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf", required=True)
    ap.add_argument("--ref", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dtype", default="float32", choices=["float32", "bfloat16"])
    a = ap.parse_args()
    import torch
    import transformers
    ref = json.load(open(os.path.join(a.ref, "reference.json"), encoding="utf-8"))
    if ref.get("data", {}).get("kind") != "chat":
        sys.exit("텍스트 데이터의 기준값은 아직 견주지 않는다(채팅 데이터만)")
    K = int(ref["top_k"])
    topk = open(os.path.join(a.ref, "topk.bin"), "rb").read()
    t0 = time.time()
    torch.set_num_threads(max(1, os.cpu_count() or 1))
    dt = torch.float32 if a.dtype == "float32" else torch.bfloat16
    model = transformers.AutoModelForCausalLM.from_pretrained(a.hf, torch_dtype=dt)
    model.eval()
    print(f"원래 모델을 읽었다 — {a.dtype} · {time.time() - t0:.1f} 초", flush=True)
    쪽 = 0
    줄 = []
    합kl = 합d = 0.0
    합top1 = 합n = 0
    큰kl = 0.0
    대화 = [e for e in ref["items"] if e.get("scored")]
    for k, e in enumerate(대화, 1):
        p, ans = e["prompt"], e["answer"]
        m = len(ans)
        ids = torch.tensor([p + ans[:-1]], dtype=torch.long)
        with torch.no_grad():
            lg = model(input_ids=ids).logits[0, len(p) - 1:, :]
        kl_i = d_i = 0.0
        t1_i = 0
        for a0 in range(0, m, 64):
            lp = torch.log_softmax(lg[a0:a0 + 64].double(), dim=-1)
            for r in range(lp.shape[0]):
                i = a0 + r
                v = struct.unpack_from(f"<{2 * K + 1}q", topk, 쪽)
                쪽 += (2 * K + 1) * 8
                row = lp[r]
                머리 = ps = qs = 0.0
                for j in range(K):
                    t = v[j]
                    if t < 0:
                        continue
                    lr = 비트(v[K + j])
                    lq = float(row[t])
                    pj = math.exp(lr)
                    머리 += pj * (lr - lq)
                    ps += pj
                    qs += math.exp(lq)
                pt = 1.0 - ps
                qt = 1.0 - qs
                kl = 머리 if pt < 바닥 else 머리 + pt * (math.log(pt) - math.log(max(qt, 바닥)))
                kl_i += kl
                큰kl = max(큰kl, kl)
                t1_i += int(int(torch.argmax(row)) == v[0])
                d_i += abs(비트(v[2 * K]) - float(row[ans[i]]))
        줄.append({"id": e["id"], "positions": m, "kl_mean": kl_i / max(m, 1), "top1": t1_i / max(m, 1)})
        합kl += kl_i
        합d += d_i
        합top1 += t1_i
        합n += m
        print(f"{k}/{len(대화)} {e['id']} — 위치 {m} · KL {kl_i / max(m, 1):.6f}", flush=True)
    if 쪽 != len(topk):
        sys.exit(f"topk.bin 의 길이가 reference.json 과 맞지 않는다({쪽} · {len(topk)})")
    out = {"format": "geulpu-eval-original/1", "hf_dir": a.hf, "dtype": a.dtype, "torch": torch.__version__,
           "transformers": transformers.__version__, "reference_key": ref["key"], "reference_model": ref["model"], "top_k": K,
           "conversations": len(대화), "positions": 합n, "kl_mean": 합kl / max(합n, 1), "kl_max": 큰kl, "top1": 합top1 / max(합n, 1),
           "answer_logprob_abs_mean": 합d / max(합n, 1), "seconds": round(time.time() - t0, 1),
           "finished": time.strftime("%Y-%m-%dT%H:%M:%S"), "items": 줄}
    with open(a.out, "w", encoding="utf-8", newline="\n") as f:
        json.dump(out, f, ensure_ascii=False)
    print(f"원래 모델과 견줌 — 대화 {len(대화)} · 위치 {합n} · KL {out['kl_mean']:.3g} · top-1 {out['top1'] * 100:.2f} %", flush=True)


if __name__ == "__main__":
    main()
