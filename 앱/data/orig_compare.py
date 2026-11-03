"""GeulPU 모델 평가 2-3(문서/모델평가.md 의 '2-3 의 판정') — 기준 모델(GGUF)의 기준값을 원래 모델(transformers)과 견준다. 앱이 'GGUF 만들기'의
가상 환경(torch · transformers — 원래 모델 폴더를 저장한 transformers 의 큰 판에 맞는 것)으로 돌린다:

    python -E -s -X utf8 orig_compare.py --hf <원래 모델 폴더> --ref <기준값 폴더> --out <결과 JSON> [--dtype float32|bfloat16]
                                          [--gguf <기준 모델의 GGUF> --gguf-id <그 모델의 앱 아이디>]

기준값 폴더의 reference.json(대화마다 프롬프트 · 답의 토큰 번호)과 topk.bin(답의 위치마다 GGUF 기준 모델의 상위 K 번호 · 로그확률 · 답 토큰의
로그확률 — 8 바이트 정수 2K + 1 개)을 읽고, 같은 토큰을 원래 모델에 넣어(teacher forcing) 위치마다 견준다:

- KL divergence — GGUF 의 상위 K 와 꼬리(1 − Σp)를 기준으로 원래 모델의 같은 번호의 확률(앱의 지표와 같은 식 · 같은 방향, 바닥 10⁻¹²). 지수 ·
  로그는 파이썬의 것이라 앱의 결과 해시와는 상관없다.
- top-1 — 원래 모델이 가장 높게 본 토큰(모든 어휘)이 GGUF 의 첫째와 같은가.
- 답 토큰의 로그확률 차의 절댓값 평균.

먼저 설정을 견준다(--gguf 의 아이디가 기준값의 모델과 같을 때): GGUF 의 하이퍼파라미터(<구조>.block_count · rope.freq_base …)를 config.json 의
값(transformers 4 · 5 의 꼴 둘 다 — rope_theta · rope_parameters)과 transformers 가 읽은 값에. transformers 가 config.json 과 다르게
읽었으면(예: transformers 5 가 쓴 rope_parameters 를 4.x 가 몰라 rope_theta 의 기본값 10000 을 씀) 돌리는 모델이 저장한 모델이 아니므로 KL 을
내지 않고 멈춘다. GGUF 만 다르면 견주고 config_check 에 적는다(그 차이가 KL 에 보인다).

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

# 견줄 하이퍼파라미터 — (GGUF 키의 구조 이름 뒤, config.json · transformers 의 이름)
설정쌍 = (("block_count", "num_hidden_layers"), ("embedding_length", "hidden_size"), ("feed_forward_length", "intermediate_size"),
         ("attention.head_count", "num_attention_heads"), ("attention.head_count_kv", "num_key_value_heads"),
         ("attention.key_length", "head_dim"), ("rope.freq_base", "rope_theta"),
         ("attention.layer_norm_rms_epsilon", "rms_norm_eps"), ("vocab_size", "vocab_size"))


def 비트(x):
    return struct.unpack("<d", struct.pack("<q", x))[0]


def 수인가(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def 같은수(a, b):
    return abs(float(a) - float(b)) <= 1e-6 * max(abs(float(a)), abs(float(b)))     # GGUF 는 f32 로 적는다


def 수글(x):
    return str(int(x)) if float(x).is_integer() else f"{float(x):.9g}"


def 로프값(rp):
    """rope_parameters 의 기준값 — 층 종류마다(transformers 5 의 Gemma 따위)면 full_attention 의 것"""
    if not isinstance(rp, dict):
        return None
    if 수인가(rp.get("rope_theta")):
        return rp["rope_theta"]
    fa = rp.get("full_attention")
    return fa["rope_theta"] if isinstance(fa, dict) and 수인가(fa.get("rope_theta")) else None


def 원래설정값(cj, 이름):
    """config.json 의 값(멀티모달이면 text_config 를 먼저) — rope 는 transformers 4 의 rope_theta 와 5 의 rope_parameters 둘 다"""
    for d in (cj.get("text_config"), cj):
        if not isinstance(d, dict):
            continue
        v = d.get(이름)
        if 이름 == "rope_theta" and not 수인가(v):
            v = 로프값(d.get("rope_parameters"))
        if 수인가(v):
            return v
    return None


def 읽은설정값(cfg, 이름, 큰판):
    """transformers 가 읽어 모델에 쓰는 값 — rope 는 5.x 가 rope_parameters, 4.x 가 rope_theta"""
    v = 로프값(getattr(cfg, "rope_parameters", None)) if 이름 == "rope_theta" and 큰판 >= 5 else None
    if v is None:
        v = getattr(cfg, 이름, None)
    return v if 수인가(v) else None


def 설정견주기(hf, cfg, 큰판, gguf_path):
    """GGUF · config.json · transformers 의 하이퍼파라미터 — (줄들, transformers 가 다르게 읽은 줄들, GGUF 만 다른가, GGUF 텐서 형식의 수)"""
    from gguf import GGUFReader                     # 가상 환경의 gguf(llama.cpp 의 requirements)
    r = GGUFReader(gguf_path)

    def 키값(k):
        f = r.fields.get(k)
        if f is None:
            return None
        try:
            return f.contents()
        except Exception:
            return None

    구조 = 키값("general.architecture")
    cj = json.load(open(os.path.join(hf, "config.json"), encoding="utf-8"))
    줄들, 읽기틀림 = [], []
    gguf다름 = False
    for gk, 이름 in 설정쌍:
        if gk == "vocab_size":
            tk = r.fields.get("tokenizer.ggml.tokens")
            g = len(tk.data) if tk is not None else None
        else:
            g = 키값(f"{구조}.{gk}")
        g = g if 수인가(g) else None                   # 층마다 다른 값(배열) · 없는 키는 견주지 않는다
        o = 원래설정값(cj, 이름)
        t = 읽은설정값(cfg, 이름, 큰판)
        if sum(x is not None for x in (g, o, t)) < 2:
            continue
        줄 = {"key": gk, "config": o, "transformers": t, "gguf": g, "same": True}
        if o is not None and t is not None and not 같은수(o, t):
            줄["same"] = False
            읽기틀림.append(줄)
        기준 = o if o is not None else t
        if g is not None and 기준 is not None and not 같은수(기준, g):
            줄["same"] = False
            gguf다름 = True
        줄들.append(줄)
    형식들 = {}
    for t in r.tensors:
        형식들[t.tensor_type.name] = 형식들.get(t.tensor_type.name, 0) + 1
    return 줄들, 읽기틀림, gguf다름, 형식들, cj.get("transformers_version")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf", required=True)
    ap.add_argument("--ref", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dtype", default="float32", choices=["float32", "bfloat16"])
    ap.add_argument("--gguf")
    ap.add_argument("--gguf-id")
    ap.add_argument("--limit", type=int, default=0)      # #219 — 처음 n 개 대화만(0 이면 모두) — GSM8K 크기 결과는 CPU 로 수십 분
    a = ap.parse_args()
    import torch
    import transformers
    큰판 = int(transformers.__version__.split(".")[0])
    ref = json.load(open(os.path.join(a.ref, "reference.json"), encoding="utf-8"))
    if ref.get("data", {}).get("kind") != "chat":
        sys.exit("텍스트 데이터의 기준값은 아직 견주지 않는다(채팅 데이터만)")
    K = int(ref["top_k"])
    topk = open(os.path.join(a.ref, "topk.bin"), "rb").read()
    t0 = time.time()
    torch.set_num_threads(max(1, os.cpu_count() or 1))
    dt = torch.float32 if a.dtype == "float32" else torch.bfloat16
    model = transformers.AutoModelForCausalLM.from_pretrained(a.hf, dtype=dt)     # 4.57 부터 torch_dtype 대신 dtype
    model.eval()
    print(f"원래 모델을 읽었다 — {a.dtype} · transformers {transformers.__version__} · {time.time() - t0:.1f} 초", flush=True)
    설정 = {"status": "skip", "rows": []}
    형식들 = None
    if a.gguf and a.gguf_id and a.gguf_id == ref.get("model", {}).get("id") and os.path.isfile(a.gguf):
        cfg = model.config.get_text_config() if hasattr(model.config, "get_text_config") else model.config
        줄들, 읽기틀림, gguf다름, 형식들, 저장판 = 설정견주기(a.hf, cfg, 큰판, a.gguf)
        설정 = {"status": "gguf_differs" if gguf다름 else "same", "rows": 줄들, "saved_with": 저장판}
        if 읽기틀림:
            틀린 = " · ".join(f"{z['key']}: config.json {수글(z['config'])} · transformers {수글(z['transformers'])}" for z in 읽기틀림)
            sys.exit(f"transformers {transformers.__version__} 가 원래 모델의 설정을 config.json 과 다르게 읽었습니다 — {틀린}. 이 폴더는 "
                     f"transformers {저장판 or '?'} 로 저장했습니다 — 저장한 큰 판의 transformers 로 견주어야 합니다(KL 을 내지 않고 멈춤)")
        print(f"설정 — 하이퍼파라미터 {len(줄들)}개 · " + ("GGUF 가 원래 모델과 다름" if gguf다름 else "같음"), flush=True)
    else:
        print("설정 — 기준값을 만든 GGUF 를 찾지 못해 견주지 않음", flush=True)
    쪽 = 0
    줄 = []
    합kl = 합d = 0.0
    합top1 = 합n = 0
    큰kl = 0.0
    대화 = [e for e in ref["items"] if e.get("scored")]
    전체 = len(대화)
    if 0 < a.limit < 전체:                               # topk.bin 은 대화 차례대로라 처음 n 개는 앞쪽을 그대로 읽는다
        대화 = 대화[:a.limit]
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
    if len(대화) == 전체 and 쪽 != len(topk):
        sys.exit(f"topk.bin 의 길이가 reference.json 과 맞지 않는다({쪽} · {len(topk)})")
    if 쪽 > len(topk):
        sys.exit(f"topk.bin 이 reference.json 보다 짧다({쪽} · {len(topk)})")
    out = {"format": "geulpu-eval-original/1", "hf_dir": a.hf, "dtype": a.dtype, "torch": torch.__version__,
           "transformers": transformers.__version__, "reference_key": ref["key"], "reference_model": ref["model"], "top_k": K,
           "conversations": len(대화), "conversations_total": 전체, "positions": 합n, "kl_mean": 합kl / max(합n, 1), "kl_max": 큰kl, "top1": 합top1 / max(합n, 1),
           "answer_logprob_abs_mean": 합d / max(합n, 1), "config_check": 설정,
           "reference_types": 형식들, "reference_16bit": None if 형식들 is None else all(x in ("F32", "F16", "BF16") for x in 형식들),
           "seconds": round(time.time() - t0, 1), "finished": time.strftime("%Y-%m-%dT%H:%M:%S"), "items": 줄}
    with open(a.out, "w", encoding="utf-8", newline="\n") as f:
        json.dump(out, f, ensure_ascii=False)
    print(f"원래 모델과 견줌 — 대화 {len(대화)} · 위치 {합n} · KL {out['kl_mean']:.3g} · top-1 {out['top1'] * 100:.2f} %", flush=True)
    if len(대화) < 전체:
        print(f"처음 {len(대화)}개 대화만 견줌(모두 {전체}개)", flush=True)


if __name__ == "__main__":
    main()
