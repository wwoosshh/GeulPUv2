#!/usr/bin/env python3
"""#96 2단계 — Jacobian lens 의 J 파일(계약 5.4 — geulpu-jlens/1)을 원본(Hugging Face) 모델에서 jlens 로 맞춰 쓴다.

    python -X utf8 도구/jlens_fit.py <원본 모델 폴더> --text <글 파일> --out <모델>.jlens.safetensors [--n 100]
           [--hf-repo Qwen/Qwen3-0.6B --hf-revision <커밋>] [--checked <JSON 글 또는 @파일>]
    python -X utf8 도구/jlens_fit.py <원본 모델 폴더> --from-pt <jlens 의 lens.pt> --out … [--fit-json <기록 JSON>] [--checked …]

맞추기: jlens(anthropics/jacobian-lens)의 fit — 원천 층 0 … L − 2, 목표 층 L − 1, 프롬프트마다 앞으로 한 번 + 뒤로 ⌈D / 8⌉ 번, 앞 16 자리 뺌,
128 토큰. 프롬프트는 --text 의 앞 절반에서 빈 줄로 나눈 문단(= 로 시작하는 제목 줄 뺌)을 이어 1,200 자 이상씩 앞에서부터 n 개
(wikitext-2 의 wiki.test.raw 이면 뒤 절반을 옮겨 감의 시험에 쓴다 — 겹치지 않게). 체크포인트(<out>.ckpt)로 끊겨도 이어 한다.
--from-pt: 이미 맞춘 jlens 의 lens.pt 를 J 파일로 바꾸기만 한다. --checked: 이 J 를 견준 GGUF 의 기록(파일 SHA-256 과 일치 — 계산에 들지 않는다).

GPU 와 torch · transformers · jlens 가 필요하다(엔진 · repro_check 는 J 파일만 읽는다 — numpy).
  pip install torch transformers "jlens @ git+https://github.com/anthropics/jacobian-lens@581d398613e5602a5af361e1c34d3a92ea82ba8e"
"""
import argparse
import hashlib
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "계약"))
import jlens읽기 as JL      # noqa: E402

JLENS커밋 = "581d398613e5602a5af361e1c34d3a92ea82ba8e"


def 조각들(경로, 수, 앞절반=True, 길이=1200):
    """글 파일의 앞(뒤) 절반에서 빈 줄로 나눈 문단(= 로 시작하는 제목 줄 뺌)을 이어 길이 자 이상씩 앞에서부터 수 개."""
    t = open(경로, encoding="utf-8").read()
    반 = len(t) // 2
    t = t[:반] if 앞절반 else t[반:]
    문단 = [p.strip() for p in t.split("\n \n") if p.strip() and not p.strip().startswith("=")]
    out, 지금 = [], ""
    for p in 문단:
        지금 = (지금 + "\n" + p).strip()
        if len(지금) >= 길이:
            out.append(지금)
            지금 = ""
        if len(out) >= 수:
            break
    return out


def 파일해시(경로):
    h = hashlib.sha256()
    with open(경로, "rb") as f:
        for 덩이 in iter(lambda: f.read(1 << 24), b""):
            h.update(덩이)
    return h.hexdigest()


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description="원본 모델에서 Jacobian lens 의 J 파일(geulpu-jlens/1)을 맞춰 쓴다")
    ap.add_argument("model", help="원본 모델 폴더(Hugging Face 꼴 — config.json · safetensors)")
    ap.add_argument("--out", required=True, help="쓸 J 파일(<모델>.jlens.safetensors)")
    ap.add_argument("--text", help="프롬프트를 자를 글 파일(예: wikitext-2 의 wiki.test.raw)")
    ap.add_argument("--n", type=int, default=100, help="프롬프트 수(기본 100)")
    ap.add_argument("--from-pt", dest="from_pt", help="이미 맞춘 jlens 의 lens.pt — 바꾸기만")
    ap.add_argument("--fit-json", dest="fit_json", help="--from-pt 의 맞춘 기록(JSON 파일 — fit 에 넣는다)")
    ap.add_argument("--hf-repo", dest="hf_repo", help="원본 모델의 Hugging Face 저장소(기록)")
    ap.add_argument("--hf-revision", dest="hf_revision", help="원본 모델의 리비전(커밋 — 기록)")
    ap.add_argument("--checked", help="견준 GGUF 의 기록 — JSON 배열 글 또는 @파일")
    a = ap.parse_args()
    import torch
    import transformers
    import jlens
    설정 = json.load(open(os.path.join(a.model, "config.json"), encoding="utf-8"))
    설정 = 설정.get("text_config", 설정)
    L = int(설정["num_hidden_layers"])
    fit = {"tool": "jlens", "jlens_commit": JLENS커밋, "source_model": {"hf_repo": a.hf_repo, "revision": a.hf_revision,
           "folder": os.path.basename(os.path.normpath(a.model)), "dtype": "bfloat16"},
           "torch": torch.__version__, "transformers": transformers.__version__}
    if a.from_pt:
        lens = jlens.JacobianLens.load(a.from_pt)
        if a.fit_json:
            fit.update(json.load(open(a.fit_json, encoding="utf-8")))
        fit["from_pt_sha256"] = 파일해시(a.from_pt)
    else:
        if not a.text:
            sys.exit("--text(프롬프트를 자를 글 파일) 또는 --from-pt 가 필요하다")
        프롬프트 = 조각들(a.text, a.n)
        if len(프롬프트) < a.n:
            sys.exit(f"글 파일에서 프롬프트 {a.n} 개를 만들지 못했다({len(프롬프트)} 개)")
        tok = transformers.AutoTokenizer.from_pretrained(a.model)
        hf = transformers.AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16).cuda()
        m = jlens.from_hf(hf, tok)
        t0 = time.time()
        lens = jlens.fit(m, prompts=프롬프트, checkpoint_path=a.out + ".ckpt")
        fit.update({"prompts": {"file": os.path.basename(a.text), "sha256": 파일해시(a.text), "n": len(프롬프트),
                                "rule": "first half; blank-line paragraphs without '=' headings joined to >= 1200 chars"},
                    "max_seq_len": 128, "skip_first": 16, "dim_batch": 8, "target_layer": L - 1, "seconds": round(time.time() - t0)})
    fit["n_prompts"] = lens.n_prompts
    if lens.d_model != int(설정["hidden_size"]) or max(lens.source_layers) > L - 2:
        sys.exit(f"lens 의 너비 · 층({lens.d_model} · {lens.source_layers[-1]})이 모델({설정['hidden_size']} · {L})과 맞지 않는다")
    checked = []
    if a.checked:
        글 = open(a.checked[1:], encoding="utf-8").read() if a.checked.startswith("@") else a.checked
        checked = json.loads(글)
        assert isinstance(checked, list), "--checked 는 JSON 배열"
    J = {l: lens.jacobians[l].to(torch.float16).numpy() for l in lens.source_layers}
    JL.쓰기(a.out, J, L, {"fit": json.dumps(fit, ensure_ascii=False), "checked": json.dumps(checked, ensure_ascii=False)})
    print(f"J 파일 {a.out}: 원천 층 {lens.source_layers[0]} … {lens.source_layers[-1]} · 너비 {lens.d_model} · 프롬프트 {lens.n_prompts} · "
          f"{os.path.getsize(a.out) / 2**20:.1f} MiB · SHA-256 {JL.해시(a.out)}", flush=True)


if __name__ == "__main__":
    main()
