#!/usr/bin/env python3
"""모델 평가 3-2(문서/모델평가.md 의 '3-2 의 판정') — 공개 과제 둘을 앱의 내장 데이터(앱/data/eval/ 의 JSONL)로 바꾼다.

  python -I -X utf8 도구/공개과제.py <GSM8K 의 test.jsonl> <MMLU-Pro 의 test-00000-of-00001.parquet> 앱/data/eval

- GSM8K test — github.com/openai/grade-school-math(커밋 3101c7d5)의 grade_school_math/data/test.jsonl, MIT(Copyright (c) 2021 OpenAI)
  → gsm8k.jsonl(1,319 문제 — 원본의 차례).
- MMLU-Pro test — huggingface.co/datasets/TIGER-Lab/MMLU-Pro(판 b189ec76)의 data/test-00000-of-00001.parquet, MIT(데이터 카드)
  → mmlu-pro-500.jsonl(14 분야를 이름 차례로 분야마다 35 문제 · 앞의 10 분야는 36 — 분야 안에서는 question_id 차례의 n 문제 중
  k = 0 … q − 1 마다 ⌊k · n / q⌋ 번째). parquet 를 읽으려면 pyarrow 가 든다(앱은 JSONL 만 읽는다).

원본의 SHA-256 이 아래와 다르면 알리고 멈춘다(같은 원본이면 바이트까지 같은 JSONL). 고지는 THIRD_PARTY_NOTICES.md.
"""
import hashlib
import json
import os
import sys

GSM8K_SHA = "3730d312f6e3440559ace48831e51066acaca737f6eabec99bccb9e4b3c39d14"
MMLU_SHA = "0e24a191921c2f453518a537a8b2117bd137e7714d4ef1565e9ba06c1ecb9ad8"
수지시 = "Solve the problem step by step, then finish your reply with 'The answer is' followed by the final number."
고름지시 = ('The following is a multiple choice question about {}. Think step by step and then finish your answer with "the answer is (X)" '
        "where X is the correct letter choice.")
MMLU_수 = 500
글자 = "ABCDEFGHIJ"


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def 쓰기(p, 줄):
    with open(p, "w", encoding="utf-8", newline="\n") as f:
        for x in 줄:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")
    print(f"{os.path.basename(p)}: {len(줄)} 줄 · {os.path.getsize(p):,} 바이트 · SHA-256 {sha(p)[:16]}…")


def gsm8k(src):
    줄 = []
    for k, l in enumerate((l for l in open(src, encoding="utf-8") if l.strip()), 1):
        x = json.loads(l)
        답 = x["answer"].split("####")[-1].strip()
        assert "####" in x["answer"] and 답, k
        줄.append({"id": f"gsm8k-{k:04d}", "category": "math", "messages": [{"role": "user", "content": x["question"].strip() + "\n\n" + 수지시}],
                  "answer": f"#### {답}", "grader": "number"})
    assert len(줄) == 1319, len(줄)
    return 줄


def mmlu_pro(src):
    import pyarrow.parquet as pq
    t = pq.read_table(src).to_pylist()
    assert len(t) == 12032, len(t)
    분야 = sorted({x["category"] for x in t})
    assert len(분야) == 14, 분야
    줄 = []
    for i, c in enumerate(분야):
        q = MMLU_수 // len(분야) + (1 if i < MMLU_수 % len(분야) else 0)
        모음 = sorted((x for x in t if x["category"] == c), key=lambda x: x["question_id"])
        n = len(모음)
        for k in range(q):
            x = 모음[k * n // q]
            assert 글자[x["answer_index"]] == x["answer"] and len(x["options"]) <= 10, x["question_id"]
            보기 = "\n".join(f"{글자[j]}. {o}" for j, o in enumerate(x["options"]))
            줄.append({"id": f"mmlu-pro-{x['question_id']}", "category": c,
                      "messages": [{"role": "user", "content": f"{고름지시.format(c)}\n\nQuestion: {x['question'].strip()}\nOptions:\n{보기}"}],
                      "answer": x["answer"], "grader": "choice"})
    assert len(줄) == MMLU_수 and len({x["id"] for x in 줄}) == MMLU_수
    return 줄


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print(__doc__)
        sys.exit(2)
    g, m, out = sys.argv[1:4]
    for p, s in ((g, GSM8K_SHA), (m, MMLU_SHA)):
        if sha(p) != s:
            sys.exit(f"원본이 다르다 — {p}: SHA-256 {sha(p)} (기대 {s})")
    쓰기(os.path.join(out, "gsm8k.jsonl"), gsm8k(g))
    쓰기(os.path.join(out, "mmlu-pro-500.jsonl"), mmlu_pro(m))
