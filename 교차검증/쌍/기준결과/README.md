# Reference results — file-pair reproducibility

The CUDA check run of GeulPU 0.11.1 (commit `bbb6cb3`) on the reference PC (RTX 4070 Ti · driver 616.56 · Windows 11): every file
gave 32/32 problems with the same tokens and logit hash at every step as [`../expected.json`](../expected.json), which an earlier run
of the same build recorded. See [docs/file-pair.md](../../../docs/file-pair.md) (Korean: [문서/파일쌍.md](../../../문서/파일쌍.md)).

- `summary.json` — per file: `items` (per problem: steps `n`, `tokens_sha256`, `logit_hashes_sha256`, `result`); `pairs` (between files).
- `environment.json` — GeulPU version · commit, SHA-256 of the contract document and its CPU reference, GPU · driver · OS.
- `<label>.run.jsonl` — run log per file (format `geulpu-run-log/1`): every request (prompt token ids, settings) and the server's
  response with its per-step logit hashes. `도구/repro_check.py` re-computes it on a CPU.
- `pair-default-<B>.json` — per problem, the per-token log-probability difference (B − A) of `default`'s answer.

기준 PC 에서 GeulPU 0.11.1(커밋 `bbb6cb3`)로 견준 실행의 결과 — 파일마다 32 문제 모두 `../expected.json` 과 같았다. 자세한 것은
[문서/파일쌍.md](../../../문서/파일쌍.md).
