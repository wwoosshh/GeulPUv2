갈래: 고침

- **텍스트 결과의 PPL · BPB 를 작업 공간끼리 · 결과 표에서 견줌**: 이어서 사전학습 · 병합 · 어휘 늘리기 가운데 어느 모델의 PPL · BPB
  가 낮은지를 볼 곳이 없어 결과를 하나씩 열어야 하던 것을 고쳤다(#232). 텍스트 데이터면 '작업 공간 견주기'의 칸마다 평가할 모델의
  PPL · BPB, 기준 모델 줄에 그 모델의 PPL · BPB, 미세조정 전 모델이 있으면 그 줄(정답률 · PPL · BPB)이 보이고, 작업 공간의 결과 표에도
  PPL · BPB 칸과 기준 모델 · 미세조정 전 줄이 생긴다. CSV 에 `chunks` · `nll_ref` · `nll` · `ppl_ref` · `ppl` · `bpb_ref` · `bpb` ·
  `baseline_ppl` · `baseline_bpb` 를 더했다. 결과 화면의 미세조정 전 모델 칸에 BPB 도 보인다.
- **텍스트 결과의 청크 수는 실제로 잰 수**: 글보다 많은 청크를 요청하면(201 청크인 글에 4096) 결과의 설정에 요청한 수가 남아, 같은
  측정이 다른 설정으로 갈려 결과 표와 작업 공간 견주기에서 나란히 보이지 않던 것을 고쳤다. 이제 `settings.chunks` 는 잰 수이고 요청이
  더 컸으면 `chunks_requested` 에 남으며, 화면에는 '청크 201(요청 4096)'으로 보인다.
- **BPB 의 바이트를 정확히**: 청크 경계에서 잘린 글자를 대체 문자(3 바이트)로 세어 BPB 의 분모가 0.06 ~ 0.09 % 많던 것을 고쳤다.
  추론 서버의 `/detokenize` 가 다듬기 전 바이트 수 `n_bytes` 를 함께 돌려주고, 평가는 그것으로 나눈다(ko-heldout 64 청크 49,172 →
  49,126 바이트 — 토크나이저로 따로 센 값과 같음).

---

- **Text results' PPL · BPB compared across workspaces and in the result table.** There was no place to see which model —
  continued pretraining, merges, vocabulary extension — has the lowest PPL · BPB, so results had to be opened one by one (#232).
  For text data, each cell of Compare workspaces now shows the evaluated model's PPL · BPB, the reference-model row shows that
  model's PPL · BPB, and a before-fine-tuning row appears when there is a baseline model (accuracy · PPL · BPB); the workspace's
  result table gains PPL · BPB columns and the reference / before-fine-tuning rows. The CSV adds `chunks` · `nll_ref` · `nll` ·
  `ppl_ref` · `ppl` · `bpb_ref` · `bpb` · `baseline_ppl` · `baseline_bpb`. The result screen's before-fine-tuning cell shows BPB
  too.
- **Text results record the chunks actually measured.** Asking for more chunks than the text has (4096 for a 201-chunk text) kept
  the requested number in the result's settings, so the same measurement split into different settings and did not line up in the
  result table or Compare workspaces. `settings.chunks` is now the measured number, `chunks_requested` keeps a larger request,
  and the screen shows "201 chunks (requested 4096)".
- **Exact bytes for BPB.** Characters cut at chunk boundaries were counted as replacement characters (3 bytes), so the BPB
  denominator was 0.06–0.09 % too large. The inference server's `/detokenize` now also returns `n_bytes`, the byte count before
  cleaning, and evaluation divides by it (ko-heldout, 64 chunks: 49,172 → 49,126 bytes — the same as an independent count with the
  tokenizer).
