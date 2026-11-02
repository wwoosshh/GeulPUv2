갈래: 더함

- **개선 측정**: 모델 평가의 작업 공간에서 양자화한 버전을 기준으로, 행렬 무리(attn_q · attn_k · attn_v · attn_output · ffn_gate+ffn_up ·
  ffn_down · token_embd · output)와 층의 세 토막마다 그 무리만 높은 형식(Q6_K · Q5_K · BF16)으로 바꾼 시험 파일을 만들어 같은 데이터로
  평가하고, 무리마다 KL divergence 가 얼마나 주는지와 파일이 얼마나 커지는지를 보고서로 보인다(시험 파일은 잰 뒤 지운다). 보고서에서 고른
  무리로 다시 만들면 기준 버전과 함께 이어서 평가한다.

---

- **Improvement measurement.** In a Model evaluation workspace, taking a quantized version as the base, each matrix group (attn_q · attn_k ·
  attn_v · attn_output · ffn_gate+ffn_up · ffn_down · token_embd · output) and each third of the layers gets a trial file with only that
  group at a higher type (Q6_K · Q5_K · BF16), evaluated on the same data; a report shows how much each group lowers the KL divergence and
  how much the file grows (trial files are deleted after measuring). Remaking with the chosen groups evaluates the new version together
  with the base version.
