갈래: 고침

- **Qwen3.5 · Gemma 4 의 첫 요청이 판을 재느라 늦던 것을 고침**: 추론 엔진이 모델을 올릴 때 층마다 쓰는 행렬곱의 판을 모두 미리 고르고
  판 기억에 남긴다. 전에는 Qwen3.5 의 GDN 행렬곱(Q8_0 파일) · β · α(k-양자화 파일) · 파일 안의 Q8_0 행렬 · MoE 의 라우터 · Gemma 4 의
  전체 층(Q8_0 파일)의 판을 행 수 묶음을 처음 쓸 때마다 그 자리에서 재(묶음마다 70 ~ 100 ms) 서버를 띄울 때마다 되풀이했다. 출력 비트는
  그대로다. 판 기억의 꼴이 바뀌어(고르기 32-3) 처음 한 번은 판을 다시 잰다.

---

- **The first Qwen3.5 and Gemma 4 requests no longer wait for kernel plan measurement.** The inference engine now picks the
  plans of every matrix multiplication the layers use when it loads the model, and keeps them in the plan memory. Before, the
  plans of the Qwen3.5 GDN projections (Q8_0 files), β · α (k-quant files), Q8_0 matrices inside k-quant files, the MoE router and
  the Gemma 4 full-attention layers (Q8_0 files) were measured on first use of each row bucket (70–100 ms each), again after every
  server start. Output bits are unchanged. The plan memory format changed (고르기 32-3), so plans are measured once more on the
  first start.
