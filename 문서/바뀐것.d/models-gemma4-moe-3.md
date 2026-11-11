갈래: 더함

- **엔진이 Gemma 4 26B-A4B(MoE)의 k-양자화 파일을 받는다**(0.24.0): 모든 층의 dense 가지와 전문가 128 개 가운데 8 개를 계약 4.12 의
  10 대로 계산한다. 그래픽 메모리 12 GB 에서는 전문가 행렬(15.1 GB)의 일부를 VRAM 에 두고 나머지는 고정 호스트 메모리에 두는 전문가
  캐시로 돈다(GLM-4.7-Flash · Kanana-2 와 같은 길). 전문가의 gate · up 이 한 텐서인 파일과, 깊이가 256 의 배수가 아니라 Q8_0 · Q5_0 으로
  든 행렬(Q5_0 은 불러올 때 Q8_0 블록으로 — 값이 같다)을 받는다. 앱의 모델 목록은 다음 단계라, 앱의 모델 찾기는 아직 이 파일을 '실행
  가능'으로 표시하지 않는다. 기록: [구조/gemma4/기록.md](../../구조/gemma4/기록.md).

---

- **The engine takes the k-quant files of Gemma 4 26B-A4B (MoE).** Every layer's dense branch and 8 of 128 experts are computed as item 10
  of contract section 4.12 says. On a 12 GB GPU part of the expert matrices (15.1 GB) stays in VRAM and the rest in pinned host memory
  under the expert cache (the same path as GLM-4.7-Flash and Kanana-2). Files whose expert gate and up are one tensor, and matrices stored
  as Q8_0 or Q5_0 because their depth is not a multiple of 256 (Q5_0 is converted to Q8_0 blocks at load — the same values), are accepted.
  The app's model list comes in the next step, so the app's model search does not show this file as runnable yet. Record:
  [구조/gemma4/기록.md](../../구조/gemma4/기록.md).
