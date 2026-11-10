갈래: 바꿈

- **그래픽 메모리에 다 들지 않는 MoE 모델이 빨라졌다 — 전문가 캐시**(0.24.0): 그래픽 메모리에 둘 전문가(expert)를 서버가 쉴 때가
  아니라 계산 중에 바꾼다. 생성에서는 그래픽 메모리에 없는 전문가를 기다리지 않고 RAM 에서 직접 읽으면서 자주 쓰는 전문가를
  백그라운드로 올리고, 프롬프트 처리에서는 그래픽 메모리의 임시 버퍼에 청크마다 한 번만 올려 쓴다(청크는 1,024 행). MLA 어텐션의
  헤드별 행렬곱 커널도 새로 짰다. RTX 4070 Ti(12 GB)의 앱 속도 측정에서 GLM-4.7-Flash 는 프롬프트 2,048 토큰 490 → 1,942 토큰/초 ·
  생성 23.2 → 48.8 토큰/초, Kanana-2-30B-A3B 는 476 → 1,593 · 28.3 → 41.5 토큰/초다. Qwen3.6-35B-A3B 의 프롬프트 처리도 약 1.4 배
  빨라진다. 모델을 불러온 직후나 글의 주제가 바뀐 직후에도 생성 속도가 바로 나온다(전에는 서버가 쉬면서 재배치한 뒤에야 나왔다).
  출력 비트는 그대로다. llama.cpp 의 가장 빠른 설정과 비교하면 긴 프롬프트는 1.5 ~ 1.75 배 빠르고 생성은 아직 0.6 ~ 0.85 배다.
  측정과 구조: [문서/MoE전문가캐시.md](../MoE전문가캐시.md).

---

- **MoE models that do not fit in graphics memory are faster — an expert cache.** The set of experts kept in graphics memory now
  changes during computation instead of only while the server is idle. During decoding, an expert that is not in graphics memory is
  read straight from RAM without waiting, while frequently used experts are uploaded in the background; during prompt processing,
  the missing experts are uploaded once per chunk into a staging buffer in graphics memory (chunks are now 1,024 rows). The per-head
  matrix products of MLA attention have new kernels. In the app's speed test on an RTX 4070 Ti (12 GB), GLM-4.7-Flash goes from 490
  to 1,942 tokens/s on a 2,048-token prompt and from 23.2 to 48.8 tokens/s in decoding; Kanana-2-30B-A3B from 476 to 1,593 and
  from 28.3 to 41.5. Prompt processing of Qwen3.6-35B-A3B is about 1.4 times faster. Decoding speed is there right after loading a
  model or after the topic changes (before, only after the server had rearranged experts while idle). Output bits are unchanged.
  Against llama.cpp's fastest configuration, long prompts are 1.5–1.75 times faster and decoding is still at 0.6–0.85 times.
  Measurements and design: [문서/MoE전문가캐시.md](../MoE전문가캐시.md).
