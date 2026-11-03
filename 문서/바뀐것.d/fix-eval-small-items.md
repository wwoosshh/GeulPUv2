갈래: 고침

- **모델 평가의 작은 거리들**(#220):
  - 변환 검증의 '고친 버전' 파일은 작업 공간의 made 폴더에 쓴다. 전에는 자료 폴더의 모델\(한글 경로)에 써서 llama.cpp 의 도구로 열 수 없었다.
  - 개선 측정의 보고서(`GET /api/eval/improve`) 줄마다 기준 버전 대비 KL 줄임 · 크기 차 · 1 MiB 당 줄임 · 정답률 차를 넣었다.
  - 개선 측정에 텍스트 데이터를 쓸 수 있다.
  - '크기와 품질'은 크기가 5 % 안쪽으로 크고 KL 이 0.7 배 이하인 버전(고른 무리의 파일보다 조금 큰 표준 Q5_K_M 같은 것)도 짚는다.
  - 작업 공간 견주기에 저마다의 기준 모델 줄(정답률)을 넣고, 두 결과(작업 공간이 달라도 — 기준 모델 자신도)의 짝지은 검정(A 만 맞음 · B 만 맞음 · 부호 검정 p)을 더했다.
  - 도구 상태를 불러오는 동안 '도구 준비를 먼저 하세요'가 잠깐 보이던 것을 고쳤다.
  - 변환이 붙인 뜻 없는 모델 이름(0.17.0 의 'Merged' 등)은 파일 이름으로 보인다.

---

- **Small model-evaluation fixes** (#220):
  - The conversion check's fixed-version file is written to the workspace's made folder. Before, it went to the data
    folder's 모델\ (a non-ASCII path) that llama.cpp's tools could not open.
  - Each row of the improvement report (`GET /api/eval/improve`) now has the KL reduction against the base version, the size
    difference, the reduction per MiB and the accuracy difference.
  - Text data can be used for improvement measurements.
  - Size and quality also points to a version at most 5 % larger with a KL of 0.7× or less (such as a standard Q5_K_M a
    little larger than the picked-groups file).
  - Compare workspaces shows each workspace's reference-model row (accuracy), and adds a paired test of two results
    (across workspaces too, reference models included): only A right · only B right · sign-test p.
  - "Set up tools first" no longer flashes while the tool state is loading.
  - Meaningless model names set by conversion (such as 0.17.0's "Merged") show the file name instead.
