갈래: 고침

- **작업 공간 견주기에 같은 형식의 버전이 모두 보임**: 한 작업 공간에 같은 형식의 버전이 여럿이면(imatrix 없이 · 글 둘로 만든
  Q4_K_M 처럼) 견주기의 칸에는 가장 나중에 평가한 버전 하나만 어느 버전인지 밝히지 않고 보였고, 짝지은 검정에서도 나머지는 고를 수
  없었다. 이제 칸마다 버전과 메모를 보이고 같은 줄의 버전은 모두 칸에 들며, 앱이 imatrix 로 만든 버전은 그 글마다 줄을 따로 둔다(예:
  "Q4_K_M · imatrix ko_imatrix.txt"). 짝지은 검정은 그 데이터 · 설정의 결과가 있는 버전을 모두 고를 수 있다.
- **텍스트 결과의 설정 이름 · 묶음 · 예전 BPB**: 여러 결과를 묶는 설정의 이름은 잰 청크 수만 보인다("청크 201" — 요청한 수는 결과마다
  다를 수 있다). 0.19.0 의 텍스트 결과(설정에 요청한 청크 수를 적음)도 잰 청크 수로 묶여 같은 설정의 새 결과와 함께 보인다. 0.20.0
  앞에 만든 텍스트 결과의 BPB 에는 * 와 까닭(예전 방식의 바이트 수 — 다시 평가하면 맞춰짐)을 붙이고, 기준 모델 · 미세조정 전 줄은 가장
  나중의 결과를 쓴다. 작업 공간 목록(`GET /api/eval`)의 결과마다 그 결과를 만든 앱의 판(`app`)이 있고, CSV 끝에 `app` 칸을 더했다.

---

- **Compare workspaces shows every version of a type.** When a workspace had several versions of one type (Q4_K_M made
  without an imatrix and with two texts, say), the comparison cell showed only the last-evaluated one without saying which,
  and the paired test could not pick the others. Each cell now shows the version and its note, every version of the row is in
  the cell, and versions the app made with an imatrix get a row per imatrix text (for example "Q4_K_M · imatrix
  ko_imatrix.txt"). The paired test can pick every version with a result for that data and setting.
- **Setting names, grouping and old BPB for text results.** The name of a setting that groups several results shows only the
  measured number of chunks ("201 chunks" — the requested number can differ per result). Text results from 0.19.0 (whose
  settings recorded the requested number of chunks) are grouped by the measured number too, together with new results of the
  same setting. BPB from text results made before 0.20.0 carries a * and the reason (bytes counted the old way — evaluating
  again corrects it), and the reference-model and before-fine-tuning rows use the latest result. Each result in the workspace
  list (`GET /api/eval`) has the app version that made it (`app`), and the CSV gains an `app` column at the end.
