갈래: 더함

- **결과 표 · 작업 공간 견주기**: 모델 평가의 작업 공간에 '결과 표'가 생겼다 — 데이터 · 설정을 하나 고르면 버전마다 크기 · KL · top-1 · 답이
  달라진 대화 · 정답률(95 % 구간)과 기준 모델의 정답률을 한 표에, '분야별'이면 분야마다의 정답률(채점 데이터) 또는 KL 을 보이고, 크기와
  품질(KL · 정답률)의 그림과 그 경계(더 작으면서 품질이 같거나 나은 버전이 없는 것)를 그린다. 결과는 CSV 로 내보낸다(분야마다의 줄 포함). 작업
  공간이 여럿이면 '작업 공간 견주기'가 같은 데이터 · 설정의 결과를 형식마다 나란히 보인다 — 미세조정 A 와 B 가운데 어느 쪽이 Q4_K_M 에서 덜 잃나(#198).

---

- **Result table and workspace comparison.** A Model evaluation workspace now has a Result table — choose one data/settings
  combination to see each version's size · KL · top-1 · answers that differ · accuracy (95 % interval) and the reference model's
  accuracy in one table, per-category accuracy (graded data) or KL with 'By category', and a size–quality chart (KL or accuracy) with its
  frontier (versions with no smaller version that is as good or better). Results export to CSV (with per-category rows). With several
  workspaces, 'Compare workspaces' lines up results with the same data and settings type by type — which fine-tune loses less at
  Q4_K_M (#198).
