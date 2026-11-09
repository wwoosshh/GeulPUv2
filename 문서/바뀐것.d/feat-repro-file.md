갈래: 더함

- **재현 파일을 CLI 에서도 만들고 확인**: 실행의 재현 파일을 `experiment repro` 로 만들고(그 실험의 답 전체 점수 실행에서 — 모델이 필요
  없음), `check repro --json @<파일>` 로 확인합니다(그 모델을 불러온 이 PC 에서 점수 · 다시 생성 · 지문을 비트까지, 다르면 처음 다른 스텝과
  설정의 차이). 실행 화면의 '재현 파일(JSON)'과 재현 › 검사도 이 두 기능을 씁니다 — 렌즈 실행이 파일에서 빠지거나 점수를 매긴 직후 내보내기가
  멈추던 일이 없어지고, 파일에 모델의 양자화(`model.quant`)가 적힙니다([사용성시험.md](../사용성시험.md) §3).

---

- **Make and check reproduction files from the CLI too.** `experiment repro` makes a run's reproduction file (from the experiment's
  whole-answer score run — no model needed) and `check repro --json @<file>` checks one (scoring, regeneration and fingerprint bit for bit
  on this PC with the file's model loaded; when they differ, the first differing step and the settings that differ). The run screen's
  reproduction file export and Reproduce › Checks use the same two operations, so an exported file no longer misses its logit lens
  runs or stops right after scoring, and it records the model's quantization (`model.quant`) ([사용성시험.md](../사용성시험.md) §3).
