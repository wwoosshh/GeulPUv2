갈래: 고침

- **모델 평가의 작은 고침**(#202): 새 작업 공간 창의 기준 모델 칸이 목록의 첫 16비트 파일을 미리 고르던 것을 '없음 — GGUF 만들기로 변환'이
  기본이 되게 했다(다른 모델 · 다른 미세조정이 기준이 되면 모든 KL 이 엉뚱한 것을 잰다). '고른 무리로 만들기'의 이어서 평가는 같은 파일 · 데이터 ·
  설정으로 이미 잰 기준 버전을 다시 재지 않는다(결과가 비트까지 같다 — 사용자가 누른 평가는 전처럼 늘 잰다). imatrix 의 청크 수(청크마다
  512 토큰)가 작업 기록과 만든 버전에 남는다. 작업 공간 머리의 '기준 모델 기준 모델 없음'은 '기준 모델 아직 없음'으로, 양자화 형식 · 평가할
  버전의 체크 상자에는 이름(접근성 이름 · 값)을 달았다. 모델 평가에서 디버거로 넘긴 실행은 점수를 왜 따로 매기는지 말한다.

---

- **Small Model evaluation fixes** (#202). The reference model field of the New workspace dialog no longer preselects the first 16-bit
  file in the list — 'None — convert with Make GGUF' is the default (with another model or another fine-tune as the reference, every KL
  measures the wrong thing). The follow-up evaluation of 'Make with the chosen groups' no longer measures the base version again when
  a result with the same file · data · settings exists (results are bit-identical — evaluations you start yourself still always run).
  The number of imatrix chunks (512 tokens each) is kept in the run log and the new versions. The workspace header says 'Reference
  not set yet' instead of repeating itself, and the quantization type and evaluated version checkboxes have names (accessible name ·
  value). A run handed from Model evaluation to the debugger says why its scores are made separately.
