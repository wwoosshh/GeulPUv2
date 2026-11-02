갈래: 고침

- **만든 GGUF 의 이름 · 형식 이름**: 미세조정 폴더 이름이 모두 'merged' 라서, 'GGUF 만들기'로 만든 파일이 모델 화면에서 모두 "Merged" 로 보이고
  파일 이름(merged-Q4_K_M-custom.gguf)도 무엇을 바꿨는지 말하지 않았다. 행렬별 형식으로 섞은 파일은 형식 이름도 틀렸다 — 층 0 만 Q6_K 로
  바꾼 Q4_K_M 파일이 "Q6_K" 로 보였다(#200). 이제 형식 이름은 llama-quantize 가 적은 형식(general.file_type — Q4_K_M · Q5_K_M 처럼)이고,
  만든 파일의 이름과 모델 이름(general.name)은 작업 공간 이름에서 짓는다. 행렬별 형식이면 파일 이름에 바꾼 행렬이 붙고(…-Q4_K_M+blk0-8.Q6_K.gguf)
  모델 화면이 그것을 형식 옆에 보인다. 만들기 · 버전 더하기에 메모 칸이 생겼고, '고른 무리로 만들기'는 고른 무리를 메모로 붙인다.

---

- **Names and type labels of made GGUFs.** Fine-tune folders are all called 'merged', so every file made with Make GGUF showed up as
  "Merged" on the Models screen and the file names (merged-Q4_K_M-custom.gguf) did not say what changed. Files with per-matrix types
  also had the wrong type label — a Q4_K_M file with only layers 0–8 at Q6_K showed up as "Q6_K" (#200). The type label now comes from
  the type llama-quantize wrote (general.file_type — Q4_K_M, Q5_K_M and so on), and made files take their file name and model name
  (general.name) from the workspace name. With per-matrix types the file name carries the changed matrices
  (…-Q4_K_M+blk0-8.Q6_K.gguf) and the Models screen shows them next to the type. Make GGUF and Add version have a note field, and
  'Make with the chosen groups' puts the chosen groups in the note.
