갈래: 고침

- **모델 찾기의 거짓 '실행됨' 표시를 고침**: 모델 찾기는 Hugging Face 의 메타데이터만 보고 저장소에 '실행됨'을 붙였고, 트렌딩 상위 50 에서
  그 23 개 가운데 14 개는 실제로 받을 수 없는 저장소였다(받는 형식의 파일이 없는 것, Gemma 4 의 E2B · E4B · 26B-A4B 변형, 분할 파일뿐인
  것). 이제 저장소의 파일 이름(받는 형식 · 분할 · unsloth 의 UD)과 Gemma 4 의 변형(매개변수 수)까지 보고, 규칙으로 아는 것은 '실행 가능',
  모르는 것은 '확인 필요'(파일을 고르면 받기 전에 앞부분을 받아 엔진에 묻는다), 받을 수 없는 것은 그 까닭으로 표시한다. 기본 목록('실행되는
  것만')은 꼬리표 qwen2 · qwen3 · llama · mistral 로 묻지 않고(Hugging Face 의 꼬리표에는 구조 이름이 거의 없어 Qwen3.5 · Qwen3.6 · Gemma 4
  저장소가 기본 목록에 나오지 않았다) 저장소마다의 구조 · 파일로 거르며, '확인 필요'인 저장소도 표시를 붙여 보인다. 빠른 점검이 2026-10-30
  조사의 트렌딩 200 저장소로 표시 규칙을 본다(시험/찾기표시.py — 옛 규칙은 31 개를 거짓으로 '실행됨', 새 규칙은 0 개).

---

- **Fixed false "Runs" labels in Find models.** Find models labelled repositories "Runs" from Hugging Face metadata alone, and in the
  trending top 50, 14 of the 23 so labelled could not be downloaded at all (no file in a supported format, the Gemma 4 E2B · E4B ·
  26B-A4B variants, or split files only). It now also reads the repository's file names (supported formats, split parts, unsloth's
  UD files) and the Gemma 4 variant (parameter count): what the rules know is "Should run", what they cannot know is "Needs a check"
  (picking a file fetches its start and asks the engine before downloading), and what cannot be loaded says why. The default list
  ("only those that run") no longer asks Hugging Face for the tags qwen2 · qwen3 · llama · mistral — Hugging Face's tags rarely
  carry the architecture, so Qwen3.5 · Qwen3.6 · Gemma 4 repositories never appeared in it — but filters each repository by its
  architecture and files, and shows "Needs a check" repositories with that mark. The quick check runs the labelling rules on the
  200 trending repositories of the 2026-10-30 survey (시험/찾기표시.py — the old rules gave 31 false "Runs", the new rules 0).
