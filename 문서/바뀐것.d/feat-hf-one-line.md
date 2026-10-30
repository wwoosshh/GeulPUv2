갈래: 더함

- **`GeulPU --hf 저장소/파일.gguf` — Hugging Face 의 GGUF 하나를 받아 바로 불러오기(한 줄 설치)**: 이를테면
  `GeulPU --hf Qwen/Qwen3-0.6B-GGUF/Qwen3-0.6B-Q8_0.gguf`(파일의 huggingface.co 주소를 그대로 붙여도 된다). 화면의 찾기 → 확인 → 다운로드 →
  불러오기와 같은 길을 간다 — 크기 · SHA-256 을 묻고, 앞부분을 받아 GGUF 헤더와 엔진으로 확인하고, 목록에 더하고, 받고, 다 받으면
  SHA-256 을 확인한 뒤 모델을 불러온다. 진행은 콘솔에, 화면에도 같은 다운로드 · 작업이 보인다. 이미 GeulPU 가 떠 있으면 그 GeulPU 가
  받는다(`POST /api/hf/pull`). Hugging Face 의 'Use this model' 목록(huggingface.js local-apps)에 GeulPU 를 올리기 위한 한 줄이다.
  `pipx install geulpu` 뒤에는 `geulpu --hf …`.

---

- **`GeulPU --hf <repo>/<file>.gguf` — download one GGUF from Hugging Face and load it (one-line install).** For example
  `GeulPU --hf Qwen/Qwen3-0.6B-GGUF/Qwen3-0.6B-Q8_0.gguf` (a pasted huggingface.co file URL works too). It takes the same path as the
  app's finder → check → download → load: it asks for the size and SHA-256, downloads the head and checks it with the GGUF header and the
  engine, adds the file to the list, downloads it, verifies the SHA-256 and loads the model. Progress is printed to the console, and the
  app shows the same download and job. If GeulPU is already running, that instance does the work (`POST /api/hf/pull`). This is the
  one-liner for listing GeulPU in Hugging Face's "Use this model" menu (huggingface.js local-apps). After `pipx install geulpu`:
  `geulpu --hf …`.
