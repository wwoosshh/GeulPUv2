// 모델 평가(문서/모델평가.md — 1-3) — 작업 공간(모델 하나의 작업: 기준 모델 · 버전들 · 결과)을 왼쪽에, 고른 것을 오른쪽에: 작업 공간 ·
// 결과 · 전후 비교 · 답이 달라진 곳. 넓은 창은 두 칸이 저마다 스크롤하고(화면은 고정), 좁은 창은 위아래로 놓인다(디버거와 같은 꼴).
// app.js 다음에 읽힌다 — app.js 의 도움(h · T · api · modal …)과 상태(S)를 쓴다. 영어 문구는 아래 사전(app.js 의 DICT 에 더한다)
const EVAL_DICT = {
  "끝 토큰": "End token",
  "도구 호출": "Tool call",
  "기준 모델 없음": "No reference model",
  "먼저 모델 화면에서 GGUF 파일을 더하세요": "Add a GGUF file on the Models screen first",
  "새 작업 공간": "New workspace",
  "기준 모델": "Reference model",
  "작업 공간 하나는 모델 하나의 작업입니다 — 기준 모델과, 그와 비교할 GGUF 파일(버전)들과 결과를 묶습니다.": "A workspace holds the work on one model — a reference model, the GGUF files (versions) compared with it, and the results.",
  "이름": "Name",
  "같은 모델의 BF16 · F16 파일이 가장 좋습니다(그래픽 메모리에 들지 않으면 Q8_0). 평가할 파일은 모두 이 모델과 비교합니다.": "A BF16 or F16 file of the same model is best (Q8_0 if it does not fit in graphics memory). Every evaluated file is compared with this model.",
  "만들기": "Create",
  "고를 수 있는 모델이 없습니다": "No model to choose",
  "기준 모델 바꾸기": "Change reference model",
  "지난 결과는 그대로 남지만, 기준 모델이 다른 결과끼리는 전후 비교를 하지 않습니다.": "Earlier results stay, but results with different reference models are not compared before/after.",
  "작업 공간 지우기": "Delete workspace",
  "{} — 목록에서 지웁니다. 결과 파일은 자료 폴더의 평가\\지운작업공간\\ 으로 옮겨 둡니다(모델 파일은 그대로).": "{} — removed from the list. Its result files are moved to the deleted-workspaces folder in the data folder (model files are not touched).",
  "더할 수 있는 모델이 없습니다 — 모델 화면에서 GGUF 파일을 더하세요": "No model to add — add a GGUF file on the Models screen",
  "평가할 GGUF": "GGUF to evaluate",
  "버전 더하기": "Add version",
  "기준 모델과 비교할 GGUF 파일을 고르세요. 파일은 옮기지 않고 앱의 모델 목록에 있는 파일을 그대로 씁니다.": "Choose a GGUF file to compare with the reference model. The file is not moved — the app uses the file in its model list.",
  "더하기": "Add",
  "버전 지우기": "Remove version",
  "{} — 작업 공간에서 뺍니다. 모델 파일과 지난 결과는 그대로 남습니다.": "{} — removed from the workspace. The model file and earlier results stay.",
  "비우면 파일 이름": "File name if empty",
  "데이터를 불러왔습니다 — {}": "Data imported — {}",
  "데이터 불러오기": "Import data",
  "JSONL — 줄마다 대화 하나: {\"messages\": […]}, 고를 수 있는 칸 id · category · tools · max_tokens · thinking, 정답을 채점하려면 answer와 grader(number · choice · exact · tool_call). TXT — 텍스트 하나를 문맥 512의 청크로 나눠 llama-perplexity와 같은 위치를 점수 매깁니다.": "JSONL — one conversation per line: {\"messages\": […]}, optional fields id · category · tools · max_tokens · thinking, and answer with grader (number · choice · exact · tool_call) to score correctness. TXT — one text split into context-512 chunks, scored at the same positions as llama-perplexity.",
  "파일 경로(.jsonl · .txt)": "File path (.jsonl · .txt)",
  "이름(영문 · 숫자 · . _ -)": "Name (letters · digits · . _ -)",
  "GeulPU가 쓴 데이터": "Written by GeulPU",
  "내장|data": "Built-in",
  // 4-1 — GGUF 만들기
  "GGUF 만들기 도구 준비": "Set up GGUF making tools",
  "GGUF 만들기": "Make GGUF",
  "GGUF 만들기와 변환 검증이 이 폴더를 씁니다. 나중에 정해도 됩니다.": "Make GGUF and the conversion check use this folder. You can set it later.",
  "GGUF를 만들려면 먼저 도구를 받습니다 — llama.cpp의 실행 파일 · 변환 스크립트 · 파이썬 패키지.": "To make GGUF files, first get the tools — llama.cpp executables · conversion scripts · Python packages.",
  "imatrix에 쓸 파일을 정하세요": "Choose the file for the imatrix",
  "imatrix에 쓸 파일": "File for the imatrix",
  "llama.cpp {} 실행 파일 — {}": "llama.cpp {} executables — {}",
  "도구 — llama.cpp {} · 파이썬 {} · torch {} · transformers {}": "Tools — llama.cpp {} · Python {} · torch {} · transformers {}",
  "도구 준비": "Set up tools",
  "transformers {} 환경 — transformers 5로 저장한 모델 폴더용(torch는 위의 것을 함께 씀)": "transformers {} environment — for model folders saved with transformers 5 (shares torch with the one above)",
  "약 30 MB(설치하면 200 MB 안쪽)": "About 30 MB (under 200 MB installed)",
  "도구 — llama.cpp {} · 파이썬 {} · torch {} · transformers {} · {}": "Tools — llama.cpp {} · Python {} · torch {} · transformers {} · {}",
  "transformers 5로 저장한 모델 폴더를 변환하려면 도구 준비를 다시 하세요 — transformers 5 환경을 더합니다(약 30 MB).": "To convert model folders saved with transformers 5, set up the tools again — this adds a transformers 5 environment (about 30 MB).",
  "받고 설치": "Download and install",
  "변환 {}": "Converted {}",
  "변환 스크립트(convert_hf_to_gguf.py 등) {}개": "{0} conversion script {0|file|files} (convert_hf_to_gguf.py and others)",
  "변환 형식": "Conversion type",
  "비우면 자동(py -3 · python · python3)": "Automatic if empty (py -3 · python · python3)",
  "약 150 ~ 250 MB(설치하면 1 GB 안쪽)": "about 150–250 MB (under 1 GB installed)",
  "없음 — GGUF 만들기로 원래 모델에서 변환": "None — convert from the original model with Make GGUF",
  "원래 모델 폴더(config.json과 가중치 .safetensors가 있는 폴더)": "Original model folder (the folder with config.json and the .safetensors weights)",
  "원래 모델 폴더(고름 — Hugging Face 형식)": "Original model folder (optional — Hugging Face format)",
  "원래 모델(Hugging Face 형식)에서 — llama.cpp의 변환 · 양자화": "From the original model (Hugging Face format) — llama.cpp conversion · quantization",
  "원래 모델(Hugging Face 형식)을 GGUF로 바꾸고 양자화하는 데 llama.cpp의 도구를 씁니다. 아래 파일을 받아 자료 폴더의 tools에 둡니다 — 앱이 정한 판이고, 받은 파일마다 SHA-256을 확인합니다.": "Converting the original model (Hugging Face format) to GGUF and quantizing it uses llama.cpp's tools. The files below go to tools in the data folder — a version fixed by the app, and the SHA-256 of every file is checked.",
  "이 PC에서 파이썬을 찾지 못했습니다 — python.org에서 파이썬 3.10 이상을 설치하거나(설치할 때 Add python.exe to PATH), 아래에 경로를 적으세요.": "No Python was found on this PC — install Python 3.10 or later from python.org (check Add python.exe to PATH during setup), or enter its path below.",
  "자료 폴더의 경로에 ASCII가 아닌 글자가 있어 llama.cpp의 도구를 쓸 수 없습니다 — ASCII 경로의 자료 폴더로 실행하세요.": "The data folder path has non-ASCII characters, so llama.cpp's tools cannot use it — run with a data folder on an ASCII path.",
  "파이썬 경로": "Python path",
  "파이썬 경로(고름)": "Python path (optional)",
  "파이썬 패키지 — {}(CPU) · transformers 등": "Python packages — {} (CPU) · transformers and others",
  "파이썬은 이 PC의 것으로 앱 전용 가상 환경만 만듭니다 — 찾은 파이썬: {}": "Python itself is not downloaded — the app makes its own virtual environment with this PC's Python. Found: {}",
  "행렬별 {}": "per matrix {}",
  "행렬별 형식은 양자화 형식과 함께 씁니다": "Per-matrix types need a quantization type",
  "변환한 파일은 새 버전이 됩니다(기준 모델이 이미 있음).": "The converted file becomes a new version (the workspace already has a reference model).",
  "변환한 파일이 이 작업 공간의 기준 모델이 됩니다.": "The converted file becomes this workspace's reference model.",
  "양자화 형식(여럿 고름 — 고르지 않으면 변환만)": "Quantization types (choose any — none means convert only)",
  "글 파일로 만들기": "Make from a text file",
  "imatrix 파일": "imatrix file",
  "글 파일의 문장으로 행렬마다 중요한 가중치를 잽니다(llama-imatrix — GPU). 모델이 쓰일 글과 비슷할수록 좋습니다.": "Measures which weights matter in each matrix from the text file's sentences (llama-imatrix — GPU). Text close to the model's real use works best.",
  "행렬별 형식(고급)": "Per-matrix types (advanced)",
  "행렬별 형식": "Per-matrix types",
  "줄마다 행렬 이름=형식(llama-quantize --tensor-type). 엔진이 받지 않는 섞임이면 버전에 까닭이 보입니다.": "One matrix name=type per line (llama-quantize --tensor-type). If the engine does not take the mix, the version shows why.",
  "만들기|make": "Make",
  "기준 모델에서 만듦": "made from the reference model",
  "엔진이 받지 않음 — {}": "Not supported by the engine — {}",
  "기준 모델과 비교할 GGUF 파일을 더하거나, 아래 GGUF 만들기로 원래 모델에서 만드세요.": "Add GGUF files to compare with the reference model, or make them from the original model with Make GGUF below.",
  "양자화": "Quantized",
  "양자화 형식을 하나 이상 고르세요": "Choose at least one quantization type",
  "출발": "Start from",
  "기준 모델 {}을 그대로 두고 양자화한 파일만 새 버전으로 — 행렬별 형식을 바꿔 다시 만들 때.": "Keeps the reference model {} and adds only the quantized files as new versions — for remaking with different per-matrix types.",
  "양자화 형식(여럿 고름)": "Quantization types (choose any)",
  // 2-3 — 원래 모델과 견주기
  "원래 모델과 견주기": "Compare with the original model",
  "계산 형식": "Compute type",
  "KL divergence {} · top-1 일치율 {} · 위치 {}개 · 대화 {}개": "KL divergence {0} · top-1 agreement {1} · {2} {2|position|positions} · {3} {3|conversation|conversations}",
  "답 토큰의 로그확률 차(절댓값 평균) {} · 가장 큰 KL {} · {} · transformers {} · {}": "Answer-token log-probability difference (mean absolute) {} · largest KL {} · {} · transformers {} · {}",
  "transformers · CPU — GGUF 만들기의 파이썬 환경": "transformers · CPU — the Make GGUF Python environment",
  "기준 모델(GGUF)의 답을 원래 모델에 같은 토큰으로 넣어 위치마다 견줍니다. 0에 가까우면 변환과 엔진이 원래 모델을 그대로 옮긴 것이고, 버전들의 KL은 그 위에 더해진 차이입니다.": "Feeds the reference model's (GGUF) answers to the original model token by token and compares every position. Near 0 means conversion and engine reproduce the original model; the versions' KL is the difference added on top.",
  "견주기": "Compare",
  // 4-2 — 개선
  "개선 — 행렬 무리마다 되돌려 측정": "Improve — measure each matrix group restored",
  "개선 측정": "Improvement measurement",
  "고른 무리로 만들기": "Make with the chosen groups",
  "고른 무리를 되돌릴 형식으로 바꿔 기준 버전과 같은 형식 · imatrix로 다시 만들고, 기준 버전과 함께 같은 데이터로 평가합니다.": "Remakes the base version's type and imatrix with the chosen groups at the restore type, then evaluates it with the base version on the same data.",
  "기준 대비": "vs base",
  "기준 버전": "Base version",
  "되돌릴 형식": "Restore type",
  "마지막 측정 — 기준 {}({}) · KL {} · 되돌릴 형식 {} · 데이터 {} · 답 {}토큰 · {}": "Last measurement — base {} ({}) · KL {} · restore type {} · data {} · answers {} tokens · {}",
  "무리 하나만 높은 형식으로 바꾼 시험 파일을 같은 데이터로 잽니다": "Measures trial files with one group at a higher type on the same data",
  "무리": "Group",
  "무리를 하나 이상 고르세요": "Choose at least one group",
  "시험 {}개쯤 — 시험마다 파일을 만들고 평가합니다(0.6B에서 시험 하나에 20초쯤, 큰 모델은 더 오래). 시험 파일은 잰 뒤 지웁니다.": "About {} trials — each makes a file and evaluates it (about 20 s per trial for a 0.6B model, longer for bigger models). Trial files are deleted after measuring.",
  "측정": "Measure",
  "측정하려면 GGUF 만들기에서 도구 준비를 먼저 하세요.": "To measure, first set up tools in Make GGUF.",
  "층의 세 토막도 잰다": "Also measure the three layer thirds",
  "1 MB 당": "Per MB",
  "미리 고른 것 — 행렬 무리 가운데 KL을 5 % 넘게 줄인 것을 1 MB 당 줄임이 큰 차례로 셋. 무리마다 잰 줄임은 더해지지 않습니다 — 만든 뒤의 평가가 실제 값입니다.": "Preselected — the three matrix groups that cut the KL by more than 5 %, in order of reduction per MB. The measured reductions do not add up — the evaluation after making gives the real value.",
  "원래 모델과 견주려면 작업 공간의 GGUF 만들기에서 도구 준비를 먼저 하세요(파이썬 · transformers).": "To compare with the original model, first set up tools in the workspace's Make GGUF (Python · transformers).",
  "대화 {}개": "{0} {0|conversation|conversations}",
  "대화 {}개 — 원본 {}개 중 분야마다 고르게": "{0} {0|conversation|conversations} — chosen evenly by category from {1}",
  "평가할 버전을 하나 이상 고르세요": "Choose at least one version to evaluate",
  "데이터를 고르세요": "Choose the data",
  "평가하는 동안 GPU를 혼자 쓰도록 불러온 채팅 모델을 내립니다. 계속할까요?": "The loaded chat model is unloaded so the evaluation has the GPU to itself. Continue?",
  "평가 실행": "Run evaluation",
  "실행하는 동안 불러온 채팅 모델은 GPU에서 내립니다": "The loaded chat model is unloaded from the GPU while it runs",
  "평가할 버전": "Versions to evaluate",
  "먼저 버전을 더하세요": "Add a version first",
  "데이터": "Data",
  "텍스트": "Text",
  "대화": "Conversations",
  "청크 수(512 토큰)": "Chunks (512 tokens)",
  "답의 최대 토큰": "Max answer tokens",
  "동시 요청": "Concurrent requests",
  "생각(thinking) 켬": "Thinking on",
  "위치마다 평가할 모델의 상위 확률도 저장(cand.bin — 다시 계산 · 분석용, 위치마다 약 0.5 KB)": "Also save the evaluated model's top probabilities at every position (cand.bin — for recomputation and analysis, about 0.5 KB per position)",
  "텍스트 데이터는 같은 토큰의 확률만 비교합니다(llama-perplexity --kl-divergence와 같은 위치).": "Text data compares only the probabilities of the same tokens (the same positions as llama-perplexity --kl-divergence).",
  "기준 모델이 낸 답을 두 모델이 점수 매기고(같은 토큰의 확률), 두 모델이 저마다의 채팅 템플릿으로 답합니다. 기준 모델의 값은 처음 한 번 만들고, 같은 기준 모델 · 데이터 · 설정이면 다시 씁니다.": "Both models score the reference model's answers (probabilities of the same tokens), and each model answers with its own chat template. The reference values are made once and reused for the same reference model, data and settings.",
  "KL divergence {} · top-1 일치율 {}": "KL divergence {} · top-1 agreement {}",
  "대화 {}개 중 {}개의 답이 기준 모델과 달라졌습니다": "{1} of {0} answers differ from the reference model",
  " — 그중 {}개는 마진이 작은 스텝(0.05 nat 미만)에서 갈라졌습니다": " — {} of them diverged at a low-margin step (below 0.05 nat)",
  "대화 {}개의 답이 모두 기준 모델과 같습니다": "All {} answers match the reference model",
  "비교한 위치가 없습니다": "No positions were compared",
  "위치 {}개": "{} positions",
  "작업 공간": "Workspaces",
  "작업 공간|one": "Workspace",
  "채팅 템플릿": "Chat template",
  " · 정답률 {}": " · accuracy {}",
  "정답률": "Accuracy",
  "채점한 대화 {}개": "{0} graded {0|conversation|conversations}",
  "95% 구간 {}": "95% interval {}",
  "차이(평가 − 기준)": "Difference (evaluated − reference)",
  "기준만 맞음 {} · 평가만 맞음 {} · p = {}": "only reference correct {} · only evaluated correct {} · p = {}",
  "같은 대화에서 한쪽만 맞은 수로 부호 검정을 합니다. 구간은 대화를 다시 뽑는 부트스트랩(1,000번)입니다.": "The sign test uses the conversations where only one model is correct. The interval is a bootstrap over the conversations (1,000 resamples).",
  "정답(기준 · 평가)|col": "Correct (ref · eval)",
  "정답이 다른 대화": "Different correctness",
  "정답률(평가할 모델)": "Accuracy (evaluated model)",
  "채점한 대화 {}개 — 틀림 → 맞음 {}개 · 맞음 → 틀림 {}개(부호 검정 p = {})": "{} graded conversations — wrong → correct {} · correct → wrong {} (sign test p = {})",
  "틀림 → 맞음 {}개": "Wrong → correct: {}",
  "맞음 → 틀림 {}개": "Correct → wrong: {}",
  "맞음": "correct",
  "틀림": "wrong",
  "채점": "Grading",
  "정답": "Expected answer",
  "기준 모델의 도구 호출": "Reference model's tool call",
  "(없음)": "(none)",
  "평가할 모델의 도구 호출": "Evaluated model's tool call",
  "사용자 메시지 하나": "One user message",
  "시스템 메시지": "System message",
  "여러 턴": "Several turns",
  "도구 정의 · 호출 · 결과": "Tool definition · call · result",
  "생성 프롬프트 없이": "Without the generation prompt",
  "생각 끔": "Thinking off",
  "생각 켬": "Thinking on",
  "특수 토큰": "Special tokens",
  "토크나이저": "Tokenizer",
  "통과": "Pass",
  "대화 경우 {}개를 두 템플릿으로 그린 글이 바이트까지 같습니다": "The {} conversation cases render byte-identically with both templates",
  " — 템플릿의 글은 다르지만 그린 결과는 같습니다.": " — the template text differs but the rendered result is the same.",
  "대화 경우 {}개 중 {}개에서 그린 글이 원래 모델과 다릅니다 — 모델이 학습 때와 다른 프롬프트를 받습니다.": "{1} of {0} conversation cases render differently from the original model — the model gets a different prompt from the one it was trained on.",
  "GGUF에 채팅 템플릿이 없습니다 — 런타임이 기본 템플릿(ChatML)을 씁니다.": "The GGUF has no chat template — runtimes fall back to a default template (ChatML).",
  "원래 모델의 템플릿을 해석기가 풀지 못해 비교하지 못했습니다: {}": "The interpreter could not parse the original model's template, so it was not compared: {}",
  "원래 모델 폴더에 채팅 템플릿(chat_template.jinja · tokenizer_config.json)이 없습니다.": "The original model folder has no chat template (chat_template.jinja · tokenizer_config.json).",
  "GGUF의 EOS {}가 원래 모델의 끝 토큰입니다.": "The GGUF's EOS {} is an end token of the original model.",
  "GGUF의 EOS {}가 원래 모델의 끝 토큰({})이 아닙니다 — 다른 런타임에서 답이 끝나지 않을 수 있습니다.": "The GGUF's EOS {} is not an end token of the original model ({}) — answers may not stop in other runtimes.",
  "GGUF에 EOS 토큰이 없습니다 — 원래 모델의 끝 토큰은 {}입니다.": "The GGUF has no EOS token — the original model's end tokens are {}.",
  "원래 모델 폴더에 끝 토큰 정보(generation_config.json · tokenizer_config.json)가 없습니다.": "The original model folder has no end-token information (generation_config.json · tokenizer_config.json).",
  "add_bos_token이 원래 모델과 같습니다({}).": "add_bos_token matches the original model ({}).",
  "add_bos_token이 GGUF {} · 원래 모델 {}로 다릅니다 — 프롬프트 앞에 BOS가 붙는지가 달라집니다.": "add_bos_token differs — GGUF {} · original model {} — so whether a BOS is put in front of the prompt changes.",
  "원래 모델 폴더에 tokenizer_config.json이 없습니다.": "The original model folder has no tokenizer_config.json.",
  "특수 토큰 {}개가 같은 번호 · 글로 있습니다.": "All {} special tokens are present with the same IDs and text.",
  "특수 토큰 {}개 중 {}개가 GGUF와 다릅니다 — 다시 변환해야 합니다(메타데이터로 고칠 수 없음).": "{1} of {0} special tokens differ in the GGUF — convert again (this cannot be fixed in the metadata).",
  "원래 모델 폴더에 tokenizer.json이 없습니다.": "The original model folder has no tokenizer.json.",
  "어휘 {}개와 병합 {}개가 GGUF와 같습니다.": "The {} vocabulary entries and {} merges match the GGUF.",
  "어휘 {}개 중 {}개의 번호 · 글이 GGUF와 다릅니다 — 다시 변환해야 합니다.": "{1} of {0} vocabulary entries differ in the GGUF — convert again.",
  "병합이 다릅니다(원래 {}개 · GGUF {}개, 다른 것 {}개) — 다시 변환해야 합니다.": "The merges differ (original {} · GGUF {}, {} different) — convert again.",
  "토크나이저 종류 {}는 비교하지 않습니다(BPE만).": "Tokenizer type {} is not compared (BPE only).",
  "설정 — 하이퍼파라미터": "Settings — hyperparameters",
  "config.json의 하이퍼파라미터 {}개가 GGUF와 같습니다.": "All {} hyperparameters in config.json match the GGUF.",
  "config.json의 하이퍼파라미터 {}개 중 {}개가 GGUF와 다릅니다 — 다시 변환해야 합니다(메타데이터로 고칠 수 없음).": "{1} of {0} hyperparameters in config.json differ in the GGUF — convert again (this cannot be fixed in the metadata).",
  "config.json과 GGUF({})에서 견줄 하이퍼파라미터를 찾지 못했습니다.": "No hyperparameters to compare were found in config.json and the GGUF ({}).",
  "원래 모델 폴더에 config.json이 없습니다.": "The original model folder has no config.json.",
  "하이퍼파라미터 {}개": "{0} {0|hyperparameter|hyperparameters}",
  "원래 {} · GGUF {}": "original {} · GGUF {}",
  "config.json — transformers {}로 저장": "config.json — saved with transformers {}",
  "설정 — 하이퍼파라미터 {}개가 config.json · transformers · GGUF에서 같습니다": "Settings — {} hyperparameters match in config.json · transformers · GGUF",
  "GGUF의 하이퍼파라미터가 원래 모델의 설정과 다릅니다 — {}. 다시 변환해야 합니다.": "The GGUF's hyperparameters differ from the original model's settings — {}. Convert again.",
  "설정 — 기준값을 만든 GGUF를 찾지 못해 견주지 않았습니다": "Settings — not compared (the GGUF that made the reference values was not found)",
  "기준 모델이 16비트인데 KL이 10⁻³보다 큽니다 — 변환이나 원래 모델 폴더의 설정이 다를 수 있습니다. 변환 검증을 보세요.": "The reference model is 16-bit but the KL is above 10⁻³ — the conversion or the original model folder's settings may differ. See the conversion check.",
  "기준 모델이 16비트면 바르게 변환했을 때 KL은 10⁻³보다 훨씬 작습니다(이 PC의 측정 3.5 × 10⁻⁶ ~ 1.0 × 10⁻⁴ — 답이 확실할수록 작음).": "With a 16-bit reference model, a correct conversion gives a KL far below 10⁻³ (measured on this PC: 3.5 × 10⁻⁶ to 1.0 × 10⁻⁴ — smaller when the answers are more certain).",
  "대화 경우 {}개": "{0} conversation {0|case|cases}",
  "{}바이트부터 다름 · GGUF {} · 원래 {}바이트": "differs from byte {} · GGUF {} · original {} bytes",
  "오류: {}": "Error: {}",
  "원래 모델": "Original model",
  "예 {}개": "{0} {0|example|examples}",
  "고친 버전 만들기": "Create fixed version",
  "{}에서 메타데이터만 바꾼 새 버전을 만듭니다 — 바꾸는 키: {}. 새 파일이 모델 폴더에 쓰입니다({}).": "Creates a new version from {} with only the metadata changed — keys: {}. A new file is written to the model folder ({}).",
  "원래 모델 폴더": "Original model folder",
  "원래 모델 폴더를 정하세요": "Choose the original model folder",
  "{}는 아직 검사하지 않았습니다.": "{} has not been checked yet.",
  "{} · {} · 문제 {} · 확인 필요 {} · 통과 {}": "{0} · {1} · {2} {2|problem|problems} · {3} to check · {4} passed",
  "고친 버전 만들기({})": "Create fixed version ({})",
  "이것만 고치기": "Fix only this",
  "원래 모델(Hugging Face 형식)의 작은 파일과 견줍니다 — 가중치는 읽지 않습니다": "Compares with the original model's small files (Hugging Face format) — the weights are not read",
  "원래 모델 폴더(tokenizer_config.json · tokenizer.json · generation_config.json이 있는 폴더)": "Original model folder (with tokenizer_config.json · tokenizer.json · generation_config.json)",
  "검사할 버전": "Version to check",
  "검사": "Check",
  "(키 없음)": "(no key)",
  "바뀐 것이 없습니다": "Nothing changed",
  "메타데이터 고치기 — {}": "Fix metadata — {}",
  "텐서는 그대로 두고 GGUF의 메타데이터(키)만 바꾼 새 파일을 앱의 모델 폴더에 쓰고 새 버전으로 더합니다. 원래 파일은 바뀌지 않습니다.": "Writes a new file to the app's model folder with only the GGUF metadata (keys) changed — the tensors stay as they are — and adds it as a new version. The original file is not changed.",
  "어휘 {}": "vocabulary {}",
  "키 {}개": "{0} {0|key|keys}",
  "채팅 템플릿(tokenizer.chat_template — jinja)": "Chat template (tokenizer.chat_template — jinja)",
  "토큰은 어휘의 제어 토큰(control · user-defined)에서 고릅니다. 고친 버전으로 다시 평가하면 앞 버전과 전후 비교할 수 있습니다.": "Tokens are chosen from the vocabulary's control tokens (control · user-defined). Evaluate the fixed version again to compare it with the previous one.",
  "새 버전 만들기": "Create new version",
  "바꾼 키 {}": "Changed keys: {}",
  "메타데이터 고치기": "Fix metadata",
  "채팅 템플릿 · 특수 토큰 — 새 버전으로": "Chat template · special tokens — as a new version",
  "기준 모델|col": "Reference",
  "평가할 모델|col": "Evaluated",
  "대화|col": "Items",
  "작업 공간이 없습니다": "No workspaces",
  "새 작업 공간을 만들고 기준 모델을 고른 뒤, 비교할 GGUF 파일을 버전으로 더하세요.": "Create a workspace, choose a reference model, then add the GGUF files to compare as versions.",
  "기준 모델 {}": "Reference {}",
  "버전 {}개": "{0} {0|version|versions}",
  "{} · 답 {}/{} 달라짐": "{} · {}/{} answers differ",
  "작업 공간 목록": "Workspace list",
  "GGUF 파일을 기준 모델과 비교합니다": "Compare GGUF files with a reference model",
  "작업 공간 만들기": "Create a workspace",
  "기준 모델을 고릅니다 — 같은 모델의 BF16 · F16 파일(그래픽 메모리에 들지 않으면 Q8_0).": "Choose the reference model — a BF16 or F16 file of the same model (Q8_0 if it does not fit in graphics memory).",
  "비교할 GGUF 파일(양자화 · 미세조정한 파일)을 v1 · v2 …로 더합니다.": "Add the GGUF files to compare (quantized or fine-tuned files) as v1 · v2 ….",
  "기준 모델이 낸 답의 토큰마다 두 모델의 확률을 비교하고(KL divergence · top-1 일치율), 두 모델이 저마다 낸 답이 어디서 갈라지는지 봅니다.": "Compare the two models' probabilities at every token of the reference model's answers (KL divergence · top-1 agreement), and see where each model's own answer diverges.",
  "결과 · 전후 비교": "Results · before/after comparison",
  "버전을 고친 뒤 같은 데이터 · 설정으로 다시 실행하면, 기준 모델과 같아진 대화와 새로 달라진 대화로 나아졌는지 판정합니다.": "After changing a version, run again with the same data and settings — the conversations that now match the reference and those that newly diverged decide whether it improved.",
  "작업 공간 메뉴": "Workspace menu",
  "기준 모델이 앱의 모델 목록에 없습니다 — 모델 화면에서 그 파일을 다시 더하거나 기준 모델을 바꾸세요.": "The reference model is not in the app's model list — add the file again on the Models screen or change the reference model.",
  "지금 상태 — {}": "Current state — {}",
  "지금 상태": "Current state",
  "아직 평가하지 않았습니다 — 아래에서 평가를 실행하세요.": "Not evaluated yet — run an evaluation below.",
  "평가할 GGUF 파일을 버전으로 더하세요.": "Add the GGUF files to evaluate as versions.",
  "답이 달라진 대화 {}": "Differing answers {}",
  "지금": "Current",
  "가져옴": "Imported",
  "메타데이터 고침": "Metadata fixed",
  "다시 양자화": "Requantized",
  "변환": "Converted",
  "답 {}/{} 달라짐 · KL {}": "{}/{} answers differ · KL {}",
  "결과 없음": "No result",
  "{} 대비 {}": "vs {} {}",
  "{}에서 만듦": "made from {}",
  "답 {}/{} 달라짐": "{}/{} answers differ",
  "{} 메뉴": "{} menu",
  "최근 결과 보기": "View latest result",
  "지금 버전으로": "Make current",
  "버전": "Version",
  "버전이 없습니다": "No versions",
  "기준 모델과 비교할 GGUF 파일을 더하세요.": "Add a GGUF file to compare with the reference model.",
  "{}개 · 둘을 고르면 전후 비교": "{} · choose two to compare before/after",
  "전후 비교": "Before/after comparison",
  "답이 달라진 대화": "Differing answers",
  "{} 고르기": "Select {}",
  "답 {}토큰": "answers {} tokens",
  "청크 {}": "{} chunks",
  " (마진 작음 {})": " (low margin {})",
  "위에서 평가를 실행하면 여기에 쌓입니다.": "Results appear here after you run an evaluation above.",
  "{} 결과": "{} result",
  "디버거로 열기": "Open in debugger",
  "{} · {}초": "{} · {} s",
  "어휘가 기준 모델과 달라 같은 토큰의 확률은 비교하지 않았습니다 — 답은 글로 비교했습니다.": "The vocabulary differs from the reference model, so token probabilities were not compared — the answers were compared as text.",
  "같은 토큰의 확률": "Probabilities of the same tokens",
  "기준 모델이 낸 답의 토큰마다": "at every token of the reference model's answers",
  "청크마다 뒤쪽 255 토큰": "the last 255 tokens of every chunk",
  "KL divergence 평균": "Mean KL divergence",
  "KL divergence 99 백분위수": "KL divergence, 99th percentile",
  "top-1 일치율": "Top-1 agreement",
  "PPL 비(평가 ÷ 기준)": "PPL ratio (evaluated ÷ reference)",
  "Δp 평균(기준 − 평가)": "Mean Δp (reference − evaluated)",
  "기준 모델의 상위 {}개 밖 확률": "Probability outside the reference top {}",
  "KL divergence는 기준 모델의 상위 {}개 토큰과 나머지 확률 하나로 계산한 값입니다 — 모든 어휘로 계산한 값보다 조금 작습니다(아래쪽 값). 위치 {}개.": "KL divergence is computed from the reference model's top {} tokens plus one remainder probability — slightly below the full-vocabulary value (a lower bound). {} positions.",
  "답이 달라진 대화 {}개": "{} differing answers",
  "{} 이상": "{} or more",
  "달라진 대화": "Differing answers",
  "마진이 작은 스텝(0.05 nat 미만)에서 갈라짐": "Diverged at a low-margin step (below 0.05 nat)",
  "글로 비교한 대화 {}개": "{} compared as text",
  "답이 길면 확률 차이가 작아도 어딘가에서 갈라지기 쉽습니다. 일찍 갈라진 대화와 마진이 작은 스텝에서 갈라진 대화를 함께 보세요.": "Long answers tend to diverge somewhere even when the probability differences are small. Look at both the early divergences and those at low-margin steps.",
  "끝난 까닭": "Finish reasons",
  "기준 모델 — 끝 토큰으로 끝남": "Reference model — ended with the end token",
  "평가할 모델 — 끝 토큰으로 끝남": "Evaluated model — ended with the end token",
  "평가할 모델이 끝 토큰(EOS)으로 끝내지 못하고 최대 길이까지 간 답이 기준 모델보다 많습니다 — GGUF 의 채팅 템플릿 · EOS 토큰을 확인하세요.": "More of the evaluated model's answers ran to the maximum length instead of ending with the end token (EOS) than the reference model's — check the GGUF's chat template and EOS token.",
  "분야": "Category",
  "답이 달라짐": "Differing answers",
  "마진이 작은 스텝에서 갈라짐": "Low-margin divergence",
  "전체": "All",
  "청크": "Chunks",
  "대화 거르기": "Filter conversations",
  "끝난 까닭(기준 · 평가)": "Finish (reference · evaluated)",
  "글로 비교": "Text only",
  "해당하는 대화가 없습니다": "No matching conversations",
  "처음 500개만 보입니다(모두 {}개).": "Showing the first 500 (of {}).",
  "재현 정보": "Reproduction info",
  "평가한 파일": "Evaluated file",
  "장치": "Device",
  "결과 해시": "Result hash",
  "같은 파일 · 데이터 · 설정 · GeulPU 버전이면 다시 실행해도 결과 해시가 같습니다.": "With the same files, data, settings and GeulPU version, a rerun gives the same result hash.",
  "전": "Before",
  "후": "After",
  "두 결과 비교": "Compare two results",
  "서로 다른 결과 두 개를 고르세요.": "Choose two different results.",
  "데이터가 다릅니다": "the data differs",
  "설정이 다릅니다": "the settings differ",
  "기준 모델이 다릅니다": "the reference model differs",
  "대화 수가 다릅니다": "the number of conversations differs",
  "나아졌습니다 — 기준 모델과 같아진 대화 {}개 · 새로 달라진 대화 {}개(부호 검정 p = {})": "Improved — {} now match the reference · {} newly diverged (sign test p = {})",
  "나빠졌습니다 — 기준 모델과 같아진 대화 {}개 · 새로 달라진 대화 {}개(부호 검정 p = {})": "Got worse — {} now match the reference · {} newly diverged (sign test p = {})",
  "뚜렷한 차이가 없습니다 — 기준 모델과 같아진 대화 {}개 · 새로 달라진 대화 {}개(부호 검정 p = {})": "No clear difference — {} now match the reference · {} newly diverged (sign test p = {})",
  "비교할 수 없습니다 — {}": "Cannot compare — {}",
  "나아짐": "Improved",
  "나빠짐": "Worse",
  "차이 없음": "No clear difference",
  "비교 불가": "Not comparable",
  "GeulPU의 결과에는 실행마다의 잡음이 없어 대화 하나하나의 바뀜은 파일을 바꾼 데서 옵니다. 어떤 대화를 골랐나에 따른 차이는 남기 때문에 부호 검정으로 판정합니다(p < 0.05).": "GeulPU results carry no run-to-run noise, so each conversation's change comes from the file change. Which conversations were chosen still matters, so the verdict uses a sign test (p < 0.05).",
  "지표": "Metric",
  "차이": "Change",
  "기준 모델과 같아진 대화 {}개": "{} now match the reference",
  "누르면 전 결과에서 달랐던 답을 엽니다": "Opens the differing answer in the before result",
  "새로 달라진 대화 {}개": "{} newly diverged",
  "누르면 후 결과의 답을 엽니다": "Opens the answer in the after result",
  "(빈 답)": "(empty answer)",
  "메시지 {}개": "{0} {0|message|messages}",
  "도구 {}개": "{0} {0|tool|tools}",
  "앞의 메시지 {}개": "{0} earlier {0|message|messages}",
  "앞 대화": "Previous conversation",
  "다음 대화": "Next conversation",
  "두 모델의 답이 토큰까지 같습니다.": "The two models' answers match token for token.",
  "두 모델의 답이 다릅니다 — 어휘가 달라 글로 비교했습니다.": "The two answers differ — compared as text because the vocabularies differ.",
  "스텝 {}에서 처음 갈라졌습니다 — 그 앞 {}토큰은 같습니다.": "First diverged at step {} — the {} tokens before it match.",
  "그 스텝에서 기준 모델의 마진(1위와 2위의 로그확률 차) {} nat": "Reference model's margin at that step (log-probability gap between 1st and 2nd) {} nat",
  "기준 모델 {}토큰({})": "Reference model {} tokens ({})",
  "평가할 모델 {}토큰({})": "Evaluated model {} tokens ({})",
  "첫 토큰(스텝 1)부터 갈라졌습니다.": "The answers diverge from the first token (step 1).",
  "이 대화의 KL divergence 평균 {} · top-1 일치율 {}": "This conversation: mean KL divergence {} · top-1 agreement {}",
  "두 모델의 답": "Both models' answer",
  "두 답": "The two answers",
  "흐린 글은 두 답이 같은 앞부분, 밝은 글은 갈라진 뒤": "Dim text is the shared beginning, bright text is after the divergence",
  "평가할 모델": "Evaluated model",
  "기준 모델이 고름": "Reference model's choice",
  "평가할 모델이 고름": "Evaluated model's choice",
  " · 기준 모델의 {}위": " · rank {} for the reference model",
  " · 기준 모델의 상위 {}개 밖": " · outside the reference model's top {}",
  "스텝 {}의 후보": "Candidates at step {}",
  "같은 앞부분에서 두 모델이 다음 토큰에 준 확률": "The probabilities both models gave the next token after the same beginning",
  "표시": "Mark",
  "그 앞뒤 스텝": "Steps around it",
  "기준 모델의 답을 두 모델이 점수 매긴 값": "Both models scoring the reference model's answer",
  "기준 모델의 토큰": "Reference token",
  "갈라진 스텝 뒤로는 기준 모델의 답을 따라간 값입니다 — 평가할 모델이 자기 답을 이어 간 값이 아닙니다.": "After the divergent step the values follow the reference model's answer — not the evaluated model's own continuation.",
  "작업 공간 {}개": "{0} {0|workspace|workspaces}",
  "GGUF 파일을 기준 모델과 비교합니다 — 다음 토큰 확률의 차이(KL divergence · top-1 일치율)와 두 모델이 낸 답. 같은 파일 · 같은 설정이면 다시 실행해도 결과가 비트까지 같습니다.": "Compare GGUF files with a reference model — the difference in next-token probabilities (KL divergence · top-1 agreement) and the answers both models give. With the same files and settings, a rerun gives bit-identical results.",
};
Object.assign(DICT, EVAL_DICT);
// 평가 작업(앱/평가.gl)의 단계 · 실행 기록 · 오류 문구 → 영어(app.js 의 TMX 뒤에)
TMX.push(
  [/\[([^\]]+)\] 추론 서버를 시작합니다\(슬롯 (\d+) · 문맥 (\d+) — 처음에는 커널 컴파일과 실행 방식 측정으로 몇 분 걸릴 수 있습니다\)/g, '[$1] Starting the inference server ($2 slots · context $3 — the first time can take a few minutes to compile kernels and measure)'],
  [/\[([^\]]+)\] 추론 서버 준비 완료\(슬롯 (\d+), ([\d.]+)초 · 어휘 (\d+)\)\./g, '[$1] Inference server ready ($2 slots, $3 s · vocabulary $4).'],
  [/\[([^\]]+)\] 슬롯 (\d+)개로는 추론 서버가 시작되지 않았습니다: /g, '[$1] The inference server did not start with $2 slots: '],
  [/ — 기준 모델의 답 (\d+)\/(\d+)/g, ' — reference answers $1/$2'],
  [/ — 기준 모델의 상위 확률 (\d+)\/(\d+)/g, ' — reference top probabilities $1/$2'],
  [/ — 같은 토큰의 확률 (\d+)\/(\d+)/g, ' — probabilities of the same tokens $1/$2'],
  [/ — 토큰 글 (\d+)\/(\d+)/g, ' — token text $1/$2'],
  [/ — 답 (\d+)\/(\d+)/g, ' — answers $1/$2'],
  [/기준 모델 — preparing kernel modules/g, 'Reference model — preparing kernel modules'],   // app.js 의 '(\S+) — 커널 모듈 준비 중' 이 먼저 바꾼 나머지
  [/텍스트 토큰 (\d+)개 → 청크 (\d+)개\(문맥 512 · 청크마다 뒤쪽 255 토큰을 점수 매김\)\./g, '$1 text tokens → $2 chunks (context 512 · the last 255 tokens of each chunk are scored).'],
  [/대화 (\d+)개의 그리디 답을 만듭니다\./g, 'Making greedy answers for $1 conversations.'],
  [/\] 답 (\d+)\/(\d+)개\./g, '] answers $1/$2.'],
  [/상위 (\d+)개의 로그확률 — (\d+)\/(\d+)개\./g, 'top-$1 log-probabilities — $2/$3.'],
  [/어휘가 기준 모델과 달라\(어휘 (\d+) · (\d+)\) 분포 비교는 하지 않고 답만 견줍니다\./g, 'The vocabulary differs from the reference model (vocabulary $1 · $2) — comparing answers only, without the distribution comparison.'],
  [/위치 (\d+) · KL divergence 평균 (\S+) · top-1 일치율 (\S+) · 답이 달라진 대화 (\d+)\/(\d+) — 결과 해시 /g, '$1 positions · mean KL divergence $2 · top-1 agreement $3 · differing answers $4/$5 — result hash '],
  [/저장해 둔 기준값을 씁니다\(같은 기준 모델 · 데이터 · 설정\): /g, 'Reusing the saved reference values (same reference model · data · settings): '],
  [/작업 공간에 없는 버전입니다\./g, 'The version is not in the workspace.'],
  [/모델이 앱의 모델 목록에 없습니다\((.*?)\)\./g, "The model is not in the app's model list ($1)."],
  [/그래픽 메모리가 모자랍니다\(파일 ([\d.]+) GB\)\./g, 'Not enough graphics memory (file $1 GB).'],
  [/커널 모듈을 만들지 못했습니다\./g, 'Could not build the kernel modules.'],
  [/추론 서버를 시작하지 못했습니다\(결과 폴더의 server-\*\.log 참고\)\./g, 'Could not start the inference server (see server-*.log in the result folder).'],
  [/모델 평가 — 버전 (\d+)\/(\d+)개 끝\((.*?)\) · 데이터 (\S+) · (exact|fast)/g, 'Model evaluation — $1/$2 versions done ($3) · data $4 · $5'],
  [/평가 요청을 읽지 못했습니다/g, 'Could not read the evaluation request'],
  [/기준 모델이 앱의 모델 목록에 없습니다 — 모델 화면에서 더한 뒤 작업 공간의 기준 모델을 고르세요/g, "The reference model is not in the app's model list — add it on the Models screen, then choose the workspace's reference model"],
  [/기준 모델 파일의 SHA-256 을 계산하지 못했습니다/g, "Could not compute the reference model file's SHA-256"],
  [/데이터 파일을 읽지 못했습니다/g, 'Could not read the data file'],
  [/기준 모델의 커널 모듈을 만들지 못했습니다/g, "Could not build the reference model's kernel modules"],
  [/데이터에 대화가 없습니다\(JSONL — 줄마다 messages\)/g, 'The data has no conversations (JSONL — messages on every line)'],
  [/기준 모델의 추론 서버를 시작하지 못했습니다\(기준값 폴더의 server-ref-\*\.log 참고\)/g, "Could not start the reference model's inference server (see server-ref-*.log in the reference folder)"],
  [/평가한 버전이 없습니다 — 작업 기록 참고/g, 'No version was evaluated — see the run log'],
  [/텍스트 파일을 읽지 못했습니다/g, 'Could not read the text file'],
  [/텍스트를 토큰으로 나누지 못했습니다\(\/tokenize\)/g, 'Could not tokenize the text (/tokenize)'],
  [/텍스트가 짧아 청크\(512 토큰\)를 만들 수 없습니다/g, 'The text is too short for a chunk (512 tokens)'],
  [/기준 모델이 답을 하나도 내지 못했습니다\(run-ref\.jsonl 참고\)/g, 'The reference model gave no answers (see run-ref.jsonl)'],
  [/기준 모델의 로그확률을 받지 못했습니다/g, "Could not get the reference model's log-probabilities"],
  [/이미 이 작업 공간에 있는 모델입니다\(기준 모델 또는 버전\)/g, 'The model is already in this workspace (reference model or version)'],
  [/기준 모델이 앱의 모델 목록에 없습니다/g, "The reference model is not in the app's model list"],
  [/모델이 앱의 모델 목록에 없습니다/g, "The model is not in the app's model list"],
  [/작업 공간을 쓰지 못했습니다/g, 'Could not write the workspace'],
  [/작업 공간이 없습니다/g, 'The workspace does not exist'],
  [/비교할 결과가 없습니다/g, 'No results to compare'],
  [/결과가 없습니다/g, 'The result does not exist'],
  [/대화가 없습니다/g, 'The conversation does not exist'],
  [/\.jsonl\(채팅 — 줄마다 messages\) 또는 \.txt\(텍스트\) 파일만 불러옵니다/g, 'Only .jsonl (chat — messages on every line) or .txt (text) files can be imported'],
  [/데이터 이름은 영문 · 숫자 · \. · _ · - 로 120 자까지/g, 'Data names use letters, digits, . _ - up to 120 characters'],
  [/데이터를 쓰지 못했습니다/g, 'Could not write the data'],
  [/데이터가 없습니다/g, 'The data does not exist'],
  [/이 결과에는 실행 로그가 없습니다\(텍스트 데이터\)/g, 'This result has no run log (text data)'],
  [/실행 로그를 디버거 폴더로 옮기지 못했습니다/g, 'Could not copy the run log to the debugger folder'],
  [/\[기준\]/g, '[reference]'],
  // 2-1 — 메타데이터 고치기(평가고침일 · 앱/GGUF쓰기.gl)
  [/원래 파일을 읽는 중/g, 'Reading the original file'],
  [/\[([^\]]+)\] (.+?) → (.+?)\(바꾸는 키: (.*?)\)/g, '[$1] $2 → $3 (keys: $4)'],
  [/새 파일을 쓰는 중 — (\d+)\/(\d+) MB/g, 'Writing the new file — $1/$2 MB'],
  [/새 파일을 모델 목록에 더하는 중/g, 'Adding the new file to the model list'],
  [/GGUF 메타데이터 고치기 — (\S+) → (\S+) · /g, 'GGUF metadata fix — $1 → $2 · '],
  [/버전의 모델이 앱의 모델 목록에 없습니다/g, "The version's model is not in the app's model list"],
  [/요청을 읽지 못했습니다/g, 'Could not read the request'],
  [/버전이 없습니다/g, 'The version does not exist'],
  [/바꿀 키가 없습니다/g, 'No keys to change'],
  [/원래 파일을 열 수 없습니다/g, 'Cannot open the original file'],
  [/새 파일을 만들 수 없습니다\(모델 폴더를 확인하세요\)/g, 'Cannot create the new file (check the model folder)'],
  [/새 파일의 크기가 맞지 않습니다\(디스크 공간을 확인하세요\)/g, 'The new file has the wrong size (check the disk space)'],
  [/GGUF 버전 3만 다룹니다/g, 'Only GGUF version 3 is handled'],
  [/GGUF 헤더가 손상되었습니다/g, 'The GGUF header is damaged'],
  [/GGUF 헤더를 끝까지 읽지 못했습니다\(파일이 잘렸거나 손상됨\)/g, 'Could not read the whole GGUF header (the file is truncated or damaged)'],
  [/GGUF 의 자료 구역이 파일 밖입니다/g, "The GGUF data section is outside the file"],
  [/GGUF 파일이 아닙니다/g, 'Not a GGUF file'],
  [/키 이름은 1 ~ 256 바이트/g, 'Key names are 1–256 bytes'],
  [/바꿀 수 있는 형은 글 · 정수 · bool 뿐입니다/g, 'Only string, integer and bool values can be changed'],
  [/(\S+) 의 형이 파일과 다릅니다\(글 · 정수 · bool 사이에서는 바꾸지 않습니다\)/g, '$1 has a different type in the file (string, integer and bool are not converted)'],
  // 2-2 — 변환 검증(평가검증일)
  [/원래 모델의 파일과 GGUF 의 머리를 견주는 중/g, 'Comparing the original model files with the GGUF header'],
  [/GGUF 변환 검증 — (\S+) · 문제 (\d+) · 확인 필요 (\d+) · 통과 (\d+)/g, 'GGUF conversion check — $1 · $2 problems · $3 to check · $4 passed'],
  [/원래 모델 폴더를 정하세요\(Hugging Face 형식 — tokenizer_config\.json 이 있는 폴더\)/g, 'Choose the original model folder (Hugging Face format — the folder with tokenizer_config.json)'],
  [/원래 모델 폴더가 없습니다: /g, 'The original model folder does not exist: '],
  [/GGUF 파일을 열 수 없습니다/g, 'Cannot open the GGUF file'],
  [/변환 검증 보고서가 없습니다/g, 'No conversion check report'],
  // 4-1 — GGUF 만들기(앱/평가도구.gl — 도구준비일 · 만들기일)
  [/llama\.cpp (\S+) 실행 파일 — 이미 있음\((.*?)\)/g, 'llama.cpp $1 executables — already there ($2)'],
  [/llama\.cpp (\S+) 실행 파일을 받는 중 — (\S+)\(([\d.]+) MB\)/g, 'Downloading llama.cpp $1 executables — $2 ($3 MB)'],
  [/압축을 푸는 중/g, 'Extracting'],
  [/압축을 풀 tar 를 찾지 못했습니다/g, 'Could not find tar to extract the archive'],
  [/llama\.cpp 의 압축을 풀지 못했습니다\(llama-quantize 가 없음\)/g, 'Could not extract llama.cpp (no llama-quantize)'],
  [/llama\.cpp (\S+) 실행 파일 — /g, 'llama.cpp $1 executables — '],
  [/변환 스크립트를 받는 중 — (\d+)\/(\d+)/g, 'Downloading the conversion scripts — $1/$2'],
  [/변환 스크립트\(llama\.cpp (\S+)\) — (\d+) 파일, 새로 받은 것 (\d+)/g, 'Conversion scripts (llama.cpp $1) — $2 files, $3 newly downloaded'],
  [/파이썬을 찾는 중/g, 'Looking for Python'],
  [/파이썬 3\.10 이상을 찾지 못했습니다 — python\.org 에서 설치한 뒤\(설치할 때 'Add python\.exe to PATH'\) 다시 누르거나, 파이썬의 경로를 적으세요/g, "Python 3.10 or later was not found — install it from python.org (check 'Add python.exe to PATH') and try again, or enter the path to Python"],
  [/(\S+) 의 파이썬 (\S+) — 3\.10 이상이 필요합니다/g, '$1 is Python $2 — 3.10 or later is needed'],
  [/파이썬 (\d+\.\d+\.\d+) — /g, 'Python $1 — '],
  [/가상 환경을 만드는 중/g, 'Creating the virtual environment'],
  [/가상 환경을 만들지 못했습니다\(리눅스라면 python3-venv 패키지를 설치하세요\)/g, 'Could not create the virtual environment (on Linux, install the python3-venv package)'],
  [/변환에 드는 파이썬 패키지를 설치하는 중\(torch · transformers …\)/g, 'Installing the Python packages for conversion (torch · transformers …)'],
  [/파이썬 패키지를 설치하지 못했습니다\(인터넷 · 디스크 — 작업 기록을 보세요\)/g, 'Could not install the Python packages (internet · disk — see the run log)'],
  [/패키지를 확인하는 중/g, 'Checking the packages'],
  [/설치한 패키지를 불러오지 못했습니다: /g, 'Could not import the installed packages: '],
  [/GGUF 만들기 도구 준비 — llama\.cpp (\S+) · 파이썬 (\S+) · torch (\S+) · transformers (\S+)/g, 'GGUF making tools ready — llama.cpp $1 · Python $2 · torch $3 · transformers $4'],
  // #196 — transformers 5 환경 · 폴더를 저장한 판으로 환경 고르기 · 설정 견주기
  [/transformers (\S+) 환경 — 이미 있음\((.+?)\)/g, 'transformers $1 environment — already there ($2)'],
  [/transformers (\S+) 환경을 만드는 중\(transformers 5 로 저장한 모델용\)/g, 'Creating the transformers $1 environment (for models saved with transformers 5)'],
  [/첫 가상 환경의 파이썬을 찾지 못했습니다/g, "Could not find the first virtual environment's Python"],
  [/첫 가상 환경의 site-packages 를 찾지 못했습니다/g, "Could not find the first virtual environment's site-packages"],
  [/transformers 5 환경의 site-packages 를 찾지 못했습니다/g, "Could not find the transformers 5 environment's site-packages"],
  [/transformers 5 환경의 패키지를 불러오지 못했습니다\(작업 기록을 보세요\)/g, 'Could not import the packages of the transformers 5 environment (see the run log)'],
  [/transformers (\S+) 환경 — /g, 'transformers $1 environment — '],
  [/이 모델 폴더는 transformers (\S+) 로 저장했습니다 — 지금 도구의 환경\(transformers (\S+)\)은 그 형식을 바르게 읽지 못합니다\. '도구 준비'를 다시 누르면 transformers 5 환경을 더합니다/g, "This model folder was saved with transformers $1 — the current tool environment (transformers $2) cannot read that format correctly. Press 'Set up tools' again to add a transformers 5 environment"],
  [/변환 환경 — transformers (\S+)\(원래 모델 폴더는 transformers (\S+) 로 저장\)/g, 'Conversion environment — transformers $1 (the original model folder was saved with transformers $2)'],
  [/변환 환경 — transformers (\S+)/g, 'Conversion environment — transformers $1'],
  [/견주기 환경 — transformers (\S+)\(원래 모델 폴더는 transformers (\S+) 로 저장\)/g, 'Comparison environment — transformers $1 (the original model folder was saved with transformers $2)'],
  [/견주기 환경 — transformers (\S+)/g, 'Comparison environment — transformers $1'],
  [/변환하지 못했습니다\(convert_hf_to_gguf\.py\): (.*) — 작업 기록에 자세히/g, 'Could not convert (convert_hf_to_gguf.py): $1 — details in the run log'],
  [/원래 모델과 견주지 못했습니다\(orig_compare\.py\): /g, 'Could not compare with the original model (orig_compare.py): '],
  [/transformers (\S+) 가 원래 모델의 설정을 config\.json 과 다르게 읽었습니다 — (.+?)\. 이 폴더는 transformers (\S+) 로 저장했습니다 — 저장한 큰 판의 transformers 로 견주어야 합니다\(KL 을 내지 않고 멈춤\)/g, 'transformers $1 read the original model settings differently from config.json — $2. This folder was saved with transformers $3 — compare with the same major version of transformers (stopped without a KL)'],
  [/ · GGUF 의 하이퍼파라미터가 원래 모델의 설정과 다릅니다/g, " · the GGUF's hyperparameters differ from the original model's settings"],
  [/설정 — 하이퍼파라미터 (\d+)개 · GGUF 가 원래 모델과 다름/g, 'Settings — $1 hyperparameters · the GGUF differs from the original model'],
  [/설정 — 하이퍼파라미터 (\d+)개 · 같음/g, 'Settings — $1 hyperparameters · same'],
  [/설정 — 기준값을 만든 GGUF 를 찾지 못해 견주지 않음/g, 'Settings — not compared (the GGUF that made the reference values was not found)'],
  [/(\S+) — 끝 코드 (-?\d+)\. 기록의 끝:/g, '$1 — exit code $2. End of the log:'],
  [/자료 폴더의 경로에 ASCII 가 아닌 글자가 있어 llama\.cpp 의 도구를 쓸 수 없습니다 — ASCII 경로의 자료 폴더\(--자료\)로 실행하세요/g, "The data folder path has non-ASCII characters, so llama.cpp's tools cannot use it — run with a data folder on an ASCII path (--자료)"],
  [/자료 폴더의 경로에 ASCII 가 아닌 글자가 있어 llama\.cpp 의 도구를 쓸 수 없습니다/g, "The data folder path has non-ASCII characters, so llama.cpp's tools cannot use it"],
  [/앱의 도구 목록\(data\\tools\.json\)에 이 체계\((\S+)\)의 llama\.cpp 가 없습니다/g, "The app's tool list (data\\tools.json) has no llama.cpp for this system ($1)"],
  [/ — 크기가 목록과 다릅니다\(받은 (\d+) · 목록 (\d+) 바이트\)/g, ' — the size differs from the list (got $1 · listed $2 bytes)'],
  [/ — SHA-256 이 목록과 다릅니다\((\S+)…\) — 쓰지 않았습니다/g, ' — the SHA-256 differs from the list ($1…) — not written'],
  [/(\S+) 를 쓰지 못했습니다\(디스크 · 폴더 권한\)/g, 'Could not write $1 (disk · folder permissions)'],
  [/ — 받지 못했습니다/g, ' — could not download'],
  [/도구 목록의 파일 경로가 맞지 않습니다/g, 'A file path in the tool list is not valid'],
  [/변환 형식은 bf16 · f16 · q8_0 중 하나입니다/g, 'The conversion type is one of bf16 · f16 · q8_0'],
  [/모르는 양자화 형식입니다: /g, 'Unknown quantization type: '],
  [/행렬별 형식은 '이름=형식' 꼴이어야 합니다: /g, "Per-matrix types must look like 'name=type': "],
  [/행렬별 형식은 양자화 형식과 함께 씁니다/g, 'Per-matrix types need a quantization type'],
  [/원래 모델 폴더\(Hugging Face 형식 — config\.json 과 가중치가 있는 폴더\)를 정하세요/g, 'Choose the original model folder (Hugging Face format — the folder with config.json and the weights)'],
  [/원래 모델 폴더에 config\.json 이 없습니다/g, 'The original model folder has no config.json'],
  [/원래 모델 폴더에 가중치 파일\(\.safetensors · \.bin\)이 없습니다/g, 'The original model folder has no weight files (.safetensors · .bin)'],
  [/GGUF 만들기 도구가 준비되지 않았습니다 — '도구 준비'를 먼저 누르세요/g, "The GGUF making tools are not set up — press 'Set up tools' first"],
  [/imatrix 에 쓸 파일이 없습니다/g, 'The file for the imatrix does not exist'],
  [/변환하는 중 — (.+?) → (\S+)/g, 'Converting — $1 → $2'],
  [/변환하지 못했습니다\(convert_hf_to_gguf\.py — 작업 기록을 보세요\)/g, 'Could not convert (convert_hf_to_gguf.py — see the run log)'],
  [/imatrix 에 쓸 글 파일을 읽지 못했습니다/g, 'Could not read the text file for the imatrix'],
  [/imatrix 를 만드는 중\(llama-imatrix — GPU\)/g, 'Making the imatrix (llama-imatrix — GPU)'],
  [/imatrix 를 만들지 못했습니다\(llama-imatrix — 작업 기록을 보세요\)/g, 'Could not make the imatrix (llama-imatrix — see the run log)'],
  [/imatrix 파일을 읽지 못했습니다/g, 'Could not read the imatrix file'],
  [/양자화하는 중 — (\S+)\((\d+)\/(\d+)\)/g, 'Quantizing — $1 ($2/$3)'],
  [/(\S+) 로 양자화하지 못했습니다\(llama-quantize — 작업 기록을 보세요\)/g, 'Could not quantize to $1 (llama-quantize — see the run log)'],
  [/만든 파일을 모델 목록과 작업 공간에 더하는 중/g, 'Adding the new files to the model list and the workspace'],
  [/변환한 파일을 모델 폴더로 옮기지 못했습니다/g, 'Could not move the converted file to the model folder'],
  [/변환한 파일을 원래 모델과 견주는 중\(변환 검증\)/g, 'Comparing the converted file with the original model (conversion check)'],
  [/(\S+) — 엔진이 받지 않습니다: /g, '$1 — not supported by the engine: '],
  [/(\S+) 를 (\S+) 로 옮기지 못했습니다/g, 'Could not move $1 to $2'],
  [/GGUF 만들기 — 변환 (\S+) → 기준 모델/g, 'Make GGUF — converted $1 → reference model'],
  [/GGUF 만들기 — 기준 모델에서/g, 'Make GGUF — from the reference model'],
  [/만들기의 출발은 convert · ref 중 하나입니다/g, 'The make source is convert or ref'],
  [/기준 모델에서 만들 때는 양자화 형식을 하나 이상 고르세요/g, 'Choose at least one quantization type when making from the reference model'],
  [/기준 모델이 없습니다 — 원래 모델 폴더에서 변환하세요/g, 'There is no reference model — convert from the original model folder'],
  [/기준 모델이 (\S+) 파일입니다 — 다시 양자화합니다\(--allow-requantize\)\. 16비트 파일에서 만든 것보다 품질이 낮을 수 있습니다/g, 'The reference model is a $1 file — requantizing (--allow-requantize). Quality can be lower than from a 16-bit file'],
  [/기준 모델을 ASCII 경로로 복사하는 중\(llama\.cpp 의 도구가 읽게\)/g, "Copying the reference model to an ASCII path (for llama.cpp's tools)"],
  [/기준 모델을 복사하지 못했습니다\(디스크 공간\)/g, 'Could not copy the reference model (disk space)'],
  [/GGUF 만들기 — 변환 (\S+) → (\S+)/g, 'Make GGUF — converted $1 → $2'],
  [/ · 변환 검증 못 함: /g, ' · conversion check failed: '],
  [/ · 변환 검증 문제 (\d+) · 확인 필요 (\d+) · 통과 (\d+)/g, ' · conversion check $1 problems · $2 to check · $3 passed'],
  [/GGUF 를 읽지 못했습니다/g, 'Could not read the GGUF'],
  [/(\S+): 형식 (\d+) — 이 길의 행렬은 (.+?) 여야 한다/g, '$1: type $2 — matrices on this path must be $3'],   // 엔진의 모델 확인(엔진/가중치K.gl)
  // 2-3 — 원래 모델과 견주기(원래비교일 · data/orig_compare.py)
  [/원래 모델에 기준 모델의 답을 넣어 견주는 중\(transformers · CPU · (\S+)\)/g, "Feeding the reference model's answers to the original model (transformers · CPU · $1)"],
  [/원래 모델과 견주기 — 위치 (\d+) · KL (\S+) · top-1 ([\d.]+) %/g, 'Compared with the original model — $1 positions · KL $2 · top-1 $3 %'],
  [/원래 모델을 읽었다 — (\S+) · transformers (\S+) · ([\d.]+) 초/g, 'Loaded the original model — $1 · transformers $2 · $3 s'],
  [/원래 모델을 읽었다 — (\S+) · ([\d.]+) 초/g, 'Loaded the original model — $1 · $2 s'],
  [/(\d+)\/(\d+) (\S+) — 위치 (\d+) · KL (\S+)/g, '$1/$2 $3 — $4 positions · KL $5'],
  [/원래 모델과 견줌 — 대화 (\d+) · 위치 (\d+) · KL (\S+) · top-1 ([\d.]+) %/g, 'Compared with the original model — $1 conversations · $2 positions · KL $3 · top-1 $4 %'],
  [/계산 형식은 float32 · bfloat16 중 하나입니다/g, 'The compute type is float32 or bfloat16'],
  [/이 결과에는 견줄 기준값\(같은 토큰의 확률\)이 없습니다/g, 'This result has no reference values to compare (probabilities of the same tokens)'],
  [/GGUF 만들기 도구\(파이썬 가상 환경\)가 준비되지 않았습니다 — '도구 준비'를 먼저 누르세요/g, "The Make GGUF tools (Python virtual environment) are not set up — press 'Set up tools' first"],
  [/앱의 data\\orig_compare\.py 가 없습니다/g, "The app's data\\orig_compare.py is missing"],
  [/원래 모델과 견주지 못했습니다\(orig_compare\.py — 작업 기록을 보세요\)/g, 'Could not compare with the original model (orig_compare.py — see the run log)'],
  [/원래 모델과 견준 값이 없습니다/g, 'No comparison with the original model yet'],
  [/결과를 쓰지 못했습니다/g, 'Could not write the result'],
  // 4-2 — 개선 측정(개선일) · 만든 뒤 이어서 평가
  [/되돌릴 형식은 q6_k · q5_k · bf16 · f16 중 하나입니다/g, 'The restore type is one of q6_k · q5_k · bf16 · f16'],
  [/기준 버전은 GGUF 만들기로 양자화한 버전이어야 합니다\(만든 설정이 있어야 함\)/g, 'The base version must be one quantized with Make GGUF (it needs its settings)'],
  [/기준 버전의 양자화 형식을 읽지 못했습니다/g, "Could not read the base version's quantization type"],
  [/기준 버전의 imatrix 파일이 없습니다: /g, "The base version's imatrix file is missing: "],
  [/개선 측정 — 기준 버전 (\S+)\((\S+)\) · 되돌릴 형식 (\S+) · 시험 (\d+) 개 · 데이터 (\S+) · 답 (\d+) 토큰/g, 'Improvement measurement — base version $1 ($2) · restore type $3 · $4 trials · data $5 · answers $6 tokens'],
  [/시험 파일을 만드는 중 — (.+?)\((\d+)\/(\d+)\)/g, 'Making trial files — $1 ($2/$3)'],
  [/\[(.+?)\] 시험 파일을 만들지 못했습니다 — 건너뜀/g, '[$1] Could not make the trial file — skipped'],
  [/시험을 평가하지 못했습니다: /g, 'Could not evaluate the trials: '],
  [/개선 측정 — (\S+) 기준 · 시험 (\d+) 개 · 가장 크게 줄인 것 (.+?)\(KL (\S+) → (\S+), −([\d.]+) %\)/g, 'Improvement measurement — base $1 · $2 trials · largest reduction $3 (KL $4 → $5, −$6 %)'],
  [/개선 측정 — (\S+) 기준 · 시험 (\d+) 개 · KL 을 줄인 시험이 없습니다/g, 'Improvement measurement — base $1 · $2 trials · no trial reduced the KL'],
  [/평가하지 못했습니다\(작업 기록 참고\)/g, 'Could not evaluate (see the run log)'],
  [/\[(.+?)\] 바뀐 행렬이 없습니다\(기준 버전과 같은 크기\) — 건너뜀/g, '[$1] No matrix changed (same size as the base version) — skipped'],
  [/바뀐 행렬이 없습니다\(기준 버전과 같은 크기\)/g, 'No matrix changed (same size as the base version)'],
  [/시험 파일을 만들지 못했습니다/g, 'Could not make the trial file'],
  [/개선 측정 보고서가 없습니다/g, 'No improvement measurement yet'],
  [/ · 이어서 평가 못 함: /g, ' · follow-up evaluation failed: '],
  [/엔진이 받지 않는 파일입니다/g, 'The engine does not support this file'],
  [/파일을 열 수 없습니다/g, 'Cannot open the file'],
);

// view: ws(작업 공간) · result(결과) · compare(전후 비교) · item(답이 달라진 곳). res · cmp · items 는 받아 둔 응답(결과 폴더는 다시 쓰지 않는다)
const EV = {
  list: null, listSig: '', at: 0, loading: false, err: '',
  sel: load('geulpu.ev.sel', ''), view: 'ws', dir: '', item: -1, cmpA: '', cmpB: '',
  res: {}, cmp: {}, items: {}, filter: 'differ', pick: [], opts: {}, jobSig: '', tick: 0, checks: {}, checkV: '', origDir: {},
  tools: undefined, mk: {},   // 4-1 — GGUF 만들기 도구(/api/eval/tools) · 작업 공간마다 만들기 양식
  orig: {}, origDt: 'float32',   // 2-3 — 결과마다 원래 모델과 견준 값(/api/eval/original) · 계산 형식
  imp: {}, impo: {},   // 4-2 — 작업 공간마다 마지막 개선 측정 보고서(/api/eval/improve) · 측정 양식 · 고른 무리
};
const EVF = new Set();   // 받는 중인 것
const evEnc = encodeURIComponent;
const evWs = id => (EV.list && EV.list.workspaces.find(w => w.id === id)) || null;
const evKL = x => (typeof x === 'number' && isFinite(x) ? (x >= 0.1 ? x.toFixed(3) : x >= 0.0001 ? x.toFixed(4) : x > 0 ? x.toExponential(1) : '0') : '-');
const evPct = x => (typeof x === 'number' && isFinite(x) ? (x * 100).toFixed(2) + '%' : '-');
const evNum = (x, d) => (typeof x === 'number' && isFinite(x) ? x.toFixed(d) : '-');
const evProb = lp => (typeof lp === 'number' && isFinite(lp) ? pct(lp) : '-');
const evDirWhen = d => { const m = /^(\d{4})(\d{2})(\d{2})-(\d{2})(\d{2})(\d{2})/.exec(String(d || '')); return m ? `${m[1]}-${m[2]}-${m[3]} ${m[4]}:${m[5]}` : String(d || ''); };
const evMode = m => (m === 'fast' ? T('빠름') : T('정밀'));
const evFinish = f => ({ stop: T('끝 토큰'), length: T('최대 길이'), tool_calls: T('도구 호출'), text: '-' })[f] || (f || '-');
const evRefLabel = w => (w.reference && w.reference.model ? labelById(w.reference.model) : T('기준 모델 없음'));
const evVerLabel = (w, vid) => { const v = w.versions.find(x => x.id === vid); return v ? `${vid} · ${modelOf(v.model) ? labelById(v.model) : v.file}` : vid; };
const evResults = w => [...(w.results || [])].sort((a, b) => byName(a.dir, b.dir));
const evLatest = (w, vid) => evResults(w).filter(r => r.version === vid).pop() || null;
const evChat = r => !!(r && r.data && r.data.kind === 'chat');
const evSameSet = (a, b) => !!(a && b && a.data && b.data && a.data.sha256 === b.data.sha256 && JSON.stringify(evSetKey(a.settings)) === JSON.stringify(evSetKey(b.settings)));
const evSetKey = s => (s ? [s.mode, s.max_tokens, s.thinking, s.top_k, s.chunks] : null);

async function evLoad() {
  if (EV.loading) return;
  EV.loading = true;
  EV.at = Date.now();
  try {
    const r = await api('/api/eval');
    const sig = JSON.stringify(r);
    EV.err = '';
    if (sig !== EV.listSig) { EV.listSig = sig; EV.list = r; }
  } catch (e) { EV.err = e.message; }
  finally { EV.loading = false; EV.at = Date.now(); renderEval(); }
}
function evFetch(kind, key, url) {
  if (EV[kind][key] !== undefined || EVF.has(kind + key)) return;
  EVF.add(kind + key);
  api(url).then(r => { EV[kind][key] = r; }, e => { EV[kind][key] = { error: e.message }; })
    .finally(() => { EVF.delete(kind + key); EV.tick++; renderEval(); });
}
const evResKey = (w, dir) => w.id + '/' + dir;
const evDirLabel = (w, dir) => { const r = (w.results || []).find(x => x.dir === dir); return r ? `${r.version} · ${evDirWhen(dir)}` : dir; };
const evRes = (w, dir) => EV.res[evResKey(w, dir)];
// 지금 보기에 드는 응답을 받는다(결과 · 비교 · 대화)
function evNeed(w) {
  if (!w) return;
  const res = dir => evFetch('res', evResKey(w, dir), `/api/eval/result?workspace=${evEnc(w.id)}&dir=${evEnc(dir)}`);
  if ((EV.view === 'result' || EV.view === 'item') && EV.dir) res(EV.dir);
  if (EV.view === 'item' && EV.dir && EV.item >= 0) evFetch('items', `${w.id}/${EV.dir}/${EV.item}`, `/api/eval/item?workspace=${evEnc(w.id)}&dir=${evEnc(EV.dir)}&i=${EV.item}`);
  if (EV.view === 'compare' && EV.cmpA && EV.cmpB) {
    evFetch('cmp', `${w.id}/${EV.cmpA}/${EV.cmpB}`, `/api/eval/compare?workspace=${evEnc(w.id)}&a=${evEnc(EV.cmpA)}&b=${evEnc(EV.cmpB)}`);
    res(EV.cmpA); res(EV.cmpB);
  }
}
// 2-2 — 버전의 마지막 변환 검증 보고서(없으면 {missing: true})
function evCheckNeed(w, vid) {
  const key = `${w.id}/${vid}`;
  if (EV.checks[key] !== undefined || EVF.has('chk' + key)) return;
  EVF.add('chk' + key);
  api(`/api/eval/check?workspace=${evEnc(w.id)}&version=${evEnc(vid)}`).then(r => { EV.checks[key] = r; }, () => { EV.checks[key] = { missing: true }; })
    .finally(() => { EVF.delete('chk' + key); EV.tick++; renderEval(); });
}
function evGo(view, o) {
  Object.assign(EV, { view }, o || {});
  if (o && o.sel !== undefined) store('geulpu.ev.sel', EV.sel);
  renderEval(true);
}
function evShow() { EV.at = 0; }

// ---------- 작업 공간 · 버전 · 데이터 — 창 ----------
const evRunnables = () => sortedModels().filter(runnable);
const evModelOpts = (ms, sel) => ms.map(m => h('option', { value: m.id, selected: m.id === sel ? true : null }, `${displayName(m)} · ${quantOf(m)} · ${fmtBytes(m.size)}`));
function evNewWs() {
  const ms = evRunnables();
  const name = h('input', { class: 'input', type: 'text', value: T('새 작업 공간'), maxlength: 80 });
  const def = ms.length ? (ms.find(m => /BF16|F16|F32/.test(quantOf(m))) || ms.find(m => quantOf(m) === 'Q8_0') || ms[0]).id : '';
  // 4-1 — 기준 모델 없이도 만든다: 원래 모델에서 'GGUF 만들기'로 변환한 파일이 기준 모델이 된다
  const ref = h('select', { class: 'input', 'aria-label': T('기준 모델') }, h('option', { value: '', selected: def ? null : true }, T('없음 — GGUF 만들기로 원래 모델에서 변환')), evModelOpts(ms, def));
  const orig = h('input', { class: 'input mono', type: 'text', placeholder: 'C:\\models\\my-finetune', 'aria-label': T('원래 모델 폴더') });
  const go = async () => {
    try {
      const w = await api('/api/eval/workspace', { action: 'create', name: name.value.trim() || T('새 작업 공간'), reference: ref.value, original_dir: orig.value.trim() });
      closeModal();
      await evLoad();
      evGo('ws', { sel: w.id });
    } catch (e) { toast(e.message, 'err'); }
  };
  modal(h('div', { class: 'col g16' }, h('h2', null, T('새 작업 공간')),
    h('p', { class: 'm0 muted' }, T('작업 공간 하나는 모델 하나의 작업입니다 — 기준 모델과, 그와 비교할 GGUF 파일(버전)들과 결과를 묶습니다.')),
    h('label', { class: 'col g8' }, h('span', { class: 'small muted' }, T('이름')), name),
    h('label', { class: 'col g8' }, h('span', { class: 'small muted' }, T('기준 모델')), ref,
      h('span', { class: 'small muted' }, T('같은 모델의 BF16 · F16 파일이 가장 좋습니다(그래픽 메모리에 들지 않으면 Q8_0). 평가할 파일은 모두 이 모델과 비교합니다.'))),
    h('label', { class: 'col g8' }, h('span', { class: 'small muted' }, T('원래 모델 폴더(고름 — Hugging Face 형식)')), orig,
      h('span', { class: 'small muted' }, T('GGUF 만들기와 변환 검증이 이 폴더를 씁니다. 나중에 정해도 됩니다.')))),
  [h('button', { class: 'btn ghost', onclick: () => closeModal() }, T('취소')), h('button', { class: 'btn primary', onclick: go }, T('만들기'))], false, 'input');
}
function evRename(w) {
  const name = h('input', { class: 'input', type: 'text', value: w.name, maxlength: 80 });
  const go = async () => {
    try { await api('/api/eval/workspace', { action: 'update', id: w.id, name: name.value.trim() || w.name }); closeModal(); evLoad(); }
    catch (e) { toast(e.message, 'err'); }
  };
  modal(h('div', { class: 'col g16' }, h('h2', null, T('이름 바꾸기')), h('label', { class: 'col g8' }, h('span', { class: 'small muted' }, T('이름')), name)),
    [h('button', { class: 'btn ghost', onclick: () => closeModal() }, T('취소')), h('button', { class: 'btn primary', onclick: go }, T('저장'))], false, 'input');
}
function evSetRef(w) {
  const ms = evRunnables().filter(m => !w.versions.some(v => v.model === m.id));
  if (!ms.length) { toast(T('고를 수 있는 모델이 없습니다')); return; }
  const ref = h('select', { class: 'input', 'aria-label': T('기준 모델') }, evModelOpts(ms, w.reference.model));
  const go = async () => {
    try { await api('/api/eval/workspace', { action: 'update', id: w.id, reference: ref.value }); closeModal(); evLoad(); }
    catch (e) { toast(e.message, 'err'); }
  };
  modal(h('div', { class: 'col g16' }, h('h2', null, T('기준 모델 바꾸기')),
    h('p', { class: 'm0 muted' }, T('지난 결과는 그대로 남지만, 기준 모델이 다른 결과끼리는 전후 비교를 하지 않습니다.')), ref),
    [h('button', { class: 'btn ghost', onclick: () => closeModal() }, T('취소')), h('button', { class: 'btn primary', onclick: go }, T('저장'))]);
}
async function evDelWs(w) {
  if (!(await confirmBox(T('작업 공간 지우기'), h('p', { class: 'muted m0' }, T`${w.name} — 목록에서 지웁니다. 결과 파일은 자료 폴더의 평가\\지운작업공간\\ 으로 옮겨 둡니다(모델 파일은 그대로).`), T('지우기'), 'danger'))) return;
  try { await api('/api/eval/workspace', { action: 'delete', id: w.id }); EV.sel = ''; store('geulpu.ev.sel', ''); await evLoad(); evGo('ws'); }
  catch (e) { toast(e.message, 'err'); }
}
function evAddVersion(w) {
  const ms = evRunnables().filter(m => m.id !== w.reference.model && !w.versions.some(v => v.model === m.id));
  if (!ms.length) { toast(T('더할 수 있는 모델이 없습니다 — 모델 화면에서 GGUF 파일을 더하세요')); return; }
  const sel = h('select', { class: 'input', 'aria-label': T('평가할 GGUF') }, evModelOpts(ms, ms[0].id));
  const go = async () => {
    try {
      const r = await api('/api/eval/version', { workspace: w.id, model: sel.value });
      closeModal();
      const o = EV.opts[w.id];
      if (o && r.current) o.versions.add(r.current);
      evLoad();
    } catch (e) { toast(e.message, 'err'); }
  };
  modal(h('div', { class: 'col g16' }, h('h2', null, T('버전 더하기')),
    h('p', { class: 'm0 muted' }, T('기준 모델과 비교할 GGUF 파일을 고르세요. 파일은 옮기지 않고 앱의 모델 목록에 있는 파일을 그대로 씁니다.')), sel),
    [h('button', { class: 'btn ghost', onclick: () => closeModal() }, T('취소')), h('button', { class: 'btn primary', onclick: go }, T('더하기'))]);
}
async function evDelVersion(w, v) {
  if (!(await confirmBox(T('버전 지우기'), h('p', { class: 'muted m0' }, T`${v.id} — 작업 공간에서 뺍니다. 모델 파일과 지난 결과는 그대로 남습니다.`), T('지우기'), 'danger'))) return;
  try { await api('/api/eval/version', { workspace: w.id, action: 'delete', id: v.id }); evLoad(); }
  catch (e) { toast(e.message, 'err'); }
}
// 2-1 — 메타데이터 고치기: 채팅 템플릿 · 특수 토큰(번호) · add_bos/eos 를 바꾼 새 GGUF(텐서는 바이트 그대로)를 새 버전으로
const EV_SPECIAL = [['bos', 'tokenizer.ggml.bos_token_id', 'BOS'], ['eos', 'tokenizer.ggml.eos_token_id', 'EOS'], ['eot', 'tokenizer.ggml.eot_token_id', 'EOT'], ['pad', 'tokenizer.ggml.padding_token_id', 'PAD']];
async function evFixMeta(w, v) {
  let M;
  try { M = await api('/api/eval/meta?model=' + evEnc(v.model)); } catch (e) { toast(e.message, 'err'); return; }
  const ctl = M.control || [];
  const tpl = h('textarea', { class: 'input mono ev-tpl', rows: 12, spellcheck: 'false', 'aria-label': T('채팅 템플릿') });
  tpl.value = M.chat_template || '';
  const sp = EV_SPECIAL.map(([k, key, label]) => {
    const cur = M.special && M.special[k];
    const ids = new Map(ctl.map(c => [c.id, c.text]));
    if (cur && !ids.has(cur.id)) ids.set(cur.id, cur.text || '');
    const sel = h('select', { class: 'input', 'aria-label': label },
      h('option', { value: '' }, T('(키 없음)')),
      [...ids].sort((a, b) => a[0] - b[0]).map(([id, text]) => h('option', { value: String(id), selected: cur && cur.id === id ? true : null }, `${id} · ${text}`)));
    return { k, key, label, cur, sel };
  });
  const flag = (key, cur) => h('select', { class: 'input', 'aria-label': key },
    h('option', { value: '', selected: cur === null ? true : null }, T('(키 없음)')),
    h('option', { value: 'true', selected: cur === true ? true : null }, 'true'), h('option', { value: 'false', selected: cur === false ? true : null }, 'false'));
  const tk = M.tokenizer || {};
  const fb = flag('add_bos_token', tk.add_bos_token === undefined ? null : tk.add_bos_token);
  const fe = flag('add_eos_token', tk.add_eos_token === undefined ? null : tk.add_eos_token);
  const go = async () => {
    const ch = [];
    if (tpl.value !== (M.chat_template || '')) ch.push(tpl.value ? { key: 'tokenizer.chat_template', type: 'string', value: tpl.value } : { key: 'tokenizer.chat_template', type: 'remove' });
    for (const s of sp) {
      const was = s.cur ? String(s.cur.id) : '';
      if (s.sel.value !== was) ch.push(s.sel.value === '' ? { key: s.key, type: 'remove' } : { key: s.key, type: 'u32', value: +s.sel.value });
    }
    for (const [sel, key, cur] of [[fb, 'tokenizer.ggml.add_bos_token', tk.add_bos_token], [fe, 'tokenizer.ggml.add_eos_token', tk.add_eos_token]]) {
      const was = cur === null || cur === undefined ? '' : String(cur);
      if (sel.value !== was) ch.push(sel.value === '' ? { key, type: 'remove' } : { key, type: 'bool', value: sel.value === 'true' });
    }
    if (!ch.length) { toast(T('바뀐 것이 없습니다')); return; }
    try {
      await api('/api/eval/fix', { workspace: w.id, from: v.id, changes: ch });
      closeModal();
      S.jobHide = '';
      EV.at = 0;
      resetSig();
      refresh();
    } catch (e) { toast(e.message, 'err'); }
  };
  modal(h('div', { class: 'col g16' }, h('h2', null, T`메타데이터 고치기 — ${v.id}`),
    h('p', { class: 'm0 muted' }, T('텐서는 그대로 두고 GGUF의 메타데이터(키)만 바꾼 새 파일을 앱의 모델 폴더에 쓰고 새 버전으로 더합니다. 원래 파일은 바뀌지 않습니다.')),
    h('div', { class: 'small muted' }, [M.arch, tk.model, tk.pre, T`어휘 ${fmtNum(tk.vocab || 0)}`, T`키 ${(M.keys || []).length}개`].filter(Boolean).join(NB)),
    h('label', { class: 'col g8' }, h('span', { class: 'small muted' }, T('채팅 템플릿(tokenizer.chat_template — jinja)')), tpl),
    h('div', { class: 'ev-specials' }, sp.map(s => h('label', { class: 'col g4' }, h('span', { class: 'small muted' }, `${s.label} · ${s.key}`), s.sel)),
      h('label', { class: 'col g4' }, h('span', { class: 'small muted' }, 'tokenizer.ggml.add_bos_token'), fb),
      h('label', { class: 'col g4' }, h('span', { class: 'small muted' }, 'tokenizer.ggml.add_eos_token'), fe)),
    h('p', { class: 'small muted m0' }, T('토큰은 어휘의 제어 토큰(control · user-defined)에서 고릅니다. 고친 버전으로 다시 평가하면 앞 버전과 전후 비교할 수 있습니다.'))),
  [h('button', { class: 'btn ghost', onclick: () => closeModal() }, T('취소')), h('button', { class: 'btn primary needs-conn', onclick: go }, T('새 버전 만들기'))], true);
}
function evImport(w) {
  const path = h('input', { class: 'input mono', type: 'text', placeholder: 'C:\\data\\my-set.jsonl' });
  const name = h('input', { class: 'input', type: 'text', placeholder: T('비우면 파일 이름') });
  const go = async () => {
    try {
      const r = await api('/api/eval/data/import', Object.assign({ path: path.value.trim() }, name.value.trim() ? { name: name.value.trim() } : {}));
      closeModal();
      evOpts(w).data = r.name;
      toast(T`데이터를 불러왔습니다 — ${r.name}`, 'ok');
      evLoad();
    } catch (e) { toast(e.message, 'err'); }
  };
  modal(h('div', { class: 'col g16' }, h('h2', null, T('데이터 불러오기')),
    h('p', { class: 'm0 muted' }, T('JSONL — 줄마다 대화 하나: {"messages": […]}, 고를 수 있는 칸 id · category · tools · max_tokens · thinking, 정답을 채점하려면 answer와 grader(number · choice · exact · tool_call). TXT — 텍스트 하나를 문맥 512의 청크로 나눠 llama-perplexity와 같은 위치를 점수 매깁니다.')),
    h('label', { class: 'col g8' }, h('span', { class: 'small muted' }, T('파일 경로(.jsonl · .txt)')), path),
    h('label', { class: 'col g8' }, h('span', { class: 'small muted' }, T('이름(영문 · 숫자 · . _ -)')), name)),
  [h('button', { class: 'btn ghost', onclick: () => closeModal() }, T('취소')), h('button', { class: 'btn primary', onclick: go }, T('불러오기'))], false, 'input');
}

// ---------- 평가 실행 ----------
function evOpts(w) {
  let o = EV.opts[w.id];
  if (!o) {
    const ds = (EV.list && EV.list.data) || [];
    const d0 = ds.find(d => d.name === 'basic60') || ds.find(d => d.kind === 'chat') || ds[0];
    o = EV.opts[w.id] = { versions: new Set(w.current ? [w.current] : w.versions.map(v => v.id)), data: d0 ? d0.name : '', mode: 'exact', max: 512, think: false, chunks: 64, slots: 8, dump: false };
  }
  for (const v of [...o.versions]) if (!w.versions.some(x => x.id === v)) o.versions.delete(v);
  return o;
}
async function evRun(w) {
  const o = evOpts(w);
  const vs = w.versions.map(v => v.id).filter(id => o.versions.has(id));
  if (!vs.length) { toast(T('평가할 버전을 하나 이상 고르세요')); return; }
  if (!o.data) { toast(T('데이터를 고르세요')); return; }
  if (S.streaming) { toast(T('채팅의 답이 끝난 뒤 실행할 수 있습니다')); return; }
  if (S.st.server.state === 'ready' && !(await confirmBox(T('채팅 모델을 내립니다'), h('p', { class: 'muted m0' }, T('평가하는 동안 GPU를 혼자 쓰도록 불러온 채팅 모델을 내립니다. 계속할까요?')), T('계속')))) return;
  try {
    await api('/api/eval/run', { workspace: w.id, versions: vs, data: o.data, mode: o.mode, max_tokens: o.max, thinking: o.think, chunks: o.chunks, slots: o.slots, dump: o.dump });
    S.jobHide = '';
    EV.at = 0;
    resetSig();
    refresh();
  } catch (e) { toast(e.message, 'err'); }
}
// ---------- 2-2 변환 검증 ----------
const evCaseName = n => ({ basic: T('사용자 메시지 하나'), system: T('시스템 메시지'), multi: T('여러 턴'), tools: T('도구 정의 · 호출 · 결과'),
  no_prompt: T('생성 프롬프트 없이'), think_off: T('생각 끔'), think_on: T('생각 켬') })[n] || n;
const evCheckTitle = id => ({ chat_template: T('채팅 템플릿'), eos: 'EOS', bos: 'BOS', special_tokens: T('특수 토큰'), tokenizer: T('토크나이저'),
  hparams: T('설정 — 하이퍼파라미터') })[id] || id;
const evHp = x => (typeof x === 'number' && isFinite(x) ? String(+x.toPrecision(7)) : '-');   // 하이퍼파라미터 값(GGUF 의 f32 를 7 자리로)
const evCheckTag = st => ({ pass: h('span', { class: 'tag ok' }, I('check'), T('통과')), problem: h('span', { class: 'tag err' }, I('alert'), T('문제')),
  check: h('span', { class: 'tag acc' }, I('info'), T('확인 필요')), skip: h('span', { class: 'tag dim' }, T('건너뜀')) })[st] || h('span', { class: 'tag' }, st);
const evTokTxt = t => (t ? `${t.text !== undefined ? t.text : '?'}(${t.id})` : '-');
function evCheckLine(it) {
  const c = it.code;
  if (it.id === 'chat_template') {
    if (c === 'same') return T`대화 경우 ${it.cases}개를 두 템플릿으로 그린 글이 바이트까지 같습니다` + (it.same_text ? '.' : T(' — 템플릿의 글은 다르지만 그린 결과는 같습니다.'));
    if (c === 'differs') return T`대화 경우 ${it.cases}개 중 ${it.differ}개에서 그린 글이 원래 모델과 다릅니다 — 모델이 학습 때와 다른 프롬프트를 받습니다.`;
    if (c === 'gguf_missing') return T('GGUF에 채팅 템플릿이 없습니다 — 런타임이 기본 템플릿(ChatML)을 씁니다.');
    if (c === 'original_error') return T`원래 모델의 템플릿을 해석기가 풀지 못해 비교하지 못했습니다: ${it.original_error || ''}`;
    return T('원래 모델 폴더에 채팅 템플릿(chat_template.jinja · tokenizer_config.json)이 없습니다.');
  }
  if (it.id === 'eos') {
    const o = (it.original || []).map(evTokTxt).join(', ');
    if (c === 'in_original') return T`GGUF의 EOS ${evTokTxt(it.gguf)}가 원래 모델의 끝 토큰입니다.`;
    if (c === 'not_in_original') return T`GGUF의 EOS ${evTokTxt(it.gguf)}가 원래 모델의 끝 토큰(${o})이 아닙니다 — 다른 런타임에서 답이 끝나지 않을 수 있습니다.`;
    if (c === 'gguf_missing') return T`GGUF에 EOS 토큰이 없습니다 — 원래 모델의 끝 토큰은 ${o}입니다.`;
    return T('원래 모델 폴더에 끝 토큰 정보(generation_config.json · tokenizer_config.json)가 없습니다.');
  }
  if (it.id === 'bos') {
    if (c === 'same') return T`add_bos_token이 원래 모델과 같습니다(${String(it.gguf_add_bos)}).`;
    if (c === 'add_bos_differs') return T`add_bos_token이 GGUF ${String(it.gguf_add_bos)} · 원래 모델 ${String(it.original_add_bos)}로 다릅니다 — 프롬프트 앞에 BOS가 붙는지가 달라집니다.`;
    return T('원래 모델 폴더에 tokenizer_config.json이 없습니다.');
  }
  if (it.id === 'special_tokens') {
    if (c === 'all_match') return T`특수 토큰 ${it.count}개가 같은 번호 · 글로 있습니다.`;
    if (c === 'mismatch') return T`특수 토큰 ${it.count}개 중 ${it.mismatched}개가 GGUF와 다릅니다 — 다시 변환해야 합니다(메타데이터로 고칠 수 없음).`;
    return T('원래 모델 폴더에 tokenizer.json이 없습니다.');
  }
  if (it.id === 'hparams') {
    if (c === 'same') return T`config.json의 하이퍼파라미터 ${it.count}개가 GGUF와 같습니다.`;
    if (c === 'differs') return T`config.json의 하이퍼파라미터 ${it.count}개 중 ${it.mismatched}개가 GGUF와 다릅니다 — 다시 변환해야 합니다(메타데이터로 고칠 수 없음).`;
    if (c === 'no_rows') return T`config.json과 GGUF(${it.architecture || '?'})에서 견줄 하이퍼파라미터를 찾지 못했습니다.`;
    return T('원래 모델 폴더에 config.json이 없습니다.');
  }
  if (c === 'same') return T`어휘 ${fmtNum(it.vocab)}개와 병합 ${fmtNum(it.merges)}개가 GGUF와 같습니다.`;
  if (c === 'vocab_differs') return T`어휘 ${fmtNum(it.vocab)}개 중 ${fmtNum(it.vocab_mismatched)}개의 번호 · 글이 GGUF와 다릅니다 — 다시 변환해야 합니다.`;
  if (c === 'merges_differs') return T`병합이 다릅니다(원래 ${fmtNum(it.merges)}개 · GGUF ${fmtNum(it.gguf_merges)}개, 다른 것 ${fmtNum(it.merges_mismatched)}개) — 다시 변환해야 합니다.`;
  if (c === 'type_not_compared') return T`토크나이저 종류 ${it.original_type || '?'}는 비교하지 않습니다(BPE만).`;
  return T('원래 모델 폴더에 tokenizer.json이 없습니다.');
}
function evCheckDetails(it) {
  if (it.id === 'chat_template' && Array.isArray(it.details)) {
    return h('details', { class: 'fold' }, h('summary', null, I('chevR', 's'), T`대화 경우 ${it.details.length}개`),
      h('div', { class: 'col g8 mt8' }, it.details.map(d => h('div', { class: 'col g4' },
        h('div', { class: 'row g8 small' }, d.same ? h('span', { class: 'tag ok' }, T('같음')) : h('span', { class: 'tag err' }, T('다름')), h('span', { class: 'ink' }, evCaseName(d.name)),
          d.same ? null : h('span', { class: 'muted' }, T`${d.at}바이트부터 다름 · GGUF ${d.gguf_len} · 원래 ${d.original_len}바이트`)),
        d.same ? null : h('div', { class: 'dbg-vs' },
          h('div', null, h('div', { class: 'eyebrow' }, 'GGUF'), h('pre', { class: 'dbg-ans sm mono' }, d.gguf_error ? T`오류: ${d.gguf_error}` : tokShow(d.gguf || ''))),
          h('div', null, h('div', { class: 'eyebrow' }, T('원래 모델')), h('pre', { class: 'dbg-ans sm mono' }, d.original_error ? T`오류: ${d.original_error}` : tokShow(d.original || ''))))))));
  }
  if (it.id === 'hparams' && Array.isArray(it.rows) && it.rows.length) {
    return h('details', { class: 'fold' }, h('summary', null, I('chevR', 's'), T`하이퍼파라미터 ${it.rows.length}개`),
      h('div', { class: 'col g4 mt8 small' }, it.saved_with ? h('div', { class: 'muted' }, T`config.json — transformers ${it.saved_with}로 저장`) : null,
        it.rows.map(r => h('div', { class: 'row g8 wrap' }, r.same ? h('span', { class: 'tag ok' }, T('같음')) : h('span', { class: 'tag err' }, T('다름')),
          h('span', { class: 'mono' }, `${r.key} · ${r.config}`), h('span', { class: 'muted' }, T`원래 ${evHp(r.original)} · GGUF ${evHp(r.gguf)}`)))));
  }
  const ex = it.examples || [];
  if (!ex.length) return null;
  return h('details', { class: 'fold' }, h('summary', null, I('chevR', 's'), T`예 ${ex.length}개`),
    h('div', { class: 'col g4 mt8 small mono' }, ex.map(e => h('div', null, `${e.id}: ${JSON.stringify(e.original)} → GGUF ${e.gguf !== undefined ? JSON.stringify(e.gguf) : '-'}${e.gguf_type !== undefined ? ` (type ${e.gguf_type})` : ''}`))));
}
async function evApplyFix(w, vid, changes) {
  const v = w.versions.find(x => x.id === vid);
  const rm = vid === 'ref' ? modelOf(w.reference.model) : null;
  const keys = [...new Set(changes.map(c => c.key))].join(', ');
  const from = vid === 'ref' ? T('기준 모델') : vid;
  if (!(await confirmBox(T('고친 버전 만들기'), h('p', { class: 'muted m0' }, T`${from}에서 메타데이터만 바꾼 새 버전을 만듭니다 — 바꾸는 키: ${keys}. 새 파일이 모델 폴더에 쓰입니다(${fmtBytes(v ? v.size : rm ? rm.size : 0)}).`), T('만들기')))) return;
  try { await api('/api/eval/fix', { workspace: w.id, from: vid, changes }); S.jobHide = ''; EV.at = 0; resetSig(); refresh(); }
  catch (e) { toast(e.message, 'err'); }
}
function evCheckCard(w, busy) {
  // 검사할 것 — 버전들과(4-1) 기준 모델('ref' — GGUF 만들기가 변환한 파일일 때가 많다)
  const opts = (w.reference.model ? [{ id: 'ref', file: w.reference.file }] : []).concat(w.versions);
  if (!opts.length) return null;
  const vid = EV.checkV && opts.some(v => v.id === EV.checkV) ? EV.checkV : (w.current || (w.versions[0] ? w.versions[0].id : 'ref'));
  evCheckNeed(w, vid);
  const R = EV.checks[`${w.id}/${vid}`];
  const dirv = EV.origDir[w.id] !== undefined ? EV.origDir[w.id] : ((w.original && w.original.dir) || '');
  const dir = h('input', { class: 'input mono', type: 'text', value: dirv, placeholder: 'C:\\models\\my-finetune', 'data-fk': 'ev-orig', 'aria-label': T('원래 모델 폴더'),
    oninput: ev => { EV.origDir[w.id] = ev.target.value; } });
  const run = async () => {
    const d = dir.value.trim();
    if (!d) { toast(T('원래 모델 폴더를 정하세요')); return; }
    try { await api('/api/eval/check', { workspace: w.id, version: vid, original_dir: d }); S.jobHide = ''; EV.at = 0; resetSig(); refresh(); }
    catch (e) { toast(e.message, 'err'); }
  };
  let body;
  if (R === undefined) body = h('div', { class: 'row g8 muted small' }, h('i', { class: 'dot busy' }), T('불러오는 중'));
  else if (R.missing) body = h('p', { class: 'small muted m0' }, T`${vid === 'ref' ? T('기준 모델') : vid}는 아직 검사하지 않았습니다.`);
  else {
    const its = R.items || [];
    const fixes = its.filter(it => it.fix && Array.isArray(it.fix.changes)).flatMap(it => it.fix.changes);
    const n = st => its.filter(it => it.status === st).length;
    body = h('div', { class: 'col g12' },
      h('div', { class: 'row g8 small' }, h('span', { class: 'muted' }, T`${R.version} · ${String(R.finished || '').slice(0, 16)} · 문제 ${n('problem')} · 확인 필요 ${n('check')} · 통과 ${n('pass')}`),
        h('span', { class: 'grow' }),
        fixes.length ? h('button', { class: 'btn sm primary needs-conn', disabled: busy ? true : null, onclick: () => evApplyFix(w, vid, fixes) }, I('edit', 's'), T`고친 버전 만들기(${fixes.length})`) : null),
      h('div', { class: 'list ev-chks' }, its.map(it => h('div', { class: 'ev-chk' },
        h('div', { class: 'row g8' }, evCheckTag(it.status), h('span', { class: 'ink' }, evCheckTitle(it.id)), h('span', { class: 'grow' }),
          it.fix ? h('button', { class: 'btn sm needs-conn', disabled: busy ? true : null, onclick: () => evApplyFix(w, vid, it.fix.changes) }, T('이것만 고치기')) : null),
        h('div', { class: 'small' }, evCheckLine(it)),
        evCheckDetails(it)))));
  }
  return h('div', { class: 'card col g16' },
    h('div', { class: 'row' }, h('span', { class: 'sec' }, T('GGUF 변환 검증')), h('span', { class: 'grow' }),
      h('span', { class: 'small muted' }, T('원래 모델(Hugging Face 형식)의 작은 파일과 견줍니다 — 가중치는 읽지 않습니다'))),
    h('div', { class: 'ev-form' },
      h('label', { class: 'col g8' }, h('span', { class: 'small muted' }, T('원래 모델 폴더(tokenizer_config.json · tokenizer.json · generation_config.json이 있는 폴더)')), dir),
      h('div', { class: 'row g12' },
        h('label', { class: 'col g4' }, h('span', { class: 'small muted' }, T('검사할 버전')),
          h('select', { class: 'input', 'data-fk': 'ev-chkv', onchange: ev => { EV.checkV = ev.target.value; renderEval(true); } },
            opts.map(v => h('option', { value: v.id, selected: v.id === vid ? true : null }, `${v.id === 'ref' ? T('기준 모델') : v.id} · ${v.file}`)))),
        h('div', { class: 'ev-chkrun' }, h('button', { class: 'btn needs-conn', 'data-fk': 'ev-chk', disabled: busy ? true : null, onclick: run }, I('shield', 's'), busy ? T('다른 작업 실행 중') : T('검사'))))),
    body);
}
// ---------- 4-1 GGUF 만들기 ----------
const EV_QUANTS = ['Q8_0', 'Q6_K', 'Q5_K_M', 'Q4_K_M', 'IQ4_XS'];
const evMB = n => `${((n || 0) / 1048576).toFixed(1)} MB`;
function evToolsNeed() {
  if (EV.tools !== undefined || EVF.has('tools')) return;
  EVF.add('tools');
  api('/api/eval/tools').then(r => { EV.tools = r; }, e => { EV.tools = { error: e.message }; })
    .finally(() => { EVF.delete('tools'); EV.tick++; renderEval(); });
}
// 받을 것(출처 · 크기)을 보이고, 누르면 작업 'tools'
function evToolsSetup() {
  const t = EV.tools || {}, b = t.build || {}, c = t.convert || {};
  const py = h('input', { class: 'input mono', type: 'text', placeholder: T('비우면 자동(py -3 · python · python3)'), 'aria-label': T('파이썬 경로') });
  const row = (what, size, src) => h('div', { class: 'ev-chk' }, h('div', { class: 'ink' }, what), h('div', { class: 'small muted' }, [size, src].filter(Boolean).join(NB)));
  const go = async () => {
    try { await api('/api/eval/tools', { python: py.value.trim() }); closeModal(); S.jobHide = ''; EV.at = 0; resetSig(); refresh(); }
    catch (e) { toast(e.message, 'err'); }
  };
  const cands = t.python_candidates || [];
  modal(h('div', { class: 'col g16' }, h('h2', null, T('GGUF 만들기 도구 준비')),
    h('p', { class: 'm0 muted' }, T('원래 모델(Hugging Face 형식)을 GGUF로 바꾸고 양자화하는 데 llama.cpp의 도구를 씁니다. 아래 파일을 받아 자료 폴더의 tools에 둡니다 — 앱이 정한 판이고, 받은 파일마다 SHA-256을 확인합니다.')),
    h('div', { class: 'list ev-chks' },
      row(T`llama.cpp ${t.tag} 실행 파일 — ${b.file}`, evMB(b.size), 'github.com/ggml-org/llama.cpp · MIT'),
      row(T`변환 스크립트(convert_hf_to_gguf.py 등) ${c.files}개`, evMB(c.bytes), 'raw.githubusercontent.com/ggml-org/llama.cpp · MIT'),
      row(T`파이썬 패키지 — ${c.torch}(CPU) · transformers 등`, T('약 150 ~ 250 MB(설치하면 1 GB 안쪽)'), 'pypi.org · download.pytorch.org'),
      c.transformers5 ? row(T`transformers ${c.transformers5} 환경 — transformers 5로 저장한 모델 폴더용(torch는 위의 것을 함께 씀)`, T('약 30 MB(설치하면 200 MB 안쪽)'), 'pypi.org') : null),
    h('p', { class: 'small muted m0' }, cands.length ? T`파이썬은 이 PC의 것으로 앱 전용 가상 환경만 만듭니다 — 찾은 파이썬: ${cands.join(', ')}`
      : T('이 PC에서 파이썬을 찾지 못했습니다 — python.org에서 파이썬 3.10 이상을 설치하거나(설치할 때 Add python.exe to PATH), 아래에 경로를 적으세요.')),
    h('label', { class: 'col g8' }, h('span', { class: 'small muted' }, T('파이썬 경로(고름)')), py)),
  [h('button', { class: 'btn ghost', onclick: () => closeModal() }, T('취소')), h('button', { class: 'btn primary', onclick: go }, T('받고 설치'))], false, 'input');
}
// 만든 설정의 한 줄(버전 · 기준 모델) — 변환: 형식 · llama.cpp, 양자화: 형식 · imatrix · 행렬별 형식
function evMadeLine(s) {
  if (!s) return '';
  const tool = s.llama_cpp ? `llama.cpp ${s.llama_cpp}` : null;
  if (s.step === 'convert') return [T`변환 ${String(s.outtype || '').toUpperCase()}`, tool, s.transformers ? `transformers ${s.transformers}` : null].filter(Boolean).join(NB);
  if (s.step === 'quantize') return [s.type, s.imatrix ? `imatrix ${s.imatrix.file || ''}`.trim() : null,
    (s.tensor_types || []).length ? T`행렬별 ${s.tensor_types.join(', ')}` : null, tool].filter(Boolean).join(NB);
  return '';
}
function evMakeCard(w, busy) {
  evToolsNeed();
  const t = EV.tools, st = t && !t.error ? t.state : null;
  const o = EV.mk[w.id] || (EV.mk[w.id] = { from: w.reference.made ? 'ref' : 'convert', outtype: 'bf16', quants: new Set(['Q4_K_M']), imode: 'none', ipath: '', tt: '' });
  if (!w.reference.model) o.from = 'convert';
  const conv = o.from === 'convert';
  const dirv = EV.origDir[w.id] !== undefined ? EV.origDir[w.id] : ((w.original && w.original.dir) || '');
  const dir = h('input', { class: 'input mono', type: 'text', value: dirv, placeholder: 'C:\\models\\my-finetune', 'data-fk': 'ev-mkdir', 'aria-label': T('원래 모델 폴더'),
    oninput: ev => { EV.origDir[w.id] = ev.target.value; } });
  const seg = (key, vals, label) => h('div', { class: 'seg-ctl', role: 'radiogroup', 'aria-label': label }, vals.map(([v, l]) => h('button', {
    class: o[key] === v ? 'on' : '', role: 'radio', 'aria-checked': o[key] === v ? 'true' : 'false', 'data-fk': `ev-mk-${key}-${v}`,
    onclick: () => { o[key] = v; renderEval(true); } }, l)));
  let tools;
  if (t === undefined) tools = h('div', { class: 'row g8 muted small' }, h('i', { class: 'dot busy' }), T('불러오는 중'));
  else if (t.error) tools = h('div', { class: 'note-line err' }, I('alert', 's'), h('span', null, TM(t.error)));
  else if (!t.ascii_ok) tools = h('div', { class: 'note-line err' }, I('alert', 's'), h('span', null, T('자료 폴더의 경로에 ASCII가 아닌 글자가 있어 llama.cpp의 도구를 쓸 수 없습니다 — ASCII 경로의 자료 폴더로 실행하세요.')));
  else if (st) tools = h('div', { class: 'col g8' },
    h('div', { class: 'small muted' }, st.python5 ? T`도구 — llama.cpp ${st.llama_cpp.tag} · 파이썬 ${st.python.version} · torch ${st.python.torch} · transformers ${st.python.transformers} · ${st.python5.transformers}`
      : T`도구 — llama.cpp ${st.llama_cpp.tag} · 파이썬 ${st.python.version} · torch ${st.python.torch} · transformers ${st.python.transformers}`),
    // #196 — transformers 5 로 저장한 폴더는 transformers 5 환경이 읽는다(앱의 목록에 있는데 아직 받지 않았으면)
    !st.python5 && t.convert && t.convert.transformers5 ? h('div', { class: 'row g12 wrap' },
      h('span', { class: 'small muted grow' }, T('transformers 5로 저장한 모델 폴더를 변환하려면 도구 준비를 다시 하세요 — transformers 5 환경을 더합니다(약 30 MB).')),
      h('button', { class: 'btn sm needs-conn', 'data-fk': 'ev-tools5', disabled: busy ? true : null, onclick: evToolsSetup }, I('download', 's'), T('도구 준비'))) : null);
  else tools = h('div', { class: 'row g12 wrap' }, h('span', { class: 'small muted grow' }, T('GGUF를 만들려면 먼저 도구를 받습니다 — llama.cpp의 실행 파일 · 변환 스크립트 · 파이썬 패키지.')),
    h('button', { class: 'btn sm primary needs-conn', 'data-fk': 'ev-tools', disabled: busy ? true : null, onclick: evToolsSetup }, I('download', 's'), T('도구 준비')));
  const run = async () => {
    const d = dir.value.trim();
    if (conv && !d) { toast(T('원래 모델 폴더를 정하세요')); return; }
    if (!conv && !o.quants.size) { toast(T('양자화 형식을 하나 이상 고르세요')); return; }
    const tt = o.tt.split(/\r?\n/).map(s => s.trim()).filter(Boolean);
    if (tt.length && !o.quants.size) { toast(T('행렬별 형식은 양자화 형식과 함께 씁니다')); return; }
    if (o.imode !== 'none' && !o.ipath.trim()) { toast(T('imatrix에 쓸 파일을 정하세요')); return; }
    try {
      await api('/api/eval/make', { workspace: w.id, from: o.from, source_dir: d, outtype: o.outtype, quants: EV_QUANTS.filter(q => o.quants.has(q)), imatrix: { mode: o.imode, path: o.ipath.trim() }, tensor_types: tt });
      S.jobHide = ''; EV.at = 0; resetSig(); refresh();
    } catch (e) { toast(e.message, 'err'); }
  };
  return h('div', { class: 'card col g16' },
    h('div', { class: 'row' }, h('span', { class: 'sec' }, T('GGUF 만들기')), h('span', { class: 'grow' }),
      h('span', { class: 'small muted' }, T('원래 모델(Hugging Face 형식)에서 — llama.cpp의 변환 · 양자화'))),
    tools,
    h('div', { class: 'ev-form' },
      w.reference.model ? h('div', { class: 'col g8' }, h('span', { class: 'small muted' }, T('출발')),
        seg('from', [['ref', T('기준 모델')], ['convert', T('원래 모델 폴더')]], T('출발')),
        h('span', { class: 'small muted' }, conv ? T('변환한 파일은 새 버전이 됩니다(기준 모델이 이미 있음).') : T`기준 모델 ${evRefLabel(w)}을 그대로 두고 양자화한 파일만 새 버전으로 — 행렬별 형식을 바꿔 다시 만들 때.`)) : null,
      conv ? h('label', { class: 'col g8' }, h('span', { class: 'small muted' }, T('원래 모델 폴더(config.json과 가중치 .safetensors가 있는 폴더)')), dir) : null,
      conv ? h('div', { class: 'col g8' }, h('span', { class: 'small muted' }, T('변환 형식')), seg('outtype', [['bf16', 'BF16'], ['f16', 'F16'], ['q8_0', 'Q8_0']], T('변환 형식')),
        w.reference.model ? null : h('span', { class: 'small muted' }, T('변환한 파일이 이 작업 공간의 기준 모델이 됩니다.'))) : null,
      h('div', { class: 'col g8' }, h('span', { class: 'small muted' }, conv ? T('양자화 형식(여럿 고름 — 고르지 않으면 변환만)') : T('양자화 형식(여럿 고름)')),
        h('div', { class: 'row g12 wrap' }, EV_QUANTS.map(q => h('label', { class: 'check' }, h('input', { type: 'checkbox', 'data-fk': 'ev-mk-q-' + q, checked: o.quants.has(q) ? true : null,
          onchange: ev => { if (ev.target.checked) o.quants.add(q); else o.quants.delete(q); } }), q)))),
      h('div', { class: 'col g8' }, h('span', { class: 'small muted' }, 'imatrix'),
        seg('imode', [['none', T('없음')], ['text', T('글 파일로 만들기')], ['file', T('imatrix 파일')]], 'imatrix'),
        o.imode !== 'none' ? h('input', { class: 'input mono', type: 'text', value: o.ipath, placeholder: o.imode === 'text' ? 'C:\\data\\calibration.txt' : 'C:\\data\\imatrix.gguf',
          'data-fk': 'ev-mk-ipath', 'aria-label': T('imatrix에 쓸 파일'), oninput: ev => { o.ipath = ev.target.value; } }) : null,
        o.imode === 'text' ? h('span', { class: 'small muted' }, T('글 파일의 문장으로 행렬마다 중요한 가중치를 잽니다(llama-imatrix — GPU). 모델이 쓰일 글과 비슷할수록 좋습니다.')) : null),
      h('details', { class: 'fold' }, h('summary', null, I('chevR', 's'), T('행렬별 형식(고급)')),
        h('div', { class: 'col g8 mt8' },
          h('textarea', { class: 'input mono', rows: '3', 'data-fk': 'ev-mk-tt', 'aria-label': T('행렬별 형식'), placeholder: 'attn_v=q6_k\nffn_down=q6_k', oninput: ev => { o.tt = ev.target.value; } }, o.tt),
          h('span', { class: 'small muted' }, T('줄마다 행렬 이름=형식(llama-quantize --tensor-type). 엔진이 받지 않는 섞임이면 버전에 까닭이 보입니다.'))))),
    h('div', null, h('button', { class: 'btn primary needs-conn', 'data-fk': 'ev-make', disabled: busy || !st ? true : null, onclick: run },
      I('play', 's'), busy ? T('다른 작업 실행 중') : T('만들기', 'make'))));
}
// ---------- 4-2 개선 — 행렬 무리마다 되돌려 측정 ----------
function evImpNeed(w) {
  if (EV.imp[w.id] !== undefined || EVF.has('imp' + w.id)) return;
  EVF.add('imp' + w.id);
  api(`/api/eval/improve?workspace=${evEnc(w.id)}`).then(r => { EV.imp[w.id] = r; }, () => { EV.imp[w.id] = { missing: true }; })
    .finally(() => { EVF.delete('imp' + w.id); EV.tick++; renderEval(); });
}
function evImproveCard(w, busy) {
  const qv = w.versions.filter(v => v.made && v.made.kind === 'quantize' && v.made.settings);
  if (!w.reference.model || !qv.length) return null;
  evToolsNeed();
  evImpNeed(w);
  const t = EV.tools, st = t && !t.error ? t.state : null;
  const o = EV.impo[w.id] || (EV.impo[w.id] = { base: '', data: 'basic60', max: 64, restore: 'q6_k', ranges: true, pick: null, pickFor: '' });
  if (!qv.some(v => v.id === o.base)) o.base = (qv.find(v => v.id === w.current) || qv[qv.length - 1]).id;
  const ds = ((EV.list && EV.list.data) || []).filter(d => d.kind === 'chat');
  const R = EV.imp[w.id];
  const rep = R && !R.missing && !R.error && R.base ? R : null;
  const seg = h('div', { class: 'seg-ctl', role: 'radiogroup', 'aria-label': T('되돌릴 형식') }, [['q6_k', 'Q6_K'], ['q5_k', 'Q5_K'], ['bf16', 'BF16']].map(([v, l]) => h('button', {
    class: o.restore === v ? 'on' : '', role: 'radio', 'aria-checked': o.restore === v ? 'true' : 'false', 'data-fk': 'ev-imp-r-' + v,
    onclick: () => { o.restore = v; renderEval(true); } }, l)));
  const run = async () => {
    try { await api('/api/eval/improve', { workspace: w.id, base: o.base, data: o.data, max_tokens: o.max, restore: o.restore, layer_ranges: o.ranges }); S.jobHide = ''; EV.at = 0; resetSig(); refresh(); }
    catch (e) { toast(e.message, 'err'); }
  };
  let table = null;
  if (rep) {
    const bk = rep.base.kl_mean;
    const rows = (rep.rows || []).map(r => {
      const d = typeof r.kl_mean === 'number' && r.kl_mean >= 0 && bk > 0 ? (bk - r.kl_mean) / bk : null;
      const ds = typeof r.size === 'number' && rep.base.size ? r.size - rep.base.size : null;
      return Object.assign({}, r, { d, ds, eff: d !== null && ds > 0 ? (d * 100) / (ds / 1048576) : null });   // 1 MB 당 KL 을 줄인 몫(%)
    });
    const sorted = [...rows].sort((a, b) => (b.d === null ? -9 : b.d) - (a.d === null ? -9 : a.d));
    // 미리 고르기 — 행렬 무리(층 토막은 행렬 무리와 겹친다) 가운데 5 % 넘게 줄인 것을 1 MB 당 줄임이 큰 차례로 셋
    if (!o.pick || o.pickFor !== rep.finished) {
      o.pick = new Set(rows.filter(r => !r.error && r.d !== null && r.d >= 0.05 && r.eff !== null && !/^blk /.test(r.name)).sort((a, b) => b.eff - a.eff).slice(0, 3).map(r => r.name));
      o.pickFor = rep.finished;
    }
    const make = async () => {
      const sel = rows.filter(r => o.pick.has(r.name));
      if (!sel.length) { toast(T('무리를 하나 이상 고르세요')); return; }
      const tt = [].concat(...sel.map(r => r.tensor_types || []), rep.base.tensor_types || []);
      try {
        await api('/api/eval/make', { workspace: w.id, from: 'ref', quants: [rep.base.type], imatrix: rep.base.imatrix ? { mode: 'file', path: rep.base.imatrix } : { mode: 'none' },
          tensor_types: tt, based_on: rep.base.version, then_eval: { data: rep.data, max_tokens: rep.max_tokens, versions: [rep.base.version] } });
        S.jobHide = ''; EV.at = 0; resetSig(); refresh();
      } catch (e) { toast(e.message, 'err'); }
    };
    table = h('div', { class: 'col g8' },
      h('div', { class: 'small muted' }, T`마지막 측정 — 기준 ${rep.base.version}(${rep.base.type}) · KL ${evKL(bk)} · 되돌릴 형식 ${String(rep.restore).toUpperCase()} · 데이터 ${rep.data} · 답 ${rep.max_tokens}토큰 · ${String(rep.finished || '').replace('T', ' ').slice(0, 16)}`),
      h('div', { class: 'list' }, h('table', { class: 'data ev-table' },
        h('thead', null, h('tr', null, h('th', null, ''), h('th', null, T('무리')), h('th', { class: 'r' }, 'KL'), h('th', { class: 'r' }, T('기준 대비')), h('th', { class: 'r' }, T('크기')),
          h('th', { class: 'r' }, T('1 MB 당')))),
        h('tbody', null, sorted.map(r => h('tr', null,
          h('td', null, r.error ? null : h('input', { type: 'checkbox', 'aria-label': r.name, 'data-fk': 'ev-imp-p-' + r.name, checked: o.pick.has(r.name) ? true : null,
            onchange: ev => { if (ev.target.checked) o.pick.add(r.name); else o.pick.delete(r.name); } })),
          h('td', { class: 'mono small' }, r.name),
          r.error ? h('td', { class: 'small err', colspan: '4' }, TM(r.error)) : [
            h('td', { class: 'r num small' }, evKL(r.kl_mean)),
            h('td', { class: 'r num small' }, r.d === null ? '-' : `${r.d >= 0 ? '−' : '+'}${Math.abs(r.d * 100).toFixed(1)}%`),
            h('td', { class: 'r num small' }, r.ds === null ? '-' : `${r.ds >= 0 ? '+' : '−'}${fmtBytes(Math.abs(r.ds))}`),
            h('td', { class: 'r num small' }, r.eff === null ? '-' : `${r.eff >= 0 ? '−' : '+'}${Math.abs(r.eff).toFixed(2)}%`)]))))),
      h('p', { class: 'small muted m0' }, T('미리 고른 것 — 행렬 무리 가운데 KL을 5 % 넘게 줄인 것을 1 MB 당 줄임이 큰 차례로 셋. 무리마다 잰 줄임은 더해지지 않습니다 — 만든 뒤의 평가가 실제 값입니다.')),
      h('div', { class: 'row g12 wrap' }, h('span', { class: 'small muted grow' }, T('고른 무리를 되돌릴 형식으로 바꿔 기준 버전과 같은 형식 · imatrix로 다시 만들고, 기준 버전과 함께 같은 데이터로 평가합니다.')),
        h('button', { class: 'btn primary needs-conn', 'data-fk': 'ev-imp-make', disabled: busy || !st ? true : null, onclick: make }, I('play', 's'), T('고른 무리로 만들기'))));
  }
  const n = 8 + (o.ranges ? 3 : 0);
  return h('div', { class: 'card col g16' },
    h('div', { class: 'row' }, h('span', { class: 'sec' }, T('개선 — 행렬 무리마다 되돌려 측정')), h('span', { class: 'grow' }),
      h('span', { class: 'small muted' }, T('무리 하나만 높은 형식으로 바꾼 시험 파일을 같은 데이터로 잽니다'))),
    st ? null : h('div', { class: 'small muted' }, T('측정하려면 GGUF 만들기에서 도구 준비를 먼저 하세요.')),
    h('div', { class: 'ev-form' },
      h('div', { class: 'row g12 wrap' },
        h('label', { class: 'col g4' }, h('span', { class: 'small muted' }, T('기준 버전')),
          h('select', { class: 'input', 'data-fk': 'ev-imp-base', onchange: ev => { o.base = ev.target.value; } }, qv.map(v => h('option', { value: v.id, selected: v.id === o.base ? true : null }, `${v.id} · ${v.file}`)))),
        h('label', { class: 'col g4' }, h('span', { class: 'small muted' }, T('데이터')),
          h('select', { class: 'input', 'data-fk': 'ev-imp-data', onchange: ev => { o.data = ev.target.value; } }, ds.map(d => h('option', { value: d.name, selected: d.name === o.data ? true : null }, d.name)))),
        h('label', { class: 'col g4 ev-num' }, h('span', { class: 'small muted' }, T('답의 최대 토큰')),
          h('input', { class: 'input num', type: 'number', min: 8, max: 1024, value: o.max, 'data-fk': 'ev-imp-max',
            onchange: ev => { const x = Math.round(+ev.target.value); o.max = Math.min(1024, Math.max(8, isFinite(x) ? x : 64)); ev.target.value = o.max; } }))),
      h('div', { class: 'row g16 wrap' }, h('div', { class: 'col g4' }, h('span', { class: 'small muted' }, T('되돌릴 형식')), seg),
        h('label', { class: 'check' }, h('input', { type: 'checkbox', 'data-fk': 'ev-imp-ranges', checked: o.ranges ? true : null, onchange: ev => { o.ranges = ev.target.checked; renderEval(true); } }), T('층의 세 토막도 잰다')))),
    h('div', { class: 'row g12 wrap' },
      h('button', { class: 'btn needs-conn', 'data-fk': 'ev-imp-run', disabled: busy || !st ? true : null, onclick: run }, I('gauge', 's'), busy ? T('다른 작업 실행 중') : T('측정')),
      h('span', { class: 'small muted grow' }, T`시험 ${n}개쯤 — 시험마다 파일을 만들고 평가합니다(0.6B에서 시험 하나에 20초쯤, 큰 모델은 더 오래). 시험 파일은 잰 뒤 지웁니다.`)),
    table);
}
// ---------- 2-3 원래 모델과 견주기 ----------
// 설정 견주기(orig_compare.py 의 config_check) — config.json · transformers 가 읽은 값 · 기준 모델의 GGUF
function evOrigCfgLine(cc) {
  const rows = cc.rows || [];
  if (cc.status === 'same') return h('div', { class: 'small muted' }, T`설정 — 하이퍼파라미터 ${rows.length}개가 config.json · transformers · GGUF에서 같습니다`);
  if (cc.status === 'gguf_differs') {
    const bad = rows.filter(r => !r.same).map(r => `${r.key} ${evHp(r.config !== null && r.config !== undefined ? r.config : r.transformers)} · GGUF ${evHp(r.gguf)}`).join(', ');
    return h('div', { class: 'note-line err' }, I('alert', 's'), h('span', null, T`GGUF의 하이퍼파라미터가 원래 모델의 설정과 다릅니다 — ${bad}. 다시 변환해야 합니다.`));
  }
  return h('div', { class: 'small muted' }, T('설정 — 기준값을 만든 GGUF를 찾지 못해 견주지 않았습니다'));
}
// 결과의 기준값(GGUF 기준 모델의 답 · 상위 K)을 원래 모델(transformers)에 같은 토큰으로 넣는다 — 변환 · 엔진이 원래 모델을 얼마나 그대로 옮겼나
function evOrigCard(w, dir) {
  evToolsNeed();
  const key = `${w.id}/${dir}`;
  if (EV.orig[key] === undefined && !EVF.has('orig' + key)) {
    EVF.add('orig' + key);
    api(`/api/eval/original?workspace=${evEnc(w.id)}&dir=${evEnc(dir)}`).then(r => { EV.orig[key] = r; }, () => { EV.orig[key] = { missing: true }; })
      .finally(() => { EVF.delete('orig' + key); EV.tick++; renderEval(); });
  }
  const O = EV.orig[key];
  const busy = !!(S.st.job && S.st.job.state === 'running');
  const t = EV.tools, st = t && !t.error ? t.state : null;
  const dirv = EV.origDir[w.id] !== undefined ? EV.origDir[w.id] : ((w.original && w.original.dir) || '');
  const inp = h('input', { class: 'input mono', type: 'text', value: dirv, placeholder: 'C:\\models\\my-finetune', 'data-fk': 'ev-origcmp', 'aria-label': T('원래 모델 폴더'),
    oninput: ev => { EV.origDir[w.id] = ev.target.value; } });
  const seg = h('div', { class: 'seg-ctl', role: 'radiogroup', 'aria-label': T('계산 형식') }, ['float32', 'bfloat16'].map(v => h('button', {
    class: EV.origDt === v ? 'on' : '', role: 'radio', 'aria-checked': EV.origDt === v ? 'true' : 'false', 'data-fk': 'ev-odt-' + v,
    onclick: () => { EV.origDt = v; renderEval(true); } }, v)));
  const run = async () => {
    const d = inp.value.trim();
    if (!d) { toast(T('원래 모델 폴더를 정하세요')); return; }
    try { await api('/api/eval/original', { workspace: w.id, dir, original_dir: d, dtype: EV.origDt }); S.jobHide = ''; EV.at = 0; resetSig(); refresh(); }
    catch (e) { toast(e.message, 'err'); }
  };
  let body = null;
  if (O === undefined) body = h('div', { class: 'row g8 muted small' }, h('i', { class: 'dot busy' }), T('불러오는 중'));
  else if (!O.missing && !O.error) {
    // #196 — 16비트 기준 모델의 바른 변환은 이 PC 에서 KL 3.5×10⁻⁶ ~ 1.0×10⁻⁴(답이 확실할수록 작다), rope 를 잘못 읽은 것은 1.5×10⁻² —
    // 10⁻³ 을 넘으면 짚는다
    const big16 = O.reference_16bit === true && typeof O.kl_mean === 'number' && O.kl_mean > 1e-3;
    body = h('div', { class: 'col g4' },
      h('div', null, T`KL divergence ${evKL(O.kl_mean)} · top-1 일치율 ${evPct(O.top1)} · 위치 ${fmtNum(O.positions || 0)}개 · 대화 ${O.conversations}개`),
      h('div', { class: 'small muted' }, T`답 토큰의 로그확률 차(절댓값 평균) ${evNum(O.answer_logprob_abs_mean, 6)} · 가장 큰 KL ${evKL(O.kl_max)} · ${O.dtype} · transformers ${O.transformers} · ${String(O.finished || '').replace('T', ' ').slice(0, 16)}`),
      O.config_check ? evOrigCfgLine(O.config_check) : null,
      O.reference_16bit !== true ? null : big16
        ? h('div', { class: 'note-line err' }, I('alert', 's'), h('span', null, T('기준 모델이 16비트인데 KL이 10⁻³보다 큽니다 — 변환이나 원래 모델 폴더의 설정이 다를 수 있습니다. 변환 검증을 보세요.')))
        : h('div', { class: 'small muted' }, T('기준 모델이 16비트면 바르게 변환했을 때 KL은 10⁻³보다 훨씬 작습니다(이 PC의 측정 3.5 × 10⁻⁶ ~ 1.0 × 10⁻⁴ — 답이 확실할수록 작음).')));
  }
  return h('div', { class: 'card col g12' },
    h('div', { class: 'row' }, h('span', { class: 'sec' }, T('원래 모델과 견주기')), h('span', { class: 'grow' }),
      h('span', { class: 'small muted' }, T('transformers · CPU — GGUF 만들기의 파이썬 환경'))),
    h('p', { class: 'small muted m0' }, T('기준 모델(GGUF)의 답을 원래 모델에 같은 토큰으로 넣어 위치마다 견줍니다. 0에 가까우면 변환과 엔진이 원래 모델을 그대로 옮긴 것이고, 버전들의 KL은 그 위에 더해진 차이입니다.')),
    body,
    st ? h('div', { class: 'ev-form' },
      h('label', { class: 'col g8' }, h('span', { class: 'small muted' }, T('원래 모델 폴더(config.json과 가중치 .safetensors가 있는 폴더)')), inp),
      h('div', { class: 'row g12 wrap' }, h('div', { class: 'col g4' }, h('span', { class: 'small muted' }, T('계산 형식')), seg),
        h('div', { class: 'ev-chkrun' }, h('button', { class: 'btn needs-conn', 'data-fk': 'ev-origrun', disabled: busy ? true : null, onclick: run }, I('target', 's'), busy ? T('다른 작업 실행 중') : T('견주기')))))
      : h('div', { class: 'small muted' }, T('원래 모델과 견주려면 작업 공간의 GGUF 만들기에서 도구 준비를 먼저 하세요(파이썬 · transformers).')));
}
// 3-2 — 내장 데이터의 출처 · 대화 수 · 라이선스(앱/data/eval/sources.json)
function evDataInfo(d) {
  if (!d || !d.items) return null;
  const parts = [d.by === 'GeulPU' ? T('GeulPU가 쓴 데이터') : d.title,
    d.of ? T`대화 ${fmtNum(d.items)}개 — 원본 ${fmtNum(d.of)}개 중 분야마다 고르게` : T`대화 ${fmtNum(d.items)}개`, d.license,
    d.url ? h('a', { href: d.url, target: '_blank', rel: 'noopener' }, d.url.replace(/^https?:\/\//, '')) : null].filter(Boolean);
  return h('div', { class: 'small muted break' }, parts.map((x, i) => [i ? ' · ' : null, x]));
}
function evRunCard(w, busy) {
  const o = evOpts(w);
  const ds = (EV.list && EV.list.data) || [];
  const d = ds.find(x => x.name === o.data);
  const text = !!(d && d.kind === 'text');
  const cuda = S.st.device && S.st.device.layer === 'cuda';
  if (!cuda) o.mode = 'exact';
  const num = (key, min, max, label) => h('label', { class: 'col g4 ev-num' }, h('span', { class: 'small muted' }, label),
    h('input', { class: 'input num', type: 'number', min, max, value: o[key], 'data-fk': 'ev-o-' + key, onchange: ev => { const x = Math.round(+ev.target.value); o[key] = Math.min(max, Math.max(min, isFinite(x) ? x : o[key])); ev.target.value = o[key]; } }));
  const seg = h('div', { class: 'seg-ctl', role: 'radiogroup', 'aria-label': T('계산 방식') }, ['exact', 'fast'].map(m => h('button', {
    class: o.mode === m ? 'on' : '', role: 'radio', 'aria-checked': o.mode === m ? 'true' : 'false', 'data-fk': 'ev-mode-' + m, disabled: m === 'fast' && !cuda ? true : null,
    onclick: () => { o.mode = m; renderEval(true); },
  }, evMode(m))));
  const ok = w.versions.length && w.reference.model && ds.length;
  return h('div', { class: 'card col g16' },
    h('div', { class: 'row' }, h('span', { class: 'sec' }, T('평가 실행')), h('span', { class: 'grow' }),
      h('span', { class: 'small muted' }, T('실행하는 동안 불러온 채팅 모델은 GPU에서 내립니다'))),
    h('div', { class: 'ev-form' },
      h('div', { class: 'col g8' }, h('span', { class: 'small muted' }, T('평가할 버전')),
        w.versions.length ? h('div', { class: 'row g12' }, w.versions.map(v => h('label', { class: 'check' }, h('input', {
          type: 'checkbox', 'data-fk': 'ev-v-' + v.id, checked: o.versions.has(v.id) ? true : null,
          onchange: ev => { if (ev.target.checked) o.versions.add(v.id); else o.versions.delete(v.id); },
        }), v.id))) : h('span', { class: 'small muted' }, T('먼저 버전을 더하세요'))),
      h('div', { class: 'col g8' }, h('span', { class: 'small muted' }, T('데이터')),
        h('div', { class: 'row g8 nowrap' },
          h('select', { class: 'input', 'data-fk': 'ev-data', 'aria-label': T('데이터'), onchange: ev => { o.data = ev.target.value; renderEval(true); } },
            ds.map(x => h('option', { value: x.name, selected: x.name === o.data ? true : null },
              `${x.name} · ${x.kind === 'text' ? T('텍스트') + ' · ' : ''}${x.source === 'builtin' ? T('내장', 'data') : T('불러옴')}`))),
          h('button', { class: 'btn sm needs-conn', onclick: () => evImport(w) }, I('folder', 's'), T('불러오기'))),
        evDataInfo(d)),
      h('div', { class: 'col g8' }, h('span', { class: 'small muted' }, T('계산 방식')), seg),
      h('div', { class: 'row g16 ev-nums' },
        text ? num('chunks', 1, 4096, T('청크 수(512 토큰)')) : num('max', 1, 8192, T('답의 최대 토큰')),
        num('slots', 1, 8, T('동시 요청')),
        text ? null : h('label', { class: 'check ev-think' }, h('input', { type: 'checkbox', 'data-fk': 'ev-think', checked: o.think ? true : null, onchange: ev => { o.think = ev.target.checked; } }), T('생각(thinking) 켬'))),
      h('label', { class: 'check' }, h('input', { type: 'checkbox', 'data-fk': 'ev-dump', checked: o.dump ? true : null, onchange: ev => { o.dump = ev.target.checked; } }),
        T('위치마다 평가할 모델의 상위 확률도 저장(cand.bin — 다시 계산 · 분석용, 위치마다 약 0.5 KB)'))),
    h('p', { class: 'small muted m0' }, text
      ? T('텍스트 데이터는 같은 토큰의 확률만 비교합니다(llama-perplexity --kl-divergence와 같은 위치).')
      : T('기준 모델이 낸 답을 두 모델이 점수 매기고(같은 토큰의 확률), 두 모델이 저마다의 채팅 템플릿으로 답합니다. 기준 모델의 값은 처음 한 번 만들고, 같은 기준 모델 · 데이터 · 설정이면 다시 씁니다.')),
    h('div', null, h('button', { class: 'btn primary needs-conn', 'data-fk': 'ev-run', disabled: busy || !ok ? true : null, onclick: () => evRun(w) },
      I('play', 's'), busy ? T('다른 작업 실행 중') : T('평가 실행'))));
}

// ---------- 판정 문장(3.7) ----------
function evVerdict(r) {
  if (!r) return null;
  const a = r.answers || {}, m = r.metrics || {};
  const dist = m.positions ? T`KL divergence ${evKL(m.kl_mean)} · top-1 일치율 ${evPct(m.top1)}` : null;
  if (evChat(r) && a.compared) {
    const s = a.differ ? T`대화 ${a.compared}개 중 ${a.differ}개의 답이 기준 모델과 달라졌습니다` + (a.near_tie ? T` — 그중 ${a.near_tie}개는 마진이 작은 스텝(0.05 nat 미만)에서 갈라졌습니다` : '')
      : T`대화 ${a.compared}개의 답이 모두 기준 모델과 같습니다`;
    return [s, dist];
  }
  return [dist || T('비교한 위치가 없습니다'), m.positions ? T`위치 ${fmtNum(m.positions)}개` : null];
}

// ---------- 화면 ----------
const EV_JOBS = ['eval', 'fix', 'check', 'tools', 'make', 'orig', 'improve'];
const evJobOf = () => { const j = S.st && S.st.job; return j && EV_JOBS.includes(j.kind) && (j.state === 'running' || jobKey(j) !== S.jobHide) ? j : null; };
function evJobCard(j) {
  const tone = { running: 'acc', done: 'ok', failed: 'err', cancelled: 'dim' }[j.state] || '';
  return h('div', { class: 'card job ev-job' },
    h('div', { class: 'head' },
      h('div', { class: 't' }, h('div', { class: 'ink h3' }, jobName(j)), h('div', { class: 'small muted', id: 'ev-job-step' }, j.state === 'running' ? TM(j.step || '') : '')),
      h('span', { class: 'tag ' + tone, id: 'ev-job-time' }, ''),
      j.state === 'running' ? h('button', { class: 'btn sm needs-conn', 'data-fk': 'ev-job-stop', onclick: () => api('/api/job/cancel', {}).then(refresh).catch(e => toast(e.message, 'err')) }, I('stop', 's'), T('중지'))
        : h('button', { class: 'icon-btn', 'data-tip': T('카드 닫기'), 'data-tip-pos': 'left', 'aria-label': T('끝난 작업 카드 닫기'), onclick: () => { S.jobHide = jobKey(j); store('geulpu.jobHide', S.jobHide); renderEval(true); } }, I('x', 's'))),
    h('div', { class: 'bar' + (j.state === 'done' ? ' done' : j.state === 'failed' ? ' fail' : '') }, h('i', { id: 'ev-job-bar', style: `width:${j.state === 'done' ? 100 : j.progress / 10}%` })),
    j.state === 'done' && j.summary ? h('div', { class: 'small' }, TM(j.summary)) : null,
    j.reason && j.state !== 'done' ? h('div', { class: 'note-line err' }, I('alert', 's'), h('span', null, TM(j.reason))) : null,
    h('details', { class: 'fold' }, h('summary', null, I('chevR', 's'), T('실행 기록')), h('pre', { class: 'log', id: 'ev-joblog' })));
}
function evJobTick() {
  const j = evJobOf();
  if (!j) return;
  const set = (id, v) => { const e = document.getElementById(id); if (e && e.textContent !== v) e.textContent = v; };
  set('ev-job-step', j.state === 'running' ? TM(j.step || '') : '');
  set('ev-job-time', `${{ running: T('실행 중'), done: T('완료'), failed: T('실패'), cancelled: T('중지됨') }[j.state] || j.state} · ${fmtClock(j.seconds)}`);
  const bar = document.getElementById('ev-job-bar');
  if (bar) bar.style.width = (j.state === 'done' ? 100 : j.progress / 10) + '%';
}
function evListPane() {
  const L = EV.list, out = [];
  out.push(h('div', { class: 'ev-lhead' }, h('span', { class: 'ink' }, T('작업 공간')), h('span', { class: 'grow' }),
    h('button', { class: 'btn sm needs-conn', 'data-fk': 'ev:new', onclick: evNewWs }, I('plus', 's'), T('새 작업 공간'))));
  if (EV.err && !L) out.push(h('div', { class: 'fd-more' }, h('div', { class: 'note-line err' }, I('alert', 's'), h('span', null, TM(EV.err)))));
  else if (!L) out.push(h('div', { class: 'fd-more' }, h('div', { class: 'row g8 muted small' }, h('i', { class: 'dot busy' }), T('불러오는 중'))));
  else if (!L.workspaces.length) out.push(h('div', { class: 'fd-empty' }, I('flask', 'l'), h('div', { class: 'big' }, T('작업 공간이 없습니다')),
    h('div', { class: 'small' }, T('새 작업 공간을 만들고 기준 모델을 고른 뒤, 비교할 GGUF 파일을 버전으로 더하세요.'))));
  else out.push(h('div', { class: 'fd-list' }, [...L.workspaces].sort((a, b) => byName(a.id, b.id)).map(w => {
    const cur = w.current ? evLatest(w, w.current) : null;
    const a = cur && cur.answers;
    return h('button', { class: 'fd-row' + (EV.sel === w.id ? ' on' : ''), 'data-fk': 'ev:ws:' + w.id, 'aria-current': EV.sel === w.id ? 'true' : null, onclick: () => evGo('ws', { sel: w.id, dir: '', item: -1 }) },
      h('span', { class: 'main' },
        h('span', { class: 'top' }, h('span', { class: 'nm' }, w.name)),
        h('span', { class: 'by' }, T`기준 모델 ${evRefLabel(w)}`),
        h('span', { class: 'foot' }, h('span', { class: 'st' }, I('layers'), T`버전 ${w.versions.length}개`),
          cur && evChat(cur) && a && a.compared ? h('span', { class: 'st' }, I('target'), T`${cur.version} · 답 ${a.differ}/${a.compared} 달라짐`)
            : cur ? h('span', { class: 'st' }, I('target'), `${cur.version} · KL ${evKL(cur.metrics && cur.metrics.kl_mean)}`) : null)));
  })));
  return h('div', { class: 'ev-list', 'aria-label': T('작업 공간 목록') }, out);
}
function evCrumb(w, parts) {
  return h('nav', { class: 'ev-crumb', 'aria-label': T('위치') }, [
    h('button', { class: 'dbg-link', 'data-fk': 'ev:crumb:ws', onclick: () => evGo('ws') }, w.name),
    parts.map(([label, fn]) => [h('span', { class: 'faint' }, '›'), fn ? h('button', { class: 'dbg-link', onclick: fn }, label) : h('span', { class: 'ink' }, label)]),
  ]);
}
const evDetailKey = () => [EV.sel, EV.view, EV.dir, EV.item, EV.cmpA, EV.cmpB].join('|');
function evDetailPane() {
  const box = h('div', { class: 'ev-detail', 'data-k': evDetailKey() });
  const w = evWs(EV.sel);
  if (!w) {
    put(box, h('div', { class: 'col g24 ev-intro' },
      h('div', { class: 'col g8' }, h('div', { class: 'eyebrow' }, T('모델 평가')), h('h2', { class: 'm0' }, T('GGUF 파일을 기준 모델과 비교합니다'))),
      h('ol', { class: 'ev-steps' },
        h('li', null, h('span', { class: 'ink' }, T('작업 공간 만들기')), h('span', { class: 'small muted' }, T('기준 모델을 고릅니다 — 같은 모델의 BF16 · F16 파일(그래픽 메모리에 들지 않으면 Q8_0).'))),
        h('li', null, h('span', { class: 'ink' }, T('버전 더하기')), h('span', { class: 'small muted' }, T('비교할 GGUF 파일(양자화 · 미세조정한 파일)을 v1 · v2 …로 더합니다.'))),
        h('li', null, h('span', { class: 'ink' }, T('평가 실행')), h('span', { class: 'small muted' }, T('기준 모델이 낸 답의 토큰마다 두 모델의 확률을 비교하고(KL divergence · top-1 일치율), 두 모델이 저마다 낸 답이 어디서 갈라지는지 봅니다.'))),
        h('li', null, h('span', { class: 'ink' }, T('결과 · 전후 비교')), h('span', { class: 'small muted' }, T('버전을 고친 뒤 같은 데이터 · 설정으로 다시 실행하면, 기준 모델과 같아진 대화와 새로 달라진 대화로 나아졌는지 판정합니다.')))),
      h('div', null, h('button', { class: 'btn primary needs-conn', onclick: evNewWs }, I('plus', 's'), T('새 작업 공간')))));
    return box;
  }
  let body;
  try {
    body = EV.view === 'result' && EV.dir ? evResultView(w) : EV.view === 'compare' ? evCompareView(w) : EV.view === 'item' && EV.dir && EV.item >= 0 ? evItemView(w) : evWsView(w);
  } catch (e) { body = h('div', { class: 'note-line err' }, I('alert', 's'), h('span', null, String(e && e.message))); console.error(e); }
  put(box, body);
  return box;
}

// 작업 공간 — 지금 상태 · 버전 · 평가 실행 · 결과
function evWsView(w) {
  const busy = !!(S.st.job && S.st.job.state === 'running');
  const rs = evResults(w);
  const cur = w.current ? evLatest(w, w.current) : null;
  const vd = evVerdict(cur);
  const head = h('div', { class: 'fd-dhead' },
    h('div', { class: 'row g8' }, h('span', { class: 'eyebrow' }, T('작업 공간', 'one')), h('span', { class: 'grow' }),
      h('button', { class: 'icon-btn', 'aria-label': T('작업 공간 메뉴'), 'aria-haspopup': 'menu', 'data-tip': T('작업 공간 메뉴'), 'data-tip-pos': 'left',
        onclick: ev => menu(ev.currentTarget, T('작업 공간 메뉴'), [
          { icon: 'edit', label: T('이름 바꾸기'), run: () => evRename(w) },
          { icon: 'target', label: T('기준 모델 바꾸기'), run: () => evSetRef(w) },
          '-', { icon: 'trash', label: T('작업 공간 지우기'), danger: true, run: () => evDelWs(w) }], 'right') }, I('dots'))),
    h('div', { class: 'fd-title' }, w.name),
    h('div', { class: 'fd-sub' }, T`기준 모델 ${evRefLabel(w)}`, w.reference.sha256 ? h('span', { class: 'mono faint small' }, ` · ${w.reference.sha256.slice(0, 12)}`) : null,
      w.reference.made ? h('span', { class: 'faint small' }, ` · ${evMadeLine(w.reference.made)}`) : null),
    w.reference.model && !modelOf(w.reference.model) ? h('div', { class: 'note-line err mt8' }, I('alert', 's'), h('span', null, T('기준 모델이 앱의 모델 목록에 없습니다 — 모델 화면에서 그 파일을 다시 더하거나 기준 모델을 바꾸세요.'))) : null);
  const state = h('div', { class: 'card col g8 ev-state' },
    h('div', { class: 'eyebrow' }, cur ? T`지금 상태 — ${evVerLabel(w, cur.version)}` : T('지금 상태')),
    vd ? h('div', { class: 'verdict' }, vd[0]) : h('div', { class: 'verdict' }, w.versions.length ? T('아직 평가하지 않았습니다 — 아래에서 평가를 실행하세요.') : T('평가할 GGUF 파일을 버전으로 더하세요.')),
    vd && vd[1] ? h('div', { class: 'small muted' }, vd[1]) : null,
    cur ? h('div', null, h('button', { class: 'btn sm', onclick: () => evGo('result', { dir: cur.dir }) }, T('결과 보기'), I('chevR', 's'))) : null);
  // 버전 — 버전마다 최근 결과와, 견줄 버전(만든 부모 — 없으면 앞 버전)의 같은 데이터 · 설정 결과 가운데 최근 것 대비
  const vrows = w.versions.map((v, k) => {
    const r = evLatest(w, v.id);
    const bo = v.made && v.made.settings && v.made.settings.based_on;   // 4-2 — 개선 측정의 기준 버전에서 다시 만든 버전은 그 버전 대비
    const base = bo && w.versions.some(x => x.id === bo) ? w.versions.find(x => x.id === bo)
      : v.made && v.made.from ? w.versions.find(x => x.id === v.made.from) : k > 0 ? w.versions[k - 1] : null;
    const prev = base && r ? evResults(w).filter(x => x.version === base.id && evSameSet(x, r)).pop() || null : null;
    const pc = !!prev;
    const a = r && r.answers, pa = prev && prev.answers;
    const m = r && r.metrics, pm = prev && prev.metrics;
    const sign = x => (x > 0 ? '+' : x < 0 ? '−' : '±') + Math.abs(x);
    const dKL = pc && m && pm && m.positions && pm.positions ? m.kl_mean - pm.kl_mean : null;
    const delta = pc ? [dKL !== null ? `KL ${dKL > 0 ? '+' : dKL < 0 ? '−' : '±'}${evKL(Math.abs(dKL))}` : null,
      evChat(r) && a && pa && a.compared && pa.compared ? T`답이 달라진 대화 ${sign(a.differ - pa.differ)}` : null].filter(Boolean).join(NB) : '';
    return h('div', { class: 'ev-vrow' + (r ? ' clk' : ''), onclick: e => { if (r && !e.target.closest('button')) evGo('result', { dir: r.dir }); } },
      h('div', { class: 'ev-vid' }, h('span', { class: 'ink' }, v.id), w.current === v.id ? h('span', { class: 'tag acc' }, T('지금')) : null),
      h('div', { class: 'ev-vfile' }, h('div', { class: 'ink ell', 'data-tip': modelOf(v.model) ? labelById(v.model) : v.file, 'data-tip-pos': 'top' }, v.file),
        h('div', { class: 'small muted' }, [fmtBytes(v.size), ({ imported: T('가져옴'), metadata: T('메타데이터 고침'), quantized: T('다시 양자화'), converted: T('변환'), convert: T('변환'), quantize: T('양자화') })[v.made && v.made.kind] || '',
          v.made && v.made.from ? (v.made.from === 'ref' ? T('기준 모델에서 만듦') : T`${v.made.from}에서 만듦`) : null].filter(Boolean).join(NB)),
        v.made && v.made.changed ? h('div', { class: 'small faint ell', 'data-tip': v.made.changed, 'data-tip-pos': 'top' }, T`바꾼 키 ${v.made.changed}`) : null,
        v.made && v.made.settings ? h('div', { class: 'small faint ell', 'data-tip': evMadeLine(v.made.settings), 'data-tip-pos': 'top' }, evMadeLine(v.made.settings)) : null,
        modelOf(v.model) && modelOf(v.model).shape && !modelOf(v.model).shape.ok ? h('div', { class: 'small err ell', 'data-tip': TM(modelOf(v.model).shape.reason || ''), 'data-tip-pos': 'top' },
          T`엔진이 받지 않음 — ${TM(modelOf(v.model).shape.reason || '')}`) : null),
      h('div', { class: 'ev-vres' }, h('div', { class: 'small' }, r ? (evChat(r) && a && a.compared ? T`답 ${a.differ}/${a.compared} 달라짐 · KL ${evKL(m && m.kl_mean)}` : `KL ${evKL(m && m.kl_mean)} · top-1 ${evPct(m && m.top1)}`) : h('span', { class: 'muted' }, T('결과 없음'))),
        delta ? h('div', { class: 'small muted' }, T`${base.id} 대비 ${delta}`) : null),
      h('div', { class: 'ev-vmenu' }, h('button', { class: 'icon-btn', 'aria-label': T`${v.id} 메뉴`, 'aria-haspopup': 'menu', onclick: ev => menu(ev.currentTarget, T`${v.id} 메뉴`, [
        r ? { icon: 'chevR', label: T('최근 결과 보기'), run: () => evGo('result', { dir: r.dir }) } : null,
        w.current !== v.id ? { icon: 'check', label: T('지금 버전으로'), run: () => api('/api/eval/workspace', { action: 'update', id: w.id, current: v.id }).then(evLoad, e => toast(e.message, 'err')) } : null,
        { icon: 'edit', label: T('메타데이터 고치기'), sub: T('채팅 템플릿 · 특수 토큰 — 새 버전으로'), run: () => evFixMeta(w, v) },
        '-', { icon: 'trash', label: T('버전 지우기'), danger: true, run: () => evDelVersion(w, v) }], 'right') }, I('dots'))));
  });
  const versions = h('div', { class: 'col g12' },
    h('div', { class: 'section-head' }, h('span', { class: 'sec' }, T('버전')), h('span', { class: 'grow' }),
      h('button', { class: 'btn sm needs-conn', 'data-fk': 'ev:addv', onclick: () => evAddVersion(w) }, I('plus', 's'), T('버전 더하기'))),
    w.versions.length ? h('div', { class: 'list ev-vlist' }, vrows)
      : h('div', { class: 'empty-box' }, h('div', { class: 'big' }, T('버전이 없습니다')), h('div', null, T('기준 모델과 비교할 GGUF 파일을 더하거나, 아래 GGUF 만들기로 원래 모델에서 만드세요.'))));
  // 결과 — 둘을 고르면 전후 비교
  const pick = EV.pick.filter(d => rs.some(r => r.dir === d));
  const results = h('div', { class: 'col g12' },
    h('div', { class: 'section-head' }, h('span', { class: 'sec' }, T('결과')), h('span', { class: 'small muted' }, rs.length ? T`${rs.length}개 · 둘을 고르면 전후 비교` : ''), h('span', { class: 'grow' }),
      h('button', { class: 'btn sm', 'data-fk': 'ev:cmp', disabled: pick.length === 2 ? null : true, onclick: () => { const [a, b] = [...pick].sort(byName); evGo('compare', { cmpA: a, cmpB: b }); } }, T('전후 비교'))),
    rs.length ? h('div', { class: 'list ev-vlist' }, [...rs].reverse().map(r => {
      const a = r.answers || {}, m = r.metrics || {}, s = r.settings || {};
      return h('div', { class: 'ev-rrow clk', onclick: e => { if (!e.target.closest('input, label')) evGo('result', { dir: r.dir }); } },
        h('label', { class: 'check ev-pick' }, h('input', { type: 'checkbox', 'aria-label': T`${r.dir} 고르기`, 'data-fk': 'ev:pick:' + r.dir, checked: pick.includes(r.dir) ? true : null,
          onchange: ev => { EV.pick = pick.filter(d => d !== r.dir); if (ev.target.checked) { EV.pick.push(r.dir); if (EV.pick.length > 2) EV.pick.shift(); } renderEval(true); } })),
        h('div', { class: 'ev-vfile' }, h('div', { class: 'ink' }, `${r.version} · ${evDirWhen(r.dir)}`),
          h('div', { class: 'small muted' }, [r.data && r.data.name, evMode(s.mode), evChat(r) ? T`답 ${s.max_tokens}토큰` : T`청크 ${s.chunks}`].filter(Boolean).join(NB))),
        h('div', { class: 'ev-vres small' },
          h('div', null, evChat(r) && a.compared ? T`답 ${a.differ}/${a.compared} 달라짐` + (a.near_tie ? T` (마진 작음 ${a.near_tie})` : '') : T`위치 ${fmtNum(m.positions || 0)}개`),
          h('div', { class: 'muted num' }, `KL ${evKL(m.kl_mean)} · top-1 ${evPct(m.top1)}` + (r.accuracy ? T` · 정답률 ${evPct(r.accuracy.cand)}` : ''))));
    }))
      : h('div', { class: 'empty-box' }, h('div', { class: 'big' }, T('아직 결과가 없습니다')), h('div', null, T('위에서 평가를 실행하면 여기에 쌓입니다.'))));
  return h('div', { class: 'col g24' }, head, state, versions, evMakeCard(w, busy), evImproveCard(w, busy), evCheckCard(w, busy), evRunCard(w, busy), results);
}

// 결과 — 판정 · 지표 · 처음 갈라진 스텝 · 끝난 까닭 · 분야 · 대화
function evResultView(w) {
  const R = evRes(w, EV.dir);
  const crumb = evCrumb(w, [[evDirLabel(w, EV.dir)]]);
  if (!R) return h('div', { class: 'col g16' }, crumb, h('div', { class: 'row g8 muted' }, h('i', { class: 'dot busy' }), T('불러오는 중')));
  if (R.error) return h('div', { class: 'col g16' }, crumb, h('div', { class: 'note-line err' }, I('alert', 's'), h('span', null, TM(R.error))));
  const s = R.summary, a = s.answers || {}, m = s.metrics || {}, st = s.settings || {}, rp = s.reproduce || {};
  const chat = evChat(s);
  const vd = evVerdict(s);
  const K = st.top_k || 64;
  const out = [crumb,
    h('div', { class: 'fd-dhead' },
      h('div', { class: 'row g8' }, h('span', { class: 'eyebrow' }, [s.version_model && s.version_model.file, s.data && s.data.name, evMode(st.mode)].filter(Boolean).join(NB)), h('span', { class: 'grow' }),
        h('button', { class: 'btn sm ghost needs-conn', onclick: () => api('/api/eval/open', { workspace: w.id, dir: EV.dir }).catch(e => toast(e.message, 'err')) }, I('folder', 's'), T('결과 폴더 열기')),
        chat ? h('button', { class: 'btn sm ghost needs-conn', onclick: () => evToDebugger(w, EV.dir, -1) }, I('bug', 's'), T('디버거로 열기')) : null),
      h('div', { class: 'fd-title' }, T`${s.version} 결과`),
      h('div', { class: 'fd-sub' }, T`${s.finished || evDirWhen(EV.dir)} · ${evNum(s.seconds, 0)}초`)),
    h('div', { class: 'card col g8' }, h('div', { class: 'verdict' }, vd[0]), vd[1] ? h('div', { class: 'small muted' }, vd[1]) : null,
      s.same_vocab === false ? h('div', { class: 'note-line' }, I('info', 's'), h('span', null, T('어휘가 기준 모델과 달라 같은 토큰의 확률은 비교하지 않았습니다 — 답은 글로 비교했습니다.'))) : null)];
  const acc = s.accuracy;
  if (acc && acc.items) {
    const ci = c => (Array.isArray(c) ? `${evPct(c[0])} ~ ${evPct(c[1])}` : '-');
    const d = acc.cand - acc.ref;
    out.push(h('div', { class: 'col g8' }, h('div', { class: 'section-head' }, h('span', { class: 'sec' }, T('정답률')), h('span', { class: 'small muted' }, T`채점한 대화 ${acc.items}개`)),
      h('div', { class: 'fd-stats ev-stats3' },
        h('div', { class: 'cell' }, h('div', { class: 'k' }, T('기준 모델')), h('div', { class: 'v' }, evPct(acc.ref)), h('div', { class: 's' }, T`95% 구간 ${ci(acc.ref_ci)}`)),
        h('div', { class: 'cell' }, h('div', { class: 'k' }, T('평가할 모델')), h('div', { class: 'v' }, evPct(acc.cand)), h('div', { class: 's' }, T`95% 구간 ${ci(acc.cand_ci)}`)),
        h('div', { class: 'cell' }, h('div', { class: 'k' }, T('차이(평가 − 기준)')), h('div', { class: 'v' }, (d > 0 ? '+' : d < 0 ? '−' : '±') + (Math.abs(d) * 100).toFixed(2) + T('%p')),
          h('div', { class: 's' }, T`기준만 맞음 ${acc.ref_only} · 평가만 맞음 ${acc.cand_only} · p = ${evNum(acc.p, 3)}`))),
      h('p', { class: 'small muted m0' }, T('같은 대화에서 한쪽만 맞은 수로 부호 검정을 합니다. 구간은 대화를 다시 뽑는 부트스트랩(1,000번)입니다.'))));
  }
  if (m.positions) out.push(h('div', { class: 'col g8' }, h('div', { class: 'section-head' }, h('span', { class: 'sec' }, T('같은 토큰의 확률')),
    h('span', { class: 'small muted' }, chat ? T('기준 모델이 낸 답의 토큰마다') : T('청크마다 뒤쪽 255 토큰'))),
    kpis([[evKL(m.kl_mean), T('KL divergence 평균')], [evKL(m.kl_p99), T('KL divergence 99 백분위수')], [evPct(m.top1), T('top-1 일치율')],
      [evNum(m.ppl_ratio, 4), T('PPL 비(평가 ÷ 기준)')], [evPct(m.dp_mean), T('Δp 평균(기준 − 평가)')], [evPct(m.tail_mean), T`기준 모델의 상위 ${K}개 밖 확률`]], true),
    h('p', { class: 'small muted m0' }, T`KL divergence는 기준 모델의 상위 ${K}개 토큰과 나머지 확률 하나로 계산한 값입니다 — 모든 어휘로 계산한 값보다 조금 작습니다(아래쪽 값). 위치 ${fmtNum(m.positions)}개.`)));
  if (chat && a.compared) {
    const its = (R.items || []).filter(x => x.same === false);
    const bins = [[1, 8], [9, 32], [33, 128], [129, 512], [513, Infinity]];
    const cnt = bins.map(([lo, hi]) => its.filter(x => x.first_divergence !== null && x.first_divergence + 1 >= lo && x.first_divergence + 1 <= hi));
    const max = Math.max(1, ...cnt.map(c => c.length));
    const textOnly = its.filter(x => x.first_divergence === null).length;
    out.push(h('div', { class: 'col g8' }, h('div', { class: 'section-head' }, h('span', { class: 'sec' }, T('처음 갈라진 스텝')),
      h('span', { class: 'small muted' }, T`답이 달라진 대화 ${a.differ}개`)),
      h('div', { class: 'card col g8 ev-bars' }, bins.map(([lo, hi], k) => h('div', { class: 'ev-bar' },
        h('span', { class: 'l small muted num' }, hi === Infinity ? T`${lo} 이상` : `${lo}–${hi}`),
        h('span', { class: 'b' }, h('i', { style: `width:${(100 * cnt[k].length / max).toFixed(1)}%` }), cnt[k].some(x => x.near_tie) ? h('i', { class: 'nt', style: `width:${(100 * cnt[k].filter(x => x.near_tie).length / max).toFixed(1)}%` }) : null),
        h('span', { class: 'n small num' }, String(cnt[k].length)))),
        h('div', { class: 'row g16 small muted' }, h('span', { class: 'row g8' }, h('i', { class: 'ev-key' }), T('달라진 대화')), h('span', { class: 'row g8' }, h('i', { class: 'ev-key nt' }), T('마진이 작은 스텝(0.05 nat 미만)에서 갈라짐')),
          textOnly ? h('span', null, T`글로 비교한 대화 ${textOnly}개`) : null)),
      h('p', { class: 'small muted m0' }, T('답이 길면 확률 차이가 작아도 어딘가에서 갈라지기 쉽습니다. 일찍 갈라진 대화와 마진이 작은 스텝에서 갈라진 대화를 함께 보세요.'))));
    const eosWarn = a.cand_stop < a.ref_stop - Math.max(2, Math.round(a.compared * 0.1));
    out.push(h('div', { class: 'col g8' }, h('div', { class: 'section-head' }, h('span', { class: 'sec' }, T('끝난 까닭'))),
      h('div', { class: 'fd-stats ev-stats2' },
        h('div', { class: 'cell' }, h('div', { class: 'k' }, T('기준 모델 — 끝 토큰으로 끝남')), h('div', { class: 'v' }, `${a.ref_stop}/${a.compared}`)),
        h('div', { class: 'cell' }, h('div', { class: 'k' }, T('평가할 모델 — 끝 토큰으로 끝남')), h('div', { class: 'v' }, `${a.cand_stop}/${a.compared}`))),
      eosWarn ? h('div', { class: 'note-line err' }, I('alert', 's'), h('span', null, T('평가할 모델이 끝 토큰(EOS)으로 끝내지 못하고 최대 길이까지 간 답이 기준 모델보다 많습니다 — GGUF 의 채팅 템플릿 · EOS 토큰을 확인하세요.'))) : null));
  }
  const cats = s.categories || [];
  if (cats.length > 1) out.push(h('div', { class: 'col g8' }, h('div', { class: 'section-head' }, h('span', { class: 'sec' }, T('분야'))),
    h('div', { class: 'list' }, h('table', { class: 'data ev-table' },
      h('thead', null, h('tr', null, h('th', null, T('분야')), h('th', { class: 'r' }, T('대화', 'col')), chat ? h('th', { class: 'r' }, T('답이 달라짐')) : null, acc ? h('th', { class: 'r' }, T('정답(기준 · 평가)', 'col')) : null, h('th', { class: 'r' }, 'KL'), h('th', { class: 'r' }, 'top-1'))),
      h('tbody', null, cats.map(c => h('tr', null, h('td', null, c.name), h('td', { class: 'r num' }, String(c.items)), chat ? h('td', { class: 'r num' }, `${c.differ}/${c.compared}`) : null,
        acc ? h('td', { class: 'r num' }, c.graded ? `${c.ref_correct} · ${c.cand_correct}/${c.graded}` : '-') : null,
        h('td', { class: 'r num' }, evKL(c.metrics && c.metrics.kl_mean)), h('td', { class: 'r num' }, evPct(c.metrics && c.metrics.top1)))))))));
  // 대화 — 거르기
  const items = R.items || [];
  const F = chat ? [['differ', T('답이 달라진 대화'), x => x.same === false], ['tie', T('마진이 작은 스텝에서 갈라짐'), x => !!x.near_tie], ['same', T('같은 답'), x => x.same === true],
    acc ? ['acc', T('정답이 다른 대화'), x => !!x.graded && x.ref_correct !== x.cand_correct] : null, ['all', T('전체'), () => true]].filter(Boolean) : [['all', T('전체'), () => true]];
  const f = F.find(x => x[0] === EV.filter) || F[0];
  const shown = items.filter(f[2]);
  out.push(h('div', { class: 'col g12' },
    h('div', { class: 'section-head' }, h('span', { class: 'sec' }, chat ? T('대화') : T('청크')), h('span', { class: 'small muted' }, String(items.length))),
    F.length > 1 ? h('div', { class: 'row g8', role: 'radiogroup', 'aria-label': T('대화 거르기') }, F.map(([k, l, fn]) => h('button', {
      class: 'fchip' + (f[0] === k ? ' on' : ''), role: 'radio', 'aria-checked': f[0] === k ? 'true' : 'false', 'data-fk': 'ev:f:' + k, onclick: () => { EV.filter = k; renderEval(true); },
    }, l, h('span', { class: 'muted num' }, String(items.filter(fn).length))))) : null,
    shown.length ? h('div', { class: 'list' }, h('table', { class: 'data ev-table' },
      h('thead', null, h('tr', null, h('th', null, chat ? T('대화') : T('청크')), h('th', { class: 'c-cat' }, T('분야')),
        chat ? h('th', { class: 'r' }, T('처음 갈라진 스텝')) : null, chat ? h('th', { class: 'r' }, T('마진')) : null,
        h('th', { class: 'r' }, 'KL'), h('th', { class: 'r' }, 'top-1'), acc ? h('th', null, T('정답(기준 · 평가)', 'col')) : null, chat ? h('th', { class: 'c-fin' }, T('끝난 까닭(기준 · 평가)')) : null)),
      h('tbody', null, shown.slice(0, 500).map(x => h('tr', { class: chat ? 'clk' : '', onclick: chat ? () => evGo('item', { item: x.i }) : null },
        h('td', null, chat ? h('button', { class: 'dbg-link', 'data-fk': 'ev:it:' + x.i, onclick: ev => { ev.stopPropagation(); evGo('item', { item: x.i }); } }, x.id) : x.id),
        h('td', { class: 'small c-cat' }, x.category),
        chat ? h('td', { class: 'r num small' }, x.same ? h('span', { class: 'muted' }, T('같음')) : x.first_divergence !== null ? String(x.first_divergence + 1) : T('글로 비교')) : null,
        chat ? h('td', { class: 'r num small' }, typeof x.ref_margin === 'number' ? h('span', { class: x.near_tie ? 'ev-tie' : '' }, evNum(x.ref_margin, 3)) : '-') : null,
        h('td', { class: 'r num small' }, x.scored ? evKL(x.kl_mean) : '-'),
        h('td', { class: 'r num small' }, x.scored ? evPct(x.top1) : '-'),
        acc ? h('td', { class: 'small nw' }, x.graded ? [evOkI(x.ref_correct), ' ', evOkI(x.cand_correct)] : '-') : null,
        chat ? h('td', { class: 'small c-fin nw' }, `${evFinish(x.ref_finish)} · ${evFinish(x.cand_finish)}`) : null)))))
      : h('div', { class: 'empty-box' }, h('div', null, T('해당하는 대화가 없습니다'))),
    shown.length > 500 ? h('p', { class: 'small muted m0' }, T`처음 500개만 보입니다(모두 ${shown.length}개).`) : null));
  if (chat && m.positions) out.push(evOrigCard(w, EV.dir));
  out.push(h('details', { class: 'fold ev-repro' }, h('summary', null, I('chevR', 's'), T('재현 정보')),
    h('div', { class: 'kv mt16' },
      [[T('기준 모델'), `${s.reference && s.reference.file} · ${s.reference && s.reference.sha256}`],
        [T('평가한 파일'), `${s.version_model && s.version_model.file} · ${s.version_model && s.version_model.sha256}`],
        [T('데이터'), `${s.data && s.data.name} · ${s.data && s.data.sha256}`],
        [T('설정'), Object.entries(st).map(([k, v]) => `${k}=${v}`).join(' ')],
        ['GeulPU', `${rp.geulpu} · ${String(rp.commit || '').slice(0, 12)}`],
        [T('장치'), `${rp.device} · ${LAYER[rp.layer] || rp.layer}`],
        [T('커널 모듈'), `${rp.module} · ${String(rp.module_sha256 || '').slice(0, 16)}`],
        [T('결과 해시'), s.result_sha256]].map(([k, v]) => [h('div', { class: 'k' }, k), h('div', { class: 'v mono small ev-wrap' }, String(v))])),
    h('p', { class: 'small muted' }, T('같은 파일 · 데이터 · 설정 · GeulPU 버전이면 다시 실행해도 결과 해시가 같습니다.'))));
  return h('div', { class: 'col g24' }, out);
}

// 전후 비교(3.7) — 같은 데이터 · 같은 설정 · 같은 기준 모델의 두 결과
function evCompareView(w) {
  const rs = evResults(w);
  const sel = (key, val) => h('select', { class: 'input', 'data-fk': 'ev:cmp:' + key, 'aria-label': key === 'cmpA' ? T('전') : T('후'),
    onchange: ev => { EV[key] = ev.target.value; renderEval(true); } },
    rs.map(r => h('option', { value: r.dir, selected: r.dir === val ? true : null }, `${r.version} · ${r.data && r.data.name} · ${evMode(r.settings && r.settings.mode)} · ${evDirWhen(r.dir)}`)));
  if ((!EV.cmpA || !EV.cmpB) && rs.length >= 2) { EV.cmpA = rs[rs.length - 2].dir; EV.cmpB = rs[rs.length - 1].dir; }
  const out = [evCrumb(w, [[T('전후 비교')]]),
    h('div', { class: 'fd-dhead' }, h('div', { class: 'eyebrow' }, T('전후 비교')), h('div', { class: 'fd-title' }, T('두 결과 비교'))),
    h('div', { class: 'ev-cmpsel' }, h('label', { class: 'col g8' }, h('span', { class: 'small muted' }, T('전')), sel('cmpA', EV.cmpA)),
      h('span', { class: 'faint ev-arrow' }, '→'), h('label', { class: 'col g8' }, h('span', { class: 'small muted' }, T('후')), sel('cmpB', EV.cmpB)))];
  if (!EV.cmpA || !EV.cmpB || EV.cmpA === EV.cmpB) { out.push(h('p', { class: 'muted' }, T('서로 다른 결과 두 개를 고르세요.'))); return h('div', { class: 'col g24' }, out); }
  const C = EV.cmp[`${w.id}/${EV.cmpA}/${EV.cmpB}`];
  if (!C) { out.push(h('div', { class: 'row g8 muted' }, h('i', { class: 'dot busy' }), T('불러오는 중'))); return h('div', { class: 'col g24' }, out); }
  if (C.error) { out.push(h('div', { class: 'note-line err' }, I('alert', 's'), h('span', null, TM(C.error)))); return h('div', { class: 'col g24' }, out); }
  const why = !C.same_data ? T('데이터가 다릅니다') : !C.same_settings ? T('설정이 다릅니다') : !C.same_reference ? T('기준 모델이 다릅니다') : T('대화 수가 다릅니다');
  const pv = evNum(C.p, 3);
  const vt = { better: T`나아졌습니다 — 기준 모델과 같아진 대화 ${C.fixed}개 · 새로 달라진 대화 ${C.newly_diverged}개(부호 검정 p = ${pv})`,
    worse: T`나빠졌습니다 — 기준 모델과 같아진 대화 ${C.fixed}개 · 새로 달라진 대화 ${C.newly_diverged}개(부호 검정 p = ${pv})`,
    no_clear_difference: T`뚜렷한 차이가 없습니다 — 기준 모델과 같아진 대화 ${C.fixed}개 · 새로 달라진 대화 ${C.newly_diverged}개(부호 검정 p = ${pv})`,
    not_comparable: T`비교할 수 없습니다 — ${why}` }[C.verdict] || C.verdict;
  const tone = { better: 'ok', worse: 'err' }[C.verdict] || '';
  out.push(h('div', { class: 'card col g8' }, h('div', { class: 'row g8' }, h('span', { class: 'tag ' + tone }, { better: T('나아짐'), worse: T('나빠짐'), no_clear_difference: T('차이 없음'), not_comparable: T('비교 불가') }[C.verdict] || ''),
    h('span', { class: 'small muted' }, `${C.a_version} → ${C.b_version}`)), h('div', { class: 'verdict' }, vt),
    C.comparable ? h('p', { class: 'small muted m0' }, T('GeulPU의 결과에는 실행마다의 잡음이 없어 대화 하나하나의 바뀜은 파일을 바꾼 데서 옵니다. 어떤 대화를 골랐나에 따른 차이는 남기 때문에 부호 검정으로 판정합니다(p < 0.05).')) : null));
  const ma = C.a_metrics || {}, mb = C.b_metrics || {}, aa = C.a_answers || {}, ab = C.b_answers || {};
  const rows = [
    aa.compared ? [T('답이 달라진 대화'), `${aa.differ}/${aa.compared}`, `${ab.differ}/${ab.compared}`, ab.differ - aa.differ, x => (x > 0 ? '+' : '') + x] : null,
    aa.compared ? [T('마진이 작은 스텝에서 갈라짐'), String(aa.near_tie), String(ab.near_tie), ab.near_tie - aa.near_tie, x => (x > 0 ? '+' : '') + x] : null,
    aa.compared ? [T('평가할 모델 — 끝 토큰으로 끝남'), `${aa.cand_stop}/${aa.compared}`, `${ab.cand_stop}/${ab.compared}`, ab.cand_stop - aa.cand_stop, x => (x > 0 ? '+' : '') + x] : null,
    ma.positions ? [T('KL divergence 평균'), evKL(ma.kl_mean), evKL(mb.kl_mean), mb.kl_mean - ma.kl_mean, x => (x > 0 ? '+' : x < 0 ? '−' : '') + evKL(Math.abs(x))] : null,
    ma.positions ? [T('KL divergence 99 백분위수'), evKL(ma.kl_p99), evKL(mb.kl_p99), mb.kl_p99 - ma.kl_p99, x => (x > 0 ? '+' : x < 0 ? '−' : '') + evKL(Math.abs(x))] : null,
    ma.positions ? [T('top-1 일치율'), evPct(ma.top1), evPct(mb.top1), mb.top1 - ma.top1, x => (x > 0 ? '+' : x < 0 ? '−' : '') + (Math.abs(x) * 100).toFixed(2) + T('%p')] : null,
    ma.positions ? [T('PPL 비(평가 ÷ 기준)'), evNum(ma.ppl_ratio, 4), evNum(mb.ppl_ratio, 4), mb.ppl_ratio - ma.ppl_ratio, x => (x > 0 ? '+' : x < 0 ? '−' : '') + Math.abs(x).toFixed(4)] : null,
    C.a_accuracy && C.b_accuracy ? [T('정답률(평가할 모델)'), evPct(C.a_accuracy.cand), evPct(C.b_accuracy.cand), C.b_accuracy.cand - C.a_accuracy.cand, x => (x > 0 ? '+' : x < 0 ? '−' : '') + (Math.abs(x) * 100).toFixed(2) + T('%p')] : null,
  ].filter(Boolean);
  out.push(h('div', { class: 'list' }, h('table', { class: 'data ev-table' },
    h('thead', null, h('tr', null, h('th', null, T('지표')), h('th', { class: 'r' }, T('전')), h('th', { class: 'r' }, T('후')), h('th', { class: 'r' }, T('차이')))),
    h('tbody', null, rows.map(([l, va, vb, d, fd]) => h('tr', null, h('td', null, l), h('td', { class: 'r num' }, va), h('td', { class: 'r num' }, vb), h('td', { class: 'r num' }, isFinite(d) ? fd(d) : '-')))))));
  const RA = evRes(w, EV.cmpA), RB = evRes(w, EV.cmpB);
  const idx = (R, id) => { const x = R && R.items && R.items.find(t => t.id === id); return x ? x.i : -1; };
  const ids = (list, R, dir, label, note) => h('div', { class: 'col g8' }, h('div', { class: 'section-head' }, h('span', { class: 'sec' }, label), h('span', { class: 'small muted' }, note)),
    list.length ? h('div', { class: 'row g8 ev-ids' }, list.map(id => { const i = idx(R, id); return i >= 0 ? h('button', { class: 'dbg-chip ev-chip', onclick: () => evGo('item', { dir, item: i }) }, id) : h('span', { class: 'dbg-chip ev-chip' }, id); }))
      : h('div', { class: 'small muted' }, T('없음')));
  if (C.comparable && C.acc_items) {
    out.push(h('div', { class: 'card col g8' }, h('div', { class: 'verdict' }, T`채점한 대화 ${C.acc_items}개 — 틀림 → 맞음 ${C.acc_fixed}개 · 맞음 → 틀림 ${C.acc_broke}개(부호 검정 p = ${evNum(C.acc_p, 3)})`)));
    out.push(ids(C.acc_fixed_items || [], RB, EV.cmpB, T`틀림 → 맞음 ${C.acc_fixed}개`, T('누르면 후 결과의 답을 엽니다')));
    out.push(ids(C.acc_broke_items || [], RB, EV.cmpB, T`맞음 → 틀림 ${C.acc_broke}개`, T('누르면 후 결과의 답을 엽니다')));
  }
  if (C.comparable && aa.compared) {
    out.push(ids(C.fixed_items || [], RA, EV.cmpA, T`기준 모델과 같아진 대화 ${C.fixed}개`, T('누르면 전 결과에서 달랐던 답을 엽니다')));
    out.push(ids(C.newly_items || [], RB, EV.cmpB, T`새로 달라진 대화 ${C.newly_diverged}개`, T('누르면 후 결과의 답을 엽니다')));
  }
  return h('div', { class: 'col g24' }, out);
}

// 답이 달라진 곳 — 대화 하나: 두 답, 처음 갈라진 스텝의 후보(두 모델의 확률), 그 앞뒤 스텝
const evText = c => (typeof c === 'string' ? c : Array.isArray(c) ? c.map(p => (p && typeof p.text === 'string' ? p.text : '')).join('') : c === null || c === undefined ? '' : JSON.stringify(c));
function evPrefix(a, b) {
  const x = Array.from(a || ''), y = Array.from(b || '');
  let k = 0;
  while (k < x.length && k < y.length && x[k] === y[k]) k++;
  return k;
}
function evAnswer(label, text, k, tone) {
  const cs = Array.from(text || '');
  const pre = cs.slice(0, k), tail = cs.slice(k).join('');
  const cut = pre.length > 240 ? '…' + pre.slice(-240).join('') : pre.join('');
  return h('div', { class: 'col g8 ev-ans' }, h('div', { class: 'eyebrow' }, label),
    h('pre', { class: 'dbg-ans' }, cut ? h('span', { class: 'ev-same' }, cut) : null, tail ? h('span', { class: 'ev-diff ' + (tone || '') }, tail) : null,
      !cs.length ? h('span', { class: 'muted' }, T('(빈 답)')) : null));
}
const evTok = s => h('span', { class: 'ev-tok mono' }, tokShow(s === null || s === undefined ? '' : s));
const evOk = v => h('span', { class: v ? 'ev-ok' : 'ev-no' }, v ? T('맞음') : T('틀림'));
const evOkI = v => h('span', { class: 'ev-oki ' + (v ? 'ev-ok' : 'ev-no'), role: 'img', 'aria-label': v ? T('맞음') : T('틀림'), 'data-tip': v ? T('맞음') : T('틀림'), 'data-tip-pos': 'top' }, I(v ? 'check' : 'x', 's'));
function evMessages(req) {
  if (!req || !Array.isArray(req.messages)) return null;
  const ms = req.messages;
  const last = ms.length ? ms[ms.length - 1] : null;
  const one = m => h('div', { class: 'dbg-msg' }, h('span', { class: 'qlab' }, m.role),
    h('span', { class: 'dbg-msg-t' }, (() => { const t = evText(m.content) || (m.tool_calls ? JSON.stringify(m.tool_calls) : ''); return t.length > 1500 ? t.slice(0, 1500) + '…' : t; })()));
  return h('div', { class: 'col g8' }, h('div', { class: 'section-head' }, h('span', { class: 'sec' }, T('대화')),
    h('span', { class: 'small muted' }, [T`메시지 ${ms.length}개`, req.tools && req.tools.length ? T`도구 ${req.tools.length}개` : null].filter(Boolean).join(NB))),
    ms.length > 1 ? h('details', { class: 'fold' }, h('summary', null, I('chevR', 's'), T`앞의 메시지 ${ms.length - 1}개`), h('div', { class: 'col g8 mt8' }, ms.slice(0, -1).map(one))) : null,
    last ? one(last) : null);
}
function evItemView(w) {
  const R = evRes(w, EV.dir);
  const X = EV.items[`${w.id}/${EV.dir}/${EV.item}`];
  const row = R && R.items ? R.items.find(x => x.i === EV.item) : null;
  const crumb = evCrumb(w, [[evDirLabel(w, EV.dir), () => evGo('result', { item: -1 })], [row ? row.id : String(EV.item)]]);
  if (!X || !R) return h('div', { class: 'col g16' }, crumb, h('div', { class: 'row g8 muted' }, h('i', { class: 'dot busy' }), T('불러오는 중')));
  if (X.error || R.error) return h('div', { class: 'col g16' }, crumb, h('div', { class: 'note-line err' }, I('alert', 's'), h('span', null, TM(X.error || R.error))));
  const it = X.item || {}, dv = X.diverge, s = R.summary;
  const items = R.items || [];
  const pos = items.findIndex(x => x.i === EV.item);
  const nav = d => { const x = items[pos + d]; return x ? () => evGo('item', { item: x.i }) : null; };
  const out = [crumb,
    h('div', { class: 'fd-dhead' },
      h('div', { class: 'row g8' }, h('span', { class: 'eyebrow' }, [it.category, evVerLabel(w, s.version)].filter(Boolean).join(NB)), h('span', { class: 'grow' }),
        h('button', { class: 'icon-btn', 'aria-label': T('앞 대화'), 'data-tip': T('앞 대화'), 'data-tip-pos': 'bottom', disabled: nav(-1) ? null : true, onclick: nav(-1) }, I('chevL')),
        h('button', { class: 'icon-btn', 'aria-label': T('다음 대화'), 'data-tip': T('다음 대화'), 'data-tip-pos': 'bottom', disabled: nav(1) ? null : true, onclick: nav(1) }, I('chevR')),
        h('button', { class: 'btn sm ghost needs-conn', onclick: () => evToDebugger(w, EV.dir, EV.item) }, I('bug', 's'), T('디버거로 열기'))),
      h('div', { class: 'fd-title' }, it.id || String(EV.item)))];
  const d = dv ? dv.first_divergence : it.first_divergence;
  let line;
  if (it.same) line = T('두 모델의 답이 토큰까지 같습니다.');
  else if (d === null || d === undefined) line = T('두 모델의 답이 다릅니다 — 어휘가 달라 글로 비교했습니다.');
  else if (d === 0) line = T('첫 토큰(스텝 1)부터 갈라졌습니다.');
  else line = T`스텝 ${d + 1}에서 처음 갈라졌습니다 — 그 앞 ${d}토큰은 같습니다.`;
  out.push(h('div', { class: 'card col g8' }, h('div', { class: 'verdict' }, line),
    !it.same && typeof it.ref_margin === 'number' ? h('div', { class: 'row g8 small' },
      h('span', { class: 'muted' }, T`그 스텝에서 기준 모델의 마진(1위와 2위의 로그확률 차) ${evNum(it.ref_margin, 3)} nat`),
      it.near_tie ? h('span', { class: 'tag acc' }, T('마진이 작은 스텝')) : null) : null,
    h('div', { class: 'small muted' }, [T`기준 모델 ${it.ref_len}토큰(${evFinish(it.ref_finish)})`, it.cand_len !== undefined ? T`평가할 모델 ${it.cand_len}토큰(${evFinish(it.cand_finish)})` : null, it.scored ? T`이 대화의 KL divergence 평균 ${evKL(it.kl_mean)} · top-1 일치율 ${evPct(it.top1)}` : null].filter(Boolean).join(NB))));
  const evToolTxt = t => { if (!t) return T('(없음)'); let g = t.arguments; if (typeof g === 'string') { try { g = JSON.parse(g); } catch (_) { } } return `${t.name}(${typeof g === 'string' ? g : JSON.stringify(g)})`; };
  if (it.graded) {
    const ex = it.expected;
    const exTxt = typeof ex === 'string' ? ex : JSON.stringify(ex);
    out.push(h('div', { class: 'card col g8' }, h('div', { class: 'row g8' }, h('span', { class: 'sec' }, T('채점')), h('span', { class: 'grow' }),
      h('span', { class: 'small' }, T('기준 모델'), ' ', evOk(it.ref_correct), NB, T('평가할 모델'), ' ', evOk(it.cand_correct))),
      h('div', { class: 'small muted' }, T('정답')), h('pre', { class: 'dbg-ans sm mono' }, exTxt),
      it.ref_tool || it.cand_tool ? h('div', { class: 'dbg-vs' },
        h('div', null, h('div', { class: 'eyebrow' }, T('기준 모델의 도구 호출')), h('pre', { class: 'dbg-ans sm mono' }, evToolTxt(it.ref_tool))),
        h('div', null, h('div', { class: 'eyebrow' }, T('평가할 모델의 도구 호출')), h('pre', { class: 'dbg-ans sm mono' }, evToolTxt(it.cand_tool)))) : null));
  }
  const msgs = evMessages(X.request);
  if (msgs) out.push(msgs);
  const candText = X.text !== null && X.text !== undefined ? X.text : dv ? dv.cand_text : '';
  if (it.same || !dv) out.push(h('div', { class: 'col g8' }, h('div', { class: 'section-head' }, h('span', { class: 'sec' }, T('답'))), evAnswer(T('두 모델의 답'), candText, 0)));
  else {
    const k = evPrefix(dv.ref_text, dv.cand_text);
    out.push(h('div', { class: 'col g8' }, h('div', { class: 'section-head' }, h('span', { class: 'sec' }, T('두 답')), h('span', { class: 'small muted' }, T('흐린 글은 두 답이 같은 앞부분, 밝은 글은 갈라진 뒤'))),
      h('div', { class: 'dbg-vs' }, evAnswer(T('기준 모델'), dv.ref_text, k), evAnswer(T('평가할 모델'), dv.cand_text, k, 'cand'))));
  }
  if (dv && Array.isArray(dv.candidates) && dv.candidates.length) {
    const ct = dv.cand_token;
    const inTop = ct && dv.candidates.some(c => c.id === ct.id);
    const rows = dv.candidates.map((c, q) => h('tr', { class: ct && c.id === ct.id ? 'ev-pickrow' : q === 0 ? 'ev-refrow' : '' },
      h('td', null, evTok(c.text)),
      h('td', { class: 'small muted' }, [q === 0 ? T('기준 모델이 고름') : null, ct && c.id === ct.id ? T('평가할 모델이 고름') : null].filter(Boolean).join(NB)),
      h('td', { class: 'r num' }, evProb(c.ref_lp)), h('td', { class: 'r num' }, evProb(c.cand_lp))));
    if (ct && !inTop) rows.push(h('tr', { class: 'ev-pickrow' }, h('td', null, evTok(ct.text)), h('td', { class: 'small muted' }, T('평가할 모델이 고름') + (ct.ref_rank >= 0 ? T` · 기준 모델의 ${ct.ref_rank + 1}위` : T` · 기준 모델의 상위 ${s.settings && s.settings.top_k}개 밖`)),
      h('td', { class: 'r num' }, ct.ref_rank >= 0 ? evProb(ct.ref_lp) : '-'), h('td', { class: 'r num' }, ct.ref_rank >= 0 ? evProb(ct.cand_lp) : '-')));
    out.push(h('div', { class: 'col g8' }, h('div', { class: 'section-head' }, h('span', { class: 'sec' }, T`스텝 ${d + 1}의 후보`), h('span', { class: 'small muted' }, T('같은 앞부분에서 두 모델이 다음 토큰에 준 확률'))),
      h('div', { class: 'list' }, h('table', { class: 'data ev-table' },
        h('thead', null, h('tr', null, h('th', null, T('토큰')), h('th', null, h('span', { class: 'sr' }, T('표시'))), h('th', { class: 'r' }, T('기준 모델', 'col')), h('th', { class: 'r' }, T('평가할 모델', 'col')))),
        h('tbody', null, rows)))));
  }
  if (dv && Array.isArray(dv.window) && dv.window.length) {
    out.push(h('div', { class: 'col g8' }, h('div', { class: 'section-head' }, h('span', { class: 'sec' }, T('그 앞뒤 스텝')), h('span', { class: 'small muted' }, T('기준 모델의 답을 두 모델이 점수 매긴 값'))),
      h('div', { class: 'list' }, h('table', { class: 'data ev-table' },
        h('thead', null, h('tr', null, h('th', { class: 'r' }, T('스텝')), h('th', null, T('기준 모델의 토큰')), h('th', { class: 'r' }, T('기준 모델', 'col')), h('th', { class: 'r' }, T('평가할 모델', 'col')), h('th', { class: 'r' }, 'KL'), h('th', { class: 'c-opt' }, 'top-1'))),
        h('tbody', null, dv.window.map(x => h('tr', { class: x.pos === d ? 'ev-refrow' : '' },
          h('td', { class: 'r num' }, String(x.pos + 1)), h('td', null, evTok(x.text)), h('td', { class: 'r num' }, evProb(x.ref_lp)), h('td', { class: 'r num' }, evProb(x.cand_lp)),
          h('td', { class: 'r num' }, evKL(x.kl)), h('td', { class: 'small c-opt' }, x.top1_same ? T('같음') : h('span', { class: 'ev-tie' }, T('다름')))))))),
      h('p', { class: 'small muted m0' }, T('갈라진 스텝 뒤로는 기준 모델의 답을 따라간 값입니다 — 평가할 모델이 자기 답을 이어 간 값이 아닙니다.'))));
  }
  return h('div', { class: 'col g24' }, out);
}
async function evToDebugger(w, dir, i) {
  try {
    const r = await api('/api/eval/debug', { workspace: w.id, dir });
    show('debug');
    dbgSelect(r.run);
    if (i >= 0) { DBG.view = 'req'; DBG.ri = i; }
    renderDebug(true);
  } catch (e) { toast(e.message, 'err'); }
}

function renderEval(force) {
  const page = $('#page-eval');
  if (!page || !S.st || S.view !== 'eval') return;
  const j = S.st.job;
  const js = j && EV_JOBS.includes(j.kind) ? `${j.id}:${j.state}` : '';
  if (js !== EV.jobSig) { EV.jobSig = js; EV.at = 0; EV.checks = {}; EV.tools = undefined; EV.orig = {}; EV.imp = {}; }   // 평가 작업이 시작 · 끝나면 목록 · 검증 보고서 · 도구 · 원래 모델과 견준 값 · 개선 보고서를 다시
  const running = !!(j && EV_JOBS.includes(j.kind) && j.state === 'running');
  if (!EV.loading && Date.now() - EV.at > (running ? 3000 : 20000)) evLoad();
  const w = evWs(EV.sel);
  if (EV.list && EV.sel && !w) { EV.sel = ''; EV.view = 'ws'; }
  evNeed(w);
  const jc = evJobOf();
  const sig = [S.conn, EV.listSig, EV.err, EV.sel, EV.view, EV.dir, EV.item, EV.cmpA, EV.cmpB, EV.filter, EV.pick.join(','), EV.tick,
    jc && [jc.id, jc.state, jc.reason, !!jc.summary], S.st.job && S.st.job.state, S.st.models.map(m => m.id).join(','), S.st.server.state, S.st.device.layer];
  if (!changed('ev', sig) && !force) { evJobTick(); return; }
  const ol = page.querySelector('.ev-list'), od = page.querySelector('.ev-detail');
  const ly = ol ? ol.scrollTop : 0, dy = od && od.dataset.k === evDetailKey() ? od.scrollTop : 0;
  rebuild(page,
    pageHead(EV.list ? T`작업 공간 ${EV.list.workspaces.length}개` : null, T('모델 평가'),
      T('GGUF 파일을 기준 모델과 비교합니다 — 다음 토큰 확률의 차이(KL divergence · top-1 일치율)와 두 모델이 낸 답. 같은 파일 · 같은 설정이면 다시 실행해도 결과가 비트까지 같습니다.')),
    jc ? evJobCard(jc) : null,
    h('div', { class: 'ev' }, h('div', { class: 'ev-body' }, evListPane(), evDetailPane())));
  S.log = { job: jc ? jc.id : 0, from: 0 };
  const nl = page.querySelector('.ev-list'), nd = page.querySelector('.ev-detail');
  if (nl) nl.scrollTop = ly;
  if (nd) nd.scrollTop = dy;
  evJobTick();
}
