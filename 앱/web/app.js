// GeulPU 앱 화면 — 관리자(GeulPU.exe)의 /api 와 /v1 만 쓴다(바깥 자원 없음 — 새 버전 알림만 GitHub API 를 묻는다).
// 디자인: x.ai 디자인 시스템(앱/DESIGN.md) — 어둠 한 가지, 알약 버튼, 8px 카드 · 머리카락 선, 강조색 선셋 오렌지 하나.
'use strict';

// ---------- 말(한국어 · 영어) ----------
// 화면 문구는 한국어로 쓰고 T('…') · T`…${x}…` 로 감싼다 — 영어 화면이면 아래 사전에서 바꾼다(열쇠 = 한국어 원문, ${} 자리는 {}).
// 같은 낱말이 자리에 따라 달라야 하면 T('파일', 'k') 처럼 문맥을 붙인다(사전 열쇠 '파일|k').
// 관리자(GeulPU.exe) · 추론 서버가 보낸 문구는 원문 그대로 두고(화면 코드가 그것을 읽어 판단한다) 보여 줄 때만 TM 으로 바꾼다.
// 말은 설정에서 고른다(자동 = 브라우저 언어 — 한국어가 아니면 영어). 이 브라우저에만 저장한다.
const LANGS = ['auto', 'ko', 'en'];
const LANGPREF = (() => { try { const v = JSON.parse(localStorage.getItem('geulpu.lang') || '"auto"'); return LANGS.includes(v) ? v : 'auto'; } catch (_) { return 'auto'; } })();
const LANG = LANGPREF !== 'auto' ? LANGPREF
  : ((navigator.languages && navigator.languages.length ? navigator.languages : [navigator.language || 'en']).some(l => /^ko\b/i.test(l)) ? 'ko' : 'en');
const EN = LANG === 'en';
document.documentElement.lang = LANG;
const HANGUL = /[가-힣]/;
const MISSING = new Set();   // 영어 화면에서 사전에 없던 문구(개발 때 보려고)
const DICT = {
  "남은 시간 계산 중": "Estimating time left",
  "1분 안에 끝남": "Less than a minute left",
  "약 {}분 남음": "About {} min left",
  "약 {}시간 {}분 남음": "About {} h {} min left",
  "알림 닫기": "Dismiss",
  "브라우저 저장 공간이 부족해 채팅을 저장하지 못했습니다 — 오래된 채팅을 삭제하세요": "Browser storage is full, so the chat could not be saved — delete some old chats",
  "복사하지 못했습니다": "Could not copy",
  "닫기": "Close",
  "취소": "Cancel",
  "확인": "OK",
  "GeulPU.exe와 연결이 끊겼습니다. 다시 연결하는 중… GeulPU.exe가 실행 중인지 확인하세요.": "Lost the connection to GeulPU.exe. Reconnecting… Make sure GeulPU.exe is running.",
  "추론 모델": "Reasoning model",
  "코드 모델": "Code model",
  "일반 모델": "Chat model",
  "하드웨어 진단": "Hardware diagnosis",
  "속도 측정": "Speed test",
  "재현성 검증": "Reproducibility check",
  "모델 불러오기": "Loading a model",
  "하드웨어 확인": "Hardware check",
  "재현성 검증 · 빠른 확인(문제 2개)": "Reproducibility check · quick (2 problems)",
  "재현성 검증 · 문제 32개": "Reproducibility check · 32 problems",
  "텐서 코어 경로": "Tensor-core path",
  "기본 경로": "Basic path",
  "CUDA 경로": "CUDA path",
  "쓸 수 있는 GPU 없음": "No usable GPU",
  "이 GPU 미지원": "Not supported on this GPU",
  "이 PC에 맞음": "Fits this PC",
  "빠듯함 · 문맥이 줄어듦": "Tight · shorter context",
  "그래픽 메모리 부족": "Not enough VRAM",
  "실행되지만 그래픽 메모리에 맞추느라 문맥 길이가 줄어듭니다": "Runs, but the context length is reduced to fit the graphics memory",
  "GeulPU.exe와 연결 끊김": "Disconnected from GeulPU.exe",
  "하드웨어 확인 중": "Checking hardware",
  "하드웨어 보기": "View hardware",
  "{} · 동시 {}개 · 문맥 {}": "{} · {} at once · context {}",
  "모델 불러오는 중 · {}": "Loading model · {}",
  "모델 불러오기 실패": "Model failed to load",
  "불러온 모델 없음": "No model loaded",
  "벤치마크 보기": "View benchmark",
  "다운로드 목록": "Downloads",
  "다운로드 {}개 · {}% · {}/s{}{}": "{} downloading · {}% · {}/s{}{}",
  " · 멈춤 {}개": " · {} paused",
  " · 실패 {}개": " · {} failed",
  "멈춤 {}개": "{} paused",
  "실패 {}개": "{} failed",
  "새 버전이 나왔습니다: {}(지금 v{}). 새 폴더에 압축을 풀고 실행하면 모델과 결과는 그대로 이어집니다.": "A new version is available: {} (you have v{}). Unzip it into a new folder and run it — your models and results carry over.",
  "다운로드 페이지": "Download page",
  "새 채팅": "New chat",
  "오늘": "Today",
  "어제": "Yesterday",
  "지난 7일": "Previous 7 days",
  "이전": "Older",
  " — 답을 쓰는 중": " — answering",
  "'{}' 메뉴": "'{}' menu",
  "채팅 메뉴": "Chat menu",
  "이름 바꾸기": "Rename",
  "삭제": "Delete",
  "답이 끝난 뒤 삭제할 수 있습니다": "You can delete it after the answer finishes",
  "아직 채팅이 없습니다": "No chats yet",
  "채팅 이름": "Chat name",
  "채팅 이름 바꾸기": "Rename chat",
  "저장": "Save",
  "채팅 삭제": "Delete chat",
  "'{}' 채팅을 삭제할까요? 채팅은 이 브라우저에만 저장되어 있어 되돌릴 수 없습니다.": "Delete the chat '{}'? Chats are stored only in this browser, so this cannot be undone.",
  "GeulPU.exe와 연결이 끊겼습니다": "Lost the connection to GeulPU.exe",
  "하드웨어 확인이 끝나면 입력할 수 있습니다": "You can type once the hardware check finishes",
  "쓸 수 있는 GPU가 없어 채팅할 수 없습니다": "No usable GPU, so chat is not available",
  "모델을 다운로드하면 채팅을 시작할 수 있습니다": "Download a model to start chatting",
  "메시지 입력": "Message",
  "Enter 보내기 · Shift+Enter 줄바꿈": "Enter to send · Shift+Enter for a new line",
  "중지": "Stop",
  "보내기": "Send",
  "중지(Esc)": "Stop (Esc)",
  "다른 채팅의 답이 끝나면 보낼 수 있습니다": "You can send once the answer in the other chat finishes",
  "벤치마크가 끝나면 보낼 수 있습니다": "You can send once the benchmark finishes",
  "보내기(Enter)": "Send (Enter)",
  "답이 끝나면 보낼 수 있습니다 · Esc 중지": "You can send when the answer finishes · Esc to stop",
  "모델 선택": "Choose a model",
  "모델 없음": "No model",
  "연결 끊김": "Disconnected",
  "벤치마크 실행 중 · 끝나면 보낼 수 있습니다": "Benchmark running · you can send when it finishes",
  "준비됨 · {}": "Ready · {}",
  "불러오는 중": "Loading",
  "이 PC에서 실행할 수 없음": "Can't run on this PC",
  "불러오기 실패 · 보내면 다시 시도합니다": "Failed to load · sending will try again",
  "보내면 이 모델로 바뀝니다": "Sending switches to this model",
  "지금 불러온 모델: {}": "Loaded now: {}",
  "보내면 불러옵니다": "Loads when you send",
  "답이 끝나면 모델을 바꿀 수 있습니다.": "You can switch models when the answer finishes.",
  ", 불러옴": ", loaded",
  ", 이 PC에서 실행할 수 없음: ": ", can't run on this PC: ",
  "지원하지 않는 파일": "Unsupported file",
  "불러옴": "Loaded",
  "아직 모델이 없습니다.": "No models yet.",
  "모델 다운로드 · 추가": "Download or add a model",
  "불러온 모델 내리기": "Unload the model",
  "GPU 메모리를 비웁니다": "Frees GPU memory",
  "온도": "Temperature",
  "최대 답변 길이(토큰)": "Max answer length (tokens)",
  "비워 두면 모델 기본값": "Leave empty for the model's default",
  "시스템 프롬프트": "System prompt",
  "생각 과정 쓰기(Qwen3)": "Thinking (Qwen3)",
  "생성 설정": "Generation settings",
  "생성 설정 · 이 브라우저에 저장": "Generation settings · saved in this browser",
  "0이면 같은 질문에 늘 같은 답": "0 gives the same answer to the same question",
  "최대 답변 길이": "Max answer length",
  "토큰 · 추론 모델은 생각 과정도 포함": "Tokens · includes the thinking of reasoning models",
  "생각 과정 쓰기": "Thinking",
  "Qwen3 · 끄면 생각 없이 바로 답합니다": "Qwen3 · off: answers directly without thinking",
  "<div class=\"codeblk\"><div class=\"ch\"><span>{}</span><button class=\"cp\" type=\"button\">{}<span>복사</span></button></div><pre><code>{}</code></pre></div>": "<div class=\"codeblk\"><div class=\"ch\"><span>{}</span><button class=\"cp\" type=\"button\">{}<span>Copy</span></button></div><pre><code>{}</code></pre></div>",
  "코드": "Code",
  "복사함": "Copied",
  "복사": "Copy",
  "같은 모델의 더 작은 파일({} · {})로 바꾸면 실행될 수 있습니다.": "A smaller file of the same model ({} · {}) may run.",
  "{} 파일로 다시 시도": "Try again with {}",
  "설정에서 동시 처리 수나 채팅마다 문맥 길이를 줄이거나, 더 작은 모델을 쓰세요.": "Lower the concurrent chats or the context length per chat in Settings, or use a smaller model.",
  "설정 열기": "Open settings",
  "모델 바꾸기": "Change model",
  "다시 시도": "Try again",
  "GeulPU.exe가 실행 중인지 확인한 뒤 다시 시도하세요.": "Make sure GeulPU.exe is running, then try again.",
  "벤치마크가 끝나거나 중지된 뒤 다시 시도하세요.": "Try again after the benchmark finishes or is stopped.",
  "자세히를 열어 원인을 확인하세요. 다른 모델로 바꾸거나 다시 시도할 수 있습니다.": "Open Details to see the cause. You can switch to another model or try again.",
  "잠시 뒤 다시 시도하세요.": "Try again in a moment.",
  "다시 시도하거나 다른 모델로 바꿔 보세요.": "Try again or switch to another model.",
  "더 작은 모델로 측정하거나 다른 프로그램의 GPU 사용을 줄여 보세요.": "Measure with a smaller model or reduce GPU use by other programs.",
  "그래픽 드라이버를 최신으로 업데이트한 뒤 다시 실행해 보세요.": "Update the graphics driver, then run it again.",
  "자세히를 열어 원인을 확인하세요.": "Open Details to see the cause.",
  "자세히": "Details",
  "처음 한 번은 이 GPU에 맞는 커널을 만들고 가장 빠른 실행 방식을 측정합니다(보통 수십 초~2분). 다음부터는 더 빨리 준비됩니다.": "The first time, GeulPU builds kernels for this GPU and measures the fastest way to run them (usually tens of seconds to 2 minutes). Later it is ready faster.",
  "수정": "Edit",
  "이 메시지 수정": "Edit this message",
  "모델을 GPU에 올리는 중입니다.": "Loading the model onto the GPU.",
  "생각하는 중": "Thinking",
  "생각 과정 · 중지됨": "Thinking · stopped",
  "생각 과정 · 끊김": "Thinking · cut off",
  "{}초 동안 생각함": "Thought for {} s",
  "생각 과정": "Thinking",
  "답을 쓰기 전에 최대 답변 길이({}토큰)에 닿았습니다.": "Reached the max answer length ({} tokens) before answering.",
  "최대 답변 길이({}토큰)에서 답이 끊겼습니다.": "The answer was cut off at the max answer length ({} tokens).",
  "이어서 생성": "Continue",
  "{}토큰으로 다시 생성": "Regenerate with {} tokens",
  "답 복사": "Copy answer",
  "복사했습니다": "Copied",
  "다시 생성 — 지금 답은 버전으로 남습니다": "Regenerate — the current answer stays as a version",
  "다시 생성": "Regenerate",
  "버전 {} / {}": "Version {} / {}",
  "이전 버전": "Previous version",
  "다음 버전": "Next version",
  "다음 문장을 자연스러운 영어로 바꿔 줘: 내일 회의는 3시로 미뤄졌습니다": "Translate into natural Korean: \"Tomorrow's meeting has been moved to 3 o'clock.\"",
  "파이썬으로 피보나치 함수를 작성해 줘": "Write a Fibonacci function in Python",
  "봄을 주제로 짧은 시를 써 줘": "Write a short poem about spring",
  "아래 글을 세 줄로 요약해 줘": "Summarize the text below in three lines",
  "아래 글을 세 줄로 요약해 줘.\n\n": "Summarize the text below in three lines.\n\n",
  "일시정지": "Pause",
  "확인 ": "Verifying ",
  "무결성 확인 중{}{}%": "Verifying integrity{}{}%",
  "{}%에서 멈춤": "Paused at {}%",
  "글(geul) 언어로 만든 로컬 LLM": "A local LLM built with the geul (글) language",
  "무엇을 도와드릴까요?": "How can I help?",
  "하드웨어를 확인하는 중입니다": "Checking your hardware",
  "처음 실행할 때 GPU와 드라이버를 확인합니다. 잠시만 기다려 주세요.": "On the first run GeulPU checks the GPU and driver. Please wait a moment.",
  "쓸 수 있는 GPU를 찾지 못했습니다": "No usable GPU found",
  "NVIDIA RTX 30 시리즈 이상(CUDA) 또는 Vulkan 1.3을 지원하는 GPU와 최신 그래픽 드라이버가 필요합니다.": "GeulPU needs an NVIDIA RTX 30 series or newer (CUDA) or a GPU with Vulkan 1.3, and an up-to-date graphics driver.",
  "확인한 내용": "What was checked",
  "이 PC에서 실행할 수 있는 모델이 없습니다": "No installed model runs on this PC",
  "설치된 모델을 이 GPU나 GeulPU가 지원하지 않습니다. 모델 화면에서 이유를 확인하고 다른 모델을 다운로드하세요.": "The installed models aren't supported by this GPU or by GeulPU. See the reason on the Models screen and download another model.",
  "모델 보기": "View models",
  "먼저 모델이 필요합니다": "First, you need a model",
  "이 PC에는 {} {}({}) 모델을 추천합니다.": "For this PC we recommend {} {} ({}).",
  "이 GPU(공유 메모리 32 KB)는 가중치가 모두 Q8_0이거나 모두 Q4_0인 파일만 실행합니다. 목록의 Q8_0 7B는 그래픽 메모리가 10 GB 이상 필요하고(이 PC {}), 흔한 Q4_0 파일은 출력 가중치가 Q6_K라 아직 받지 않습니다. 지금은 이 PC에서 실행할 수 있는 파일이 없습니다.": "This GPU (32 KB of shared memory) only runs files whose weights are all Q8_0 or all Q4_0. The Q8_0 7B files in the list need 10 GB of graphics memory or more (this PC: {}), and common Q4_0 files keep the output weights in Q6_K, which is not supported yet. No file in the list runs on this PC right now.",
  "목록의 모델 가운데 이 PC의 그래픽 메모리({})에 맞는 것이 없습니다. Hugging Face에서 더 작은 파일을 찾아볼 수 있습니다 — 받기 전에 실행되는지 확인합니다.": "None of the listed models fit this PC's graphics memory ({}). You can look for smaller files on Hugging Face — GeulPU checks that a file runs before downloading it.",
  "추천 모델 다운로드": "Download the recommended model",
  "모델 찾기": "Find models",
  "GGUF 파일 추가": "Add a GGUF file",
  "다운로드 진행 보기": "View download progress",
  "모델 둘러보기": "Browse models",
  "모델을 불러오지 못했습니다 — ": "Could not load the model — ",
  "중지했습니다": "Stopped",
  "모델을 불러오지 못했습니다": "Could not load the model",
  "{} 불러오는 중 · {}{}": "Loading {} · {}{}",
  " · 커널 준비 중": " · preparing kernels",
  "GeulPU.exe와 연결이 끊겨 보낼 수 없습니다": "Can't send — disconnected from GeulPU.exe",
  "위의 모델 선택에서 모델을 고르세요": "Choose a model with the model button above",
  "먼저 모델 화면에서 모델을 다운로드하거나 추가하세요": "First download or add a model on the Models screen",
  "이 모델은 이 PC에서 실행할 수 없습니다 — ": "This model can't run on this PC — ",
  "메시지 수정": "Edit message",
  "보내면 지금 대화는 이전 버전으로 남습니다 · Esc 취소": "Sending keeps the current conversation as an earlier version · Esc to cancel",
  "보내면 지금 답은 이전 버전으로 남습니다 · Esc 취소": "Sending keeps the current answer as an earlier version · Esc to cancel",
  "{} 불러오는 중": "Loading {}",
  "답변이 끝났습니다": "The answer is finished",
  "중지됨": "Stopped",
  "오류": "Error",
  "불러오기를 중지했습니다": "Loading stopped",
  "생각을 먼저 쓰고 답하는 추론 모델{}수학{}논리": "Reasoning model that thinks before answering{}math{}logic",
  "코드 작성{}설명": "Writing code{}explaining",
  "생각 켬{}끔을 고르는 대화 모델{}한국어{}영어": "Chat model with thinking on{}off{}Korean{}English",
  "한국어{}영어 일반 대화": "Everyday chat in Korean{}English",
  "모델 화면": "Models screen",
  "내 모델": "My models",
  "모델": "Models",
  "모델 폴더 열기": "Open the model folder",
  "받는 중": "Downloading",
  "설치된 모델": "Installed models",
  "일부 모델은 이 GPU에서 실행할 수 없습니다. 꼬리표에 마우스를 올리면 까닭이 보입니다.": "Some models can't run on this GPU. Hover over a tag to see why.",
  "아직 설치된 모델이 없습니다": "No models installed yet",
  "Hugging Face의 모델을 둘러보고 받거나, 이미 있는 GGUF 파일을 추가하세요.": "Browse and download models from Hugging Face, or add a GGUF file you already have.",
  "이어서 다운로드": "Resume download",
  "다시 다운로드": "Download again",
  "받은 것 지우기": "Delete the partial file",
  "받은 것 지우기: {} {}": "Delete the partial file: {} {}",
  "{}{}{}문맥 {}{}": "{}{}{}context {}{}",
  " (최대 {})": " (max {})",
  "무결성 확인 중": "Verifying integrity",
  "무결성 확인 안 됨": "Integrity not verified",
  "이 모델로 채팅: {} {}": "Chat with this model: {} {}",
  "채팅": "Chat",
  "메뉴: {} {}": "Menu: {} {}",
  "모델 메뉴": "Model menu",
  "모델 내리기": "Unload model",
  "파일 무결성 다시 확인": "Verify file integrity again",
  "받은 그대로임을 확인했습니다(SHA-256) · 몇 초": "Verified as downloaded (SHA-256) · a few seconds",
  "받은 그대로인지 SHA-256으로 확인 · 몇 초": "Check with SHA-256 that it is unchanged · a few seconds",
  "내리고 목록에서 제거": "Unload and remove from the list",
  "목록에서 제거": "Remove from the list",
  "모델을 내렸습니다": "Model unloaded",
  "파일 무결성(SHA-256)을 확인합니다 — 8 GB 파일에 몇 초 걸립니다": "Checking file integrity (SHA-256) — a few seconds for an 8 GB file",
  "{} {} — {}목록에서 제거합니다.": "{} {} — {}removes it from the list.",
  "불러온 모델을 내리고 ": "unloads it and ",
  "파일도 삭제 ({})": "Also delete the file ({})",
  "GeulPU 모델 폴더 밖의 파일은 삭제하지 않고 목록에서만 제거합니다.": "Files outside GeulPU's model folder are not deleted, only removed from the list.",
  "모델 제거": "Remove model",
  "제거": "Remove",
  "서버가 응답하지 않음(시간 초과)": "the server did not respond (timeout)",
  "서버 주소를 찾지 못함": "server address not found",
  "서버에 연결하지 못함": "could not connect to the server",
  "연결이 끊김": "the connection was lost",
  "서버가 연결을 끊음": "the server closed the connection",
  "서버의 응답이 올바르지 않음": "invalid response from the server",
  "보안 연결(HTTPS)을 맺지 못함": "could not establish a secure (HTTPS) connection",
  "WinHTTP 오류 {}{}": "WinHTTP error {}{}",
  "'이어서 다운로드'": "'Resume download'",
  "'다시 다운로드'": "'Download again'",
  "연결이 끊겨 {}%에서 멈췄습니다. 인터넷 연결을 확인한 뒤 {}를 누르세요.": "The connection dropped at {}%. Check your internet connection, then click {}.",
  "다운로드 서버에 연결하지 못했습니다. 인터넷 · 방화벽 · 프록시를 확인한 뒤 {}를 누르세요.": "Couldn't connect to the download server. Check your internet, firewall and proxy, then click {}.",
  "다운로드 서버가 오류(HTTP {})를 보냈습니다. 잠시 뒤 {}를 누르세요.": "The download server returned an error (HTTP {}). Wait a moment, then click {}.",
  "{}. 여유 공간과 폴더 권한을 확인한 뒤 {}를 누르세요.": "{}. Check the free space and folder permissions, then click {}.",
  "받은 파일이 원본과 다릅니다(SHA-256 불일치). 받은 것을 지웠으니 {}를 누르세요.": "The downloaded file differs from the original (SHA-256 mismatch). It was deleted — click {}.",
  "다운로드하지 못했습니다": "Download failed",
  "Hugging Face에서 파일 앞부분을 받지 못했습니다. 인터넷 연결을 확인한 뒤 다시 확인하세요.": "Couldn't fetch the start of the file from Hugging Face. Check your internet connection and check again.",
  "Hugging Face가 오류(HTTP {})를 보냈습니다. 잠시 뒤 다시 확인하세요.": "Hugging Face returned an error (HTTP {}). Check again in a moment.",
  "확인하지 못했습니다": "Check failed",
  "지우기": "Delete",
  "다운로드": "Download",
  "진행 중인 다운로드가 없습니다.": "No downloads in progress.",
  "모델 화면에서 보기": "View on the Models screen",
  "이 PC에서는 쓸 수 있는 GPU를 찾지 못해 이 모델을 실행할 수 없습니다. 다른 PC로 옮겨 쓸 파일이면 그대로 다운로드할 수 있습니다.": "No usable GPU was found on this PC, so this model can't run here. You can still download it to use on another PC.",
  "이 PC의 그래픽 메모리({})로는 실행되지 않을 수 있습니다. 그래픽 메모리 {} GB 이상을 권장합니다.": "This PC's graphics memory ({}) may not be enough. {} GB or more is recommended.",
  "이 PC의 그래픽 메모리({})에서는 문맥 길이가 줄어듭니다.": "On this PC's graphics memory ({}) the context length is reduced.",
  "파일": "Files",
  "크기": "Size",
  "{} ({} 바이트)": "{} ({} bytes)",
  "저장 위치": "Storage",
  "{}{}여유 {}": "{}{}{} free",
  "알 수 없음": "unknown",
  "출처": "Source",
  "라이선스": "License",
  "기술 정보": "Technical details",
  "이 파일로는 아직 GeulPU에서 실행해 보지 않았습니다. 다운로드가 끝나면 지원 여부를 바로 알려 드립니다.": "GeulPU hasn't run this file yet. Support is checked as soon as the download finishes.",
  "중간에 끊겨도 다시 누르면 이어서 받고, 다 받으면 SHA-256으로 파일이 맞는지 확인합니다.": "If it stops, clicking again resumes; when it finishes, the file is verified with SHA-256.",
  "모델 다운로드": "Download model",
  "상위 폴더": "Parent folder",
  "추가했습니다: {}": "Added: {}",
  "하위 폴더가 없습니다": "No subfolders",
  "이 폴더에는 .gguf 파일이나 하위 폴더가 없습니다": "No .gguf files or subfolders here",
  "폴더 경로(예: D:\\models)": "Folder path (e.g. D:\\models)",
  "폴더 경로": "Folder path",
  "모델 폴더 고르기": "Choose the model folder",
  "받은 모델을 둘 폴더를 고르세요. 다음 실행부터 적용됩니다.": "Choose where downloaded models go. Applies from the next start.",
  "GeulPU는 qwen2 구조(Qwen2.5 · DeepSeek-R1-Distill-Qwen, 7B 이상), qwen3 구조(Qwen3 1.7B · 8B · 14B), llama 구조(Llama 3.1 8B · Llama 3.2 3B · Mistral 7B v0.3)의 GGUF를 실행합니다. 추가하면 지원 여부를 바로 확인합니다.": "GeulPU runs GGUF files of the qwen2 architecture (Qwen2.5 · DeepSeek-R1-Distill-Qwen, 7B and up), the qwen3 architecture (Qwen3 1.7B · 8B · 14B) and the llama architecture (Llama 3.1 8B · Llama 3.2 3B · Mistral 7B v0.3). Support is checked as soon as you add a file.",
  "이동": "Go",
  "이 폴더 선택": "Choose this folder",
  "구조 정보 없음": "No architecture info",
  "{} · 미지원 구조": "{} · unsupported architecture",
  "대화용 모델 아님": "Not a chat model",
  "이미지 생성용 텍스트 인코더 · 임베딩 · 음성처럼 채팅에 쓰는 모델이 아닙니다.": "Not a model for chat — e.g. a text encoder for image generation, an embedding or a speech model.",
  "GeulPU에서 실행됨": "Runs in GeulPU",
  "이 크기는 아직 미지원": "Size not supported yet",
  "Qwen3의 0.6B · 4B · 32B는 질의 폭이 너비와 달라 GeulPU가 아직 실행하지 않습니다.": "Qwen3 0.6B · 4B · 32B have a query width different from the hidden size, which GeulPU doesn't run yet.",
  "크기 확인 필요": "Size needs a check",
  "흔한 Qwen3 크기(1.7B · 8B · 14B)가 아닙니다. 받기 전에 파일 앞부분을 받아 실행되는지 확인합니다.": "Not a common Qwen3 size (1.7B · 8B · 14B). GeulPU fetches the start of the file to check that it runs before downloading.",
  "헤드 크기가 64인 작은 Qwen2(0.5B 등)는 GeulPU가 아직 실행하지 않습니다.": "GeulPU doesn't run small Qwen2 models with head size 64 (such as 0.5B) yet.",
  "헤드 크기가 64인 작은 llama 구조(Llama 3.2 1B · TinyLlama 등)는 GeulPU가 아직 실행하지 않습니다.": "GeulPU doesn't run small llama-architecture models with head size 64 (such as Llama 3.2 1B · TinyLlama) yet.",
  "흔한 Llama 3 · Mistral 크기(3B · 7~8B · 70B)가 아닙니다. 받기 전에 파일 앞부분을 받아 실행되는지 확인합니다.": "Not a common Llama 3 · Mistral size (3B · 7–8B · 70B). GeulPU fetches the start of the file to check that it runs before downloading.",
  "{}일 전": "{0} {0|day|days} ago",
  "{}주 전": "{0} {0|week|weeks} ago",
  "{}개월 전": "{0} {0|month|months} ago",
  "{}년 전": "{0} {0|year|years} ago",
  "요즘 뜨는": "Trending",
  "Hugging Face의 트렌드 점수 순": "By Hugging Face's trending score",
  "인기": "Popular",
  "최근 30일 다운로드가 많은 순": "Most downloads in the last 30 days",
  "좋아요": "Likes",
  "좋아요가 많은 순": "Most likes",
  "최근 업데이트": "Recently updated",
  "최근에 고친 저장소 순": "Most recently updated repositories",
  "모든 크기": "All sizes",
  "모델 목록": "Model list",
  "모델 자세히": "Model details",
  "이름으로 좁히기 — 예: qwen3, coder, r1, 14b": "Filter by name — e.g. qwen3, coder, r1, 14b",
  "모델 이름으로 좁히기": "Filter by model name",
  "검색어 지우기": "Clear the search",
  "실행 안 되는 모델도 보기": "Also show models that don't run",
  "끄면 GeulPU에서 실행되는 구조(qwen2 · qwen3 · llama)와 크기만 보입니다": "When off, only architectures (qwen2 · qwen3 · llama) and sizes that GeulPU runs are shown",
  "정렬": "Sort",
  "GeulPU 확인됨": "Verified by GeulPU",
  "GeulPU에서 실제로 돌려 본 파일이 있는 저장소만": "Only repositories with a file GeulPU has actually run",
  "이 PC에 맞는 크기": "Fits this PC",
  "매개변수 수로 어림한 Q4_K_M 파일이 이 PC의 그래픽 메모리에 들어가는 것만": "Only models whose Q4_K_M file (estimated from the parameter count) fits this PC's graphics memory",
  "Hugging Face에 연결하지 못했습니다. 인터넷 연결을 확인한 뒤 다시 시도하세요.": "Couldn't reach Hugging Face. Check your internet connection and try again.",
  "GeulPU에서 실행되는 것": "runs in GeulPU",
  "실행 안 되는 것 포함": "including those that don't run",
  "{}개{}": "{0}{1} {0|repository|repositories}",
  " 이상": "+",
  "0개": "0 repositories",
  "{}{}순{}{}": "{0}sorted by {1}{2}{3}",
  "Hugging Face의 GGUF 저장소": "GGUF repositories on Hugging Face",
  "불러오는 중…": "Loading…",
  "조건에 맞는 모델이 없습니다": "No models match",
  "거르기를 줄이거나, '실행 안 되는 모델도 보기'를 켜면 Hugging Face의 모든 GGUF 모델이 보입니다(실행되지 않는 것은 흐리게).": "Loosen the filters, or switch on \"Also show models that don't run\" to see every GGUF model on Hugging Face (those that don't run are dimmed).",
  "검색어나 거르기를 바꿔 보세요.": "Try another search or other filters.",
  "더 보기": "Load more",
  "{} 기반": "based on {}",
  "GeulPU에서 실제로 돌려 본 파일이 있습니다": "Has a file GeulPU has actually run",
  "실행됨": "Runs",
  "구조 알 수 없음": "Unknown architecture",
  "최근 30일 다운로드": "Downloads in the last 30 days",
  "업데이트": "Updated",
  "이 PC 추천": "Recommended",
  "모델을 고르면 자세한 정보가 여기에 보입니다": "Choose a model to see its details here",
  "크기{}구조{}다운로드 수{}라이선스{}파일마다 이 PC에 맞는지{}모델 설명": "Size{}architecture{}downloads{}license{}whether each file fits this PC{}model card",
  "{} 구조": "{} architecture",
  "매개변수 {}": "{} parameters",
  "문맥 최대 {}": "context up to {}",
  "GeulPU에서 실행되는 구조": "Architecture GeulPU runs",
  "최근 30일": "last 30 days",
  "누적 다운로드": "All-time downloads",
  "이 저장소의 구조({})는 아직 GeulPU 엔진에 없습니다. 정보는 볼 수 있지만 받을 수 없습니다.": "This repository's architecture ({}) isn't in the GeulPU engine yet. You can read the details but not download it.",
  "Hugging Face가 이 저장소의 GGUF 구조를 알려 주지 않았습니다. 파일을 받기 전에 앞부분을 받아 확인합니다.": "Hugging Face didn't report this repository's GGUF architecture. GeulPU fetches the start of a file to check it before downloading.",
  "파일 목록을 받지 못했습니다. 인터넷 연결을 확인한 뒤 모델을 다시 고르세요.": "Couldn't get the file list. Check your internet connection and choose the model again.",
  "파일 목록을 불러오는 중…": "Loading the file list…",
  "저장소": "Repository",
  "저장소에 적혀 있지 않음": "not stated in the repository",
  "바탕 모델": "Base model",
  " 외 {}개": " and {} more",
  "언어": "Languages",
  "용도": "Task",
  "만든 날": "Created",
  "모델 설명": "Model card",
  "Hugging Face의 README": "README on Hugging Face",
  "접기": "Collapse",
  "모두 보기": "Show all",
  "모델 설명을 불러오는 중…": "Loading the model card…",
  "양자화마다 하나 — 고르면 받기 전에 실행되는지 확인": "One per quantization — checked to run before downloading",
  "정보": "Details",
  "이 저장소에는 GGUF 파일이 없습니다.": "This repository has no GGUF files.",
  "지원하지 않는 구조": "Unsupported architecture",
  "아직 미지원 크기": "Size not supported yet",
  "대화용 모델이 아님": "Not a chat model",
  "여러 조각으로 나뉜 파일 — 합친 파일만 받습니다": "Split into parts — only single-file models are supported",
  "{} 형식은 아직 미지원": "{} not supported yet",
  "양자화 형식을 알 수 없는 파일": "Unknown quantization format",
  "SHA-256 정보가 없는 파일": "No SHA-256 information",
  "설치됨": "Installed",
  "실행되는지 확인하는 중": "Checking that it runs",
  "이 PC에 추천": "Recommended for this PC",
  "GeulPU에서 이 파일로 실제로 돌려 보았습니다": "GeulPU has actually run this file",
  "실행 확인됨": "Verified to run",
  "이 엔진이 받지 않는 파일입니다": "The engine doesn't accept this file",
  "확인이 너무 오래 걸립니다. 잠시 뒤 다시 누르세요.": "The check is taking too long. Click again in a moment.",
  "{} {}{}{}B{}문맥 최대 {}": "{} {}{}{}B{}context up to {}",
  "저장소에 적혀 있지 않음 — 저장소에서 확인하세요": "not stated in the repository — check the repository",
  "이 파일의 답 품질은 GeulPU에서 확인하지 않았습니다. 구조와 형식은 받기 전에 확인했고, 다 받으면 SHA-256으로 파일이 맞는지 확인합니다.": "GeulPU hasn't checked the answer quality of this file. The architecture and format were checked before downloading, and the file is verified with SHA-256 once downloaded.",
  "다운로드를 시작했습니다 — 진행은 상태 막대와 '내 모델'에서 볼 수 있습니다": "Download started — follow it in the status bar and under My models",
  "받다 만 파일 지우기": "Delete the partial download",
  "{} {} — 지금까지 받은 {}를 지웁니다.{}": "{} {} — deletes the {} downloaded so far.{}",
  " Hugging Face에서 더한 항목이라 목록에서도 뺍니다.": " It was added from Hugging Face, so it is also removed from the list.",
  "1~6분": "1–6 min",
  "이 PC의 메모리 대역폭과 정수 연산 성능을 측정하고, 엔진의 각 연산이 하드웨어 한계에 얼마나 가까운지 보여 줍니다.": "Measures this PC's memory bandwidth and integer throughput, and shows how close each engine operation comes to the hardware limit.",
  "1~3분": "1–3 min",
  "프롬프트 처리(512 · 2048토큰)와 답 생성 속도를 토큰/초로 세 번씩 측정합니다.": "Measures prompt processing (512 · 2048 tokens) and generation speed in tokens/s, three times each.",
  "경로마다 5~20분": "5–20 min per path",
  "DeepSeek-R1-Distill-Qwen-7B Q8_0으로 수학 문제 32개를 풀어 기준 PC의 출력과 비트 단위로 비교합니다. 빌드와 커널의 해시도 함께 확인합니다.": "Solves 32 math problems with DeepSeek-R1-Distill-Qwen-7B Q8_0 and compares the output with the reference PC bit for bit. The build and kernel hashes are checked too.",
  "생성 토큰/초": "Generation tokens/s",
  "프롬프트 토큰/초": "Prompt tokens/s",
  "프롬프트 2048토큰": "Prompt 2048 tokens",
  "메모리 대역폭": "Memory bandwidth",
  "정수 텐서": "Integer tensor",
  "생성 · 하드웨어 한계 대비": "Generation · of the hardware limit",
  "한계 대비 생성": "Generation vs limit",
  "프롬프트 · 하드웨어 한계 대비": "Prompt · of the hardware limit",
  "한계 대비 프롬프트": "Prompt vs limit",
  "결과 {}개": "{0} {0|result|results}",
  "벤치마크": "Benchmark",
  "한 번에 하나씩 실행됩니다. 실행하는 동안 불러온 채팅 모델은 GPU에서 내립니다. 결과는 zip으로 내보낼 수 있습니다.": "Runs one at a time. While it runs, the loaded chat model is unloaded from the GPU. Results can be exported as a zip.",
  "측정할 모델": "Model to measure",
  "하드웨어 진단 · 속도 측정에 씁니다": "Used by the hardware diagnosis and the speed test",
  "하드웨어 진단 · 속도 측정에 쓸 모델": "Model for the hardware diagnosis and the speed test",
  "실행할 수 있는 모델이 없습니다": "No model that can run",
  "최근 결과": "Recent results",
  "결과 폴더 열기": "Open the results folder",
  "끝났습니다": "Finished",
  "실패했습니다": "Failed",
  "실행 중": "Running",
  "완료": "Done",
  "실패": "Failed",
  "같은 설정으로 다시 실행": "Run again with the same settings",
  "카드 닫기": "Close card",
  "끝난 작업 카드 닫기": "Close the finished job card",
  "빠른 확인 — ": "Quick check — ",
  "결과 보기": "View result",
  "zip 내보내기": "Export zip",
  "실행 기록": "Run log",
  "도구 출력": "Tool output (raw)",
  "빠른 확인 — 문제 2개 · 64토큰(전체 검증 아님)": "Quick check — 2 problems · 64 tokens (not the full check)",
  "검증할 경로": "Paths to check",
  "모든 경로(권장)": "All paths (recommended)",
  "텐서 코어 경로만": "Tensor-core path only",
  "기본 경로만": "Basic path only",
  "DeepSeek-R1-Distill-Qwen-7B Q8_0 모델이 필요합니다.": "Needs the DeepSeek-R1-Distill-Qwen-7B Q8_0 model.",
  "다운로드({})": "Download ({})",
  "먼저 모델을 설치하세요.": "Install a model first.",
  "채팅의 답이 끝난 뒤 실행할 수 있습니다": "Available after the chat answer finishes",
  "채팅 모델을 내립니다": "Unload the chat model",
  "측정하는 동안 GPU를 혼자 쓰도록 불러온 채팅 모델을 내립니다. 계속할까요?": "To have the GPU to itself while measuring, the loaded chat model is unloaded. Continue?",
  "계속": "Continue",
  "다른 작업 실행 중": "Another job is running",
  "실행": "Run",
  "불일치": "Mismatch",
  "앞부분 일치": "Start matches",
  "일치": "Match",
  "빠른 확인": "Quick check",
  "아직 결과가 없습니다": "No results yet",
  "위에서 측정을 하나 실행하면 여기에 쌓입니다.": "Run a measurement above and it will appear here.",
  "종류": "Type",
  "일시": "Date",
  "결과": "Result",
  "동작": "Actions",
  "동시 {}개 생성 · 문맥 {}": "{} concurrent generations · context {}",
  "{}행 일괄 처리 · 문맥 {}": "{}-row batch · context {}",
  "생성 · 문맥 {}": "Generation · context {}",
  "프롬프트 {}토큰": "Prompt {} tokens",
  " · 대체 커널": " · alternative kernel",
  " · 동시 {}개 · 문맥 {}": " · {} at once · context {}",
  " · 빠른 확인(문제 2개 · 64토큰)": " · quick check (2 problems · 64 tokens)",
  "하드웨어 진단 결과": "Hardware diagnosis result",
  "속도 측정 결과": "Speed test result",
  "앞 64토큰이 기준 PC와 같습니다 — 전체 검증은 아닙니다": "The first 64 tokens match the reference PC — not the full check",
  "앞부분부터 기준 PC와 다릅니다": "Differs from the reference PC from the start",
  "문제 {}개 모두 기준 PC와 출력이 같습니다": "All {} problems give the same output as the reference PC",
  "기준 PC와 출력이 다른 문제가 있습니다": "Some problems give a different output from the reference PC",
  "문제의 앞 64토큰이 같음": "problems match in the first 64 tokens",
  "문제 · 경로 {}개 모두 같음": "all {} problem · path pairs match",
  "문제가 같은 출력": "problems give the same output",
  "{}회 중 {}회 일치(문제 {} × 경로 {})": "{1} of {0} runs match ({2} problems × {3} paths)",
  "동시 {}개 · {}": "{} at once · {}",
  "1분 미만": "under a minute",
  "분": " min",
  "문제": "Problem",
  "판정": "Verdict",
  "토큰": "Tokens",
  "기준": "Reference",
  "토큰마다 로짓(점수)의 SHA-256": "SHA-256 of the logits (scores) at every token",
  "단계별 해시": "Per-step hash",
  "파일 {}.json": "File {}.json",
  "같음": "Same",
  "다름 · {}번째 글자부터": "Differs · from character {}",
  "모든 문제 {}개 보기": "Show all {} problems",
  "문제 {}개 보기": "Show {} problems",
  "빌드 · 커널 해시": "Build · kernel hashes",
  "프롬프트 토큰/초(512토큰)": "Prompt tokens/s (512 tokens)",
  "프롬프트 토큰/초(2048토큰)": "Prompt tokens/s (2048 tokens)",
  "측정값": "Measurements",
  "세 번 측정한 가운데 값": "Median of three runs",
  "항목": "Item",
  "프롬프트 처리": "Prompt processing",
  "토큰당 생성": "Per-token generation",
  "토큰/초": "Tokens/s",
  "연산별 시간": "Time per operation",
  "100% = 하드웨어 한계 · 높을수록 좋습니다": "100% = hardware limit · higher is better",
  "연산": "Operation",
  "합계": "Total",
  "한계 대비": "Of limit",
  "같은 계산을 다른 방식으로 하는 후보 커널 — 비교하려고 함께 잽니다": "A candidate kernel doing the same computation another way — measured for comparison",
  "진단 도구의 이름: ": "Name in the diagnosis tool: ",
  "결과.txt 원문": "Raw 결과.txt (result file)",
  "폴더 열기": "Open folder",
  "하드웨어": "Hardware",
  "다시 확인": "Check again",
  "확인하는 중…": "Checking…",
  "NVIDIA RTX 30 시리즈 이상(CUDA) 또는 Vulkan 1.3을 지원하는 GPU가 필요합니다.": "GeulPU needs an NVIDIA RTX 30 series or newer (CUDA) or a GPU with Vulkan 1.3.",
  "백엔드": "Backend",
  "NVIDIA 드라이버(nvcuda.dll)": "NVIDIA driver (nvcuda.dll)",
  "Vulkan 드라이버(vulkan-1.dll)": "Vulkan driver (vulkan-1.dll)",
  "그래픽 메모리": "Graphics memory",
  "드라이버": "Driver",
  "커널": "Kernels",
  "공유 메모리 32 KB용": "For 32 KB shared memory",
  "표준": "Standard",
  "CUDA용": "For CUDA",
  "처음 한 번 이 GPU에 맞춰 만듭니다": "Built once for this GPU",
  "행렬 가속": "Matrix acceleration",
  "텐서 코어({})": "Tensor cores ({})",
  "없음 — 일반 연산": "None — regular arithmetic",
  "텐서 코어(정수)": "Tensor cores (integer)",
  "텐서 코어를 쓸 수 없어 일반 연산으로 대신합니다": "Tensor cores are not available, so regular arithmetic is used",
  "Vulkan 장치": "Vulkan devices",
  "CUDA 장치": "CUDA devices",
  "내장": "Integrated",
  "없음": "none",
  "사용 중": "In use",
  "지원": "Supported",
  "지원 안 함": "Not supported",
  "내장 GPU는 시스템 메모리를 나눠 씁니다": "An integrated GPU shares the system memory",
  "Vulkan이 알려 준 장치 메모리 — 드라이버가 예약한 만큼 작게 보일 수 있습니다": "Device memory reported by Vulkan — may look smaller by what the driver reserves",
  "공유 ": "Shared ",
  "Compute Capability {}.{}{}SM {}개{}CUDA {}.{}": "Compute Capability {}.{}{}{} SMs{}CUDA {}.{}",
  "8.0 미만 · 지원 안 함": "Below 8.0 · not supported",
  "CUDA 없음 — NVIDIA 드라이버(nvcuda.dll)가 필요합니다": "No CUDA — needs the NVIDIA driver (nvcuda.dll)",
  "저장했습니다 · 다음에 모델을 불러올 때 적용됩니다": "Saved · applies the next time a model is loaded",
  "GPU 백엔드": "GPU backend",
  "자동": "Auto",
  "저장했습니다 · 하드웨어를 다시 확인합니다": "Saved · checking the hardware again",
  "자동 — 첫 번째 외장 GPU": "Auto — the first discrete GPU",
  "동시 처리 수": "Concurrent chats",
  "채팅마다 문맥 길이": "Context length per chat",
  "기본 폴더 사용": "Use the default folder",
  "모델 폴더": "Model folder",
  "저장했습니다 · 다음 실행부터 적용됩니다": "Saved · applies from the next start",
  "모델 폴더 찾아보기": "Browse for the model folder",
  "찾아보기": "Browse",
  "새 버전 알림": "New-version notice",
  "저장했습니다": "Saved",
  "저장됨": "Saved",
  " · 새 버전 {} 있음": " · new version {} available",
  " · 최신 버전": " · up to date",
  "설정": "Settings",
  "바꾼 값은 바로 저장됩니다(숫자 · 글 칸은 다 쓰고 칸을 떠날 때).": "Changes are saved right away (number and text fields when you leave the field).",
  "자동은 NVIDIA GPU면 CUDA, 그 밖에는 Vulkan을 씁니다.": "Auto uses CUDA on an NVIDIA GPU and Vulkan otherwise.",
  "GPU가 여러 개일 때 Vulkan으로 쓸 장치": "Which device Vulkan uses when there are several GPUs",
  "동시에 답할 수 있는 채팅 수(1~8) · 다음에 모델을 불러올 때 적용": "How many chats can answer at once (1–8) · applies the next time a model is loaded",
  "토큰 · 그래픽 메모리에 맞지 않으면 엔진이 자동으로 줄입니다. 다음에 모델을 불러올 때 적용": "Tokens · the engine reduces it if it doesn't fit the graphics memory. Applies the next time a model is loaded",
  "다음 실행부터 적용됩니다. 기본: {}": "Applies from the next start. Default: {}",
  "데이터 폴더": "Data folder",
  "열기": "Open",
  "앱을 열 때 GitHub에서 최신 버전 번호만 확인합니다.": "When the app opens, it checks only the latest version number on GitHub.",
  "커밋 ": "commit ",
  "개발 빌드": "development build",
  "릴리스": "Releases",
  "소스": "Source",
  "API 주소": "API address",
  "다른 프로그램은 OpenAI 호환 API로 쓸 수 있습니다(이 PC 안에서만).": "Other programs can use the OpenAI-compatible API (from this PC only).",
  "주소 복사": "Copy address",
  "API 주소 복사": "Copy the API address",
  "모델|th": "Model",
  "파일|k": "File",
  "정보|about": "About",
  "실행|sec": "Engine",
  "다운로드|list": "Downloads",
  "없음|empty": "None",
  "Vulkan 장치|one": "Vulkan device",
  "글": "글",
  "맨 아래로": "Scroll to the bottom",
  "메뉴": "Menu",
  "메시지": "Message",
  "목록 닫기": "Close the list",
  "상태": "Status",
  "채팅 목록": "Chats",
  "채팅 목록 닫기": "Close the chat list",
  "채팅 목록 열기": "Open the chat list",
  "채팅은 이 브라우저에만 저장됩니다": "Chats are stored only in this browser",
  "Vulkan {}{}드라이버 {}{}공유 메모리 {} KB{}텐서 코어 {}": "Vulkan {}{}driver {}{}shared memory {} KB{}tensor cores {}",
  "그래픽 메모리가 부족합니다": "Not enough graphics memory",
  "다른 작업이 실행 중입니다": "Another job is running",
  "모델이 아직 준비되지 않았습니다": "The model is not ready yet",
  "답을 받지 못했습니다": "No answer was received",
  "GPU 드라이버가 작업을 멈췄습니다": "The GPU driver stopped the work",
  "측정을 끝내지 못했습니다": "The measurement did not finish",
  "설치 {}개": "{} installed",
  "받는 중 {}개": "{} downloading",
  "여유 {}": "{} free",
  "화면에 쓰는 말입니다. 바꾸면 화면을 다시 불러옵니다.": "The language of the screens. Changing it reloads the page.",
  "자동 — 브라우저 언어": "Auto — browser language",
  "재현성 검증의 기준 모델입니다. 수학 문제 32개의 출력이 기준 PC(NVIDIA · CUDA)와 비트 단위로 같은지 확인할 수 있습니다. 답하기 전에 생각 과정(<think>)을 길게 쓰는 추론 모델입니다.": "The reference model of the reproducibility check. You can check that its outputs on 32 math problems are bit-identical to the reference PC (NVIDIA · CUDA). A reasoning model that writes a long thought process (<think>) before answering.",
  "한국어 · 영어 일반 대화 모델입니다. Q8_0 파일이라 공유 메모리가 32 KB인 GPU(AMD 등)에서도 실행됩니다. 이 파일은 CUDA(RTX 4070 Ti)에서 실행을 확인했습니다(2026-10-15).": "A general chat model for Korean and English. As a Q8_0 file it also runs on GPUs with 32 KB of shared memory (AMD etc.). This file was run and checked on CUDA (RTX 4070 Ti) (2026-10-15).",
  "4비트 k-양자화 파일이라 VRAM 8 GB급 GPU에서도 실행됩니다. k-양자화 모델은 CUDA나 공유 메모리 48 KB 이상인 Vulkan GPU에서 실행됩니다(AMD는 아직 지원하지 않음). 이 파일은 CUDA(RTX 4070 Ti)에서 실행을 확인했습니다(2026-10-15).": "A 4-bit k-quant file, so it also runs on GPUs with about 8 GB of VRAM. k-quant models run on CUDA or on Vulkan GPUs with 48 KB or more of shared memory (AMD is not supported yet). This file was run and checked on CUDA (RTX 4070 Ti) (2026-10-15).",
  "추론 모델의 4비트 k-양자화 파일입니다. CUDA나 공유 메모리 48 KB 이상인 Vulkan GPU에서 실행됩니다. 이 파일은 CUDA(RTX 4070 Ti)에서 실행을 확인했습니다(2026-10-15).": "A 4-bit k-quant file of the reasoning model. Runs on CUDA or on Vulkan GPUs with 48 KB or more of shared memory. This file was run and checked on CUDA (RTX 4070 Ti) (2026-10-15).",
  "코드 작성 · 설명에 맞춘 모델입니다. Q8_0 파일이라 공유 메모리가 32 KB인 GPU에서도 실행됩니다. 이 파일은 CUDA(RTX 4070 Ti)에서 실행을 확인했습니다(2026-10-15).": "A model tuned for writing and explaining code. As a Q8_0 file it also runs on GPUs with 32 KB of shared memory. This file was run and checked on CUDA (RTX 4070 Ti) (2026-10-15).",
  "14B 모델입니다. 문맥 길이는 엔진이 그래픽 메모리에 맞춰 자동으로 줄입니다. CUDA나 공유 메모리 48 KB 이상인 Vulkan GPU에서 실행됩니다. 이 파일은 CUDA(RTX 4070 Ti)에서 실행을 확인했습니다(2026-10-15).": "A 14B model. The engine reduces the context length automatically to fit the graphics memory. Runs on CUDA or on Vulkan GPUs with 48 KB or more of shared memory. This file was run and checked on CUDA (RTX 4070 Ti) (2026-10-15).",
  "14B 추론 모델입니다. 문맥 길이는 엔진이 그래픽 메모리에 맞춰 자동으로 줄입니다. CUDA나 공유 메모리 48 KB 이상인 Vulkan GPU에서 실행됩니다. 이 파일은 CUDA(RTX 4070 Ti)에서 실행을 확인했습니다(2026-10-15).": "A 14B reasoning model. The engine reduces the context length automatically to fit the graphics memory. Runs on CUDA or on Vulkan GPUs with 48 KB or more of shared memory. This file was run and checked on CUDA (RTX 4070 Ti) (2026-10-15).",
  "Qwen3 대화 모델입니다. 답하기 전에 생각 과정을 쓰며, 생성 설정의 '생각 과정 쓰기'를 끄면 바로 답합니다. Q8_0 파일이라 공유 메모리가 32 KB인 GPU(AMD 등)에서도 실행되지만 그래픽 메모리가 16 GB 이상이어야 합니다. 이 파일은 CUDA(RTX 4070 Ti)와 Vulkan에서 실행을 확인했습니다(2026-10-15).": "A Qwen3 chat model. It writes a thought process before answering; turn off 'Thinking' in the generation settings to get the answer directly. As a Q8_0 file it also runs on GPUs with 32 KB of shared memory (AMD etc.), but it needs 16 GB or more of graphics memory. This file was run and checked on CUDA (RTX 4070 Ti) and Vulkan (2026-10-15).",
  "Qwen3-8B의 작은 파일입니다. 답하기 전에 생각 과정을 쓰며, 생성 설정에서 끌 수 있습니다. CUDA나 공유 메모리 48 KB 이상인 Vulkan GPU에서 실행됩니다. 이 파일은 CUDA(RTX 4070 Ti)와 Vulkan에서 실행을 확인했습니다(2026-10-15).": "A smaller file of Qwen3-8B. It writes a thought process before answering, which you can turn off in the generation settings. Runs on CUDA or on Vulkan GPUs with 48 KB or more of shared memory. This file was run and checked on CUDA (RTX 4070 Ti) and Vulkan (2026-10-15).",
  "작은 Qwen3 모델입니다(1.8 GB). 그래픽 메모리가 적은 PC에서도 실행됩니다. 생각 과정을 쓰며, 생성 설정에서 끌 수 있습니다. 이 파일은 CUDA(RTX 4070 Ti) · Vulkan · 공유 메모리 32 KB 장치의 한도를 흉내 낸 Vulkan에서 실행을 확인했습니다(2026-10-15).": "A small Qwen3 model (1.8 GB). It runs even on PCs with little graphics memory. It writes a thought process, which you can turn off in the generation settings. This file was run and checked on CUDA (RTX 4070 Ti), Vulkan, and Vulkan emulating the limits of a 32 KB shared-memory device (2026-10-15).",
  "14B Qwen3 모델입니다. 이 모양(너비 5120 · 헤드 40/8 · FFN 17408)은 같은 모양의 합성 모델로 출력이 비트 단위로 맞는지 확인했지만, 이 파일로는 아직 실행해 보지 않았습니다. 문맥 길이는 엔진이 그래픽 메모리에 맞춰 자동으로 줄입니다.": "A 14B Qwen3 model. This shape (hidden 5120 · heads 40/8 · FFN 17408) was checked for bit-exact output with a synthetic model of the same shape, but this file has not been run yet. The engine reduces the context length automatically to fit the graphics memory.",
  "Meta의 Llama 3.1 대화 모델(4비트 k-양자화)입니다. 공식 지원 언어(영어 등 8개)에 한국어는 들지 않습니다. Llama 3.1 커뮤니티 라이선스를 따릅니다 — 사용 조건과 이용 정책(Acceptable Use Policy)이 있고, 월 사용자 7억 명이 넘는 서비스는 Meta의 허가가 필요합니다. CUDA나 공유 메모리 48 KB 이상인 Vulkan GPU에서 실행됩니다. 이 파일은 CUDA(RTX 4070 Ti)와 Vulkan에서 실행을 확인했습니다(2026-10-16).": "Meta's Llama 3.1 chat model (4-bit k-quant). Korean is not among its officially supported languages (English and seven others). It follows the Llama 3.1 Community License — there are usage conditions and an Acceptable Use Policy, and services with more than 700 million monthly users need permission from Meta. Runs on CUDA or on Vulkan GPUs with 48 KB or more of shared memory. This file was checked to run on CUDA (RTX 4070 Ti) and Vulkan (2026-10-16).",
  "Llama 3.1 대화 모델의 Q8_0 파일입니다. Llama 3.1 커뮤니티 라이선스(사용 조건 · 이용 정책)를 따릅니다. Q8_0 파일이라 공유 메모리가 32 KB인 GPU(AMD 등)에서도 실행됩니다. 이 파일은 CUDA(RTX 4070 Ti)와 Vulkan에서 실행을 확인했고, 같은 글의 모든 토큰에서 다음 토큰 확률이 llama.cpp와 거의 같음(평균 KL 0.0004 이하)을 확인했습니다(2026-10-16).": "The Q8_0 file of the Llama 3.1 chat model. It follows the Llama 3.1 Community License (usage conditions · Acceptable Use Policy). As a Q8_0 file it also runs on GPUs with 32 KB of shared memory (AMD and others). This file was checked to run on CUDA (RTX 4070 Ti) and Vulkan, and its next-token probabilities on every token of the same text were checked to be nearly the same as llama.cpp's (mean KL 0.0004 or less) (2026-10-16).",
  "작은 Llama 3.2 대화 모델입니다(3.4 GB). 그래픽 메모리가 적은 PC에서도 실행됩니다. 공식 지원 언어(영어 등 8개)에 한국어는 들지 않습니다. Llama 3.2 커뮤니티 라이선스(사용 조건 · 이용 정책)를 따릅니다. 이 파일은 CUDA(RTX 4070 Ti) · Vulkan · 공유 메모리 32 KB 장치의 한도를 흉내 낸 Vulkan에서 실행을 확인했습니다(2026-10-16).": "A small Llama 3.2 chat model (3.4 GB). It runs on PCs with little graphics memory too. Korean is not among its officially supported languages (English and seven others). It follows the Llama 3.2 Community License (usage conditions · Acceptable Use Policy). This file was checked to run on CUDA (RTX 4070 Ti) · Vulkan · Vulkan emulating the limits of a device with 32 KB of shared memory (2026-10-16).",
  "Mistral AI의 Mistral 7B Instruct v0.3(4비트 k-양자화)입니다. 채팅 틀에 시스템 역할이 없어 시스템 프롬프트는 첫 질문 앞에 붙여 보냅니다. CUDA나 공유 메모리 48 KB 이상인 Vulkan GPU에서 실행됩니다. 이 파일은 CUDA(RTX 4070 Ti)와 Vulkan에서 실행을 확인했습니다(2026-10-16).": "Mistral AI's Mistral 7B Instruct v0.3 (4-bit k-quant). Its chat template has no system role, so the system prompt is put in front of the first question. Runs on CUDA or on Vulkan GPUs with 48 KB or more of shared memory. This file was checked to run on CUDA (RTX 4070 Ti) and Vulkan (2026-10-16).",
  "Mistral 7B Instruct v0.3의 Q8_0 파일입니다. 시스템 프롬프트는 첫 질문 앞에 붙여 보냅니다. Q8_0 파일이라 공유 메모리가 32 KB인 GPU(AMD 등)에서도 실행됩니다. 이 파일은 CUDA(RTX 4070 Ti)와 Vulkan에서 실행을 확인했고, 같은 글의 모든 토큰에서 다음 토큰 확률이 llama.cpp와 거의 같음을 확인했습니다(2026-10-16).": "The Q8_0 file of Mistral 7B Instruct v0.3. The system prompt is put in front of the first question. As a Q8_0 file it also runs on GPUs with 32 KB of shared memory (AMD and others). This file was checked to run on CUDA (RTX 4070 Ti) and Vulkan, and its next-token probabilities on every token of the same text were checked to be nearly the same as llama.cpp's (2026-10-16).",
  "Hugging Face에서 고른 파일 — 받기 전에 GGUF 머리로 GeulPU가 받는 모양인지 확인했습니다.": "A file picked from Hugging Face — its GGUF header was checked before downloading to confirm GeulPU accepts its shape.",
  // 38단계 — 연구 패널
  "연구 패널의 셈이 끝나면 보낼 수 있습니다": "You can send once the research panel finishes computing",
  "연구 패널에서 보기 — 토큰마다 확률 · 개입 · 비교 · 재현": "Open in the research panel — per-token probabilities · intervene · compare · reproduce",
  "이 답을 연구 패널에서 보기": "Open this answer in the research panel",
  "숨은 버전": "hidden version",
  "답이 흐르는 동안에는 모델을 바꿔 불러올 수 없습니다 — 끝난 뒤에 다시 누르세요": "Can't switch models while an answer is streaming — try again when it finishes",
  "벤치마크가 끝나면 셈할 수 있습니다": "You can compute once the benchmark finishes",
  "이 답을 만든 모델이 설치되어 있지 않습니다": "The model that wrote this answer is not installed",
  "이 답이 있던 채팅을 찾지 못했습니다": "Couldn't find the chat this answer came from",
  "숨은 버전의 답입니다 — 채팅에서 그 버전을 연 뒤에 셈하세요": "This answer is in a hidden version — open that version in the chat first",
  "이 답을 만든 요청을 똑같이 다시 지을 수 없습니다 — 앞 대화가 바뀌었습니다": "Can't rebuild the exact request behind this answer — the earlier conversation changed",
  "이 답의 연구 기록이 없습니다": "No research record for this answer",
  "프롬프트를 다시 짓는 중": "Rebuilding the prompt",
  "토큰마다 확률 셈하기": "Compute per-token probabilities",
  "연구 패널": "Research panel",
  "연구 패널 닫기": "Close the research panel",
  "관찰": "Observe",
  "개입": "Intervene",
  "비교": "Compare",
  "재현": "Reproduce",
  "내보내기": "Export",
  "탐욕(온도 0)": "Greedy (temperature 0)",
  "분석할 답이 없습니다": "No answers to analyze",
  "이 판에서 받은 답부터 연구 기록이 남습니다. 질문을 보내 답을 받은 뒤 답 아래의 플라스크 단추를 누르세요.": "Research records start with answers received in this version of the app. Send a question, then press the flask button under the answer.",
  "분석할 답": "Answer to analyze",
  "거기서 갈래 — 이 토큰을 고른 뒤 같은 설정으로 이어 생성(지금 답은 버전으로 남음)": "Branch here — pick this token and continue with the same settings (the current answer stays as a version)",
  "고른 토큰": "Chosen token",
  "확률": "Probability",
  "로그확률": "log-prob",
  "순위": "Rank",
  "엔트로피": "Entropy",
  "여유": "Margin",
  "로짓 해시": "Logit hash",
  "토큰마다 확률 · 순위 · 대안 · 엔트로피를 보려면 이 답을 다시 셈합니다(점수 매기기). 생성할 때와 같은 비트의 값이고, GPU를 잠깐 씁니다.": "To see each token's probability, rank, alternatives and entropy, this answer is recomputed (scoring). The values are bit-identical to those at generation time; it uses the GPU briefly.",
  "평균 로그확률": "Mean log-prob",
  "토큰마다 로그확률의 평균(nat)": "Mean of the per-token log-probabilities (nats)",
  "퍼플렉시티": "Perplexity",
  "exp(−평균 로그확률)": "exp(−mean log-prob)",
  "평균 엔트로피": "Mean entropy",
  "걸음마다 모델 분포의 엔트로피 평균(nat)": "Mean entropy of the model's distribution per step (nats)",
  "1위와 2위의 로그확률 차가 가장 작은 걸음": "The step with the smallest gap between the top two log-probs",
  "1위와 2위의 차가 δ보다 작은 걸음 — 셈이나 설정이 조금만 달라도 다른 토큰이 될 자리": "Steps where the top-two gap is below δ — where a slightly different computation or setting could pick another token",
  "1위가 아닌 토큰": "Non-top tokens",
  "뽑기(온도 > 0)가 1위가 아닌 토큰을 고른 걸음": "Steps where sampling (temperature > 0) picked a token that wasn't ranked first",
  "확률은 모델의 분포(온도 1)입니다 — 뽑기에 쓴 온도 · top-k · top-p로 거른 분포가 아닙니다. 값은 계약 5.2로 정해져 어느 PC · GPU에서 셈해도 같습니다.": "Probabilities are the model's distribution (temperature 1), not the distribution filtered by the sampling temperature, top-k or top-p. Contract 5.2 defines the values, so they are the same on any PC or GPU.",
  "아슬아슬의 기준 δ(nat)": "Close-call threshold δ (nats)",
  "토큰을 누르면 그 걸음의 대안과 수가 보입니다. 띠에서 화살표 키로 걸음을 옮깁니다.": "Click a token to see that step's alternatives and numbers. In the strip, the arrow keys move between steps.",
  "답의 토큰": "Answer tokens",
  "아슬아슬의 기준 δ": "Close-call threshold δ",
  "앞의 아슬아슬한 걸음": "Previous close call",
  "다음 아슬아슬한 걸음": "Next close call",
  "짙을수록 확률이 낮음": "Darker = less likely",
  "밑줄 = 아슬아슬": "Underline = close call",
  "테두리 = 1위가 아닌 토큰": "Outline = non-top token",
  "이 답은 지금 보이지 않는 버전에 있습니다 — 채팅에서 그 버전을 연 뒤에 개입할 수 있습니다.": "This answer is in a version that isn't shown — open that version in the chat to intervene.",
  "거기서 갈래": "Branch here",
  "어느 걸음에서 다른 토큰을 고르고 같은 설정으로 이어 생성합니다. 뽑기(온도 > 0)면 같은 씨앗 · 같은 걸음 번호의 난수를 써서 바뀐 것은 고른 토큰 하나뿐입니다. 새 답은 그 질문의 새 버전으로 남고, 지금 답은 앞 버전으로 남습니다.": "Pick a different token at any step and continue generating with the same settings. When sampling (temperature > 0), the same seed and the same per-step random numbers are used, so the chosen token is the only change. The new answer becomes a new version of that question; the current answer stays as the previous version.",
  "관찰에서 걸음을 고르거나": "Pick a step in Observe, or",
  "설정을 바꿔 다시 생성": "Regenerate with different settings",
  "같은 질문 · 같은 대화 · 같은 모델로 온도 · 씨앗 · 최대 길이만 바꿔 새 버전을 만듭니다. 그다음 비교에서 처음 갈린 걸음을 봅니다. 씨앗을 비우면 서버가 고릅니다(기록에 남음).": "Makes a new version with the same question, conversation and model, changing only the temperature, seed and max length. Then see the first diverging step in Compare. Leave the seed empty to let the server choose one (it is recorded).",
  "씨앗": "Seed",
  "무작위": "random",
  "최대 길이": "Max length",
  "새 버전 생성": "Generate a new version",
  "질문을 바꿔": "Change the question",
  "채팅에서 질문의 연필 단추로 고쳐 보내면 새 버전이 생깁니다. 비교의 'A의 답을 B의 프롬프트로 점수'가 바꾼 곳 하나의 효과를 토큰마다 잽니다 — 같은 답 토큰의 확률이 어떻게 바뀌는가.": "Edit the question with its pencil button in the chat and send it to make a new version. 'Score A's answer under B's prompt' in Compare measures the effect of that one change token by token — how the probabilities of the same answer tokens move.",
  "갈래의 답이 끝났습니다": "The branch's answer is finished",
  "온도는 0 ~ 2 사이의 수여야 합니다": "Temperature must be a number from 0 to 2",
  "최대 길이는 1 ~ 32768 사이의 정수여야 합니다": "Max length must be an integer from 1 to 32768",
  "씨앗은 0 ~ 2⁶⁴−1 사이의 정수여야 합니다": "The seed must be an integer from 0 to 2⁶⁴−1",
  "비교하려면 먼저 이 답(A)을 셈합니다.": "To compare, first compute this answer (A).",
  "두 답 맞대기": "Two answers side by side",
  "버전 · 갈래끼리 견줍니다. 프롬프트가 같으면 처음 갈린 걸음을, 다르면(질문을 고친 버전) 한쪽 답을 다른 쪽 프롬프트 뒤에서 점수 매겨 바꾼 곳 하나의 효과를 토큰마다 잽니다.": "Compares versions and branches. With the same prompt it finds the first diverging step; with different prompts (an edited question) it scores one answer after the other prompt to measure the effect of that one change token by token.",
  "견줄 다른 답이 없습니다 — 개입에서 갈래를 내거나 설정을 바꿔 다시 생성하면 버전이 생깁니다.": "No other answer to compare — branch in Intervene or regenerate with different settings to create versions.",
  "견줄 답(B)": "Answer to compare (B)",
  "견줄 답(B) 고르기": "Choose answer B",
  "B도 셈하기": "Compute B too",
  "같은 글을 다른 모델로 점수": "Score the same text with another model",
  "이 답의 토큰을 다른 모델(예: 같은 모델의 다른 양자화)로 다시 셈합니다. 셈의 잡음이 없으므로 토큰마다의 차이는 모델 파일 하나의 효과입니다. 모델을 바꿔 불러옵니다(수십 초).": "Recomputes this answer's tokens with another model (e.g., another quantization of the same model). With no computational noise, the per-token differences are the effect of the model file alone. Switches the loaded model (tens of seconds).",
  "어휘와 채팅 틀이 같은 다른 모델이 설치되어 있지 않습니다.": "No other installed model shares this vocabulary and chat template.",
  "견줄 모델": "Model to compare",
  "모델 고르기": "Choose a model",
  "점수": "Score",
  "(끝)": "(end)",
  "A의 이 걸음 보기": "View this step of A",
  "프롬프트와 앞 토큰이 모두 같으므로 이 걸음의 분포는 두 답에서 같은 비트입니다 — 고른 토큰만 다릅니다.": "The prompt and all earlier tokens are the same, so this step's distribution is bit-identical in both answers — only the chosen token differs.",
  "두 답의 프롬프트가 다릅니다(질문 · 앞 대화 · 시스템 프롬프트 · 생각 켬끔).": "The two answers have different prompts (question, earlier conversation, system prompt or thinking on/off).",
  "A의 답을 B의 프롬프트로 점수": "Score A's answer under B's prompt",
  "B의 프롬프트에서": "under B's prompt",
  "두 답의 모델이 다릅니다 — 아래 '같은 글을 다른 모델로 점수'로 같은 토큰을 견주세요.": "The two answers come from different models — use 'Score the same text with another model' below to compare the same tokens.",
  "합 Δ로그확률": "Total Δ log-prob",
  "답 전체의 로그 확률비 — log P(그쪽) − log P(이 답)": "Log probability ratio of the whole answer — log P(other) − log P(this answer)",
  "평균 Δ로그확률": "Mean Δ log-prob",
  "1위가 바뀐 걸음": "Steps with a new top token",
  "두 조건에서 가장 그럴듯한 토큰이 다른 걸음 수": "Number of steps where the most likely token differs between the two conditions",
  "점수 매길 글을 붙여 넣으세요": "Paste the text to score",
  "점수 매길 글": "Text to score",
  "맥락": "Context",
  "고른 답의 자리에(같은 질문 · 대화 뒤 — 그 답의 모델)": "In place of the selected answer (after the same question and conversation — that answer's model)",
  "고른 답의 자리에(먼저 관찰에서 셈하세요)": "In place of the selected answer (compute it in Observe first)",
  "맥락 없이(글만 — 첫 토큰은 확률 없음)": "No context (text only — the first token has no probability)",
  "점수 매길 모델": "Model to score with",
  "글의 토큰마다 로그확률과 퍼플렉시티를 셉니다 — 모델이 이 글을 얼마나 그럴듯하게 보는가. 같은 글 · 같은 모델이면 어느 PC에서든 같은 값입니다.": "Computes each token's log-probability and the perplexity — how plausible the model finds this text. The same text with the same model gives the same values on any PC.",
  "점수 매길 글을 넣으세요": "Enter the text to score",
  "토큰이 둘 이상인 글을 넣으세요(맥락 없이는 첫 토큰의 확률이 없습니다)": "Enter text with at least two tokens (without context, the first token has no probability)",
  "고른 답의 자리에": "in place of the selected answer",
  "맥락 없이": "no context",
  "합 로그확률": "Total log-prob",
  "점수 매긴 글의 토큰": "Tokens of the scored text",
  "지문": "Fingerprints",
  "같은 모델 파일 · 같은 프롬프트 · 같은 결과면 어느 GPU · 백엔드에서 셈해도 같은 지문입니다(계약). 장치는 지문에 넣지 않고 만든 곳으로 따로 적습니다.": "The same model file, prompt and result give the same fingerprint on any GPU or backend (contract). The device isn't part of the fingerprint; it is recorded separately as where the answer was made.",
  "토큰 지문": "Token fingerprint",
  "셈하면 보입니다": "Shown after computing",
  "로짓 지문": "Logit fingerprint",
  "모델 파일": "Model file",
  "SHA-256 모름": "SHA-256 unknown",
  "만든 곳": "Made on",
  "직접 추가한 모델 파일은 SHA-256을 아직 모릅니다 — 지문 · 캡슐에는 모델 파일의 해시가 필요합니다.": "The SHA-256 of a model file you added yourself isn't known yet — fingerprints and capsules need the model file's hash.",
  "SHA-256 셈하는 중": "Computing SHA-256",
  "SHA-256 셈하기": "Compute SHA-256",
  "셈하기": "Compute",
  "다시 돌려 확인": "Re-run and check",
  "같은 요청(같은 프롬프트 토큰 · 매개변수 · 씨앗)을 캐시 없이 다시 생성하고 걸음마다 토큰과 로짓 해시를 견줍니다.": "Regenerates the same request (same prompt tokens, parameters and seed) without the cache and compares the token and the logit hash at every step.",
  "재현 캡슐": "Reproduction capsule",
  "요청 · 결과 · 로그확률 · 로짓 해시 · 만든 곳을 JSON 파일 하나로. 받은 사람이 자기 GPU에서(아래 '캡슐 불러와 확인') 또는 GPU 없이 파이썬(도구/캡슐확인.py)으로 비트까지 같은지 확인합니다.": "The request, result, log-probabilities, logit hashes and where it was made, in one JSON file. The recipient checks that it is bit-identical on their own GPU ('Load capsule and check' below) or without a GPU in Python (도구/캡슐확인.py).",
  "캡슐 내보내기": "Export capsule",
  "캡슐 파일": "Capsule file",
  "캡슐 불러와 확인": "Load capsule and check",
  "해시를 셈하는 중": "Computing hashes",
  "엔진 · 커널 해시 보기": "Show engine and kernel hashes",
  "같은 요청을 캐시 없이 다시 생성하는 중": "Regenerating the same request without the cache",
  "JSON 파일이 아닙니다": "Not a JSON file",
  "GeulPU 재현 캡슐이 아닙니다": "Not a GeulPU reproduction capsule",
  "캡슐의 토큰을 이 PC에서 다시 셈하는 중": "Recomputing the capsule's tokens on this PC",
  "캡슐의 요청으로 다시 생성하는 중": "Regenerating from the capsule's request",
  "다시 생성한 토큰 · 로짓 해시가 다름": "regenerated tokens or logit hashes differ",
  "지문이 다름": "fingerprints differ",
  "기준과 같음": "Matches reference",
  "기준과 다름": "Differs from reference",
  "기준 없음": "No reference",
  "엔진 바이너리의 SHA-256을 재현성 검증의 기준 빌드와 견줍니다. 직접 빌드한 판은 다를 수 있습니다.": "Compares the SHA-256 of the engine binaries with the reference build of the reproducibility check. A build you made yourself may differ.",
  "커널 모듈": "Kernel module",
  "커밋": "commit",
  "계약": "contract",
  "기준 PC · 하드웨어": "Reference PC · hardware",
  "재현성 검증을 아직 하지 않았습니다 — 이 PC의 출력이 기준 PC와 비트까지 같은지 벤치마크에서 확인합니다.": "No reproducibility check yet — the benchmark checks whether this PC's output is bit-identical to the reference PC's.",
  "이 모델의 하드웨어 진단이 없습니다 — 생성이 하드웨어 한계에 얼마나 가까운지 벤치마크에서 잽니다.": "No hardware diagnosis for this model — the benchmark measures how close generation is to the hardware limit.",
  "벤치마크로": "Go to Benchmark",
  "이 답의 토큰 표": "This answer's token table",
  "요청 · 결과 · 만든 곳 — 재현에서 확인하는 파일": "Request, result and where it was made — the file checked in Reproduce",
  "같은 토큰의 두 모델 로그확률과 차(Δ = 그 모델 − 이 답)": "Both models' log-probs for the same tokens and their difference (Δ = that model − this answer)",
  "같은 답 토큰의 두 프롬프트 로그확률과 차(Δ = B의 프롬프트 − A)": "Log-probs of the same answer tokens under both prompts and their difference (Δ = B's prompt − A)",
  "이 답을 셈하면 토큰 표 · 캡슐을 내보낼 수 있습니다.": "Compute this answer to export its token table and capsule.",
  "점수 매긴 글": "Scored text",
  "앱에서 모델을 불러온 상태에서": "with a model loaded in the app",
  "이 패널의 값과 비트까지 같다": "bit-identical to this panel's values",
  "하늘은 왜 파란가?": "Why is the sky blue?",
  "파이썬": "Python",
  "패널의 값은 모두 OpenAI 꼴 API로도 받습니다 — 어느 PC · GPU에서든 같은 비트입니다.": "Every value in the panel is also available through the OpenAI-style API — bit-identical on any PC or GPU.",
  "이 답을 점수 매기기(echo)": "Score this answer (echo)",
  "생성하며 로그확률 받기": "Get log-probs while generating",
  "버전 {}/{}": "version {}/{}",
  "질문 {}": "Question {}",
  "갈래 — 걸음 {}": "branch — step {}",
  "토큰 {}개를 다시 셈하는 중(점수 매기기)": "Recomputing {0} {0|token|tokens} (scoring)",
  "온도 {} · 씨앗 {}": "temperature {} · seed {}",
  "토큰 {}개": "{0} {0|token|tokens}",
  "{}토큰/초": "{} tokens/s",
  "갈래 — 걸음 {}부터": "branch from step {}",
  "걸음 {} · 확률 {} · 여유 {}": "step {} · probability {} · margin {}",
  "이 토큰으로 갈래: {}": "Branch with this token: {}",
  "걸음 {}": "Step {}",
  "{}위 밖": "outside the top {}",
  "{}위": "#{}",
  "대안 상위 {}개": "Top {} alternatives",
  "이 답의 모델({})을 불러옵니다.": "This loads the answer's model ({}).",
  "가장 작은 여유 (걸음 {})": "Smallest margin (step {})",
  "아슬아슬 (여유 < {})": "Close calls (margin < {})",
  "가장 아슬아슬한 걸음 {} 고르기": "Pick the closest call (step {})",
  "{}에서": "with {}",
  "프롬프트가 같고 두 답의 토큰 {}개가 모두 같습니다.": "Same prompt, and all {0} {0|token is|tokens are} the same in both answers.",
  "확률 {} · {}위": "probability {} · rank {}",
  "처음 갈린 걸음 {}": "First diverging step: {}",
  "토큰마다 Δ로그확률 — {}": "Per-token Δ log-prob — {}",
  "걸음 {} · Δ로그확률 {}": "step {} · Δ log-prob {}",
  "같은 토큰 {}개 — {}": "Same {0} {0|token|tokens} — {1}",
  "가장 큰 |Δ| (걸음 {})": "Largest |Δ| (step {})",
  "초록 = 그쪽에서 더 높은 확률, 분홍 = 더 낮은 확률, 짙을수록 큰 차이. 차이 0(비트까지 같음) {}걸음.": "Green = more likely there, pink = less likely; darker = bigger difference. Zero difference (bit-identical): {0} {0|step|steps}.",
  "토큰 {}개를 셈하는 중": "Computing {0} {0|token|tokens}",
  "가장 낮은 확률 (걸음 {})": "Lowest probability (step {})",
  "{}걸음 모두 같음 — 토큰 · 로짓 해시(슬롯 {} · 캐시 없이)": "All {0} {0|step matches|steps match} — tokens and logit hashes (slot {1}, no cache)",
  "걸음 {}에서 다름 — 같은 토큰 {}/{} · 같은 로짓 해시 {}/{}": "Differs at step {} — matching tokens {}/{} · matching logit hashes {}/{}",
  "크기가 같은 모델({})이 있지만 SHA-256을 모릅니다 — 모델 화면에서 무결성을 확인한 뒤 다시 불러오세요": "A model of the same size ({}) is installed, but its SHA-256 isn't known — verify its integrity on the Models screen, then load the capsule again",
  "같은 모델 파일(SHA-256 {}…)이 설치되어 있지 않습니다 — {} {}": "The same model file (SHA-256 {}…) is not installed — {} {}",
  " (캡슐에 적힌 만든 곳 {} · {} → 여기 {})": " (made on {} · {} as the capsule states → here {})",
  "캡슐의 씨앗(seed)을 정수로 읽지 못했습니다 — 정밀도를 잃은 사본일 수 있습니다": "Couldn't read the capsule's seed as an integer — this may be a copy that lost precision",
  "점수는 같으니 씨앗을 확인하세요 — 다른 도구로 다시 쓴 캡슐은 씨앗(64비트 정수)이 자리를 잃었을 수 있습니다": "scoring matches, so check the seed — a capsule rewritten by another tool may have lost digits of the seed (a 64-bit integer)",
  "비트까지 같음 — {}걸음의 로그확률 · 엔트로피 · 상위 {} · 로짓 해시, 다시 생성한 토큰, 지문": "Bit-identical — log-probs, entropy, top {1} and logit hashes for {0} {0|step|steps}, the regenerated tokens and the fingerprints",
  "점수 매기기가 걸음 {}에서 다름": "scoring differs at step {}",
  "재현성 검증 {}: {} — {}": "Reproducibility check {}: {} — {}",
  "하드웨어 진단 {}: {}": "Hardware diagnosis {}: {}",
  "걸음마다 토큰 · 로그확률 · 확률 · 순위 · 엔트로피 · 여유 · 로짓 해시 · 상위 {}": "Per step: token, log-prob, probability, rank, entropy, margin, logit hash, top {}",
  "다른 모델로 점수 — {}": "Scored with another model — {}",
  "UTF-8 바이트 조각인 토큰입니다 — 이웃 토큰과 합쳐 한 글자가 됩니다.": "A UTF-8 byte-fragment token — it forms one character together with its neighbors.",
  "토큰 {}개(끝 토큰 포함)": "{0} {0|token|tokens} (including the end token)",
};
function T(s, ...v) {
  if (typeof s === 'string') {
    if (!EN) return s;
    if (v.length && typeof v[0] === 'string') { const c = DICT[s + '|' + v[0]]; if (c !== undefined) return c; }
    const e = DICT[s];
    if (e === undefined) { MISSING.add(s); return s; }
    return e;
  }
  const ko = () => { let o = s[0]; for (let i = 0; i < v.length; i++) o += v[i] + s[i + 1]; return o; };
  if (!EN) return ko();
  const key = s.join('{}');
  const e = DICT[key];
  if (e === undefined) { MISSING.add(key); return ko(); }
  // {} 는 차례대로, {0} 은 자리 번호로. {0|one|many} 는 0번 값이 1이면 one, 아니면 many(영어의 단수 · 복수)
  let k = 0;
  return e.replace(/\{(\d*)(?:\|([^|}]*)\|([^}]*))?\}/g, (_, d, one, many) => {
    const x = String(v[d === '' ? k++ : +d]);
    return one === undefined ? x : x === '1' ? one : many;
  });
}
// 관리자 · 서버 문구 → 영어(사전에 통째로 있으면 그것, 아니면 아래 바꾸기를 차례로). 영어 화면이 아니면 그대로
const TMX = [
  // 작업 제목
  [/모델 불러오기 — /g, 'Loading a model — '],
  [/하드웨어 진단 — /g, 'Hardware diagnosis — '],
  [/속도 측정 — /g, 'Speed test — '],
  [/재현성 검증\(빠른 확인\) — 문제 2개 · 64토큰/g, 'Reproducibility check (quick) — 2 problems · 64 tokens'],
  [/재현성 검증 — 문제 32개/g, 'Reproducibility check — 32 problems'],
  // 결과 요약(교차검증)
  [/결과: 기준 PC와 다른 출력이 있거나 끝나지 못한 실행이 있습니다 — /g, 'Result: some outputs differ from the reference PC, or some runs did not finish — '],
  [/결과: 모든 문제가 기준 PC\(NVIDIA · CUDA\)와 똑같은 출력입니다 — /g, 'Result: every problem gives exactly the same output as the reference PC (NVIDIA · CUDA) — '],
  [/결과\(빠른 확인 — 문제 2개 · 64토큰\): 기준 PC의 출력과 앞부분이 같습니다 — /g, 'Result (quick check — 2 problems · 64 tokens): the start matches the reference PC — '],
  [/(\S+) 추론 서버가 시작되지 않음/g, '$1: the inference server did not start'],
  [/\[([^\]]+)\] (\d+)\/(\d+) 문제가 기준 PC와 같은 출력 · 슬롯 (\d+) · 생성 ([\d.]+)분/g, '[$1] $2/$3 problems match the reference PC · $4 slots · generation $5 min'],
  [/협력 행렬 모듈 (\S+)/g, 'cooperative-matrix module $1'],
  [/기본 모듈/g, 'basic module'],
  [/CUDA 모듈/g, 'CUDA module'],
  // 작업 단계 · 기록(일 · 벤치)
  [/커널 모듈 준비 중\(처음 한 번만 — 이 GPU에 맞춰 만듭니다\)/g, 'Preparing kernel modules (first time only — built for this GPU)'],
  [/커널 모듈 준비 중\(협력 행렬\)/g, 'Preparing kernel modules (cooperative matrix)'],
  [/커널 모듈 준비 중\(기본\)/g, 'Preparing kernel modules (basic)'],
  [/커널 소스를 만듭니다: (\S+) \(생성기 — 모양 ([^,]+), ([^)]+)\)/g, 'Generating kernel source: $1 (generator — shape $2, $3)'],
  [/커널을 컴파일합니다: (\S+) → (\S+)\(글\(geul\) 컴파일러의 (\S+) 백엔드, 1~3분\)/g, "Compiling kernels: $1 → $2 (the geul compiler's $3 backend, 1–3 min)"],
  [/생성기 실패\(종료 코드 (-?\d+)\): /g, 'Generator failed (exit code $1): '],
  [/백엔드 실패\(종료 코드 (-?\d+)\): /g, 'Backend failed (exit code $1): '],
  [/모듈 완성: (\S+) \(([\d.]+) MB, SHA-256 ([0-9a-f]+)…\)/g, 'Module ready: $1 ($2 MB, SHA-256 $3…)'],
  [/모듈이 이미 있습니다: /g, 'Module already exists: '],
  [/모델을 불러오는 중에 추론 서버가 종료되었습니다\(종료 코드 (-?\d+)\)\. ?/g, 'The inference server exited while loading the model (exit code $1). '],
  [/모델을 불러오는 중\(([^·()]+) · ([^·()]+) · 슬롯 (\d+) · 문맥 (\d+)\)/g, 'Loading the model ($1 · $2 · $3 slots · context $4)'],
  [/모델 준비 완료\(([\d.]+)초, 슬롯당 문맥 (\d+)\)\./g, 'Model ready ($1 s, context $2 per slot).'],
  [/30분이 지나도 모델이 준비되지 않았습니다\./g, 'The model was not ready after 30 minutes.'],
  [/이 모델은 이미 불러와 있습니다\./g, 'This model is already loaded.'],
  [/추론 서버를 실행하지 못했습니다: /g, 'Could not run the inference server: '],
  [/추론 서버가 종료되었습니다\(종료 코드 (-?\d+)\)([.:]) ?/g, 'The inference server exited (exit code $1)$2 '],
  [/실행하지 못했습니다: /g, 'Could not run: '],
  [/작업 스레드를 시작하지 못했습니다/g, 'Could not start the job thread'],
  [/^실패: /gm, 'Failed: '],
  [/^중지했습니다\.$/gm, 'Stopped.'],
  [/Vulkan 장치를 읽지 못했습니다\(종료 코드 (-?\d+)\) — 그래픽 드라이버가 Vulkan을 지원하는지 확인하세요/g, 'Could not read the Vulkan devices (exit code $1) — check that the graphics driver supports Vulkan'],
  [/Vulkan 장치 확인 중/g, 'Checking Vulkan devices'],
  [/CUDA 장치 확인 중/g, 'Checking CUDA devices'],
  [/CUDA를 쓸 수 없습니다\(NVIDIA 드라이버의 nvcuda\.dll이 없거나 열리지 않음\) — Vulkan으로 실행합니다/g, "CUDA is not available (the NVIDIA driver's nvcuda.dll is missing or won't open) — running on Vulkan"],
  [/사용할 GPU: (.+?) · (\S+) · 메모리 ([\d.]+) GB · 커널 변형 (\S*) · 협력 행렬 ?(\S*)/g, 'GPU in use: $1 · $2 · memory $3 GB · kernel variant $4 · cooperative matrix $5'],
  [/kernel variant 보통/g, 'kernel variant standard'],
  [/kernel variant 공유32/g, 'kernel variant shared32'],
  // Vulkan 장치를 막는 까닭(장치 확인 도구)
  [/서브그룹 크기 32 를 고를 수 없다\((\d+) … (\d+)\)/g, 'Cannot select subgroup size 32 ($1 … $2)'],
  [/없는 것: /g, 'Missing: '],
  [/공유 메모리 (\d+) 바이트 — 32 KB 가 필요하다/g, 'Shared memory is $1 bytes — 32 KB is required'],
  [/푸시 상수 (\d+) 바이트 — 128 바이트가 필요하다/g, 'Push constants are $1 bytes — 128 bytes are required'],
  [/한 번 할당의 한도 (\d+) 바이트 — 가중치 한 층\(256 MiB\)도 잡지 못한다/g, 'The single-allocation limit is $1 bytes — not even one weight layer (256 MiB) fits'],
  [/Vulkan 인스턴스를 만들지 못했다\(VkResult (-?\d+)\) — Vulkan 1\.3 드라이버가 필요하다/g, 'Could not create a Vulkan instance (VkResult $1) — a Vulkan 1.3 driver is required'],
  [/GEULPU_VULKAN=(.*?) 인 장치가 없다/g, 'No device matches GEULPU_VULKAN=$1'],
  [/Vulkan 장치가 없다/g, 'No Vulkan device'],
  [/CUDA ?를 열지 못했습니다/g, 'Could not open CUDA'],
  [/\s?\(nvcuda\.dll 없음\)/g, ' (nvcuda.dll missing)'],
  // 추론 서버 · 엔진의 오류(오류 카드의 자세히 · 상태 막대의 도움말)
  [/프롬프트가 슬롯의 문맥보다 길다/g, "The prompt is longer than the slot's context"],
  [/프롬프트가 비었다/g, 'The prompt is empty'],
  [/줄이 가득 찼다 — 잠시 뒤에/g, 'The queue is full — try again shortly'],
  [/장치 메모리에 맞춰 문맥을 (\d+) → (\d+) 로 줄였다\(남은 ([\d.]+) GB\)/g, 'Reduced the context from $1 to $2 to fit the device memory ($3 GB left)'],
  [/장치 메모리 (\d+) 바이트가 장치\((.+?)\)의 한 번 할당 한도 (\d+) 바이트를 넘는다/g, 'Device memory of $1 bytes exceeds the single-allocation limit of $3 bytes on $2'],
  [/장치 메모리가 모자라다/g, 'Not enough device memory'],
  [/모델을 올리는 중…/g, 'Loading the model…'],
  [/^오류: /gm, 'Error: '],
  [/(.+?)의 Compute Capability는 (\d+)\.(\d+)입니다 — GeulPU의 CUDA 모듈은 8\.0\(RTX 30 시리즈\) 이상이 필요합니다/g, "$1 has Compute Capability $2.$3 — GeulPU's CUDA modules need 8.0 (RTX 30 series) or newer"],
  [/실행할 수 있는 GPU가 없습니다(:| —) /g, 'No GPU can run GeulPU$1 '],
  [/nvcuda\.dll이 없습니다/g, 'nvcuda.dll is missing'],
  [/CUDA 장치가 없습니다/g, 'No CUDA device'],
  [/CUDA를 열지 못했습니다/g, 'Could not open CUDA'],
  [/Vulkan 장치를 읽지 못했습니다/g, 'Could not read the Vulkan devices'],
  [/GeulPU 요구 사항을 만족하는 Vulkan 장치가 없습니다/g, "No Vulkan device meets GeulPU's requirements"],
  [/GPU를 비우는 중\(채팅 모델을 내립니다\)/g, 'Freeing the GPU (unloading the chat model)'],
  [/하드웨어 한계\(메모리 대역폭 · 정수 텐서\)와 연산별 한계 대비를 측정하는 중\(GPU에 따라 1~6분\)/g, 'Measuring the hardware limits (memory bandwidth · integer tensor) and each operation against them (1–6 min depending on the GPU)'],
  [/하드웨어 진단 중\(GPU에 따라 1~6분\)/g, 'Running the hardware diagnosis (1–6 min depending on the GPU)'],
  [/하드웨어 진단에 실패했습니다/g, 'The hardware diagnosis failed'],
  [/하드웨어 진단: (\S+)\(모듈 ([^)]+)\)/g, 'Hardware diagnosis: $1 (module $2)'],
  [/진단 실패\(종료 코드 (-?\d+)\): /g, 'Diagnosis failed (exit code $1): '],
  [/추론 서버 응답 HTTP (\d+): /g, 'Inference server response HTTP $1: '],
  [/예열 중/g, 'Warming up'],
  [/측정 중: /g, 'Measuring: '],
  [/측정된 항목이 없습니다 — 실행 기록을 확인하세요/g, 'Nothing was measured — see the run log'],
  [/모델을 불러오지 못했습니다 — 실행 기록을 확인하세요/g, 'Could not load the model — see the run log'],
  [/\[([^\]]+)\] 추론 서버를 시작합니다\(슬롯 (\d+) — 처음에는 커널 컴파일과 실행 방식 측정으로 몇 분 걸릴 수 있습니다\)/g, '[$1] Starting the inference server ($2 slots — the first time can take a few minutes to compile kernels and measure)'],
  [/\[([^\]]+)\] 추론 서버 준비 완료\(슬롯 (\d+), ([\d.]+)초\)\./g, '[$1] Inference server ready ($2 slots, $3 s).'],
  [/\[([^\]]+)\] 슬롯 (\d+)개로는 추론 서버가 시작되지 않았습니다: /g, '[$1] The inference server did not start with $2 slots: '],
  [/추론 서버를 시작하지 못했습니다\(logs 폴더의 server-(\S+)-\*\.log 참고\)/g, 'Could not start the inference server (see server-$1-*.log in the logs folder)'],
  [/\[([^\]]+)\] 문제 (\d+)개를 보냅니다\(최대 답변 길이 (\d+)토큰\)\./g, '[$1] Sending $2 problems (max answer length $3 tokens).'],
  [/생성 도중 추론 서버가 종료되었습니다\(logs 폴더의 server-(\S+)-\*\.log 참고\)/g, 'The inference server exited during generation (see server-$1-*.log in the logs folder)'],
  [/생성 중 — (.+?)\(GPU에 따라 5~20분\)/g, 'Generating — $1 (5–20 min depending on the GPU)'],
  [/재현성 검증 기준 데이터\(data\\crosscheck\)를 읽지 못했습니다/g, 'Could not read the reproducibility reference data (data\\crosscheck)'],
  [/재현성 검증에 필요한 모델\(DeepSeek-R1-Distill-Qwen-7B Q8_0, 8\.1 GB\)이 없습니다 — 모델 화면에서 다운로드하거나 파일을 추가하세요/g, 'The model needed for the reproducibility check (DeepSeek-R1-Distill-Qwen-7B Q8_0, 8.1 GB) is missing — download it or add the file on the Models screen'],
  [/모델 파일 무결성 확인 중\(SHA-256\)/g, "Verifying the model file's integrity (SHA-256)"],
  [/모델 파일의 SHA-256이 기준과 다릅니다\(([0-9a-f]+)…\) — 파일을 지우고 다시 다운로드하세요/g, "The model file's SHA-256 differs from the reference ($1…) — delete the file and download it again"],
  [/모델 파일을 읽지 못했습니다/g, 'Could not read the model file'],
  [/모델 확인: SHA-256이 기준과 같습니다\./g, 'Model check: SHA-256 matches the reference.'],
  [/VRAM이 (\d+) MiB라 모델\(8\.1 GB\)이 들어가지 않습니다 — 12 GB 이상이 필요합니다/g, "The model (8.1 GB) doesn't fit in $1 MiB of VRAM — 12 GB or more is needed"],
  [/빌드 해시를 기준 PC와 비교하는 중/g, 'Comparing build hashes with the reference PC'],
  [/커널 모듈을 만들지 못했습니다/g, 'Could not build the kernel modules'],
  [/ {2}\[통과\] (.+?) 소스: 기준과 같음 /g, '  [pass] $1 source: same as the reference '],
  [/ {2}\[경고\] (.+?) 소스: /g, '  [warning] $1 source: '],
  [/ {2}\[통과\] (.+?): 기준과 같음 /g, '  [pass] $1: same as the reference '],
  [/ {2}\[경고\] (.+?): 기준과 다름 (\S+) \(기준 /g, '  [warning] $1: differs from the reference $2 (reference '],
  [/ {2}\[경고\] /g, '  [warning] '],
  [/ {2}\[정보\] (.+?): 기준 해시 없음 /g, '  [info] $1: no reference hash '],
  [/ {2}\[없음\] /g, '  [missing] '],
  [/\(기준 없음\)/g, '(no reference)'],
  // 속도 · 진단 요약
  [/메모리 대역폭 ([\d.]+) GB\/s · 정수 텐서 ([\d.]+) TOPS/g, 'memory bandwidth $1 GB/s · integer tensor $2 TOPS'],
  [/생성\(문맥 1024\) 한계의 (\d+)%/g, 'generation (context 1024) $1% of the limit'],
  [/슬롯 8개 생성\(문맥 2048\) 한계의 (\d+)%/g, '8-slot generation (context 2048) $1% of the limit'],
  [/프롬프트 2048토큰 한계의 (\d+)%/g, 'prompt 2048 tokens $1% of the limit'],
  [/프롬프트 ([\d.]+) 토큰\/초\(512토큰\)/g, 'prompt $1 tokens/s (512 tokens)'],
  [/생성 ([\d.]+) 토큰\/초/g, 'generation $1 tokens/s'],
  [/: 토큰당 ([\d.]+) ms · ([\d.]+) 토큰\/초/g, ': $1 ms per token · $2 tokens/s'],
  [/: ([\d.]+) ms · ([\d.]+) 토큰\/초/g, ': $1 ms · $2 tokens/s'],
  [/프롬프트 512토큰/g, 'prompt 512 tokens'],
  [/프롬프트 2048토큰/g, 'prompt 2048 tokens'],
  [/생성 128토큰\(짧은 문맥\)/g, 'generation 128 tokens (short context)'],
  [/생성 64토큰\(문맥 1024 이후\)/g, 'generation 64 tokens (after context 1024)'],
  // 모델 파일(GGUF 머리) 까닭
  [/여러 조각으로 나뉜 GGUF의 일부입니다\((\d+)조각\) — 하나로 합친 파일만 지원합니다\(llama-gguf-split --merge\)/g, 'Part of a GGUF split into $1 parts — only merged single files are supported (llama-gguf-split --merge)'],
  [/지원하지 않는 모델 구조입니다\((.+?)\) — 지금은 qwen2 구조\(Qwen2\.5 · DeepSeek-R1-Distill-Qwen\), qwen3 구조\(Qwen3\), llama 구조\(Llama 3\.1 · 3\.2 · Mistral\)만 지원합니다/g, 'Unsupported model architecture ($1) — only the qwen2 (Qwen2.5 · DeepSeek-R1-Distill-Qwen), qwen3 (Qwen3) and llama (Llama 3.1 · 3.2 · Mistral) architectures are supported for now'],
  [/지원하지 않는 헤드 크기입니다\((\d+) · (\d+)\) — 128만 지원합니다/g, 'Unsupported head size ($1 · $2) — only 128 is supported'],
  [/질의 폭\(헤드 (\d+) × 128 = (\d+)\)이 너비 (\d+)와 다른 Qwen3\(0\.6B · 4B · 32B\)는 아직 지원하지 않습니다/g, 'Qwen3 models whose query width (heads $1 × 128 = $2) differs from the hidden size $3 (0.6B · 4B · 32B) are not supported yet'],
  [/RMSNorm eps가 10⁻⁶이 아닌 Qwen3 파일은 지원하지 않습니다/g, 'Qwen3 files whose RMSNorm eps is not 10⁻⁶ are not supported'],
  [/RMSNorm eps가 10⁻⁵이 아닌 llama 구조 파일은 지원하지 않습니다/g, 'llama-architecture files whose RMSNorm eps is not 10⁻⁵ are not supported'],
  [/질의 폭\(헤드 (\d+) × 128 = (\d+)\)이 너비 (\d+)와 다른 파일은 지원하지 않습니다/g, 'Files whose query width (heads $1 × 128 = $2) differs from the hidden size $3 are not supported'],
  [/RoPE 늘이기\((.+?)\)가 들어간 파일은 지원하지 않습니다\(늘이기 인자는 rope_freqs 텐서로만\)/g, 'Files with RoPE scaling ($1) are not supported (scaling factors only through the rope_freqs tensor)'],
  [/전문가 섞기\(MoE — 전문가 (\d+)개\) 모델은 아직 지원하지 않습니다/g, 'Mixture-of-experts models (MoE — $1 experts) are not supported yet'],
  [/채팅 틀이 Llama 3 · Mistral의 것과 다른 파일입니다\(ChatML 등을 쓰는 미세조정\) — 아직 지원하지 않습니다/g, 'The chat template differs from the Llama 3 · Mistral ones (a fine-tune using ChatML or similar) — not supported yet'],
  [/RoPE 늘이기\((.+?)\)가 들어간 파일은 지원하지 않습니다/g, 'Files with RoPE scaling ($1) are not supported'],
  [/지원하지 않는 모델 크기입니다\(너비 (\d+) · 헤드 (\d+) · KV 헤드 (\d+) · FFN (\d+)\) — 헤드 크기 128인 모델\(Qwen2 7B 이상 · Llama 3\.x · Mistral 7B 등\)만 지원합니다/g, 'Unsupported model size (hidden $1 · heads $2 · KV heads $3 · FFN $4) — only models with head size 128 (Qwen2 7B and up · Llama 3.x · Mistral 7B and the like) are supported'],
  [/지원하지 않는 토크나이저입니다\((.*?)\) — gpt2\(바이트 BPE\)와 llama\(SentencePiece\)만 지원합니다/g, 'Unsupported tokenizer ($1) — only gpt2 (byte-level BPE) and llama (SentencePiece) are supported'],
  [/지원하지 않는 토크나이저 사전 처리입니다\((.*?)\) — qwen2 · deepseek-r1-qwen · llama-bpe만 지원합니다/g, 'Unsupported tokenizer pre-processing ($1) — only qwen2 · deepseek-r1-qwen · llama-bpe are supported'],
  [/지원하지 않는 GGUF 버전입니다\((\d+)\) — 버전 3만 지원합니다/g, 'Unsupported GGUF version ($1) — only version 3 is supported'],
  [/GGUF 헤더를 끝까지 읽지 못했습니다\(파일이 잘렸거나 손상됨\)/g, 'Could not read the whole GGUF header (the file is truncated or damaged)'],
  [/GGUF 헤더가 손상되었습니다/g, 'The GGUF header is damaged'],
  [/GGUF 파일이 아닙니다/g, 'Not a GGUF file'],
  [/파일을 열 수 없습니다/g, 'Cannot open the file'],
  [/(\S+) 텐서가 없습니다/g, 'The $1 tensor is missing'],
  [/k-양자화 모델은 너비와 FFN 폭이 256의 배수여야 합니다/g, 'For k-quant models the hidden size and FFN width must be multiples of 256'],
  [/k-양자화 파일의 임베딩 형식\((.+?)\)은 지원하지 않습니다 — Q4_K · Q5_K · Q6_K여야 합니다/g, 'The embedding format ($1) of this k-quant file is not supported — it must be Q4_K · Q5_K · Q6_K'],
  [/k-양자화 파일의 출력 가중치 형식\((.+?)\)은 지원하지 않습니다 — Q4_K · Q5_K · Q6_K여야 합니다/g, 'The output weight format ($1) of this k-quant file is not supported — it must be Q4_K · Q5_K · Q6_K'],
  [/k-양자화 파일의 층 가중치 (\d+)개가 Q4_K · Q5_K · Q6_K 형식이 아닙니다/g, '$1 layer weights of this k-quant file are not Q4_K · Q5_K · Q6_K'],
  [/임베딩 형식\((.+?)\)이 다른 가중치\((.+?)\)와 다릅니다 — 모든 가중치가 같은 형식인 파일만 지원합니다/g, 'The embedding format ($1) differs from the other weights ($2) — only files whose weights all share one format are supported'],
  [/출력 가중치 형식\((.+?)\)이 다른 가중치\((.+?)\)와 다릅니다 — 모든 가중치가 같은 형식인 파일만 지원합니다\(공식 q4_0 파일은 출력이 Q6_K\)/g, 'The output weight format ($1) differs from the other weights ($2) — only files whose weights all share one format are supported (official q4_0 files keep the output in Q6_K)'],
  [/층 가중치 (\d+)개가 다른 형식입니다\(기준 (.+?)\)/g, '$1 layer weights use a different format (expected $2)'],
  [/지원하지 않는 가중치 형식입니다\((.+?)\) — Q8_0 · Q4_0 · Q4_K · Q5_K · Q6_K만 지원합니다/g, 'Unsupported weight format ($1) — only Q8_0 · Q4_0 · Q4_K · Q5_K · Q6_K are supported'],
  [/GeulPU가 지원하지 않는 파일입니다/g, "GeulPU doesn't support this file"],
  [/이 GPU에서는 Q4_K_M 같은 k-양자화 파일을 아직 실행할 수 없습니다\. 같은 모델의 Q8_0 · Q4_0 파일은 실행됩니다\(공유 메모리 32 KB인 GPU의 제약\)\./g, "This GPU can't run k-quant files such as Q4_K_M yet. The same model's Q8_0 · Q4_0 files run (a limit of GPUs with 32 KB of shared memory)."],
  // 받기
  [/ — 다시 누르면 이어서 받습니다/g, ' — click again to resume'],
  [/요청을 열지 못했습니다\(오류 (\d+)\)/g, 'Could not open the request (error $1)'],
  [/연결하지 못했습니다\(오류 (\d+)\) — 인터넷 · 방화벽 · 프록시를 확인해 주세요/g, 'Could not connect (error $1) — check your internet, firewall and proxy'],
  [/다운로드 서버가 HTTP (\d+) 오류를 보냈습니다/g, 'The download server returned HTTP $1'],
  [/서버가 HTTP (\d+) 오류를 보냈습니다/g, 'The server returned HTTP $1'],
  [/모델 파일을 쓸 수 없습니다\(폴더 권한 · 디스크를 확인해 주세요\)/g, 'Cannot write the model file (check the folder permissions and the disk)'],
  [/다운로드 중 연결이 끊겼습니다\(오류 (\d+)\)/g, 'The connection dropped during the download (error $1)'],
  [/받는 중 연결이 끊겼습니다\(오류 (\d+)\)/g, 'The connection dropped while downloading (error $1)'],
  [/디스크에 쓰지 못했습니다\(여유 공간을 확인해 주세요\)/g, 'Could not write to the disk (check the free space)'],
  [/일시정지했습니다/g, 'Paused'],
  [/다운로드한 파일을 읽지 못했습니다/g, 'Could not read the downloaded file'],
  [/SHA-256이 목록과 다릅니다\(([0-9a-f]+)…, 목록 ([0-9a-f]+)…\)\. 파일을 지웠으니 다시 다운로드하세요/g, 'SHA-256 differs from the list ($1…, list $2…). The file was deleted — download it again'],
  [/다운로드한 파일의 이름을 바꾸지 못했습니다/g, 'Could not rename the downloaded file'],
  [/https · http 주소가 아닙니다/g, 'Not an https or http address'],
  [/WinHTTP를 열지 못했습니다/g, 'Could not open WinHTTP'],
  [/(\S+)에 연결하지 못했습니다/g, 'Could not connect to $1'],
  [/다운로드를 마치지 못했습니다/g, 'The download did not finish'],
  [/다운로드한 파일이 목록의 크기보다 큽니다 — 파일을 지우고 다시 다운로드하세요/g, 'The downloaded file is larger than listed — delete it and download again'],
  [/파일의 앞부분을 받지 못했습니다/g, 'Could not fetch the start of the file'],
  [/다운로드 스레드를 시작하지 못했습니다/g, 'Could not start the download thread'],
  [/확인 스레드를 시작하지 못했습니다/g, 'Could not start the check thread'],
  // 관리자의 거절(/api)
  [/다른 작업이 실행 중입니다 — 끝나거나 중지한 뒤 다시 시도하세요/g, 'Another job is running — try again after it finishes or is stopped'],
  [/다른 작업이 실행 중입니다 — 끝난 뒤 다시 시도하세요/g, 'Another job is running — try again after it finishes'],
  [/다른 작업이 실행 중입니다/g, 'Another job is running'],
  [/불러온 모델이 없습니다 — 앱 화면에서 모델을 먼저 불러오세요/g, 'No model is loaded — load a model in the app first'],
  [/Host 헤더가 127\.0\.0\.1이 아닙니다/g, 'The Host header is not 127.0.0.1'],
  [/다른 출처에서 온 요청은 받지 않습니다/g, 'Requests from other origins are refused'],
  [/GET · POST 요청만 받습니다/g, 'Only GET and POST requests are accepted'],
  [/결과 파일이 없습니다/g, 'The result file does not exist'],
  [/결과가 없습니다/g, 'The result does not exist'],
  [/zip 파일을 만들지 못했습니다/g, 'Could not create the zip file'],
  [/없는 경로입니다: /g, 'Unknown path: '],
  [/Content-Type은 application\/json이어야 합니다/g, 'Content-Type must be application/json'],
  [/요청 본문이 JSON 객체가 아닙니다/g, 'The request body is not a JSON object'],
  [/폴더가 없습니다 — 있는 폴더의 경로를 입력하세요/g, 'The folder does not exist — enter the path of an existing folder'],
  [/모델이 없거나 이미 확인 중입니다/g, 'The model does not exist or is already being verified'],
  [/지금 불러온 모델입니다 — 먼저 모델을 내린 뒤 제거하세요/g, 'This model is loaded now — unload it before removing it'],
  [/무결성을 확인하는 중입니다 — 끝난 뒤 다시 시도하세요/g, 'Integrity is being verified — try again after it finishes'],
  [/목록에 없는 모델입니다/g, 'This model is not in the list'],
  [/이미 설치된 모델입니다 — 모델 메뉴에서 제거하세요/g, 'Already installed — remove it from the model menu'],
  [/이미 설치된 모델입니다/g, 'This model is already installed'],
  [/이미 다운로드 중입니다/g, 'Already downloading'],
  [/다운로드 중입니다 — 먼저 일시정지하세요/g, 'Downloading — pause it first'],
  [/디스크 공간이 부족합니다\(여유 ([\d.]+) GB · 필요 ([\d.]+) GB\)/g, 'Not enough disk space ($1 GB free · $2 GB needed)'],
  [/Hugging Face 저장소 · 파일 이름이 올바르지 않습니다/g, 'The Hugging Face repository or file name is invalid'],
  [/다른 파일을 확인하는 중입니다 — 잠시 뒤 다시 시도하세요/g, 'Another file is being checked — try again in a moment'],
  [/SHA-256 값이 없습니다 — Hugging Face가 알려 준 파일만 받습니다/g, 'No SHA-256 value — only files with a SHA-256 from Hugging Face are downloaded'],
  [/먼저 이 파일을 확인하세요/g, 'Check this file first'],
  [/모델 폴더에 같은 이름의 파일이 이미 있습니다: /g, 'A file with the same name is already in the model folder: '],
  [/같은 이름의 모델이 이미 목록에 있습니다/g, 'A model with the same name is already in the list'],
  [/이미 추가한 파일입니다: /g, 'Already added: '],
  [/\.gguf 파일만 추가할 수 있습니다/g, 'Only .gguf files can be added'],
  [/요청 헤더가 너무 깁니다/g, 'The request headers are too long'],
  [/요청을 읽을 수 없습니다/g, 'Cannot read the request'],
  [/Content-Length가 있는 요청만 받습니다/g, 'Only requests with Content-Length are accepted'],
  [/요청 본문은 16 MB까지 받습니다/g, 'Request bodies are limited to 16 MB'],
  [/동시 연결이 너무 많습니다 — 잠시 뒤 다시 시도하세요/g, 'Too many simultaneous connections — try again in a moment'],
  [/추론 서버에 연결하지 못했습니다/g, 'Could not connect to the inference server'],
  [/모델을 불러오지 못했습니다/g, 'Could not load the model'],
  [/파일이 없습니다/g, 'The file does not exist'],
  [/모델이 없습니다/g, 'The model does not exist'],
  [/하드웨어 확인/g, 'Hardware check'],
];
function TM(s) {
  if (!EN || s === null || s === undefined) return s;
  let t = String(s);
  if (!HANGUL.test(t)) return t;
  const d = DICT[t];
  if (d !== undefined) return d;
  for (const [re, rep] of TMX) { if (HANGUL.test(t)) t = t.replace(re, rep); else break; }
  return t;
}
// index.html 의 정적 문구(속성 · 글)를 영어로 — 앱 코드보다 먼저 한 번
(function applyStatic() {
  if (!EN) return;
  for (const el of document.querySelectorAll('[aria-label],[data-tip],[placeholder],[title]')) {
    for (const a of ['aria-label', 'data-tip', 'placeholder', 'title']) {
      const v = el.getAttribute(a);
      if (v && HANGUL.test(v)) el.setAttribute(a, T(v));
    }
  }
  const w = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT);
  for (let n = w.nextNode(); n; n = w.nextNode()) {
    const t = n.nodeValue.trim();
    if (t && HANGUL.test(t)) n.nodeValue = n.nodeValue.replace(t, T(t));
  }
})();

// ---------- 아이콘(24 × 24, 선 1.7 — 이 앱에서 그린 것) ----------
const IC = {
  chat: '<path d="M20.5 11.5c0 4.4-3.8 8-8.5 8-1.2 0-2.3-.2-3.3-.6L3.5 20l1.3-4c-.8-1.3-1.3-2.8-1.3-4.5 0-4.4 3.8-8 8.5-8s8.5 3.6 8.5 8z"/>',
  box: '<path d="M12 2.8 20.2 7.3v9.4L12 21.2l-8.2-4.5V7.3z"/><path d="M3.8 7.3 12 11.8l8.2-4.5M12 11.8v9.4"/>',
  pulse: '<path d="M3 12h3.5l2.5-7 4 14 2.5-7H21"/>',
  cpu: '<rect x="5" y="5" width="14" height="14" rx="2"/><rect x="9" y="9" width="6" height="6" rx="1"/><path d="M9 2v3M15 2v3M9 19v3M15 19v3M2 9h3M2 15h3M19 9h3M19 15h3"/>',
  gear: '<path d="M19.15 10.08 L21.51 10.66 L21.51 13.34 L19.15 13.92 L18.41 15.70 L19.67 17.78 L17.78 19.67 L15.70 18.41 L13.92 19.15 L13.34 21.51 L10.66 21.51 L10.08 19.15 L8.30 18.41 L6.22 19.67 L4.33 17.78 L5.59 15.70 L4.85 13.92 L2.49 13.34 L2.49 10.66 L4.85 10.08 L5.59 8.30 L4.33 6.22 L6.22 4.33 L8.30 5.59 L10.08 4.85 L10.66 2.49 L13.34 2.49 L13.92 4.85 L15.70 5.59 L17.78 4.33 L19.67 6.22 L18.41 8.30 Z"/><circle cx="12" cy="12" r="3"/>',
  sliders: '<path d="M4 7h9M17 7h3M4 17h3M11 17h9"/><circle cx="15" cy="7" r="2"/><circle cx="9" cy="17" r="2"/>',
  plus: '<path d="M12 5v14M5 12h14"/>',
  up: '<path d="M12 19V5M6 11l6-6 6 6"/>',
  down: '<path d="M12 5v14M6 13l6 6 6-6"/>',
  stop: '<rect x="7.5" y="7.5" width="9" height="9" rx="1.5" fill="currentColor" stroke="none"/>',
  pause: '<path d="M9 6.5v11M15 6.5v11"/>',
  download: '<path d="M12 4v11M7 10.5l5 5 5-5M5 20h14"/>',
  check: '<path d="M5 12.5l4.5 4.5L19 7.5"/>',
  checkc: '<circle cx="12" cy="12" r="8.5"/><path d="m8.5 12.2 2.4 2.4 4.6-4.8"/>',
  x: '<path d="M6 6l12 12M18 6 6 18"/>',
  trash: '<path d="M4 7h16M9.5 7V4.5h5V7M6.5 7l1 12.5h9l1-12.5M10 11v5.5M14 11v5.5"/>',
  edit: '<path d="M4.5 19.5l1-4 10-10 3 3-10 10z"/><path d="M13.5 7.5l3 3"/>',
  dots: '<circle cx="5.5" cy="12" r="1.3" fill="currentColor" stroke="none"/><circle cx="12" cy="12" r="1.3" fill="currentColor" stroke="none"/><circle cx="18.5" cy="12" r="1.3" fill="currentColor" stroke="none"/>',
  folder: '<path d="M3.5 7.5A1.5 1.5 0 0 1 5 6h4.2l2 2H19a1.5 1.5 0 0 1 1.5 1.5v8A1.5 1.5 0 0 1 19 19H5a1.5 1.5 0 0 1-1.5-1.5z"/>',
  file: '<path d="M6.5 3.5h7l4 4v13h-11z"/><path d="M13.5 3.5v4h4"/>',
  refresh: '<path d="M19.5 12a7.5 7.5 0 1 1-2.2-5.3M19.5 4.5v4h-4"/>',
  chevD: '<path d="M7 10l5 5 5-5"/>',
  chevR: '<path d="M10 7l5 5-5 5"/>',
  chevL: '<path d="M14 7l-5 5 5 5"/>',
  code: '<path d="m8.5 7.5-5 4.5 5 4.5M15.5 7.5l5 4.5-5 4.5"/>',
  type: '<path d="M5.5 6.5V5h13v1.5M12 5v14M9 19h6"/>',
  hash: '<path d="M10 4 8 20M16 4l-2 16M4.5 9h15M3.5 15h15"/>',
  target: '<circle cx="12" cy="12" r="8.5"/><circle cx="12" cy="12" r="4.5"/><circle cx="12" cy="12" r="1" fill="currentColor" stroke="none"/>',
  chevU: '<path d="M7 14l5-5 5 5"/>',
  ext: '<path d="M14 4.5h5.5V10M19.5 4.5 11 13M17.5 14v4.5a1 1 0 0 1-1 1h-11a1 1 0 0 1-1-1v-11a1 1 0 0 1 1-1H10"/>',
  alert: '<path d="M12 4 2.8 19.5h18.4z"/><path d="M12 10v4.2M12 16.8v.2"/>',
  info: '<circle cx="12" cy="12" r="8.5"/><path d="M12 11v5M12 8v.2"/>',
  play: '<path d="M8 5.5v13l10.5-6.5z"/>',
  shield: '<path d="M12 3.5 19 6v5.5c0 4.4-3 7.6-7 8.9-4-1.3-7-4.5-7-8.9V6z"/><path d="M9 12l2.2 2.2L15.5 10"/>',
  copy: '<rect x="8.5" y="8.5" width="11" height="11" rx="2"/><path d="M15.5 8.5V6a1.5 1.5 0 0 0-1.5-1.5H6A1.5 1.5 0 0 0 4.5 6v8A1.5 1.5 0 0 0 6 15.5h2.5"/>',
  panel: '<rect x="3.5" y="4.5" width="17" height="15" rx="2"/><path d="M9.5 4.5v15"/>',
  spark: '<path d="M12 3.5l1.9 5.1 5.1 1.9-5.1 1.9-1.9 5.1-1.9-5.1-5.1-1.9 5.1-1.9z"/>',
  gauge: '<path d="M3.5 16a8.5 8.5 0 1 1 17 0"/><path d="M12 16l4.5-5"/><circle cx="12" cy="16" r="1.4"/><path d="M5 20h14"/>',
  layers: '<path d="M12 3.5 21 8l-9 4.5L3 8z"/><path d="M3 12.5l9 4.5 9-4.5M3 16.5 12 21l9-4.5"/>',
  bolt: '<path d="M13 3 5 13.5h6L10 21l8-10.5h-6z"/>',
  star: '<path d="M12 3.8l2.5 5.2 5.7.8-4.1 4 1 5.6L12 16.7l-5.1 2.7 1-5.6-4.1-4 5.7-.8z"/>',
  terminal: '<rect x="3.5" y="4.5" width="17" height="15" rx="2"/><path d="m7.5 9.5 3 2.5-3 2.5M12.5 15h4"/>',
  eject: '<path d="M12 5.5 5 13.5h14z"/><path d="M5 18.5h14"/>',
  forward: '<path d="M5 12h12M12.5 6.5 18 12l-5.5 5.5"/>',
  search: '<circle cx="10.5" cy="10.5" r="6.5"/><path d="m15.5 15.5 5 5"/>',
  heart: '<path d="M12 19.6s-7.6-4.5-7.6-9.8A4.2 4.2 0 0 1 12 7.4a4.2 4.2 0 0 1 7.6 2.4c0 5.3-7.6 9.8-7.6 9.8z"/>',
  clock: '<circle cx="12" cy="12" r="8.5"/><path d="M12 7.5V12l3 2"/>',
  flask: '<path d="M9.5 3.5h5M10.5 3.5v5.4l-5.3 9.2a1.6 1.6 0 0 0 1.4 2.4h10.8a1.6 1.6 0 0 0 1.4-2.4l-5.3-9.2V3.5"/><path d="M7.6 14.5h8.8"/>',
};
const svg = (n, cls) => `<svg class="i ${cls || ''}" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">${IC[n] || ''}</svg>`;

// ---------- 도움 ----------
const $ = (s, r = document) => r.querySelector(s);
const keep = a => a.flat(Infinity).filter(x => x !== null && x !== undefined && x !== false);
function h(tag, attrs, ...kids) {
  const e = document.createElement(tag);
  if (attrs) for (const [k, v] of Object.entries(attrs)) {
    if (v === null || v === undefined || v === false) continue;
    if (k === 'class') e.className = v;
    else if (k === 'html') e.innerHTML = v;
    else if (k.startsWith('on')) e.addEventListener(k.slice(2), v);
    else e.setAttribute(k, v === true ? '' : v);
  }
  for (const k of keep(kids)) e.append(k instanceof Node ? k : document.createTextNode(String(k)));
  return e;
}
// 자식 갈기 — null · undefined · false 는 빼고(화면에 "null" 이 찍히지 않게)
const put = (el, ...kids) => { if (el) el.replaceChildren(...keep(kids)); return el; };
const I = (n, cls) => { const t = document.createElement('template'); t.innerHTML = svg(n, cls); return t.content.firstChild; };
const esc = s => String(s ?? '').replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
const NB = '\u00a0· ';   // 가운뎃점이 줄 머리에 오지 않게
const GIB = 1073741824;
// 크기는 1024 단위(윈도 탐색기와 같은 수 — VRAM 12 GB 카드는 12.0 GB)
function fmtBytes(n) {
  if (!(n >= 0)) return '-';
  const M = 1048576, K = 1024;
  if (n >= GIB) return (n / GIB).toFixed(n >= 100 * GIB ? 0 : 1) + '\u00a0GB';
  if (n >= M) return Math.round(n / M) + '\u00a0MB';
  if (n >= K) return Math.round(n / K) + '\u00a0KB';
  return n + '\u00a0B';
}
const fmtNum = (x, d) => Number(x).toLocaleString('ko-KR', { minimumFractionDigits: d || 0, maximumFractionDigits: d || 0 });
// 하드웨어 한계 대비 — 진단 도구의 '배'(잰 시간 ÷ 한계 시간)를 백분율로(높을수록 한계에 가깝다)
const effOf = x => (x > 0 ? Math.round(100 / x) : NaN);
const fmtEff = p => (p >= 0 ? p + '%' : '-');
function fmtClock(s) {
  if (!(s >= 0) || !isFinite(s)) return '-';
  s = Math.round(s);
  const m = Math.floor(s / 60), r = s % 60;
  if (m < 60) return m + ':' + String(r).padStart(2, '0');
  return Math.floor(m / 60) + ':' + String(m % 60).padStart(2, '0') + ':' + String(r).padStart(2, '0');
}
function fmtLeft(s) {
  if (!(s >= 0) || !isFinite(s)) return T('남은 시간 계산 중');
  s = Math.round(s);
  if (s < 60) return T('1분 안에 끝남');
  const m = Math.round(s / 60);
  if (m < 60) return T`약 ${m}분 남음`;
  return T`약 ${Math.floor(m / 60)}시간 ${m % 60}분 남음`;
}
const fmtCtx = n => (n >= 1024 && n % 1024 === 0 ? (n / 1024) + 'K' : fmtNum(n || 0));
async function api(path, body) {
  const opt = body === undefined ? {} : { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) };
  const r = await fetch(path, opt);
  let j = null;
  try { j = await r.json(); } catch (_) { }
  if (!r.ok) throw new Error((j && j.error && j.error.message) || ('HTTP ' + r.status));
  return j;
}
function toast(msg, kind) {
  const close = h('button', { class: 'icon-btn', 'aria-label': T('알림 닫기') }, I('x', 's'));
  const d = h('div', { class: kind || '', role: kind === 'err' ? 'alert' : 'status' }, I(kind === 'err' ? 'alert' : kind === 'ok' ? 'check' : 'info', 's'), h('span', null, TM(msg)), close);
  close.addEventListener('click', () => d.remove());
  $('#toast').append(d);
  setTimeout(() => d.remove(), kind === 'err' ? 10000 : 4000);
}
const announce = t => { const e = $('#sr-live'); if (e) { e.textContent = ''; setTimeout(() => { e.textContent = t; }, 50); } };
let storeWarned = false;
function store(key, val) {
  try { localStorage.setItem(key, JSON.stringify(val)); }
  catch (_) {
    // 채팅이 먼저다 — 자리가 모자라면 연구 기록(38단계)을 비우고 한 번 더
    if (key === 'geulpu.rooms') { try { if (localStorage.getItem('geulpu.traces')) { localStorage.removeItem('geulpu.traces'); localStorage.setItem(key, JSON.stringify(val)); return; } } catch (_) { } }
    if (key === 'geulpu.rooms' && !storeWarned) { storeWarned = true; toast(T('브라우저 저장 공간이 부족해 채팅을 저장하지 못했습니다 — 오래된 채팅을 삭제하세요'), 'err'); } }
}
function load(key, dflt) { try { const v = localStorage.getItem(key); return v ? JSON.parse(v) : dflt; } catch (_) { return dflt; } }
const SIG = {};
function changed(name, obj) { const s = JSON.stringify(obj); if (SIG[name] === s) return false; SIG[name] = s; return true; }
const resetSig = () => { for (const k of Object.keys(SIG)) delete SIG[k]; };
async function copyText(text) {
  try { await navigator.clipboard.writeText(text); return true; } catch (_) { toast(T('복사하지 못했습니다'), 'err'); return false; }
}

// 창(가운데) — 머리(닫기)와 바닥(단추)은 그대로, 가운데만 스크롤. 초점은 창 안에 가두고, 닫으면 연 자리로 돌려준다
let modalReturn = null, modalOnClose = null;
function modal(content, buttons, wide, focusSel, onClose) {
  const box = $('#modalbox');
  if ($('#modal').hidden) modalReturn = document.activeElement;
  modalOnClose = onClose || null;
  box.className = 'box' + (wide ? ' wide' : '');
  const body = h('div', { class: 'mb' }, content);
  put(box, body, buttons && buttons.length ? h('div', { class: 'foot' }, buttons) : null,
    h('button', { class: 'icon-btn x', 'aria-label': T('닫기'), onclick: () => closeModal() }, I('x')));
  const t = box.querySelector('h2');
  if (t) t.id = 'modal-title';
  $('#modal').hidden = false;
  body.scrollTop = 0;
  const f = (focusSel && box.querySelector(focusSel)) || box.querySelector('.foot .primary') || box.querySelector('.mb input, .mb textarea') || box.querySelector('.foot button, .foot a') || box.querySelector('.x');
  if (f) setTimeout(() => { f.focus({ preventScroll: true }); if (f.select && f.type === 'text') f.select(); }, 0);
}
function closeModal() {
  if ($('#modal').hidden) return;
  $('#modal').hidden = true;
  put($('#modalbox'));
  const cb = modalOnClose;
  modalOnClose = null;
  if (cb) cb();
  if (modalReturn && modalReturn.isConnected && modalReturn.focus) modalReturn.focus({ preventScroll: true });
  modalReturn = null;
}
// kind: 'danger'(지우기 등) · 'quiet'(권하지 않는 동작 — 흰 채움 없이, 초점은 취소에)
function confirmBox(title, body, okText, kind) {
  return new Promise(res => {
    let done = false;
    const fin = v => { if (!done) { done = true; res(v); } };
    const no = h('button', { class: 'btn ghost', onclick: () => closeModal() }, T('취소'));
    const ok = h('button', { class: 'btn ' + (kind === 'danger' ? 'danger' : kind === 'quiet' ? '' : 'primary'), onclick: () => { fin(true); closeModal(); } }, okText || T('확인'));
    modal(h('div', null, h('h2', null, title), body), [no, ok], false, kind ? '.foot .btn.ghost' : null, () => fin(false));
  });
}
$('#modal').addEventListener('click', e => { if (e.target.id === 'modal') closeModal(); });
$('#modal').addEventListener('keydown', e => {
  if (e.key !== 'Tab') return;
  const f = [...$('#modalbox').querySelectorAll('button, a[href], input, textarea, select, summary')].filter(x => !x.disabled && x.offsetParent);
  if (!f.length) return;
  const first = f[0], last = f[f.length - 1];
  if (e.shiftKey && document.activeElement === first) { e.preventDefault(); last.focus(); }
  else if (!e.shiftKey && document.activeElement === last) { e.preventDefault(); first.focus(); }
});

// 뜨는 판(메뉴) — 기준 요소 아래
let POP = null, POPANCHOR = null;
function closePop() {
  if (POP) { POP.remove(); POP = null; }
  if (POPANCHOR) { POPANCHOR.setAttribute('aria-expanded', 'false'); POPANCHOR = null; }
}
// 판의 자리 — 기준 요소 아래, 모자라면 위. 내용이 자란 뒤에도 다시 부른다(다운로드 목록)
function placePop(p, anchor, align) {
  const r = anchor.getBoundingClientRect();
  const w = p.offsetWidth, hh = p.offsetHeight;
  let x = align === 'right' ? r.right - w : r.left;
  x = Math.max(8, Math.min(x, innerWidth - w - 8));
  let y = r.bottom + 6;
  if (y + hh > innerHeight - 8) y = Math.max(8, r.top - hh - 6);
  p.style.left = x + 'px';
  p.style.top = y + 'px';
}
function openPop(anchor, content, align) {
  closePop();
  const p = h('div', { class: 'pop' }, content);
  document.body.append(p);
  placePop(p, anchor, align);
  POP = p;
  POPANCHOR = anchor;
  anchor.setAttribute('aria-expanded', 'true');
  // 판 밖으로 초점이 나가면 닫는다(창 꼴의 판 — 메뉴는 Tab 에서 닫힌다)
  p.addEventListener('focusout', e => { if (POP === p && e.relatedTarget && !p.contains(e.relatedTarget) && e.relatedTarget !== POPANCHOR) closePop(); });
  const f = p.querySelector('.opt.sel:not(:disabled)') || p.querySelector('.opt:not(:disabled), input, textarea, button');
  if (f) setTimeout(() => f.focus({ preventScroll: true }), 0);
  return p;
}
function menu(anchor, label, items, align) {
  const opts = items.filter(Boolean).map(it => it === '-' ? h('div', { class: 'sep', role: 'separator' }) : h('button', {
    class: 'opt' + (it.danger ? ' danger' : ''), role: 'menuitem', disabled: it.disabled ? true : null,
    onclick: () => { closePop(); it.run(); },
  }, I(it.icon, 's'), h('span', { class: 'main' }, h('div', { class: 't' }, it.label), it.sub ? h('div', { class: 's' }, it.sub) : null)));
  openPop(anchor, h('div', { role: 'menu', 'aria-label': label }, opts), align);
}
document.addEventListener('mousedown', e => { if (POP && !POP.contains(e.target) && !(POPANCHOR && POPANCHOR.contains(e.target))) closePop(); });
document.addEventListener('keydown', e => {
  if (e.key === 'Escape') {
    if (POP) { const a = POPANCHOR; closePop(); if (a) a.focus(); }
    else if (!$('#modal').hidden) closeModal();
    else if (overlayConvos() && !$('#convos').classList.contains('closed')) setConvos(false);
    else if (LAB.open && labOverlay() && S.view === 'chat') labToggle(false);
    else if (streamingHere() && S.view === 'chat') stopStream();
  }
  // 메뉴: 위아래 화살표로 고르고, Tab 은 메뉴를 닫고 다음 요소로(메뉴 항목은 Tab 정거장이 아니다)
  if (POP && POP.querySelector('[role=menu]') && POP.contains(document.activeElement)) {
    if (e.key === 'ArrowDown' || e.key === 'ArrowUp' || e.key === 'Home' || e.key === 'End') {
      const opts = [...POP.querySelectorAll('.opt:not(:disabled)')];
      if (!opts.length) return;
      e.preventDefault();
      let i = opts.indexOf(document.activeElement);
      i = e.key === 'Home' ? 0 : e.key === 'End' ? opts.length - 1 : e.key === 'ArrowDown' ? (i + 1) % opts.length : (i - 1 + opts.length) % opts.length;
      opts[i].focus();
    } else if (e.key === 'Tab') { const a = POPANCHOR; closePop(); if (a) a.focus(); }
  }
});

// 도움말(말풍선) — 떠 있는 요소 하나로(넘침을 잘라 내는 상자 안에서도). 마우스는 잠깐 머문 뒤, 키보드 초점은 바로
const TIP = h('div', { id: 'tip', role: 'tooltip', hidden: true });
document.body.append(TIP);
let tipEl = null, tipTimer = 0;
const noHover = matchMedia('(hover: none)');
function hideTip() { clearTimeout(tipTimer); if (tipEl) tipEl.removeAttribute('aria-describedby'); tipEl = null; TIP.hidden = true; }
function showTip(el) {
  if (!el.isConnected || !el.dataset.tip) return;
  tipEl = el;
  TIP.textContent = el.dataset.tip;
  TIP.hidden = false;
  if (!el.hasAttribute('aria-label') || el.getAttribute('aria-label') !== el.dataset.tip) el.setAttribute('aria-describedby', 'tip');
  const r = el.getBoundingClientRect(), w = TIP.offsetWidth, hh = TIP.offsetHeight, gap = 8;
  const fits = { right: r.right + gap + w <= innerWidth - 8, left: r.left - gap - w >= 8, top: r.top - gap - hh >= 8, bottom: r.bottom + gap + hh <= innerHeight - 8 };
  const want = el.dataset.tipPos || 'right';
  const p = [want, { right: 'left', left: 'right', top: 'bottom', bottom: 'top' }[want], 'bottom', 'top'].find(o => fits[o]) || want;
  let x, y;
  if (p === 'right' || p === 'left') { x = p === 'right' ? r.right + gap : r.left - gap - w; y = r.top + r.height / 2 - hh / 2; }
  else { x = r.left + r.width / 2 - w / 2; y = p === 'bottom' ? r.bottom + gap : r.top - gap - hh; }
  TIP.style.left = Math.round(Math.max(8, Math.min(x, innerWidth - w - 8))) + 'px';
  TIP.style.top = Math.round(Math.max(8, Math.min(y, innerHeight - hh - 8))) + 'px';
}
document.addEventListener('mouseover', e => {
  if (noHover.matches) return;
  const el = e.target.closest ? e.target.closest('[data-tip]') : null;
  if (el === tipEl) return;
  hideTip();
  if (el) tipTimer = setTimeout(() => showTip(el), 450);
});
document.addEventListener('focusin', e => {
  const el = e.target.closest ? e.target.closest('[data-tip]') : null;
  hideTip();
  if (el && el === e.target && el.matches(':focus-visible')) showTip(el);
});
document.addEventListener('focusout', hideTip);
document.addEventListener('mousedown', hideTip, true);
document.addEventListener('scroll', hideTip, true);
document.addEventListener('keydown', e => { if (e.key === 'Escape' || e.key === 'Enter') hideTip(); }, true);

// ---------- 상태 ----------
// streaming: 지금 흐르는 답 하나 — { roomId, msg, el(지금 그려진 요소), notice(불러오기 카드), ctrl, aborted, loading }
// editing: 고치는 중인 사용자 메시지 { roomId, k } · jobHide: 닫은 '끝난 작업' 카드
const S = { st: null, view: 'chat', rooms: [], roomId: null, streaming: null, editing: null, log: { job: 0, from: 0 }, results: null, resultsAt: 0, update: null, conn: true, setDirty: false, vk: null, benchModel: load('geulpu.benchModel', ''), catSel: {}, jobHide: load('geulpu.jobHide', '') };
const streamingHere = () => !!(S.streaming && S.streaming.roomId === S.roomId);
let failCount = 0;
async function refresh() {
  try {
    const st = await api('/api/state');
    S.st = st;
    failCount = 0;
    if (!S.conn) { S.conn = true; setOffline(false); }
    renderAll();
    await pollLog();
  } catch (_) {
    failCount++;
    if (failCount >= 2 && S.conn) { S.conn = false; setOffline(true); }
  }
}
setInterval(() => { if (!document.hidden) refresh(); }, 1000);
function setOffline(off) {
  document.body.classList.toggle('offline', off);
  renderStatus();
  updateComposer();
  const b = $('#banner');
  if (off) {
    put(b, I('alert', 's'), h('span', null, T('GeulPU.exe와 연결이 끊겼습니다. 다시 연결하는 중… GeulPU.exe가 실행 중인지 확인하세요.')));
    b.className = 'offline';
    b.hidden = false;
  } else if (b.classList.contains('offline')) { b.className = ''; b.hidden = true; checkUpdate(); }
}
function renderAll() {
  if (!S.st) return;
  renderStatus();
  if (S.view === 'chat') { renderChatTop(); if (!curRoom() || !curRoom().messages.length) renderMsgsIfChanged(); labSync(); }
  else if (S.view === 'models') renderModels();
  else if (S.view === 'bench') renderBench();
  else if (S.view === 'device') renderDevice();
  else if (S.view === 'settings') renderSettings();
  if (POP && POP.dataset.kind === 'dl') refreshDlPop();
  if (POPANCHOR && !POPANCHOR.isConnected) closePop();   // 다시 그려 기준 단추가 사라졌으면 판도 닫는다
  updateDlBars();
}
// 화면을 다시 그려도 키보드 초점을 같은 자리(data-fk)로 돌려준다
function rebuild(page, ...kids) {
  const a = document.activeElement;
  const key = a && page.contains(a) ? a.dataset.fk : null;
  put(page, ...kids);
  if (key) { const b = page.querySelector(`[data-fk="${CSS.escape(key)}"]`); if (b) b.focus({ preventScroll: true }); }
}

// ---------- 이름 ----------
const LAYER = { cuda: 'CUDA', vulkan: 'Vulkan' };
// GGUF 의 이름(띄어 씀)을 목록의 꼴(하이픈)로 — 한 모델은 어디서나 한 이름
function prettyName(n) { return String(n || '').trim().replace(/\s+/g, '-'); }
function catalogOf(m) { return m && m.catalog && S.st ? S.st.catalog.find(c => c.id === m.catalog) : null; }
function displayName(m) {
  if (!m) return '';
  const c = catalogOf(m);
  return c ? c.name : prettyName(m.name || m.id);
}
function modelOf(id) { return S.st ? S.st.models.find(m => m.id === id) : null; }
function labelById(id) {
  const m = modelOf(id);
  if (m) return `${displayName(m)} · ${quantOf(m)}`;
  const c = S.st && S.st.catalog.find(x => x.id === id);
  return c ? `${c.name} · ${c.quant}` : (id || '');
}
// 양자화 이름 — k-양자화 파일은 관리자가 "K" 로만 알려 주므로 목록의 이름(Q4_K_M 등)이나 가중치 형식으로 채운다
function quantOf(m) {
  const s = (m && m.shape) || {};
  if (s.quant && s.quant !== 'K') return s.quant;
  const c = catalogOf(m);
  if (c && c.quant) return c.quant;
  return s.weight_type || s.quant || '';
}
const QORDER = ['F32', 'F16', 'BF16', 'Q8_0', 'Q6_K', 'Q5_K_M', 'Q5_K_S', 'Q4_K_M', 'Q4_K_S', 'Q4_0'];
const qRank = q => { const i = QORDER.indexOf(q); return i < 0 ? 99 : i; };
const byName = (a, b) => a.localeCompare(b, 'en', { numeric: true });
// 같은 모델의 양자화들이 붙어 있게: 이름(숫자는 크기 순) → 양자화
function sortedModels() {
  return [...S.st.models].sort((a, b) => byName(displayName(a), displayName(b)) || qRank(quantOf(a)) - qRank(quantOf(b)));
}
// 모델의 종류 — 추론(생각을 먼저 쓰는 R1) · 코드(Coder) · 일반 대화. 이름은 GGUF(띄어 씀)와 목록(하이픈) 어느 꼴이든.
// 이름 앞에 종류 아이콘(동그라미)은 두지 않는다 — 종류는 글로 보인다
function kindKey(x) {
  const n = [x && x.catalog, x && x.id, x && x.name].filter(Boolean).join(' ').toLowerCase();
  if ((x && x.template === 'deepseek-r1') || /deepseek[\s_-]*r1/.test(n)) return 'think';
  if (/(^|[^a-z])coder/.test(n)) return 'code';          // 'text-encoder' 는 코드 모델이 아니다
  if ((x && x.template === 'qwen3') || /qwen3/.test(n)) return 'think';   // Qwen3 — 생각 켬 · 끔을 고르는 모델
  return 'chat';
}
const qtag = m => (quantOf(m) ? h('span', { class: 'qlab' }, quantOf(m)) : null);
const kindOf = m => ({ think: T('추론 모델'), code: T('코드 모델'), chat: T('일반 모델') })[kindKey(m)];
function jobName(j) { return { diag: T('하드웨어 진단'), speed: T('속도 측정'), crosscheck: T('재현성 검증'), server: T('모델 불러오기'), device: T('하드웨어 확인') }[j.kind] || TM(j.title); }
const jobModel = j => String(j.title || '').split(' — ')[1] || '';
function jobTitle(j) {
  if (j.kind === 'diag' || j.kind === 'speed' || j.kind === 'server') return `${jobName(j)} · ${labelById(jobModel(j))}`;
  if (j.kind === 'crosscheck') return /줄인|빠른/.test(j.title) ? T('재현성 검증 · 빠른 확인(문제 2개)') : T('재현성 검증 · 문제 32개');
  return TM(j.title);
}
// 재현성 검증의 경로 이름(관리자의 "협력 행렬 모듈 …" 등)을 쓰는 사람의 말로
function pathLabel(l) {
  return String(l || '').replace(/협력 행렬 모듈\s*([0-9x]*)/g, (_, s) => T('텐서 코어 경로') + (s ? ` (${s})` : '')).replace(/기본 모듈/g, T('기본 경로')).replace(/CUDA 모듈/g, T('CUDA 경로'));
}
// 10B 이상이면 문맥 · 작업 메모리를 더 크게 어림한다(목록의 "14B" · Hugging Face 에서 더한 "14.8B" 모두)
const isBig = c => { const p = parseFloat(String(c.params || '').replace(/[^\d.]/g, '')); return p >= 10 || (!(p > 0) && /\b(1[0-9]|[2-9]\d)b\b/i.test(c.id || '')); };
// 이 PC 의 그래픽 메모리에 맞는가(목록의 크기 + 문맥 · 작업 메모리 어림)
function fitOf(c) {
  const d = S.st && S.st.device;
  if (!d || !d.checked) return null;
  if (!d.layer) return { k: 'nogpu', t: T('쓸 수 있는 GPU 없음') };
  if (c.devices === 'k' && d.layer === 'vulkan' && d.variant === '공유32') return { k: 'unsup', t: T('이 GPU 미지원') };
  const big = isBig(c);
  const mem = d.memory / GIB, size = c.size / GIB;
  if (mem >= size + (big ? 4 : 2.5)) return { k: 'ok', t: T('이 PC에 맞음') };
  if (mem >= size + (big ? 1.5 : 1)) return { k: 'tight', t: T('빠듯함 · 문맥이 줄어듦') };
  return { k: 'no', t: T('그래픽 메모리 부족') };
}
const FITRANK = { ok: 0, tight: 1, no: 2, unsup: 3, nogpu: 4 };
function fitTag(f) {
  if (!f) return null;
  if (f.k === 'ok') return h('span', { class: 'tag' }, I('check'), f.t);
  if (f.k === 'tight') return h('span', { class: 'tag', 'data-tip': T('실행되지만 그래픽 메모리에 맞추느라 문맥 길이가 줄어듭니다'), 'data-tip-pos': 'top' }, I('info'), f.t);
  return h('span', { class: 'tag dim' }, I('alert'), f.t);
}

// ---------- 레일 · 상태 막대 ----------
const RAIL_ICON = { chat: 'chat', models: 'box', bench: 'gauge', device: 'cpu', settings: 'gear' };
document.querySelectorAll('.rail-btn').forEach(b => {
  b.innerHTML = svg(RAIL_ICON[b.dataset.view], 'l') + `<span class="lbl" aria-hidden="true">${esc(b.getAttribute('aria-label'))}</span>`;
  b.addEventListener('click', () => show(b.dataset.view));
});
function show(view) {
  closePop();
  const same = S.view === view;
  S.view = view;
  document.querySelectorAll('.rail-btn').forEach(b => {
    const on = b.dataset.view === view;
    b.classList.toggle('active', on);
    if (on) b.setAttribute('aria-current', 'page'); else b.removeAttribute('aria-current');
  });
  document.querySelectorAll('.view').forEach(v => v.classList.toggle('active', v.id === 'view-' + view));
  resetSig();
  renderAll();
  if (view === 'bench') loadResults(true);
  if (view === 'settings') loadVk();
  if (view === 'chat') { if (!same) renderMsgs(); setTimeout(() => { if (!$('#input').disabled) $('#input').focus(); }, 0); }
  try { sessionStorage.setItem('geulpu.view', view); } catch (_) { }
}
// 상태 막대는 칸마다 제자리에서 고친다 — 초점이 있는 단추를 갈아 끼우지 않게
function renderStatus() {
  const bar = $('#statusbar');
  const segs = [];
  if (!S.conn) segs.push({ key: 'off', dot: 'err', text: T('GeulPU.exe와 연결 끊김') });
  else if (S.st) {
    const st = S.st, d = st.device, sv = st.server, j = st.job;
    if (!d.checked) segs.push({ key: 'gpu', dot: 'busy', text: T('하드웨어 확인 중') });
    else if (!d.layer) segs.push({ key: 'gpu', btn: () => show('device'), tip: TM(d.reason), dot: 'err', text: T('쓸 수 있는 GPU 없음') });
    else segs.push({ key: 'gpu', btn: () => show('device'), tip: T('하드웨어 보기'), dot: 'ok', text: `${d.name} · ${LAYER[d.layer]} · ${fmtBytes(d.memory)}` });
    const svDot = { starting: 'busy', ready: 'ok', failed: 'err' }[sv.state] || '';
    const svTxt = sv.state === 'ready' ? T`${labelById(sv.model)} · 동시 ${sv.slots}개 · 문맥 ${fmtCtx(sv.context)}`
      : sv.state === 'starting' ? T`모델 불러오는 중 · ${fmtClock(sv.seconds)}` : sv.state === 'failed' ? T('모델 불러오기 실패') : T('불러온 모델 없음');
    segs.push({ key: 'srv', tip: TM(sv.reason || ''), dot: svDot, text: svTxt });
    if (j && j.state === 'running' && j.kind !== 'server' && j.kind !== 'device') segs.push({ key: 'job', btn: () => show('bench'), tip: T('벤치마크 보기'), mini: j.progress / 10, text: `${jobName(j)} ${Math.round(j.progress / 10)}%` });
    // 다운로드 — 받는 중이면 진행, 멈춤 · 실패만 남았으면 그 수. 누르면 다운로드 목록
    const all = dlList(), act = all.filter(c => c.download.state === 'downloading' || c.download.state === 'verifying');
    const openDl = ev => openDownloads(ev.currentTarget);
    if (act.length) {
      const tot = act.reduce((a, c) => a + c.download.total, 0), got = act.reduce((a, c) => a + c.download.received, 0);
      const bps = act.reduce((a, c) => a + (c.download.bytes_per_s || 0), 0);
      const fail = all.filter(c => dlStateOf(c) === 'fail').length, pause = all.length - act.length - fail;
      segs.push({ key: 'dl', btn: openDl, tip: T('다운로드 목록'), icon: 'download',
        text: T`다운로드 ${act.length}개 · ${Math.round(100 * got / Math.max(1, tot))}% · ${fmtBytes(bps)}/s${pause ? T` · 멈춤 ${pause}개` : ''}${fail ? T` · 실패 ${fail}개` : ''}` });
    } else if (all.length) {
      const fail = all.filter(c => dlStateOf(c) === 'fail').length, pause = all.length - fail;
      segs.push({ key: 'dl', btn: openDl, tip: T('다운로드 목록'), dot: fail ? 'err' : undefined, icon: fail ? null : 'pause',
        text: [T('다운로드', 'list'), pause ? T`멈춤 ${pause}개` : null, fail ? T`실패 ${fail}개` : null].filter(Boolean).join(' · ') });
    }
    segs.push({ key: 'grow', grow: true }, { key: 'ver', text: 'v' + st.app.version });
  }
  const old = new Map([...bar.children].map(e => [e.dataset.key, e]));
  let prev = null;
  for (const sp of segs) {
    const tag = sp.grow ? 'span' : sp.btn ? 'button' : 'span';
    let e = old.get(sp.key);
    if (e && e.tagName.toLowerCase() !== tag) { e.remove(); e = null; }
    if (!e) e = h(tag, { class: sp.grow ? 'grow' : 'seg', 'data-key': sp.key });
    old.delete(sp.key);
    if (!sp.grow) {
      e.onclick = sp.btn || null;
      if (sp.key === 'dl') e.setAttribute('aria-haspopup', 'dialog'); else e.removeAttribute('aria-haspopup');
      if (sp.tip) { e.dataset.tip = sp.tip; e.dataset.tipPos = 'top'; } else delete e.dataset.tip;
      const sig = JSON.stringify([sp.dot, sp.icon, sp.mini !== undefined, sp.text]);
      if (e.dataset.sig !== sig) {
        e.dataset.sig = sig;
        put(e, sp.dot !== undefined ? h('i', { class: 'dot ' + sp.dot }) : null, sp.icon ? I(sp.icon, 's') : null,
          sp.mini !== undefined ? h('span', { class: 'mini' }, h('i', { style: `width:${sp.mini}%` })) : null, h('span', null, sp.text));
      }
    }
    if (prev ? prev.nextSibling !== e : bar.firstChild !== e) { if (prev) prev.after(e); else bar.prepend(e); }
    prev = e;
  }
  for (const e of old.values()) e.remove();
}

// 새 버전 알림
function verNum(v) { return String(v || '').replace(/^v/, '').split('.').map(x => parseInt(x, 10) || 0); }
function newer(a, b) { const x = verNum(a), y = verNum(b); for (let i = 0; i < 3; i++) { if ((x[i] || 0) !== (y[i] || 0)) return (x[i] || 0) > (y[i] || 0); } return false; }
async function checkUpdate() {
  if (!S.st || !S.st.settings.update_check || !S.conn) return;
  let rel = null;
  try { const c = JSON.parse(sessionStorage.getItem('geulpu.rel') || 'null'); if (c && Date.now() - c.t < 3600e3) rel = c.r; } catch (_) { }
  if (!rel) {
    try {
      const r = await fetch('https://api.github.com/repos/wwoosshh/GeulPU/releases/latest', { headers: { Accept: 'application/vnd.github+json' } });
      if (!r.ok) return;
      const j = await r.json();
      rel = { tag: j.tag_name, url: j.html_url, name: j.name };
      try { sessionStorage.setItem('geulpu.rel', JSON.stringify({ t: Date.now(), r: rel })); } catch (_) { }
    } catch (_) { return; }
  }
  S.update = rel;
  const b = $('#banner');
  if (b.classList.contains('offline')) return;
  if (rel && newer(rel.tag, S.st.app.version)) {
    put(b, h('i', { class: 'dot acc' }), h('span', null, T`새 버전이 나왔습니다: ${rel.tag}(지금 v${S.st.app.version}). 새 폴더에 압축을 풀고 실행하면 모델과 결과는 그대로 이어집니다.`),
      h('span', { class: 'grow' }), h('a', { class: 'btn sm', href: rel.url, target: '_blank', rel: 'noopener' }, T('다운로드 페이지'), I('ext', 's')),
      h('button', { class: 'icon-btn', 'aria-label': T('알림 닫기'), onclick: () => { b.hidden = true; } }, I('x', 's')));
    b.hidden = false;
  } else b.hidden = true;
}

// ---------- 채팅 ----------
const R1_BOS = '<｜begin▁of▁sentence｜>', R1_USER = '<｜User｜>', R1_ASSIST = '<｜Assistant｜>', R1_EOS = '<｜end▁of▁sentence｜>';
const THINK_END_ID = 151649;   // R1-Distill-Qwen 의 </think> — GGUF 토큰 종류 4(사용자 정의). 서버는 글로 내지 않는다.
const OPT = Object.assign({ temp: 0, max: 4096, system: '', think: true }, load('geulpu.opt', {}));
function saveRooms() { store('geulpu.rooms', S.rooms); store('geulpu.room', S.roomId); }
function curRoom() { return S.rooms.find(r => r.id === S.roomId) || null; }
function runnable(m) { return !!(m && m.shape && m.shape.ok && !m.device_reason); }
function defaultModel() {
  const st = S.st;
  if (st && st.server.state === 'ready' && st.server.model) return st.server.model;
  const r = st ? sortedModels().filter(runnable) : [];
  return r.length ? r[0].id : '';
}
const overlayConvos = () => innerWidth <= 900;
function setConvos(open, remember) {
  $('#convos').classList.toggle('closed', !open);
  $('#show-convos').hidden = open;
  $('#scrim').hidden = !(open && overlayConvos());
  if (remember !== false && !overlayConvos()) store('geulpu.convos', open);
  if (open && overlayConvos()) setTimeout(() => $('#newroom').focus(), 0);
}
$('#scrim').addEventListener('click', () => setConvos(false));
let wasOverlay = overlayConvos();
addEventListener('resize', () => { const o = overlayConvos(); if (o !== wasOverlay) { wasOverlay = o; setConvos(o ? false : load('geulpu.convos', true), false); } });
// 답이 흐르는 동안에도 다른 채팅을 보거나 새 채팅을 열 수 있다 — 답은 제 채팅에 계속 쌓인다
function pickRoom(id) {
  S.rooms = S.rooms.filter(r => r.messages.length || r.id === id);
  S.roomId = id;
  S.editing = null;
  saveRooms(); renderRooms(); renderMsgs(); renderChatTop(true);
  if (overlayConvos()) setConvos(false);
}
function newRoom() {
  const cur = curRoom();
  if (cur && !cur.messages.length) { if (overlayConvos()) setConvos(false); $('#input').focus(); return; }
  S.rooms = S.rooms.filter(r => r.messages.length);
  const r = { id: Date.now().toString(36) + Math.random().toString(36).slice(2, 6), title: '새 채팅', model: (cur && cur.model) || defaultModel(), messages: [], created: Date.now(), updated: Date.now() };
  S.rooms.unshift(r);
  S.roomId = r.id;
  S.editing = null;
  saveRooms();
  renderRooms();
  renderMsgs();
  renderChatTop(true);
  if (overlayConvos()) setConvos(false);
  $('#input').focus();
}
put($('#newroom'), I('plus', 's'), h('span', null, T('새 채팅')));
$('#newroom').addEventListener('click', newRoom);
$('#hide-convos').innerHTML = svg('panel', 's');
$('#show-convos').innerHTML = svg('panel', 's');
$('#hide-convos').addEventListener('click', () => setConvos(false));
$('#show-convos').addEventListener('click', () => setConvos(true));
function dayGroup(t) {
  const d0 = new Date(); d0.setHours(0, 0, 0, 0);
  const day = 864e5;
  if (t >= d0.getTime()) return T('오늘');
  if (t >= d0.getTime() - day) return T('어제');
  if (t >= d0.getTime() - 7 * day) return T('지난 7일');
  return T('이전');
}
function renderRooms() {
  const list = $('#roomlist');
  const shown = S.rooms.filter(r => r.messages.length).sort((a, b) => (b.updated || b.created) - (a.updated || a.created));
  const out = [];
  let g = null;
  for (const r of shown) {
    const gg = dayGroup(r.updated || r.created);
    if (gg !== g) { g = gg; out.push(h('div', { class: 'room-group eyebrow' }, gg)); }
    const busy = S.streaming && S.streaming.roomId === r.id;
    out.push(h('div', { class: 'room' + (r.id === S.roomId ? ' active' : '') },
      h('button', { class: 'pick', 'data-tip': r.title.length > 18 ? r.title : null, 'aria-current': r.id === S.roomId ? 'true' : null, onclick: () => pickRoom(r.id) },
        busy ? h('i', { class: 'dot busy', 'aria-hidden': 'true' }) : null, h('span', null, r.title), busy ? h('span', { class: 'sr' }, T(' — 답을 쓰는 중')) : null),
      h('button', {
        class: 'icon-btn more', 'aria-label': T`'${r.title}' 메뉴`, 'aria-haspopup': 'menu', 'aria-expanded': 'false',
        onclick: ev => menu(ev.currentTarget, T('채팅 메뉴'), [
          { icon: 'edit', label: T('이름 바꾸기'), run: () => renameRoom(r) },
          { icon: 'trash', label: T('삭제'), danger: true, disabled: busy, sub: busy ? T('답이 끝난 뒤 삭제할 수 있습니다') : null, run: () => delRoom(r.id) }], 'right'),
      }, I('dots', 's'))));
  }
  if (!shown.length) out.push(h('div', { class: 'room-group eyebrow' }, T('아직 채팅이 없습니다')));
  put(list, out);
}
function renameRoom(r) {
  const inp = h('input', { type: 'text', value: r.title, class: 'w100', 'aria-label': T('채팅 이름') });
  const ok = () => { const v = inp.value.trim(); if (v) { r.title = v; saveRooms(); renderRooms(); } closeModal(); };
  inp.addEventListener('keydown', e => { if (e.key === 'Enter') ok(); });
  modal(h('div', { class: 'col g16' }, h('h2', null, T('채팅 이름 바꾸기')), inp), [h('button', { class: 'btn ghost', onclick: () => closeModal() }, T('취소')), h('button', { class: 'btn primary', onclick: ok }, T('저장'))], false, 'input');
}
async function delRoom(id) {
  if (S.streaming && S.streaming.roomId === id) return;
  const r = S.rooms.find(x => x.id === id);
  if (r && r.messages.length && !(await confirmBox(T('채팅 삭제'), h('p', { class: 'muted m0' }, T`'${r.title}' 채팅을 삭제할까요? 채팅은 이 브라우저에만 저장되어 있어 되돌릴 수 없습니다.`), T('삭제'), 'danger'))) return;
  S.rooms = S.rooms.filter(x => x.id !== id);
  if (S.roomId === id) { S.roomId = null; newRoom(); }
  saveRooms(); renderRooms(); renderMsgs(); renderChatTop(true);
}

// 입력창: 보낼 수 없는 상태면 막고 이유를 자리 글로
function updateComposer() {
  const inp = $('#input');
  const hs = helloState(S.st);
  let why = '';
  if (!S.conn) why = T('GeulPU.exe와 연결이 끊겼습니다');
  else if (hs === 'checking' || hs === 'loading') why = T('하드웨어 확인이 끝나면 입력할 수 있습니다');
  else if (hs === 'nogpu') why = T('쓸 수 있는 GPU가 없어 채팅할 수 없습니다');
  else if (hs === 'nomodel' || hs === 'unsupported') why = T('모델을 다운로드하면 채팅을 시작할 수 있습니다');
  inp.disabled = !!why;
  inp.placeholder = why || T('메시지 입력');
  updateSend();
}
const HINT = T('Enter 보내기 · Shift+Enter 줄바꿈');
// 보내기 단추 · 도움말 — 이 채팅이 답을 쓰는 중이면 중지, 다른 채팅이 쓰는 중이면 끝날 때까지 기다림
function updateSend() {
  const b = $('#sendbtn'), inp = $('#input'), hint = $('#composer-hint');
  const here = streamingHere(), other = !!S.streaming && !here;
  b.classList.toggle('stop', here);
  const icon = here ? 'stop' : 'up';
  if (b.dataset.icon !== icon) { b.innerHTML = svg(icon, 's'); b.dataset.icon = icon; }
  b.setAttribute('aria-label', here ? T('중지') : T('보내기'));
  const bench = !here && !other && benchBusy();   // 벤치마크 중에도 미리 써 둘 수는 있다
  const lab = !here && !other && !bench && !!LAB.busy;   // 연구 패널이 셈하는 동안(모델을 바꿔 불러올 수 있다)
  b.dataset.tip = here ? T('중지(Esc)') : other ? T('다른 채팅의 답이 끝나면 보낼 수 있습니다') : bench ? T('벤치마크가 끝나면 보낼 수 있습니다') : lab ? T('연구 패널의 셈이 끝나면 보낼 수 있습니다') : T('보내기(Enter)');
  b.dataset.tipPos = 'top';
  b.disabled = here ? false : (other || bench || lab || !inp.value.trim() || !S.conn || inp.disabled);
  hint.textContent = here ? T('답이 끝나면 보낼 수 있습니다 · Esc 중지') : other ? T('다른 채팅의 답이 끝나면 보낼 수 있습니다') : bench ? T('벤치마크가 끝나면 보낼 수 있습니다') : lab ? T('연구 패널의 셈이 끝나면 보낼 수 있습니다') : HINT;
  hint.hidden = inp.disabled || (!S.streaming && !bench && !!load('geulpu.hinted', false));
}
function renderChatTop(force) {
  const room = curRoom();
  const st = S.st;
  if (!st) return;
  if (room && (!room.model || !st.models.some(m => m.id === room.model)) && st.models.length) { room.model = defaultModel(); saveRooms(); }
  const m = room && modelOf(room.model);
  const sv = st.server, d = st.device;
  if (!changed('chattop', [room && room.model, m && [displayName(m), quantOf(m), m.device_reason, runnable(m)], sv.state, sv.model, sv.layer, st.models.length, d.checked, d.layer, S.conn, !!S.streaming, benchBusy()]) && !force) return;
  put($('#model-pill'), h('span', { class: 'name' }, m ? displayName(m) : (st.models.length ? T('모델 선택') : T('모델 없음'))), m && quantOf(m) ? h('span', { class: 'q' }, quantOf(m)) : null, I('chevD', 's'));
  const ms = $('#model-state');
  delete ms.dataset.tip;
  if (!S.conn) put(ms, h('i', { class: 'dot err' }), h('span', null, T('연결 끊김')));
  else if (!d.checked) put(ms, h('i', { class: 'dot busy' }), h('span', null, T('하드웨어 확인 중')));
  else if (benchBusy() && !(sv.model === (m && m.id) && sv.state === 'ready')) put(ms, h('i', { class: 'dot busy' }), h('span', null, T('벤치마크 실행 중 · 끝나면 보낼 수 있습니다')));
  else if (!d.layer) put(ms, h('i', { class: 'dot err' }), h('span', null, T('쓸 수 있는 GPU 없음')));
  else if (!m) put(ms);
  else if (sv.model === m.id && sv.state === 'ready') put(ms, h('i', { class: 'dot ok' }), h('span', null, T`준비됨 · ${LAYER[sv.layer] || ''}`));
  else if (sv.model === m.id && sv.state === 'starting') put(ms, h('i', { class: 'dot busy' }), h('span', null, T('불러오는 중')));
  else if (!runnable(m)) put(ms, h('i', { class: 'dot err' }), h('span', null, T('이 PC에서 실행할 수 없음')));
  else if (sv.state === 'failed') put(ms, h('i', { class: 'dot err' }), h('span', null, T('불러오기 실패 · 보내면 다시 시도합니다')));
  else if (sv.state === 'ready' || sv.state === 'starting') { put(ms, h('i', { class: 'dot' }), h('span', null, T('보내면 이 모델로 바뀝니다'))); ms.dataset.tip = T`지금 불러온 모델: ${labelById(sv.model)}`; ms.dataset.tipPos = 'bottom'; }
  else put(ms, h('i', { class: 'dot' }), h('span', null, T('보내면 불러옵니다')));
  updateComposer();
}
$('#model-pill').addEventListener('click', e => {
  const st = S.st;
  if (!st) return;
  if (POP) { closePop(); return; }
  const room = curRoom();
  const sv = st.server;
  const locked = streamingHere();
  const groups = [];
  for (const m of sortedModels()) {
    const name = displayName(m);
    let g = groups[groups.length - 1];
    if (!g || g.name !== name) { g = { name, items: [] }; groups.push(g); }
    g.items.push(m);
  }
  const body = [];
  if (locked) body.push(h('div', { class: 'pad small muted' }, T('답이 끝나면 모델을 바꿀 수 있습니다.')));
  // 모델마다 머리(이름 · 종류) 아래에 양자화 줄 — 이 PC 에서 도는 모델이 먼저
  groups.sort((a, b) => (a.items.some(runnable) ? 0 : 1) - (b.items.some(runnable) ? 0 : 1));
  for (const g of groups) {
    const g0 = g.items[0];
    body.push(h('div', { role: 'group', 'aria-label': g.name },
      h('div', { class: 'mhd', 'aria-hidden': 'true' }, h('span', { class: 'nm' }, g.name), h('span', { class: 'kd' }, kindOf(g0))),
      g.items.map(m => {
        const ok = runnable(m);
        const sel = room && room.model === m.id;
        const loaded = sv.model === m.id && sv.state === 'ready';
        const reason = m.device_reason || (m.shape && m.shape.reason) || '';
        const label = `${g.name} ${quantOf(m)}, ${fmtBytes(m.size)}${loaded ? T(', 불러옴') : ''}${ok ? '' : T(', 이 PC에서 실행할 수 없음: ') + TM(reason)}`;
        return h('button', {
          class: 'opt q' + (sel ? ' sel' : ''), role: 'menuitemradio', 'aria-checked': sel ? 'true' : 'false', 'aria-label': label, disabled: (ok && !locked) ? null : true,
          onclick: () => { closePop(); if (!room) return; room.model = m.id; saveRooms(); renderChatTop(true); renderMsgs(); $('#input').focus(); },
        }, h('span', { class: 'qlab' }, quantOf(m)), h('span', { class: 'sz' }, fmtBytes(m.size)),
          ok ? null : h('span', { class: 'why' }, m.device_reason ? T('이 GPU 미지원') : T('지원하지 않는 파일')),
          h('span', { class: 'grow' }), loaded ? h('span', { class: 'tag' }, h('i', { class: 'dot ok' }), T('불러옴')) : null, sel ? I('check', 's') : null);
      })));
  }
  if (!st.models.length) body.push(h('div', { class: 'pad muted' }, T('아직 모델이 없습니다.')));
  body.push(h('div', { class: 'sep', role: 'separator' }), h('button', { class: 'opt', role: 'menuitem', onclick: () => { closePop(); show('models'); } }, I('download', 's'), h('span', { class: 'main' }, h('div', { class: 't' }, T('모델 다운로드 · 추가')))));
  if (sv.state === 'ready' || sv.state === 'starting') body.push(h('button', { class: 'opt', role: 'menuitem', disabled: S.streaming ? true : null, onclick: () => { closePop(); unloadModel(); } }, I('eject', 's'), h('span', { class: 'main' }, h('div', { class: 't' }, T('불러온 모델 내리기')), h('div', { class: 's' }, T('GPU 메모리를 비웁니다')))));
  openPop(e.currentTarget, h('div', { role: 'menu', 'aria-label': T('모델 선택') }, body));
});
$('#optbtn').innerHTML = svg('sliders');
$('#optbtn').addEventListener('click', e => {
  if (POP) { closePop(); return; }
  const temp = h('input', { type: 'number', min: '0', max: '2', step: '0.1', value: OPT.temp, 'aria-label': T('온도') });
  const max = h('input', { type: 'number', min: '16', max: '32768', step: '256', value: OPT.max, 'aria-label': T('최대 답변 길이(토큰)') });
  const sys = h('textarea', { rows: '4', placeholder: T('비워 두면 모델 기본값'), 'aria-label': T('시스템 프롬프트'), class: 'w100' }, OPT.system || '');
  const think = h('input', { type: 'checkbox', 'aria-label': T('생각 과정 쓰기(Qwen3)'), onchange: () => { OPT.think = think.checked; store('geulpu.opt', OPT); } });
  think.checked = OPT.think !== false;
  // 쓰는 동안은 그대로 두고, 칸을 떠날 때(또는 Enter) 범위에 맞춰 칸에도 다시 적는다(온도 0~2 · 길이 16~32768)
  const clampNum = (v, lo, hi, d) => { const n = Number(v); return Number.isFinite(n) ? Math.min(hi, Math.max(lo, n)) : d; };
  const save = () => { OPT.temp = clampNum(parseFloat(temp.value), 0, 2, 0); OPT.max = Math.round(clampNum(parseInt(max.value, 10), 16, 32768, 4096)); OPT.system = sys.value; store('geulpu.opt', OPT); };
  const settle = () => { save(); temp.value = OPT.temp; max.value = OPT.max; };
  [temp, max, sys].forEach(x => x.addEventListener('input', save));
  [temp, max].forEach(x => { x.addEventListener('change', settle); x.addEventListener('keydown', ev => { if (ev.key === 'Enter') settle(); }); });
  openPop(e.currentTarget, h('div', { class: 'pad w360', role: 'dialog', 'aria-label': T('생성 설정') },
    h('div', { class: 'eyebrow' }, T('생성 설정 · 이 브라우저에 저장')),
    h('div', { class: 'lrow' }, h('div', null, h('div', { class: 'ink' }, T('온도')), h('div', { class: 'small muted' }, T('0이면 같은 질문에 늘 같은 답'))), temp),
    h('div', { class: 'lrow' }, h('div', null, h('div', { class: 'ink' }, T('최대 답변 길이')), h('div', { class: 'small muted' }, T('토큰 · 추론 모델은 생각 과정도 포함'))), max),
    h('div', { class: 'lrow' }, h('div', null, h('div', { class: 'ink' }, T('생각 과정 쓰기')), h('div', { class: 'small muted' }, T('Qwen3 · 끄면 생각 없이 바로 답합니다'))), h('label', { class: 'switch' }, think, h('i'))),
    h('div', { class: 'col' }, h('div', { class: 'ink' }, T('시스템 프롬프트')), sys)), 'right');
});

// ---------- 마크다운 · 수식(앱 안에서 — 바깥 라이브러리 없음, 글은 늘 탈출해서 넣는다) ----------
const TEXSYM = {
  times: '×', cdot: '·', div: '÷', pm: '±', mp: '∓', leq: '≤', le: '≤', geq: '≥', ge: '≥', neq: '≠', ne: '≠', approx: '≈', equiv: '≡',
  sim: '∼', simeq: '≃', cong: '≅', propto: '∝', infty: '∞', to: '→', rightarrow: '→', leftarrow: '←', Rightarrow: '⇒', Leftarrow: '⇐',
  leftrightarrow: '↔', Leftrightarrow: '⇔', iff: '⟺', implies: '⟹', mapsto: '↦', ldots: '…', cdots: '⋯', dots: '…', vdots: '⋮',
  sum: '∑', prod: '∏', int: '∫', oint: '∮', partial: '∂', nabla: '∇', forall: '∀', exists: '∃', in: '∈', notin: '∉', ni: '∋',
  subset: '⊂', subseteq: '⊆', supset: '⊃', supseteq: '⊇', cup: '∪', cap: '∩', emptyset: '∅', varnothing: '∅', angle: '∠', circ: '∘',
  degree: '°', perp: '⊥', parallel: '∥', mid: '∣', nmid: '∤', lfloor: '⌊', rfloor: '⌋', lceil: '⌈', rceil: '⌉', langle: '⟨', rangle: '⟩',
  neg: '¬', lnot: '¬', land: '∧', lor: '∨', wedge: '∧', vee: '∨', oplus: '⊕', otimes: '⊗', star: '⋆', ast: '∗', bullet: '•', prime: '′',
  hbar: 'ℏ', ell: 'ℓ', aleph: 'ℵ', therefore: '∴', because: '∵', checkmark: '✓',
  alpha: 'α', beta: 'β', gamma: 'γ', delta: 'δ', epsilon: 'ϵ', varepsilon: 'ε', zeta: 'ζ', eta: 'η', theta: 'θ', vartheta: 'ϑ', iota: 'ι',
  kappa: 'κ', lambda: 'λ', mu: 'μ', nu: 'ν', xi: 'ξ', pi: 'π', varpi: 'ϖ', rho: 'ρ', varrho: 'ϱ', sigma: 'σ', varsigma: 'ς', tau: 'τ',
  upsilon: 'υ', phi: 'ϕ', varphi: 'φ', chi: 'χ', psi: 'ψ', omega: 'ω', Gamma: 'Γ', Delta: 'Δ', Theta: 'Θ', Lambda: 'Λ', Xi: 'Ξ', Pi: 'Π',
  Sigma: 'Σ', Upsilon: 'Υ', Phi: 'Φ', Psi: 'Ψ', Omega: 'Ω',
  quad: '\u2003', qquad: '\u2003\u2003', ',': '\u2009', ';': '\u2005', ':': '\u2005', '>': '\u2005', '!': '', ' ': ' ',
  '{': '{', '}': '}', '%': '%', '$': '$', '#': '#', '&': '&amp;', '_': '_', '|': '‖',
};
const TEXFN = new Set(['sin', 'cos', 'tan', 'cot', 'sec', 'csc', 'arcsin', 'arccos', 'arctan', 'sinh', 'cosh', 'tanh', 'log', 'ln', 'lg', 'exp', 'lim',
  'max', 'min', 'sup', 'inf', 'det', 'gcd', 'lcm', 'deg', 'dim', 'ker', 'arg', 'Pr', 'mod', 'bmod']);
function tex(src) {
  const s = String(src);
  let i = 0;
  const group = () => {   // '{' 다음부터 짝 '}' 까지
    let depth = 1;
    const a = i;
    while (i < s.length && depth) { if (s[i] === '\\') { i += 2; continue; } if (s[i] === '{') depth++; else if (s[i] === '}') depth--; i++; }
    return s.slice(a, depth ? i : i - 1);
  };
  const raw = () => { while (s[i] === ' ') i++; if (s[i] === '{') { i++; return group(); } return s[i++] || ''; };
  const opt = () => { if (s[i] !== '[') return null; const j = s.indexOf(']', i); if (j < 0) return null; const v = s.slice(i + 1, j); i = j + 1; return v; };
  const arg = () => {
    while (s[i] === ' ') i++;
    if (s[i] === '{') { i++; return tex(group()); }
    if (s[i] === '\\') { const m = /^\\([A-Za-z]+|.)/.exec(s.slice(i)); if (!m) { i++; return ''; } i += m[0].length; return cmd(m[1]); }
    return i < s.length ? chr(s[i++]) : '';
  };
  const chr = c => (/[A-Za-z]/.test(c) ? `<i>${c}</i>` : c === '&' ? ' ' : c === '~' ? '\u00a0' : esc(c));
  function cmd(n) {
    switch (n) {
      case 'frac': case 'dfrac': case 'tfrac': case 'cfrac': { const a = arg(), b = arg(); return `<span class="frac"><span>${a}</span><span>${b}</span></span>`; }
      case 'binom': case 'dbinom': case 'tbinom': { const a = arg(), b = arg(); return `(<span class="frac nb"><span>${a}</span><span>${b}</span></span>)`; }
      case 'sqrt': { const k = opt(); const a = arg(); return (k ? `<sup>${esc(k)}</sup>` : '') + `√<span class="ov">${a}</span>`; }
      case 'boxed': case 'fbox': case 'framebox': return `<span class="boxed">${arg()}</span>`;
      case 'text': case 'textrm': case 'textnormal': case 'mathrm': case 'operatorname': case 'textit': case 'mbox': case 'textsf': case 'texttt':
        return `<span class="tx">${esc(raw())}</span>`;
      case 'textbf': case 'mathbf': case 'boldsymbol': case 'bm': return `<b>${arg()}</b>`;
      case 'mathit': case 'emph': return arg();
      case 'mathbb': { const a = raw(); return esc({ R: 'ℝ', N: 'ℕ', Z: 'ℤ', Q: 'ℚ', C: 'ℂ', P: 'ℙ' }[a] || a); }
      case 'mathcal': case 'mathscr': case 'mathfrak': case 'mathsf': case 'mathtt': return arg();
      case 'overline': case 'bar': return `<span class="ov">${arg()}</span>`;
      case 'hat': case 'widehat': return arg() + '\u0302';
      case 'vec': return arg() + '\u20d7';
      case 'dot': return arg() + '\u0307';
      case 'ddot': return arg() + '\u0308';
      case 'tilde': case 'widetilde': return arg() + '\u0303';
      case 'pmod': return `\u2005(<span class="fn">mod</span> ${arg()})`;
      case 'not': return arg() + '\u0338';
      case 'left': case 'right': case 'big': case 'Big': case 'bigg': case 'Bigg': case 'bigl': case 'bigr': case 'Bigl': case 'Bigr':
      case 'displaystyle': case 'textstyle': case 'scriptstyle': case 'limits': case 'nolimits': case 'nonumber': case 'notag': return '';
      case 'begin': case 'end': raw(); if (s[i] === '{') { i++; group(); } return '';
      case '\\': return '<br>';
      default:
        if (TEXFN.has(n)) return `<span class="fn">${n === 'bmod' ? 'mod' : n}</span>`;
        if (n in TEXSYM) return TEXSYM[n];
        return `<span class="tx">${esc(n)}</span>`;
    }
  }
  let out = '';
  while (i < s.length) {
    const c = s[i];
    if (c === '\\') { const m = /^\\([A-Za-z]+|.)/.exec(s.slice(i)); if (!m) { i++; continue; } i += m[0].length; out += cmd(m[1]); continue; }
    if (c === '^') { i++; out += `<sup>${arg()}</sup>`; continue; }
    if (c === '_') { i++; out += `<sub>${arg()}</sub>`; continue; }
    if (c === '{') { i++; out += tex(group()); continue; }
    if (c === '}') { i++; continue; }
    out += chr(c);
    i++;
  }
  return out;
}
const mathInline = t => `<span class="math">${tex(t)}</span>`;
const mathBlock = t => `<div class="math block">${tex(t)}</div>`;
function codeBlock(code, lang) {
  return T`<div class="codeblk"><div class="ch"><span>${esc(lang || T('코드'))}</span><button class="cp" type="button">${svg('copy', 's')}<span>복사</span></button></div><pre><code>${esc(code)}</code></pre></div>`;
}
function inline(src) {
  const slots = [];
  const hold = html => { slots.push(html); return '\u0000' + (slots.length - 1) + '\u0000'; };
  let s = String(src);
  s = s.replace(/`([^`\n]+)`/g, (_, c) => hold('<code class="inl">' + esc(c) + '</code>'));
  s = s.replace(/\\\((.+?)\\\)/g, (_, t) => hold(mathInline(t)));
  s = s.replace(/\\\[(.+?)\\\]/g, (_, t) => hold(mathInline(t)));
  s = s.replace(/\$\$(.+?)\$\$/g, (_, t) => hold(mathInline(t)));
  s = s.replace(/(^|[^\\$\w])\$([^\s$](?:[^$\n]*?[^\s$])?)\$(?![\w$])/g, (all, pre, t) =>
    (/[\\^_={}]/.test(t) || /^[A-Za-z]{1,2}$/.test(t) || (/\d/.test(t) && /[A-Za-z+\-*/<>]/.test(t))) ? pre + hold(mathInline(t)) : all);
  s = s.replace(/\[([^\]\n]+)\]\((https?:\/\/[^\s)]+)\)/g, (_, t, u) => hold(`<a href="${esc(u)}" target="_blank" rel="noopener noreferrer">${esc(t)}</a>`));
  s = esc(s);
  s = s.replace(/\*\*([^*\n]+)\*\*/g, '<b>$1</b>').replace(/__([^_\n]+)__/g, '<b>$1</b>')
    .replace(/(^|[\s(])\*([^*\n]+)\*(?=[\s).,!?:;]|$)/g, '$1<i>$2</i>')
    .replace(/~~([^~\n]+)~~/g, '<s>$1</s>');
  return s.replace(/\u0000(\d+)\u0000/g, (_, n) => slots[+n]);
}
const inlineLines = t => t.split('\n').map(inline).join('<br>');
// 목록 항목 안에 여러 줄 수식(\[ … \]) · 코드 블록이 있으면 블록으로 그린다
const itemHtml = t => (/(^|\n)\s*(\\\[|\$\$|```|~~~)/.test(t) ? md(t) : inlineLines(t));
const LIST_RE = /^(\s*)([-*+]|(\d+)[.)])\s+(.*)$/;
function parseList(lines, i) {
  const items = [];
  while (i < lines.length) {
    const ln = lines[i];
    const m = LIST_RE.exec(ln);
    if (m) { items.push({ ind: m[1].replace(/\t/g, '    ').length, ord: !!m[3], num: m[3] ? parseInt(m[3], 10) : 0, text: m[4] }); i++; continue; }
    if (!ln.trim()) {
      let j = i + 1;
      while (j < lines.length && !lines[j].trim()) j++;
      if (j < lines.length && (LIST_RE.test(lines[j]) || /^\s{2,}\S/.test(lines[j]))) { i = j; continue; }
      break;
    }
    if (items.length && (/^\s{2,}\S/.test(ln) || !/^\s*(#|>|```|~~~|\||\\\[|\$\$)/.test(ln))) { items[items.length - 1].text += '\n' + ln.trim(); i++; continue; }
    break;
  }
  return { html: buildList(items, 0, items.length), next: i };
}
function buildList(items, a, b) {
  let html = '';
  let k = a;
  while (k < b) {
    const base = items[k].ind, ord = items[k].ord;
    html += ord ? (items[k].num > 1 ? `<ol start="${items[k].num}">` : '<ol>') : '<ul>';
    while (k < b && items[k].ind < base + 2 && items[k].ord === ord) {
      let c = k + 1;
      while (c < b && items[c].ind >= base + 2) c++;
      html += '<li>' + itemHtml(items[k].text) + (c > k + 1 ? buildList(items, k + 1, c) : '') + '</li>';
      k = c;
    }
    html += ord ? '</ol>' : '</ul>';
  }
  return html;
}
function md(text) {
  const lines = String(text || '').replace(/\r\n?/g, '\n').split('\n');
  let out = '';
  const para = [];
  const flush = () => { if (para.length) { out += '<p>' + para.map(inline).join('<br>') + '</p>'; para.length = 0; } };
  let i = 0;
  while (i < lines.length) {
    const ln = lines[i];
    let m;
    if ((m = /^\s*(```+|~~~+)\s*([\w#+.-]*)\s*$/.exec(ln))) {
      flush();
      const fence = m[1], buf = [];
      i++;
      while (i < lines.length && !lines[i].trim().startsWith(fence)) { buf.push(lines[i]); i++; }
      i++;
      out += codeBlock(buf.join('\n'), m[2]);
      continue;
    }
    if ((m = /^\s*(\\\[|\$\$)(.*)$/.exec(ln))) {
      flush();
      const close = m[1] === '\\[' ? '\\]' : '$$';
      const first = m[2];
      const j0 = first.indexOf(close);
      if (j0 >= 0) {
        out += mathBlock(first.slice(0, j0));
        const rest = first.slice(j0 + close.length).trim();
        if (rest) para.push(rest);
        i++;
        continue;
      }
      const buf = [first];
      i++;
      while (i < lines.length && lines[i].indexOf(close) < 0) { buf.push(lines[i]); i++; }
      if (i < lines.length) { buf.push(lines[i].slice(0, lines[i].indexOf(close))); i++; }
      out += mathBlock(buf.join(' '));
      continue;
    }
    if (!ln.trim()) { flush(); i++; continue; }
    if ((m = /^\s{0,3}(#{1,6})\s+(.*?)\s*#*\s*$/.exec(ln))) { flush(); out += `<h4>${inline(m[2])}</h4>`; i++; continue; }
    if (/^\s{0,3}([-*_])(\s*\1){2,}\s*$/.test(ln)) { flush(); out += '<hr>'; i++; continue; }
    if (/^\s{0,3}>/.test(ln)) {
      flush();
      const buf = [];
      while (i < lines.length && /^\s{0,3}>/.test(lines[i])) { buf.push(lines[i].replace(/^\s{0,3}>\s?/, '')); i++; }
      out += '<blockquote>' + md(buf.join('\n')) + '</blockquote>';
      continue;
    }
    if (ln.includes('|') && i + 1 < lines.length && /^\s*\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)*\|?\s*$/.test(lines[i + 1])) {
      flush();
      const cells = s => s.trim().replace(/^\|/, '').replace(/\|$/, '').split('|').map(x => x.trim());
      const head = cells(ln);
      const al = cells(lines[i + 1]).map(x => (x.startsWith(':') && x.endsWith(':') ? 'center' : x.endsWith(':') ? 'right' : ''));
      // 짧은 칸(18자 이하)은 한 줄로 — 좁으면 표가 가로로 스크롤한다(낱말 · 음절 단위로 부서지지 않게)
      const at = (k, t) => (al[k] ? ` style="text-align:${al[k]}"` : '') + (String(t || '').length <= 18 ? ' class="nw"' : '');
      i += 2;
      const rows = [];
      while (i < lines.length && lines[i].includes('|') && lines[i].trim()) { rows.push(cells(lines[i])); i++; }
      out += '<div class="tbl"><table><thead><tr>' + head.map((c, k) => `<th${at(k, c)}>${inline(c)}</th>`).join('') + '</tr></thead><tbody>'
        + rows.map(r => '<tr>' + head.map((_, k) => `<td${at(k, r[k])}>${inline(r[k] || '')}</td>`).join('') + '</tr>').join('') + '</tbody></table></div>';
      continue;
    }
    if (LIST_RE.test(ln)) { flush(); const r = parseList(lines, i); out += r.html; i = r.next; continue; }
    para.push(ln);
    i++;
  }
  flush();
  return out;
}
// 흐르는 중: 아직 닫히지 않은 수식(\( \[ $$)은 닫힐 때까지 그리지 않는다(날 TeX 가 잠깐 보이지 않게)
function holdOpenMath(t) {
  const last = (open, close) => { const a = t.lastIndexOf(open); return a >= 0 && t.indexOf(close, a + open.length) < 0 ? a : -1; };
  let cut = -1;
  for (const [o, c] of [['\\(', '\\)'], ['\\[', '\\]']]) { const a = last(o, c); if (a >= 0 && (cut < 0 || a < cut)) cut = a; }
  const dd = t.split('$$').length - 1;
  if (dd % 2 === 1) { const a = t.lastIndexOf('$$'); if (cut < 0 || a < cut) cut = a; }
  return cut >= 0 ? t.slice(0, cut) : t;
}
$('#msgs').addEventListener('click', async e => {
  const b = e.target.closest('.codeblk .cp');
  if (!b) return;
  if (await copyText(b.closest('.codeblk').querySelector('code').textContent)) {
    b.lastChild.textContent = T('복사함');
    setTimeout(() => { b.lastChild.textContent = T('복사'); }, 1200);
  }
});

// R1 의 생각 · 답 나누기. 서버는 <think> · </think>(토큰 종류 4)를 글로 내지 않아 "\n</think>\n\n" 자리에 줄바꿈 셋이 남는다 —
// 교차검증 응답 64개를 토큰 번호로 풀어 보니 </think> 가 있는 40개 모두 첫 "\n\n\n" 이 바로 그 경계였다(다른 곳에 먼저 나온 것 0).
// Qwen3 — 화면이 geulpu.special 로 <think> · </think> 를 글로 받으므로 토큰 경계(</think>, 151668) 그대로 나눈다. 생각 끔이면
// 답이 <think> 없이 바로 온다(채팅 틀이 빈 생각을 프롬프트에 넣는다).
function splitThinkQwen3(t0, done) {
  if (!t0.startsWith('<think>')) return { think: null, answer: t0 };
  const t = t0.slice(7);
  const k = t.indexOf('</think>');
  if (k >= 0) return { think: t.slice(0, k).trim() || null, answer: t.slice(k + 8).replace(/^\s+/, '') };
  if (!done) return { think: t.trim(), answer: '', open: true };
  return { think: t.trim(), answer: '', unfinished: true };
}
function splitThink(m, done) {
  const t0 = String(m.content || '');
  if (m.template === 'qwen3') return splitThinkQwen3(t0, done);
  if (m.template !== 'deepseek-r1') return { think: null, answer: t0 };
  let t = t0.startsWith('<think>') ? t0.slice(7) : t0;
  let k = t.indexOf('</think>');
  if (k >= 0) return { think: t.slice(0, k).trim(), answer: t.slice(k + 8).replace(/^\s+/, '') };
  k = t.indexOf('\n\n\n');
  if (k >= 0) return { think: t.slice(0, k).trim(), answer: t.slice(k).replace(/^\s+/, '') };
  if (!done) return { think: t.trim(), answer: '', open: true };
  if (m.thinkEnded) return { think: null, answer: t.replace(/^\s+/, '') };
  return { think: t.trim(), answer: '', unfinished: true };
}
// 같은 이름에서 더 작은(설치되고 실행되는) 양자화
function smallerSibling(m) {
  if (!m || !S.st) return null;
  const name = displayName(m);
  return sortedModels().filter(x => x.id !== m.id && displayName(x) === name && x.size < m.size && runnable(x)).sort((a, b) => b.size - a.size)[0] || null;
}
// 채팅 오류를 쉬운 말로 — [제목, 도움말, 단추들]
function humanError(msg, room) {
  const s = String(msg || '');
  const m = room && modelOf(room.model);
  if (/out of memory|OUT_OF_MEMORY|메모리가 (모자|부족)|ERROR_OUT_OF_DEVICE_MEMORY/i.test(s)) {
    const sib = smallerSibling(m);
    if (sib) return [T('그래픽 메모리가 부족합니다'), T`같은 모델의 더 작은 파일(${quantOf(sib)} · ${fmtBytes(sib.size)})로 바꾸면 실행될 수 있습니다.`,
      [h('button', { class: 'btn sm', onclick: () => { room.model = sib.id; saveRooms(); renderChatTop(true); retryLast(); } }, I('refresh', 's'), T`${quantOf(sib)} 파일로 다시 시도`)]];
    return [T('그래픽 메모리가 부족합니다'), T('설정에서 동시 처리 수나 채팅마다 문맥 길이를 줄이거나, 더 작은 모델을 쓰세요.'),
      [h('button', { class: 'btn sm', onclick: () => show('settings') }, I('gear', 's'), T('설정 열기')), h('button', { class: 'btn sm ghost', onclick: () => $('#model-pill').click() }, I('box', 's'), T('모델 바꾸기'))]];
  }
  const retry = h('button', { class: 'btn sm needs-conn', onclick: () => retryLast() }, I('refresh', 's'), T('다시 시도'));
  const swap = h('button', { class: 'btn sm ghost', onclick: () => $('#model-pill').click() }, I('box', 's'), T('모델 바꾸기'));
  if (/Failed to fetch|NetworkError|연결이 끊/.test(s)) return [T('GeulPU.exe와 연결이 끊겼습니다'), T('GeulPU.exe가 실행 중인지 확인한 뒤 다시 시도하세요.'), [retry]];
  if (/다른 (일이 돌고|작업이 실행 중)/.test(s)) return [T('다른 작업이 실행 중입니다'), T('벤치마크가 끝나거나 중지된 뒤 다시 시도하세요.'), [retry, h('button', { class: 'btn sm ghost', onclick: () => show('bench') }, I('gauge', 's'), T('벤치마크 보기'))]];
  if (/모델을 불러오지|추론 서버|서버/.test(s)) return [T('모델을 불러오지 못했습니다'), T('자세히를 열어 원인을 확인하세요. 다른 모델로 바꾸거나 다시 시도할 수 있습니다.'), [retry, swap]];
  if (/HTTP 503/.test(s)) return [T('모델이 아직 준비되지 않았습니다'), T('잠시 뒤 다시 시도하세요.'), [retry]];
  return [T('답을 받지 못했습니다'), T('다시 시도하거나 다른 모델로 바꿔 보세요.'), [retry, swap]];
}
// 벤치마크 실패 까닭 — 아는 꼴이면 쉬운 말로(원문은 자세히에)
function humanJob(reason) {
  const s = String(reason || '');
  if (/out of memory|OUT_OF_MEMORY|ERROR_OUT_OF_DEVICE_MEMORY|들어가지 않습니다/i.test(s)) return [T('그래픽 메모리가 부족합니다'), T('더 작은 모델로 측정하거나 다른 프로그램의 GPU 사용을 줄여 보세요.')];
  if (/DEVICE_LOST|장치를 잃/.test(s)) return [T('GPU 드라이버가 작업을 멈췄습니다'), T('그래픽 드라이버를 최신으로 업데이트한 뒤 다시 실행해 보세요.')];
  if (/중지/.test(s)) return [T('중지했습니다'), ''];
  return [T('측정을 끝내지 못했습니다'), T('자세히를 열어 원인을 확인하세요.')];
}
// live: 새로 생긴 오류만 화면 읽기 프로그램에 알린다(다시 그릴 때마다 지난 오류를 또 읽지 않게)
function errorCard(title, desc, actions, raw, live) {
  return h('div', { class: 'ecard', role: live ? 'alert' : null },
    h('div', { class: 't' }, I('alert', 's'), h('span', null, title)), desc ? h('div', { class: 'd' }, desc) : null,
    actions && actions.length ? h('div', { class: 'row' }, actions) : null,
    raw ? h('details', { class: 'fold' }, h('summary', null, I('chevR', 's'), T('자세히')), h('pre', { class: 'log sans' }, TM(raw))) : null);
}
function addCaret(root) {
  let last = root;
  for (;;) {
    const c = last.lastElementChild;
    if (!c) break;
    if (/^(P|LI|UL|OL|H4|BLOCKQUOTE)$/.test(c.tagName)) { last = c; continue; }
    if (c.classList.contains('codeblk')) { last = c.querySelector('code'); }
    break;
  }
  last.append(h('span', { class: 'caret', 'aria-hidden': 'true' }));
}
const KERNEL_NOTE = T('처음 한 번은 이 GPU에 맞는 커널을 만들고 가장 빠른 실행 방식을 측정합니다(보통 수십 초~2분). 다음부터는 더 빨리 준비됩니다.');
const FRESH = new WeakSet();   // 방금 생긴 오류(한 번만 알린다)
function msgEl(m, room, opts) {
  opts = opts || {};
  if (m.role === 'user') {
    const k = room.messages.indexOf(m);
    if (S.editing && S.editing.roomId === room.id && S.editing.k === k) return editBox(m, room, k);
    return h('div', { class: 'msg user' },
      !S.streaming ? h('button', { class: 'icon-btn edit', 'data-tip': T('수정'), 'data-tip-pos': 'top', 'aria-label': T('이 메시지 수정'), onclick: () => startEdit(room, k) }, I('edit', 's')) : null,
      h('div', { class: 'bubble' }, m.content));
  }
  if (m.role === 'notice') {
    return h('div', { class: 'msg notice' }, h('div', { class: 'ncard', role: 'status' },
      h('div', { class: 'row' }, h('i', { class: 'dot busy' }), h('span', { class: 't' }, m.text || m.content)),
      h('div', { class: 'bar' }, h('i', { class: 'nbar', style: `width:${m.pct || 0}%` })),
      h('div', { class: 's' }, m.kernel ? KERNEL_NOTE : T('모델을 GPU에 올리는 중입니다.'))));
  }
  if (m.role === 'note') return h('div', { class: 'msg note' }, I('info', 's'), h('span', null, TM(m.content)));
  if (m.role === 'error') {
    const [t, d, acts] = humanError(m.content, room);
    const live = FRESH.has(m);
    FRESH.delete(m);
    return h('div', { class: 'msg error' }, errorCard(t, d, opts.last ? acts : null, m.content, live));
  }
  const mm = modelOf(m.model || room.model);
  const e = h('div', { class: 'msg assistant' },
    opts.who ? h('div', { class: 'who' }, h('span', { class: 'eyebrow' }, mm ? `${displayName(mm)} · ${quantOf(mm)}` : labelById(m.model || room.model))) : null,
    h('div', { class: 'body' }), h('div', { class: 'tools' }));
  e._last = !!opts.last;
  e._vk = opts.vk;   // 이 답이 끝맺는 차례(사용자 메시지 번호) — 버전 넘기기
  if (S.streaming && S.streaming.msg === m) S.streaming.el = e;   // 다시 그려도 흐름은 새 요소에 이어 그린다
  fillMsg(e, m, room, !m.partial);
  return e;
}
// 멈춘 자리에 남은 짝 없는 마크다운 표시(**, `, _)는 지운다
const trimDangling = s => String(s).replace(/(\*\*|\*|__|`+|~~)\s*$/, '');
function fillMsg(e, m, room, done) {
  const body = e.querySelector('.body');
  if (!body) return;
  if (!m.template) m.template = (modelOf(m.model || room.model) || {}).template;
  const sp = splitThink(m, done && !m.stopped);
  if (done && m.stopped) sp.answer = trimDangling(sp.answer);
  // 흐르는 중이거나 중지로 끊긴 답: 닫히지 않은 수식은 날 TeX 로 보이지 않게 그 앞에서 자른다
  if (!done || m.stopped) { sp.answer = holdOpenMath(sp.answer); if (sp.think) sp.think = holdOpenMath(sp.think); }
  put(body);
  const cut = done && !m.stopped && (m.finish === 'length' || sp.unfinished || /최대 길이에서 끊김/.test(m.stats || ''));
  if (sp.think) {
    const stopped = done && m.stopped && !sp.answer;
    const live = sp.open && !done;
    const label = live ? T('생각하는 중') : stopped ? T('생각 과정 · 중지됨') : (sp.unfinished || (sp.open && done)) ? T('생각 과정 · 끊김')
      : m.thinkMs ? T`${Math.max(1, Math.round(m.thinkMs / 1000))}초 동안 생각함` : T('생각 과정');
    const user = e.dataset.thinkOpen;
    body.append(h('details', { class: 'think', open: (user ? user === '1' : live) ? true : null },
      h('summary', { onclick: ev => { e.dataset.thinkOpen = ev.currentTarget.parentElement.open ? '0' : '1'; } },
        live ? h('i', { class: 'dot busy' }) : I('spark', 's'), h('span', null, label), I('chevR', 's chev')),
      h('div', { class: 'tb md', html: md(trimDangling(sp.think)) })));
  }
  const a = h('div', { class: 'md', html: md(sp.answer) });
  if (!done && (!sp.open || !sp.think)) addCaret(a);
  body.append(a);
  if (cut) {
    const r1 = m.template === 'deepseek-r1';
    const noAns = (r1 || m.template === 'qwen3') && !sp.answer;
    const max = (m.max || OPT.max);
    body.append(h('div', { class: 'cut-note' }, I('info', 's'),
      h('div', { class: 'col g12' },
        h('span', null, noAns ? T`답을 쓰기 전에 최대 답변 길이(${fmtNum(max)}토큰)에 닿았습니다.` : T`최대 답변 길이(${fmtNum(max)}토큰)에서 답이 끊겼습니다.`),
        e._last && !S.streaming ? h('div', { class: 'row' },
          r1 ? h('button', { class: 'btn sm needs-conn', onclick: continueLast }, I('forward', 's'), T('이어서 생성')) : null,
          h('button', { class: 'btn sm ghost needs-conn', onclick: () => retryLast(Math.min(32768, max * 2)) }, I('refresh', 's'), T`${fmtNum(Math.min(32768, max * 2))}토큰으로 다시 생성`)) : null)));
  }
  const tools = e.querySelector('.tools');
  put(tools,
    e._vk !== undefined ? verSwitch(room, e._vk) : null,
    done && m.content ? h('button', {
      class: 'icon-btn', 'data-tip': T('답 복사'), 'data-tip-pos': 'top', 'aria-label': T('답 복사'), onclick: async ev => {
        const btn = ev.currentTarget;
        if (await copyText(sp.answer || m.content)) { btn.innerHTML = svg('check', 's'); announce(T('복사했습니다')); setTimeout(() => { btn.innerHTML = svg('copy', 's'); }, 1200); }
      },
    }, I('copy', 's')) : null,
    done && e._last && !S.streaming ? h('button', { class: 'icon-btn needs-conn', 'data-tip': T('다시 생성 — 지금 답은 버전으로 남습니다'), 'data-tip-pos': 'top', 'aria-label': T('다시 생성'), onclick: () => retryLast() }, I('refresh', 's')) : null,
    done && m.tid && TRACES[m.tid] ? h('button', { class: 'icon-btn', 'data-tip': T('연구 패널에서 보기 — 토큰마다 확률 · 개입 · 비교 · 재현'), 'data-tip-pos': 'top', 'aria-label': T('이 답을 연구 패널에서 보기'), onclick: () => labOpen(m.tid) }, I('flask', 's')) : null,
    m.stats ? h('span', { class: 'stats' }, statsText(m.stats)) : null);
}
// 지난 기록의 통계 글도 지금 꼴로(“412 토큰” → “412토큰”)
// 숫자와 '토큰'은 붙여 쓴다(“412토큰 · 59.8토큰/초”) — 지난 기록의 띄어 쓴 꼴도 이렇게 보인다
const statsText = s => {
  const t = String(s).replace(/ · 최대 길이에서 끊김$/, '').replace(/(\d) 토큰/g, '$1토큰');
  if (!EN) return t;
  return t.replace(/^이어서 /, 'Continued · ').replace(/([\d.,]+)토큰\/초/g, '$1 tokens/s').replace(/첫 토큰 /g, 'first token ')
    .replace(/([\d.,]+)토큰/g, (_, n) => `${n} ${n === '1' ? 'token' : 'tokens'}`).replace(/([\d.]+)초/g, '$1 s').replace(/^중지됨$/, 'Stopped').replace(/^오류$/, 'Error');
};
// 버전 넘기기 ‹ 2 / 3 › — 고쳐 보내기 · 다시 생성이 남긴 앞 버전
function verSwitch(room, k) {
  const u = room.messages[k];
  if (!u || !u.vers || u.vers.length < 2) return null;
  const busy = !!S.streaming;
  return h('span', { class: 'vers', role: 'group', 'aria-label': T`버전 ${u.vi + 1} / ${u.vers.length}` },
    h('button', { class: 'icon-btn', 'data-tip': T('이전 버전'), 'data-tip-pos': 'top', 'aria-label': T('이전 버전'), disabled: (busy || u.vi === 0) ? true : null, onclick: () => switchVersion(room, k, u.vi - 1) }, I('chevL', 's')),
    h('span', { class: 'n', 'aria-hidden': 'true' }, `${u.vi + 1} / ${u.vers.length}`),
    h('button', { class: 'icon-btn', 'data-tip': T('다음 버전'), 'data-tip-pos': 'top', 'aria-label': T('다음 버전'), disabled: (busy || u.vi === u.vers.length - 1) ? true : null, onclick: () => switchVersion(room, k, u.vi + 1) }, I('chevR', 's')));
}
// 제안은 입력창을 채우기만 한다(보내기는 사람이 — 모델을 불러오는 일이 뜻밖에 시작되지 않게)
const SUGGEST = [
  { t: T('다음 문장을 자연스러운 영어로 바꿔 줘: 내일 회의는 3시로 미뤄졌습니다') },
  { t: T('파이썬으로 피보나치 함수를 작성해 줘') },
  { t: T('봄을 주제로 짧은 시를 써 줘') },
  { t: T('아래 글을 세 줄로 요약해 줘'), fill: T('아래 글을 세 줄로 요약해 줘.\n\n') },
];
function fillComposer(text) {
  const inp = $('#input');
  if (inp.disabled) return;
  inp.value = text;
  autosize();
  inp.focus();
  inp.setSelectionRange(text.length, text.length);
}
// 이 PC 에 맞는 추천: 맞는 것 가운데 7B 일반 모델 Q8_0 → Q4_K_M → 맞는 어떤 것이든(작은 것부터)
function recommendId(st) {
  const d = st.device;
  if (!d || !d.layer) return null;
  const pref = ['qwen2.5-7b-instruct-q8_0', 'qwen2.5-7b-instruct-q4_k_m'];
  for (const id of pref) { const c = st.catalog.find(x => x.id === id); if (c) { const f = fitOf(c); if (f && f.k === 'ok') return id; } }
  const ok = st.catalog.filter(c => { const f = fitOf(c); return f && (f.k === 'ok' || f.k === 'tight'); }).sort((a, b) => a.size - b.size);
  return ok.length ? ok[0].id : null;
}
function helloState(st) {
  if (!st) return 'loading';
  if (!st.device.checked) return 'checking';
  if (!st.device.layer) return 'nogpu';
  if (st.models.some(runnable)) return 'ready';
  if (st.models.length) return 'unsupported';
  return 'nomodel';
}
function renderMsgsIfChanged() {
  const st = S.st;
  if (changed('hello', [helloState(st), recommendId(st), st.catalog.map(c => [c.installed, c.download && c.download.state])])) renderMsgs();
}
// bare: 일시정지 단추 없이(다운로드 목록은 단추를 따로 둔다)
function dlBox(c, bare) {
  const d = c.download || {};
  const paused = d.state === 'cancelled' || d.state === 'failed';
  return h('div', { class: 'dl', 'data-dl': c.id },
    h('div', { class: 'bar' + (paused ? ' paused' : '') }, h('i', { style: 'width:0%' })),
    h('div', { class: 'line' }, h('span', { class: 'txt' }, ''),
      !bare && d.state === 'downloading' ? h('button', { class: 'btn sm ghost needs-conn', 'data-fk': 'pause:' + c.id, onclick: () => pauseDownload(c) }, I('pause', 's'), T('일시정지')) : null),
    h('div', { class: 'sub' }, ''));
}
// 내려받기 막대는 1초마다 글과 폭만 고친다(모델 화면 · 첫 화면 · 다운로드 목록 · 양자화 칸)
function updateDlBars() {
  if (!S.st) return;
  for (const el of document.querySelectorAll('[data-dlp]')) {
    const c = S.st.catalog.find(x => x.id === el.dataset.dlp);
    const d = c && c.download;
    if (!d) continue;
    const ver = d.state === 'verifying';
    const pct = d.total ? 100 * (ver ? d.verify_bytes : d.received) / d.total : 0;
    const t = (ver ? T('확인 ') : '') + pct.toFixed(0) + '%';
    if (el.textContent !== t) el.textContent = t;
  }
  for (const el of document.querySelectorAll('[data-dl]')) {
    const c = S.st.catalog.find(x => x.id === el.dataset.dl);
    const d = c && c.download;
    if (!d) continue;
    const ver = d.state === 'verifying', paused = d.state === 'cancelled' || d.state === 'failed';
    const pct = ver ? (d.total ? 100 * d.verify_bytes / d.total : 0) : (d.total ? 100 * d.received / d.total : 0);
    el.querySelector('.bar > i').style.width = pct.toFixed(1) + '%';
    const eta = d.bytes_per_s > 0 ? (d.total - d.received) / d.bytes_per_s : NaN;
    el.querySelector('.txt').textContent = ver ? T`무결성 확인 중${NB}${pct.toFixed(0)}%` : paused ? T`${pct.toFixed(0)}%에서 멈춤` : `${pct.toFixed(0)}%${NB}${fmtLeft(eta)}`;
    el.querySelector('.sub').textContent = ver ? `SHA-256${NB}${fmtBytes(d.total)}` : paused ? `${fmtBytes(d.received)} / ${fmtBytes(d.total)}` : `${fmtBytes(d.received)} / ${fmtBytes(d.total)}${NB}${fmtBytes(d.bytes_per_s)}/s`;
  }
}
let stickNext = false;   // 다음 그리기는 맨 아래로(보낼 때)
function renderMsgs() {
  const box = $('#msgs');
  const room = curRoom();
  const chat = $('#chat');
  if (!room || !room.messages.length) {
    box.dataset.room = room ? room.id : '';
    chat.classList.add('empty');
    $('#tobottom').hidden = true;
    const st = S.st;
    const hs = helloState(st);
    const rec = st && recommendId(st);
    const recC = rec && st.catalog.find(c => c.id === rec);
    const recDl = recC && recC.download && (recC.download.state === 'downloading' || recC.download.state === 'verifying');
    const head = h('div', { class: 'row g12' }, h('span', { class: 'eyebrow caps' }, 'GeulPU'), h('span', { class: 'eyebrow' }, T('글(geul) 언어로 만든 로컬 LLM')));
    let title, lede, actions = null;
    if (hs === 'ready') {
      title = T('무엇을 도와드릴까요?');
      lede = null;
      actions = h('div', { class: 'chips' }, SUGGEST.map(s => h('button', { class: 'chip-btn needs-conn', onclick: () => fillComposer(s.fill || s.t) }, s.t)));
    } else if (hs === 'checking' || hs === 'loading') {
      title = T('하드웨어를 확인하는 중입니다');
      lede = T('처음 실행할 때 GPU와 드라이버를 확인합니다. 잠시만 기다려 주세요.');
    } else if (hs === 'nogpu') {
      title = T('쓸 수 있는 GPU를 찾지 못했습니다');
      lede = T('NVIDIA RTX 30 시리즈 이상(CUDA) 또는 Vulkan 1.3을 지원하는 GPU와 최신 그래픽 드라이버가 필요합니다.');
      actions = h('div', { class: 'col g16 mt20' },
        st.device.reason ? h('details', { class: 'fold' }, h('summary', null, I('chevR', 's'), T('확인한 내용')), h('pre', { class: 'log sans' }, TM(st.device.reason))) : null,
        h('div', { class: 'row' }, h('button', { class: 'btn', onclick: () => show('device') }, I('cpu', 's'), T('하드웨어 보기'))));
    } else if (hs === 'unsupported') {
      title = T('이 PC에서 실행할 수 있는 모델이 없습니다');
      lede = T('설치된 모델을 이 GPU나 GeulPU가 지원하지 않습니다. 모델 화면에서 이유를 확인하고 다른 모델을 다운로드하세요.');
      actions = h('div', { class: 'row mt24' }, h('button', { class: 'btn', onclick: () => show('models') }, I('box', 's'), T('모델 보기')));
    } else {
      title = T('먼저 모델이 필요합니다');
      // 맞는 것이 없을 때 — 공유 메모리 32 KB GPU(AMD 등)는 형식 제약까지 그대로 밝힌다(흔한 Q4_0 파일도 출력이 Q6_K 라 아직 받지 않음)
      lede = recC ? T`이 PC에는 ${recC.name} ${recC.quant}(${fmtBytes(recC.size)}) 모델을 추천합니다.`
        : st.device.variant === '공유32' ? T`이 GPU(공유 메모리 32 KB)는 가중치가 모두 Q8_0이거나 모두 Q4_0인 파일만 실행합니다. 목록의 Q8_0 7B는 그래픽 메모리가 10 GB 이상 필요하고(이 PC ${fmtBytes(st.device.memory)}), 흔한 Q4_0 파일은 출력 가중치가 Q6_K라 아직 받지 않습니다. 지금은 이 PC에서 실행할 수 있는 파일이 없습니다.`
          : T`목록의 모델 가운데 이 PC의 그래픽 메모리(${fmtBytes(st.device.memory)})에 맞는 것이 없습니다. Hugging Face에서 더 작은 파일을 찾아볼 수 있습니다 — 받기 전에 실행되는지 확인합니다.`;
      actions = h('div', { class: 'col g0' },
        recDl ? dlBox(recC) : null,
        h('div', { class: 'row mt24' },
          recC && !recC.installed && !recDl ? h('button', { class: 'btn primary needs-conn', onclick: () => startDownload(recC) }, I('download', 's'), T('추천 모델 다운로드')) : null,
          !recC && st.device.variant !== '공유32' ? h('button', { class: 'btn primary needs-conn', onclick: openHfSearch }, I('search', 's'), T('모델 찾기')) : null,
          !recC ? h('button', { class: 'btn needs-conn', onclick: () => browse('') }, I('plus', 's'), T('GGUF 파일 추가')) : null,
          !recC && st.device.variant === '공유32' ? h('button', { class: 'btn ghost', onclick: () => show('device') }, I('cpu', 's'), T('하드웨어 보기')) : null,
          recC ? h('button', { class: 'btn' + (!recDl ? ' ghost' : ''), onclick: () => show('models') }, recDl ? T('다운로드 진행 보기') : T('모델 둘러보기')) : null));
    }
    put(box, h('div', { class: 'hello' }, head, h('h1', null, title), lede ? h('p', null, lede) : null, actions));
    updateDlBars();
    updateComposer();
    labSync();
    return;
  }
  chat.classList.remove('empty');
  const th = h('div', { class: 'thread' });
  let prevModel = null;
  const msgs = room.messages, n = msgs.length;
  // 차례 = 사용자 메시지 + 그 뒤의 답 · 오류. 차례의 마지막 요소에 버전 넘기기를 단다
  let uk = -1;
  msgs.forEach((m, k) => {
    if (m.role === 'user') uk = k;
    const vk = (k === n - 1 || msgs[k + 1].role === 'user') && uk >= 0 && m.role !== 'user' ? uk : undefined;
    if (m.role === 'assistant') {
      const mid = m.model || room.model;
      th.append(msgEl(m, room, { who: mid !== prevModel, last: k === n - 1, vk }));
      prevModel = mid;
    } else {
      th.append(msgEl(m, room, { last: k === n - 1 }));
      const vs = vk !== undefined ? verSwitch(room, vk) : null;
      if (vs) th.append(h('div', { class: 'msg-vers' }, vs));
    }
  });
  const sn = S.streaming && S.streaming.roomId === room.id && S.streaming.notice;
  if (sn) { sn.el = msgEl(sn, room); th.append(sn.el); }
  // 맨 아래로는 새 채팅을 열 때 · 보낼 때 · 이미 맨 아래 근처일 때만 — 위를 읽는 중이면 그 자리를 지킨다(고치기 취소 · 모델 바꾸기 등)
  const y = box.scrollTop, near = box.scrollHeight - y - box.clientHeight < 160, other = box.dataset.room !== room.id;
  put(box, th);
  box.scrollTop = stickNext || other || near ? box.scrollHeight : y;
  stickNext = false;
  box.dataset.room = room.id;
  $('#tobottom').hidden = box.scrollHeight - box.scrollTop - box.clientHeight < 240;
  updateComposer();
  labSync();
}
$('#tobottom').innerHTML = svg('down', 's');
$('#tobottom').addEventListener('click', () => { const b = $('#msgs'); b.scrollTo({ top: b.scrollHeight, behavior: 'smooth' }); });
$('#msgs').addEventListener('scroll', () => {
  const b = $('#msgs');
  $('#tobottom').hidden = $('#chat').classList.contains('empty') || b.scrollHeight - b.scrollTop - b.clientHeight < 240;
});
function buildR1Prompt(room, system) {
  let p = R1_BOS + (system || '');
  for (const m of room.messages) {
    if (m.role === 'user') p += R1_USER + m.content;
    else if (m.role === 'assistant' && !m.partial) {
      const ans = splitThink(Object.assign({ template: 'deepseek-r1' }, m), true).answer.trim();   // 앞 차례의 생각은 다시 보내지 않는다
      if (ans) p += R1_ASSIST + ans + R1_EOS;
    }
  }
  return p + R1_ASSIST;
}
async function ensureServer(modelId, notice) {
  const st = S.st;
  if (st.server.state === 'ready' && st.server.model === modelId) return;
  if (S.streaming) S.streaming.loading = true;
  if (!(st.server.state === 'starting' && st.server.model === modelId)) {
    try { await api('/api/server/start', { model: modelId }); }
    catch (e) { throw new Error('모델을 불러오지 못했습니다 — ' + e.message); }
  }
  for (;;) {
    await new Promise(r => setTimeout(r, 1000));
    if (S.streaming && S.streaming.aborted) throw new Error('중지했습니다');
    await refresh();
    const s2 = S.st;
    if (s2.server.state === 'ready' && s2.server.model === modelId) break;
    const j = s2.job;
    if (j && j.kind === 'server' && j.state !== 'running') {
      if (j.state === 'done' && s2.server.state === 'ready') break;
      throw new Error((j.reason || '모델을 불러오지 못했습니다') + (s2.server.reason ? ' — ' + s2.server.reason.slice(-400) : ''));
    }
    if (s2.server.state === 'failed') throw new Error('모델을 불러오지 못했습니다 — ' + (s2.server.reason || ''));
    if (notice) {
      notice.kernel = notice.kernel || !!(j && /커널/.test(j.step || ''));
      notice.text = T`${labelById(modelId)} 불러오는 중 · ${fmtClock(j ? j.seconds : s2.server.seconds)}${notice.kernel ? T(' · 커널 준비 중') : ''}`;
      notice.pct = j ? Math.min(100, j.progress / 10) : 0;
      const el = notice.el;
      if (el && el.isConnected) {
        el.querySelector('.t').textContent = notice.text;
        el.querySelector('.nbar').style.width = notice.pct + '%';
        if (notice.kernel) el.querySelector('.s').textContent = KERNEL_NOTE;
      }
    }
  }
  if (S.streaming) S.streaming.loading = false;
}
// 벤치마크 · 재현성 검증이 GPU 를 쓰는 중인가(그동안 채팅 모델을 불러올 수 없다)
const benchBusy = () => { const j = S.st && S.st.job; return !!(j && j.state === 'running' && j.kind !== 'server' && j.kind !== 'device'); };
// 보낼 수 없는 까닭(없으면 '') — 아무것도 바꾸지 않고 먼저 묻는다
function sendBlock(room) {
  if (S.streaming) return T('다른 채팅의 답이 끝나면 보낼 수 있습니다');
  if (!S.conn) return T('GeulPU.exe와 연결이 끊겨 보낼 수 없습니다');
  if (benchBusy()) return T('벤치마크가 끝나면 보낼 수 있습니다');
  if (LAB.busy) return T('연구 패널의 셈이 끝나면 보낼 수 있습니다');
  const m = room && modelOf(room.model);
  if (!m) return S.st && S.st.models.length ? T('위의 모델 선택에서 모델을 고르세요') : T('먼저 모델 화면에서 모델을 다운로드하거나 추가하세요');
  if (!runnable(m)) return T('이 모델은 이 PC에서 실행할 수 없습니다 — ') + TM(m.device_reason || (m.shape && m.shape.reason) || '');
  return '';
}
// 차례 k(사용자 메시지)의 새 버전을 연다 — 지금 뒤의 대화(답 · 그 뒤의 차례)는 앞 버전으로 남긴다.
// 버전은 사용자 메시지에 { content, rest } 로: rest 는 그 버전의 뒤 대화(지금 버전은 room.messages 에 있어 null)
function branch(room, k, text) {
  const u = room.messages[k];
  const rest = room.messages.slice(k + 1);
  if (rest.some(x => (x.role === 'assistant' && x.content) || x.role === 'user')) {   // 답이 없던(오류뿐인) 꼴은 남기지 않는다
    if (!u.vers) { u.vers = [{ content: u.content, rest: null }]; u.vi = 0; }
    u.vers[u.vi] = { content: u.content, rest };
    u.vers.push({ content: text, rest: null });
    u.vi = u.vers.length - 1;
  }
  u.content = text;
  room.messages.length = k + 1;
}
function switchVersion(room, k, j) {
  if (S.streaming) return;
  const u = room.messages[k];
  if (!u || !u.vers || j < 0 || j >= u.vers.length || j === u.vi) return;
  u.vers[u.vi] = { content: u.content, rest: room.messages.slice(k + 1) };
  const v = u.vers[j];
  u.content = v.content;
  room.messages.splice(k + 1, room.messages.length, ...(v.rest || []));
  v.rest = null;
  u.vi = j;
  S.editing = null;
  saveRooms();
  const keepY = $('#msgs').scrollTop;
  renderMsgs();
  $('#msgs').scrollTop = keepY;
  announce(T`버전 ${j + 1} / ${u.vers.length}`);
}
function retryLast(maxOverride) {
  const room = curRoom();
  if (!room || S.streaming) return;
  const k = room.messages.map(x => x.role).lastIndexOf('user');
  if (k < 0) return;
  const why = sendBlock(room);
  if (why) { toast(why, 'err'); return; }
  branch(room, k, room.messages[k].content);
  replyTo(room, maxOverride);
}
// 메시지 고치기 — 말풍선 자리에서 고치고, 보내면 새 버전(지금 대화는 앞 버전으로 남는다). 취소하면 아무것도 바뀌지 않는다
function startEdit(room, k) {
  if (S.streaming) return;
  S.editing = { roomId: room.id, k };
  renderMsgs();
}
function editBox(m, room, k) {
  const ta = h('textarea', { class: 'edit-input', rows: '1', 'aria-label': T('메시지 수정') });
  ta.value = m.content;
  const later = room.messages.slice(k + 1).some(x => x.role === 'user');   // 이 뒤에 다른 차례가 있는가
  const send = h('button', { class: 'btn sm primary needs-conn', onclick: () => submitEdit(room, k, ta.value) }, T('보내기'));
  const cancel = () => { S.editing = null; renderMsgs(); };
  const fit = () => { ta.style.height = 'auto'; ta.style.height = Math.min(320, ta.scrollHeight) + 'px'; send.disabled = !ta.value.trim(); };
  ta.addEventListener('input', fit);
  ta.addEventListener('keydown', e => {
    if (e.key === 'Escape') { e.preventDefault(); e.stopPropagation(); cancel(); }
    else if (e.key === 'Enter' && !e.shiftKey && !e.isComposing) { e.preventDefault(); submitEdit(room, k, ta.value); }
  });
  setTimeout(() => { fit(); ta.focus(); ta.setSelectionRange(ta.value.length, ta.value.length); }, 0);
  return h('div', { class: 'msg user editing' }, h('div', { class: 'edit-box' }, ta,
    h('div', { class: 'edit-bar' }, h('span', { class: 'hint' }, later ? T('보내면 지금 대화는 이전 버전으로 남습니다 · Esc 취소') : T('보내면 지금 답은 이전 버전으로 남습니다 · Esc 취소')),
      h('span', { class: 'grow' }), h('button', { class: 'btn sm ghost', onclick: cancel }, T('취소')), send)));
}
function submitEdit(room, k, text) {
  text = String(text || '').trim();
  if (!text) return;
  const why = sendBlock(room);
  if (why) { toast(why, 'err'); return; }
  S.editing = null;
  branch(room, k, text);
  replyTo(room);
}
// 흘림 하나를 받아 asst 에 붙인다(보내기 · 이어서 생성이 함께 쓴다). 그릴 요소는 늘 S.streaming.el(다시 그려도 이어진다)
// 38단계 — 돌려주는 것: { stats(통계 글), gp(geulpu 덧붙임), seed(씨앗의 원문 숫자 — JSON 수로 읽으면 2⁵³ 위에서 바뀐다), bodyStr, finish }.
// text: 완성 꼴의 조각(c.text)을 읽는다(R1 · 연구 패널의 갈래). bodyStr 를 주면 그 글을 그대로 보낸다(씨앗 숫자를 그대로 — 갈래)
async function streamReply(asst, room, url, body, opt) {
  opt = opt || {};
  const tpl = asst.template;
  const textChunks = tpl === 'deepseek-r1' || !!opt.text;
  const ctrl = new AbortController();
  S.streaming.ctrl = ctrl;
  const t0 = performance.now();
  let tFirst = 0, usage = null, gp = null, finish = null, seed = '';
  const bodyStr = opt.bodyStr || JSON.stringify(body);
  const res = await fetch(url, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: bodyStr, signal: ctrl.signal });
  if (!res.ok) {
    let msg = 'HTTP ' + res.status;
    try { const j = await res.json(); msg = (j.error && j.error.message) || msg; } catch (_) { }
    throw new Error(msg);
  }
  const reader = res.body.getReader();
  const dec = new TextDecoder();
  let buf = '', lastPaint = 0;
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buf += dec.decode(value, { stream: true });
    let k;
    while ((k = buf.indexOf('\n\n')) >= 0) {
      const evt = buf.slice(0, k);
      buf = buf.slice(k + 2);
      for (const line of evt.split('\n')) {
        if (!line.startsWith('data:')) continue;
        const data = line.slice(5).trim();
        if (data === '[DONE]') continue;
        let j;
        try { j = JSON.parse(data); } catch (_) { continue; }
        if (j.geulpu) { const sm = /"seed":(\d+)/.exec(data); if (sm) seed = sm[1]; }
        const c = j.choices && j.choices[0];
        const piece = c ? (textChunks ? (c.text || '') : ((c.delta && c.delta.content) || '')) : '';
        if (piece) {
          if (!tFirst) tFirst = performance.now();
          asst.content += piece;
          if (tpl === 'deepseek-r1' && !asst.thinkMs && (asst.content.includes('</think>') || asst.content.includes('\n\n\n'))) asst.thinkMs = performance.now() - tFirst;
          if (tpl === 'qwen3' && !asst.thinkMs && asst.content.startsWith('<think>') && asst.content.includes('</think>')) asst.thinkMs = performance.now() - tFirst;
        }
        if (c && c.finish_reason) finish = c.finish_reason;
        if (j.usage) usage = j.usage;
        if (j.geulpu) gp = j.geulpu;
      }
    }
    const now = performance.now();
    if (now - lastPaint > 60) {
      lastPaint = now;
      const el = S.streaming && S.streaming.el;
      if (el && el.isConnected) {
        const box = $('#msgs');
        const near = box.scrollHeight - box.scrollTop - box.clientHeight < 120;
        fillMsg(el, asst, room, false);
        if (near) box.scrollTop = box.scrollHeight;
      }
    }
  }
  delete asst.partial;
  asst.finish = finish;
  if (tpl === 'deepseek-r1' && gp && Array.isArray(gp.tokens) && gp.tokens.includes(THINK_END_ID)) asst.thinkEnded = true;
  const n = (usage && usage.completion_tokens) || (gp && gp.tokens && gp.tokens.length) || 0;
  // 서버의 generate_us 는 첫 토큰 뒤의 모든 걸음 — 멈춘 EOS 를 뽑은 걸음까지(gp.tokens 에는 있고 completion_tokens 에는 없다). 연구 패널과 같은 셈
  const steps = (gp && Array.isArray(gp.tokens) ? gp.tokens.length : n) - 1;
  let stats = '';
  if (gp && gp.generate_us > 0 && steps > 0) {
    const tps = steps / (gp.generate_us / 1e6);
    stats = `${fmtNum(n)}토큰 · ${tps.toFixed(1)}토큰/초 · 첫 토큰 ${(gp.first_token_ms / 1000).toFixed(0)} ms`;   // 서버의 이 칸은 µs(이름만 ms)
  }
  else if (tFirst) stats = `${fmtNum(n)}토큰 · ${((performance.now() - t0) / 1000).toFixed(1)}초`;
  return { stats, gp, seed, bodyStr, finish };
}
async function send(textOverride) {
  if (S.streaming) { if (!streamingHere()) toast(T('다른 채팅의 답이 끝나면 보낼 수 있습니다')); return; }
  const input = $('#input');
  const text = (typeof textOverride === 'string' ? textOverride : input.value).trim();
  if (!text) return;
  let room = curRoom();
  if (!room) { newRoom(); room = curRoom(); }
  const why = sendBlock(room);
  if (why) { toast(why, 'err'); return; }
  if (typeof textOverride !== 'string') { input.value = ''; autosize(); }
  if (!room.messages.length) room.title = text.slice(0, 40);
  room.messages.push({ role: 'user', content: text });
  await replyTo(room);
}
// 답 하나의 요청 몸 — 보낼 때와 연구 패널이 다시 지을 때 같은 함수다(같은 글 → 같은 해시). upto = 이 답 앞까지의 메시지 수.
// geulpu.prompt_tokens — 서버가 채팅 틀을 씌운 프롬프트 토큰을 돌려준다(연구 기록에 남겨, 숨은 버전 · 고친 대화의 답도 셈할 수 있게)
function replyBody(room, upto, p) {
  if (p.tpl === 'deepseek-r1') {
    const hist = Object.assign({}, room, { messages: room.messages.slice(0, upto) });
    return { url: '/v1/completions', body: { prompt: buildR1Prompt(hist, p.system), max_tokens: p.max, temperature: p.temp, stream: true, geulpu: { prompt_tokens: true } } };
  }
  const msgs = [];
  if (p.system) msgs.push({ role: 'system', content: p.system });
  for (const x of room.messages.slice(0, upto)) if (x.role === 'user' || (x.role === 'assistant' && !x.partial)) msgs.push({ role: x.role, content: x.content });
  const body = { messages: msgs, max_tokens: p.max, temperature: p.temp, stream: true, stream_options: { include_usage: true }, geulpu: { prompt_tokens: true } };
  if (p.tpl === 'qwen3') {                 // 서버의 Qwen3 틀 — 앞 답의 생각은 서버가 뗀다, <think> 는 글로 받아 접는다
    body.geulpu.special = true;
    body.chat_template_kwargs = { enable_thinking: p.think !== false };
  }
  return { url: '/v1/chat/completions', body };
}
// 방의 마지막 사용자 메시지에 답한다(보내기 · 다시 생성 · 고쳐 보내기가 함께 쓴다).
// ov — 연구 패널의 '설정을 바꿔 다시'(38단계): { model, temp, seed(원문 숫자), system, think } 가운데 준 것만 바꾼다
async function replyTo(room, maxOverride, ov) {
  ov = ov || {};
  const mid = ov.model || room.model;
  const m = modelOf(mid);
  const max = maxOverride || OPT.max;
  room.updated = Date.now();
  stickNext = true;
  S.streaming = { aborted: false, ctrl: null, loading: false, roomId: room.id, msg: null, el: null, notice: null };
  saveRooms(); renderRooms(); renderMsgs(); renderChatTop(true);
  try {
    if (!(S.st.server.state === 'ready' && S.st.server.model === mid)) {
      const notice = { role: 'notice', text: T`${labelById(mid)} 불러오는 중`, pct: 0, kernel: false, el: null };
      S.streaming.notice = notice;
      if (S.roomId === room.id) renderMsgs();
      await ensureServer(mid, notice);
      S.streaming.notice = null;
      if (notice.el) notice.el.remove();
    }
    const tpl = m.template;
    const asst = { role: 'assistant', content: '', template: tpl, model: mid, max, partial: true };
    const rp = {
      tpl, max, temp: ov.temp ?? OPT.temp, system: ov.system ?? (OPT.system || ''), think: ov.think ?? (OPT.think !== false), model: mid,
    };
    const { url, body } = replyBody(room, room.messages.length, rp);
    room.messages.push(asst);
    S.streaming.msg = asst;
    if (S.roomId === room.id) renderMsgs();
    const rr = await streamReply(asst, room, url, body, ov.seed ? { bodyStr: seedBody(body, ov.seed) } : undefined);
    asst.stats = rr.stats;
    traceAfter(asst, rp, rr, body);     // 38단계 — 연구 기록(토큰 · 씨앗 원문 · 요청 몸의 해시 — 씨앗은 빼고)
    room.updated = Date.now();
    store('geulpu.hinted', true);
    saveRooms();
    announce(T('답변이 끝났습니다'));
  } catch (e) {
    const aborted = S.streaming && S.streaming.aborted;
    const last = room.messages[room.messages.length - 1];
    if (last && last.role === 'assistant' && last.partial) {
      delete last.partial;
      if (aborted) { last.stats = '중지됨'; last.stopped = true; }
      else last.stats = '오류';
      if (!last.content) room.messages.pop();
    }
    if (aborted && room.messages[room.messages.length - 1].role === 'user') room.messages.push({ role: 'note', content: '불러오기를 중지했습니다' });
    else if (!aborted) { const em = { role: 'error', content: e.message }; FRESH.add(em); room.messages.push(em); }
    saveRooms();
  } finally {
    S.streaming = null;
    renderRooms();
    renderMsgs();
    renderChatTop(true);
  }
}
// R1 이 길이에서 끊겼을 때 — 받은 글을 그대로 앞에 두고 이어서 생성한다
async function continueLast() {
  const room = curRoom();
  if (!room || S.streaming || !S.conn) return;
  const k = room.messages.length - 1;
  const m = room.messages[k];
  if (!m || m.role !== 'assistant' || m.template !== 'deepseek-r1') return;
  const mm = modelOf(m.model || room.model);
  if (!runnable(mm)) return;
  S.streaming = { aborted: false, ctrl: null, loading: false, roomId: room.id, msg: m, el: null, notice: null };
  m.partial = true;
  m.finish = null;
  m.max = OPT.max;
  renderRooms(); renderMsgs(); renderChatTop(true);
  const before = m.content.length;
  try {
    if (!(S.st.server.state === 'ready' && S.st.server.model === mm.id)) await ensureServer(mm.id, null);
    const hist = Object.assign({}, room, { messages: room.messages.slice(0, k) });
    const prompt = buildR1Prompt(hist, OPT.system) + '<think>' + m.content;
    const stat = (await streamReply(m, room, '/v1/completions', { prompt, max_tokens: OPT.max, temperature: OPT.temp, stream: true, geulpu: {} })).stats;
    m.stats = stat ? '이어서 ' + stat : m.stats;
    delete m.tid;                     // 38단계 — 이어서 생성한 답은 한 요청이 아니라 연구 기록을 떼어 둔다
    room.updated = Date.now();
    saveRooms();
    announce(T('답변이 끝났습니다'));
  } catch (e) {
    delete m.partial;
    if (!(S.streaming && S.streaming.aborted)) { const em = { role: 'error', content: e.message }; FRESH.add(em); room.messages.push(em); }
    else if (m.content.length > before) { m.stopped = true; m.stats = '중지됨'; }
    saveRooms();
  } finally {
    S.streaming = null;
    renderRooms();
    renderMsgs();
    renderChatTop(true);
  }
}
function stopStream() {
  if (!S.streaming) return;
  S.streaming.aborted = true;
  if (S.streaming.ctrl) S.streaming.ctrl.abort();
  if (S.streaming.loading) api('/api/job/cancel', {}).catch(() => { });
}
function autosize() { const t = $('#input'); t.style.height = 'auto'; t.style.height = Math.min(240, t.scrollHeight) + 'px'; updateSend(); }
$('#input').addEventListener('input', autosize);
// 답이 흐르는 동안 Enter 는 아무것도 하지 않는다(입력은 그대로 — 끝나면 보낸다). 중지는 단추나 Esc.
$('#input').addEventListener('keydown', e => { if (e.key === 'Enter' && !e.shiftKey && !e.isComposing) { e.preventDefault(); if (!S.streaming) send(); } });
$('#sendbtn').addEventListener('click', () => (streamingHere() ? stopStream() : send()));
document.addEventListener('keydown', e => {
  if ((e.ctrlKey || e.metaKey) && e.shiftKey && e.key.toLowerCase() === 'o') { e.preventDefault(); show('chat'); newRoom(); }
});

// ---------- 모델 ----------
function pageHead(eyebrow, title, lede, actions) {
  return h('header', { class: 'page-head' }, h('div', { class: 't' }, h('div', { class: 'eyebrow' }, eyebrow || '\u00a0'), h('h1', null, title), lede ? h('p', null, lede) : null),
    actions ? h('div', { class: 'row' }, actions) : null);
}
function catDesc(c) {
  return /r1/i.test(c.id) ? T`생각을 먼저 쓰고 답하는 추론 모델${NB}수학${NB}논리` : /coder/i.test(c.id) ? T`코드 작성${NB}설명`
    : /qwen3/i.test(c.id) ? T`생각 켬${NB}끔을 고르는 대화 모델${NB}한국어${NB}영어` : T`한국어${NB}영어 일반 대화`;
}
const vramNeed = c => Math.ceil((c.size + (isBig(c) ? 4 : 2.5) * GIB) / GIB);
// 모델 화면은 두 칸 — 내 모델(받는 중 · 설치된 모델) · 모델 찾기(Hugging Face 의 GGUF 저장소 전체를 정렬해 둘러보고, 고르면 옆에 자세히)
function modelsTab() {
  if (!S.mtab) S.mtab = load('geulpu.mtab', '') || (S.st && (S.st.models.length || dlList().length) ? 'mine' : 'find');
  return S.mtab;
}
function setModelsTab(t, focus) {
  if (S.mtab !== t) {
    S.mtab = t;
    store('geulpu.mtab', t);
    resetSig();
    renderModels();
  }
  if (focus) { const b = document.querySelector(`[data-fk="mtab:${t}"]`); if (b) b.focus({ preventScroll: true }); }
}
function modelTabs(tab) {
  const n = S.st.models.length + dlList().length;
  const b = (t, label, extra) => h('button', {
    role: 'tab', class: tab === t ? 'on' : '', 'aria-selected': tab === t ? 'true' : 'false', tabindex: tab === t ? '0' : '-1', 'data-fk': 'mtab:' + t,
    onclick: () => setModelsTab(t),
  }, label, extra);
  return h('div', {
    class: 'seg-ctl mtabs', role: 'tablist', 'aria-label': T('모델 화면'),
    onkeydown: ev => { if (ev.key === 'ArrowLeft' || ev.key === 'ArrowRight') { ev.preventDefault(); setModelsTab(tab === 'mine' ? 'find' : 'mine', true); } },
  }, b('mine', T('내 모델'), n ? h('span', { class: 'cnt' }, String(n)) : null), b('find', T('모델 찾기')));
}
function renderModels() {
  const st = S.st;
  const page = $('#page-models');
  const tab = modelsTab();
  const rec = recommendId(st);
  const dls = dlList();
  const eyebrow = T`설치 ${st.models.length}개` + (dls.length ? NB + T`받는 중 ${dls.length}개` : '') + (st.app.disk_free >= 0 ? NB + T`여유 ${fmtBytes(st.app.disk_free)}` : '');
  // 받는 동안 바뀌는 수(받은 바이트 · 여유 공간)는 넣지 않는다 — 제자리에서 고친다(다시 그리면 초점 · 열린 메뉴가 사라진다).
  // 모델 찾기는 칸(목록 · 자세히)을 한 번 만들어 두고 다시 쓴다 — 상태가 바뀌어도 목록의 스크롤 · 고른 것이 그대로다
  const struct = tab === 'mine'
    ? [tab, st.models.map(m => [m.id, m.name, m.size, m.verified, !!m.verifying, m.sha256, m.catalog, m.device_reason, m.shape && m.shape.ok]),
      dls.map(c => [c.id, dlStateOf(c), c.download.state, c.download.reason]), st.device.checked, st.device.layer, st.server.model, st.server.state,
      st.settings.context, !!S.streaming]
    : [tab, st.models.length + dls.length];
  if (changed('models', struct)) {
    page.classList.toggle('wide', tab === 'find');
    rebuild(page,
      pageHead(eyebrow, T('모델'), null,
        [h('button', { class: 'btn needs-conn', 'data-fk': 'add', onclick: () => browse('') }, I('plus', 's'), T('GGUF 파일 추가')),
          h('button', { class: 'icon-btn needs-conn', 'data-tip': T('모델 폴더 열기'), 'data-tip-pos': 'bottom', 'aria-label': T('모델 폴더 열기'), onclick: () => api('/api/open', { what: 'models' }).catch(e => toast(e.message, 'err')) }, I('folder'))]),
      modelTabs(tab),
      tab === 'mine' ? mineView() : finderEl());
    if (tab === 'find') finderStart();
  } else {
    const e = page.querySelector('.page-head .eyebrow');
    if (e && e.textContent !== eyebrow) e.textContent = eyebrow;
  }
  // 모델 찾기의 파일 줄(설치됨 · 받는 중 · 이 PC 에 맞는지)은 목록 · 장치가 바뀔 때만 다시 그린다(받는 % 는 updateDlBars 가 제자리에서)
  if (tab === 'find' && changed('finder-st', [st.device.checked, st.device.layer, st.device.variant, st.device.memory, rec,
    st.catalog.map(c => [c.id, c.installed, dlStateOf(c), c.download && c.download.state]), st.models.map(m => m.path)])) fdRedraw();
  for (const m of st.models) {
    const e = document.getElementById('vf-' + m.id);
    if (e && m.verifying) e.textContent = T`무결성 확인 중${NB}${m.size > 0 ? Math.round(100 * m.verify_bytes / m.size) : 0}%`;
  }
}
// 내 모델 — 받는 중(멈춤 · 실패 포함)과 설치된 모델
function mineView() {
  const st = S.st;
  const nInst = st.models.length;
  const reasons = [...new Set(st.models.filter(m => m.shape && m.shape.ok && m.device_reason).map(m => m.device_reason))];
  const dls = dlList();
  return h('div', { class: 'mine' },
    dls.length ? h('div', { class: 'section mt0' }, h('div', { class: 'section-head' }, h('span', { class: 'sec' }, T('받는 중'))),
      h('div', { class: 'list dllist' }, dls.map(dlRow))) : null,
    h('div', { class: 'section' + (dls.length ? '' : ' mt0') }, h('div', { class: 'section-head' }, h('span', { class: 'sec' }, T('설치된 모델'))),
      // 이 GPU 에서 못 도는 까닭은 여기 한 번만(줄에는 꼬리표만)
      reasons.length ? h('div', { class: 'note-line note-gap' }, I('info', 's'), h('span', null, reasons.length === 1 ? TM(reasons[0]) : T('일부 모델은 이 GPU에서 실행할 수 없습니다. 꼬리표에 마우스를 올리면 까닭이 보입니다.'))) : null,
      nInst ? h('div', { class: 'list mlist' }, modelBlocks())
        : h('div', { class: 'empty-box' }, h('div', { class: 'big' }, T('아직 설치된 모델이 없습니다')),
          h('div', null, T('Hugging Face의 모델을 둘러보고 받거나, 이미 있는 GGUF 파일을 추가하세요.')),
          h('div', { class: 'row center mt16' },
            h('button', { class: 'btn primary needs-conn', onclick: () => setModelsTab('find') }, I('search', 's'), T('모델 찾기')),
            h('button', { class: 'btn needs-conn', onclick: () => browse('') }, I('plus', 's'), T('GGUF 파일 추가'))))));
}
// 받는 중인 파일 한 줄 — 상태 막대의 다운로드 목록과 같은 단추(이름 앞 동그라미 아이콘은 두지 않는다)
function dlRow(c) {
  const ds = dlStateOf(c), d = c.download;
  return h('div', { class: 'item dlitem' },
    h('div', { class: 'main' },
      h('div', { class: 'title' }, h('span', { class: 'nm' }, c.name), h('span', { class: 'qlab' }, c.quant)),
      ds === 'fail' ? dlWhyLine(d.reason, c) : dlBox(c, true)),
    h('div', { class: 'acts needs-conn' },
      ds === 'run' && d.state === 'downloading' ? h('button', { class: 'btn sm ghost', onclick: () => pauseDownload(c) }, I('pause', 's'), T('일시정지')) : null,
      ds === 'pause' ? h('button', { class: 'btn sm', onclick: () => resumeDownload(c) }, I('download', 's'), T('이어서 다운로드')) : null,
      ds === 'fail' ? h('button', { class: 'btn sm', onclick: () => startDownload(c) }, I('refresh', 's'), T('다시 다운로드')) : null,
      ds !== 'run' ? h('button', { class: 'icon-btn', 'data-tip': T('받은 것 지우기'), 'data-tip-pos': 'top', 'aria-label': T`받은 것 지우기: ${c.name} ${c.quant}`, onclick: () => discardDownload(c) }, I('trash', 's')) : null));
}
// 설치된 모델 — 모델마다 머리(이름 · 크기 · 종류 · 문맥) 하나, 그 아래 양자화 줄(양자화 · 파일 크기 · 상태 · 동작)
function modelBlocks() {
  const groups = [];
  for (const m of sortedModels()) {
    const name = displayName(m);
    const g = groups[groups.length - 1];
    if (g && g.name === name) g.items.push(m); else groups.push({ name, items: [m] });
  }
  return groups.map(g => {
    const head = g.items.find(m => m.shape && m.shape.ok) || g.items[0];
    const s = head.shape || {};
    const ctxNow = Math.min(s.context || 0, S.st.settings.context || s.context || 0);
    const sub = s.ok ? T`${s.params ? (s.params / 1e9).toFixed(1) + 'B' + NB : ''}${kindOf(head)}${NB}문맥 ${fmtCtx(ctxNow)}${s.context > ctxNow ? T` (최대 ${fmtCtx(s.context)})` : ''}` : kindOf(head);
    return h('div', { class: 'mblock' },
      h('div', { class: 'mhead' },
        h('div', { class: 'main' }, h('div', { class: 'title' }, h('span', { class: 'nm' }, g.name)), h('div', { class: 'sub' }, sub))),
      g.items.map(quantLine));
  });
}
function quantLine(m) {
  const s = m.shape || {};
  const ok = runnable(m);
  const srv = S.st.server;
  const loaded = srv.model === m.id && (srv.state === 'ready' || srv.state === 'starting');
  const name = displayName(m), q = quantOf(m);
  const tags = [];
  if (loaded) tags.push(h('span', { class: 'tag' }, h('i', { class: srv.state === 'ready' ? 'dot ok' : 'dot busy' }), srv.state === 'ready' ? T('불러옴') : T('불러오는 중')));
  if (!s.ok) tags.push(h('span', { class: 'tag err' }, I('alert'), T('지원하지 않는 파일')));
  else if (m.device_reason) tags.push(h('span', { class: 'tag dim', 'data-tip': TM(m.device_reason), 'data-tip-pos': 'top' }, T('이 GPU 미지원')));
  if (m.verifying) tags.push(h('span', { class: 'tag', id: 'vf-' + m.id }, T('무결성 확인 중')));
  else if (m.catalog && m.sha256 && !m.verified) tags.push(h('span', { class: 'tag err' }, I('alert'), T('무결성 확인 안 됨')));
  return h('div', { class: 'qline' + (s.ok && m.device_reason ? ' dimmed' : '') },
    h('span', { class: 'qlab', 'data-tip': m.path, 'data-tip-pos': 'top' }, q || '-'),
    h('span', { class: 'sz' }, fmtBytes(m.size)),
    tags.length ? h('span', { class: 'qtags' }, tags) : null,
    h('span', { class: 'grow' }),
    h('div', { class: 'acts needs-conn' },
      h('button', { class: 'btn sm ghost', 'data-fk': 'chat:' + m.id, 'aria-label': T`이 모델로 채팅: ${name} ${q}`, disabled: ok ? null : true, onclick: () => chatWith(m.id) }, I('chat', 's'), T('채팅')),
      h('button', {
        class: 'icon-btn', 'data-fk': 'more:' + m.id, 'aria-label': T`메뉴: ${name} ${q}`, 'aria-haspopup': 'menu', 'aria-expanded': 'false',
        onclick: ev => menu(ev.currentTarget, T('모델 메뉴'), [
          loaded ? { icon: 'eject', label: T('모델 내리기'), sub: T('GPU 메모리를 비웁니다'), disabled: !!S.streaming, run: unloadModel } : null,
          { icon: 'hash', label: T('파일 무결성 다시 확인'), sub: m.verified ? T('받은 그대로임을 확인했습니다(SHA-256) · 몇 초') : T('받은 그대로인지 SHA-256으로 확인 · 몇 초'), disabled: m.verifying, run: () => verifyModel(m.id) },
          '-',
          { icon: 'trash', label: loaded ? T('내리고 목록에서 제거') : T('목록에서 제거'), danger: true, disabled: loaded && !!S.streaming, run: () => removeModel(m, loaded) }], 'right'),
      }, I('dots'))),
    !s.ok && s.reason ? h('div', { class: 'qwhy' }, TM(s.reason)) : null);
}
async function unloadModel() {
  if (S.streaming) return;
  try { await api('/api/server/stop', {}); toast(T('모델을 내렸습니다'), 'ok'); resetSig(); refresh(); }
  catch (e) { toast(e.message, 'err'); }
}
// 이 모델로 채팅 — 늘 새 채팅(흐르는 채팅의 모델을 바꾸지 않는다)
function chatWith(id) {
  show('chat');
  const cur = curRoom();
  if (cur && !cur.messages.length && !(S.streaming && S.streaming.roomId === cur.id)) cur.model = id;
  else {
    S.rooms = S.rooms.filter(r => r.messages.length);
    const r = { id: Date.now().toString(36) + Math.random().toString(36).slice(2, 6), title: '새 채팅', model: id, messages: [], created: Date.now(), updated: Date.now() };
    S.rooms.unshift(r);
    S.roomId = r.id;
  }
  S.editing = null;
  saveRooms(); renderRooms(); renderMsgs(); renderChatTop(true);
  $('#input').focus();
}
async function verifyModel(id) {
  try { await api('/api/models/verify', { id }); toast(T('파일 무결성(SHA-256)을 확인합니다 — 8 GB 파일에 몇 초 걸립니다'), 'ok'); resetSig(); refresh(); }
  catch (e) { toast(e.message, 'err'); }
}
async function removeModel(m, loaded) {
  const inside = m.path.toLowerCase().startsWith(S.st.app.models_dir.toLowerCase());
  const cb = h('input', { type: 'checkbox' });
  const body = h('div', { class: 'col g16' }, h('p', { class: 'muted m0' }, T`${displayName(m)} ${quantOf(m)} — ${loaded ? T('불러온 모델을 내리고 ') : ''}목록에서 제거합니다.`),
    inside ? h('label', { class: 'check' }, cb, T`파일도 삭제 (${fmtBytes(m.size)})`) : h('p', { class: 'small muted m0' }, T('GeulPU 모델 폴더 밖의 파일은 삭제하지 않고 목록에서만 제거합니다.')));
  if (!(await confirmBox(T('모델 제거'), body, T('제거'), 'danger'))) return;
  try {
    if (loaded) await api('/api/server/stop', {});
    await api('/api/models/remove', { id: m.id, delete_file: inside && cb.checked });
    resetSig(); refresh();
  } catch (e) { toast(e.message, 'err'); }
}
// 받는 일의 상태 — 'run'(받는 중 · 확인 중) · 'pause'(멈춤 — 받은 데서 이어받음) · 'fail'(처음부터 다시) · null
function dlStateOf(c) {
  const d = c && c.download;
  if (!d || c.installed) return null;
  if (d.state === 'downloading' || d.state === 'verifying') return 'run';
  const again = d.state === 'failed' && /SHA-256|지웠|크기보다/.test(d.reason || '');
  if ((d.state === 'cancelled' || (d.state === 'failed' && !again)) && d.received > 0) return 'pause';
  if (d.state === 'failed') return 'fail';
  return null;
}
const dlList = () => ((S.st && S.st.catalog) || []).filter(c => dlStateOf(c));
// 다운로드 실패 까닭을 한 문장으로 — 누를 단추(이 상태의 단추)를 말하고, 윈도(WinHTTP) 오류 번호는 도움말로.
// 관리자가 붙이는 "— 다시 누르면 이어서 받습니다"는 뺀다(받은 것이 없으면 이어받기가 아니다)
const WINHTTP = { 12002: T('서버가 응답하지 않음(시간 초과)'), 12007: T('서버 주소를 찾지 못함'), 12029: T('서버에 연결하지 못함'), 12030: T('연결이 끊김'),
  12031: T('서버가 연결을 끊음'), 12152: T('서버의 응답이 올바르지 않음'), 12175: T('보안 연결(HTTPS)을 맺지 못함') };
function dlWhy(reason, c) {
  const s = String(reason || '').replace(/\s*—\s*다시 누르면 이어서 받습니다\.?\s*$/, '');
  const m = /\(오류 (\d+)\)/.exec(s);
  const tip = m ? T`WinHTTP 오류 ${m[1]}${WINHTTP[+m[1]] ? ' — ' + WINHTTP[+m[1]] : ''}` : '';
  const ds = c ? dlStateOf(c) : null;
  const btn = ds === 'pause' ? T("'이어서 다운로드'") : T("'다시 다운로드'");
  const d = c && c.download;
  const pct = d && d.total ? Math.floor(100 * d.received / d.total) : 0;
  let text;
  if (/연결이 끊겼|연결하지 못했|요청을 열지/.test(s)) text = ds === 'pause' ? T`연결이 끊겨 ${pct}%에서 멈췄습니다. 인터넷 연결을 확인한 뒤 ${btn}를 누르세요.`
    : T`다운로드 서버에 연결하지 못했습니다. 인터넷 · 방화벽 · 프록시를 확인한 뒤 ${btn}를 누르세요.`;
  else if (/HTTP (\d+)/.test(s)) text = T`다운로드 서버가 오류(HTTP ${/HTTP (\d+)/.exec(s)[1]})를 보냈습니다. 잠시 뒤 ${btn}를 누르세요.`;
  else if (/디스크에 쓰지|쓸 수 없습니다/.test(s)) text = T`${TM(s.replace(/\s*\([^)]*\)\s*$/, ''))}. 여유 공간과 폴더 권한을 확인한 뒤 ${btn}를 누르세요.`;
  else if (/SHA-256/.test(s)) text = T`받은 파일이 원본과 다릅니다(SHA-256 불일치). 받은 것을 지웠으니 ${btn}를 누르세요.`;
  else text = TM(s.replace(/\(오류 \d+\)/, '').trim()) || T('다운로드하지 못했습니다');
  return { text, tip };
}
const dlWhyLine = (reason, c) => { const w = dlWhy(reason, c); return h('div', { class: 'note-line err', 'data-tip': w.tip || null, 'data-tip-pos': 'top' }, I('alert', 's'), h('span', null, w.text)); };
// 받기 전 확인이 실패한 까닭(파일 앞부분을 받지 못함)
function checkWhy(reason) {
  const s = String(reason || '');
  if (/연결이 끊겼|연결하지 못했|요청을 열지/.test(s)) return T('Hugging Face에서 파일 앞부분을 받지 못했습니다. 인터넷 연결을 확인한 뒤 다시 확인하세요.');
  const m = /HTTP (\d+)/.exec(s);
  if (m) return T`Hugging Face가 오류(HTTP ${m[1]})를 보냈습니다. 잠시 뒤 다시 확인하세요.`;
  return TM(s) || T('확인하지 못했습니다');
}
async function resumeDownload(c) {
  try { await api('/api/models/download', { id: c.id }); resetSig(); refresh(); }
  catch (e) { toast(e.message, 'err'); }
}
function pauseDownload(c) {
  api('/api/models/cancel', { id: c.id }).then(() => { resetSig(); refresh(); }).catch(e => toast(e.message, 'err'));
}
// 단일 선택(radiogroup) — 화살표로 옮기고 고른다(다시 그려지면 새로 고른 것에 초점)
function radioKeys(ev) {
  if (!['ArrowLeft', 'ArrowRight', 'ArrowUp', 'ArrowDown'].includes(ev.key)) return;
  const bs = [...ev.currentTarget.querySelectorAll('[role=radio]')];
  if (!bs.length) return;
  const i = Math.max(0, bs.findIndex(b => b.getAttribute('aria-checked') === 'true'));
  const nx = bs[(i + (ev.key === 'ArrowRight' || ev.key === 'ArrowDown' ? 1 : bs.length - 1)) % bs.length];
  ev.preventDefault();
  const fk = nx.dataset.fk;
  nx.click();
  const again = fk ? document.querySelector(`[data-fk="${CSS.escape(fk)}"]`) : null;
  (again || nx).focus();
}
// 다운로드 목록(상태 막대에서 연다) — 받는 중 · 멈춤 · 실패를 모두, 그 자리에서 멈추고 잇는다
function dlPopBody() {
  const rows = dlList().map(c => {
    const ds = dlStateOf(c), d = c.download;
    return h('div', { class: 'dlrow' },
      h('div', { class: 'dlh' }, h('span', { class: 'nm' }, c.name), h('span', { class: 'qlab' }, c.quant)),
      ds === 'fail' ? dlWhyLine(d.reason, c) : dlBox(c, true),
      h('div', { class: 'row' },
        ds === 'run' && d.state === 'downloading' ? h('button', { class: 'btn sm ghost needs-conn', onclick: () => pauseDownload(c) }, I('pause', 's'), T('일시정지')) : null,
        ds === 'pause' ? h('button', { class: 'btn sm needs-conn', onclick: () => resumeDownload(c) }, I('download', 's'), T('이어서 다운로드')) : null,
        ds !== 'run' ? h('button', { class: 'btn sm ghost needs-conn', onclick: () => { closePop(); discardDownload(c); } }, I('trash', 's'), T('지우기')) : null,
        ds === 'fail' ? h('button', { class: 'btn sm needs-conn', onclick: () => { closePop(); startDownload(c); } }, I('refresh', 's'), T('다시 다운로드')) : null));
  });
  return h('div', { class: 'dlpop', role: 'dialog', 'aria-label': T('다운로드', 'list') },
    h('div', { class: 'pop-title' }, T('다운로드', 'list')),
    rows.length ? rows : h('div', { class: 'pad small muted' }, T('진행 중인 다운로드가 없습니다.')),
    h('div', { class: 'sep', role: 'separator' }),
    h('button', { class: 'opt', onclick: () => { closePop(); show('models'); } }, I('box', 's'), h('span', { class: 'main' }, h('div', { class: 't' }, T('모델 화면에서 보기')))));
}
const dlSig = () => JSON.stringify(dlList().map(c => [c.id, dlStateOf(c), c.download.state, c.download.reason]));
function openDownloads(anchor) {
  if (POP && POPANCHOR === anchor) { closePop(); return; }
  const p = openPop(anchor, dlPopBody(), 'left');
  p.dataset.kind = 'dl';
  p.dataset.sig = dlSig();
  updateDlBars();
  placePop(p, anchor, 'left');   // 막대의 글이 채워져 키가 자란 뒤에 자리를 다시
}
function refreshDlPop() {
  const s = dlSig();
  if (POP.dataset.sig === s) return;
  POP.dataset.sig = s;
  put(POP, dlPopBody());
  updateDlBars();
  if (POPANCHOR) placePop(POP, POPANCHOR, 'left');
}
async function startDownload(c) {
  const st = S.st;
  const f = fitOf(c);
  const bad = f && (f.k === 'no' || f.k === 'nogpu');
  const warn = !f ? null : f.k === 'nogpu' ? T('이 PC에서는 쓸 수 있는 GPU를 찾지 못해 이 모델을 실행할 수 없습니다. 다른 PC로 옮겨 쓸 파일이면 그대로 다운로드할 수 있습니다.')
    : f.k === 'no' ? T`이 PC의 그래픽 메모리(${fmtBytes(st.device.memory)})로는 실행되지 않을 수 있습니다. 그래픽 메모리 ${vramNeed(c)} GB 이상을 권장합니다.`
      : f.k === 'tight' ? T`이 PC의 그래픽 메모리(${fmtBytes(st.device.memory)})에서는 문맥 길이가 줄어듭니다.` : null;
  const body = h('div', { class: 'col g20' },
    h('p', { class: 'muted m0' }, `${c.name} ${c.quant}${NB}${catDesc(c)}`),
    warn ? h('div', { class: 'note-line' + (bad ? ' err' : '') }, I(bad ? 'alert' : 'info', 's'), h('span', null, warn)) : null,
    h('div', { class: 'kv' },
      h('div', { class: 'k' }, T('파일', 'k')), h('div', { class: 'v mono small break' }, c.file),
      h('div', { class: 'k' }, T('크기')), h('div', { class: 'v' }, T`${fmtBytes(c.size)} (${fmtNum(c.size)} 바이트)`),
      h('div', { class: 'k' }, T('저장 위치')), h('div', { class: 'v small break' }, T`${st.app.models_dir}${NB}여유 ${st.app.disk_free >= 0 ? fmtBytes(st.app.disk_free) : T('알 수 없음')}`),
      h('div', { class: 'k' }, T('출처')), h('div', { class: 'v small break' }, h('a', { href: c.url, target: '_blank', rel: 'noopener' }, c.source)),
      h('div', { class: 'k' }, T('라이선스')), h('div', { class: 'v' }, c.license)),
    h('details', { class: 'fold' }, h('summary', null, I('chevR', 's'), T('기술 정보')),
      h('div', { class: 'col g8' }, h('div', { class: 'small muted' }, TM(c.note)), h('div', { class: 'small mono muted break' }, 'SHA-256 ' + c.sha256))),
    c.check === 'untested' ? h('div', { class: 'note-line' }, I('info', 's'), h('span', null, T('이 파일로는 아직 GeulPU에서 실행해 보지 않았습니다. 다운로드가 끝나면 지원 여부를 바로 알려 드립니다.'))) : null,
    h('div', { class: 'note-line' }, I('info', 's'), h('span', null, T('중간에 끊겨도 다시 누르면 이어서 받고, 다 받으면 SHA-256으로 파일이 맞는지 확인합니다.'))));
  if (!(await confirmBox(T('모델 다운로드'), body, T('다운로드'), bad ? 'quiet' : null))) return;
  try { await api('/api/models/download', { id: c.id }); resetSig(); refresh(); }
  catch (e) { toast(e.message, 'err'); }
}
// 파일 고르기 — pick 이 있으면 폴더 고르기(설정의 모델 폴더): 파일은 감추고 '이 폴더 선택'
async function browse(path, pick) {
  let j;
  try { j = await api('/api/fs?path=' + encodeURIComponent(path)); }
  catch (e) { toast(e.message, 'err'); return; }
  const list = h('div', { class: 'fs-list' });
  if (j.path) list.append(h('button', { class: 'fs-item', onclick: () => browse(j.parent || '', pick) }, I('chevU', 's'), h('span', { class: 'n' }, T('상위 폴더'))));
  const entries = pick ? j.entries.filter(e => e.dir) : j.entries;
  for (const e of entries) {
    list.append(h('button', {
      class: 'fs-item', onclick: async () => {
        if (e.dir) { browse(e.path, pick); return; }
        closeModal();
        try { await api('/api/models/add', { path: e.path }); toast(T`추가했습니다: ${e.name}`, 'ok'); resetSig(); refresh(); }
        catch (err) { toast(err.message, 'err'); }
      },
    }, I(e.dir ? 'folder' : 'file', 's'), h('span', { class: 'n' }, e.name), e.dir ? I('chevR', 's') : h('span', { class: 'num muted small' }, fmtBytes(e.size))));
  }
  if (!entries.length) list.append(h('div', { class: 'fs-item muted' }, pick ? T('하위 폴더가 없습니다') : T('이 폴더에는 .gguf 파일이나 하위 폴더가 없습니다')));
  const inp = h('input', { type: 'text', value: j.path || '', placeholder: T('폴더 경로(예: D:\\models)'), 'aria-label': T('폴더 경로') });
  const go = () => browse(inp.value.trim(), pick);
  inp.addEventListener('keydown', ev => { if (ev.key === 'Enter') go(); });
  modal(h('div', null, h('h2', null, pick ? T('모델 폴더 고르기') : T('GGUF 파일 추가')),
    h('p', { class: 'small muted m0 mb12' }, pick ? T('받은 모델을 둘 폴더를 고르세요. 다음 실행부터 적용됩니다.') : T('GeulPU는 qwen2 구조(Qwen2.5 · DeepSeek-R1-Distill-Qwen, 7B 이상), qwen3 구조(Qwen3 1.7B · 8B · 14B), llama 구조(Llama 3.1 8B · Llama 3.2 3B · Mistral 7B v0.3)의 GGUF를 실행합니다. 추가하면 지원 여부를 바로 확인합니다.')),
    h('div', { class: 'fs-path' }, inp, h('button', { class: 'btn', onclick: go }, T('이동'))), list),
    [h('button', { class: 'btn ghost', onclick: () => closeModal() }, T('닫기')),
      pick ? h('button', { class: 'btn primary', disabled: j.path ? null : true, onclick: () => { closeModal(); pick(j.path); } }, T('이 폴더 선택')) : null].filter(Boolean), false, '.fs-item');
}

// ---------- 모델 찾기 — Hugging Face 의 GGUF 저장소 전체를 정렬해 둘러본다(검색은 좁히는 도구) ----------
// 목록 · 자세히 · 파일 목록 · 모델 설명(README)은 브라우저가 Hugging Face API 에 바로 묻는다(정렬 · 거르기 · 검색어가 Hugging Face 로 간다).
// 받기 전 확인(파일 앞 몇 MB 의 GGUF 머리)과 받기는 관리자가 — 주소는 관리자가 저장소 · 파일 이름으로 짓는다.
const HFHOST = 'https://huggingface.co';
// GeulPU 가 받는 형식(엔진: Q8_0 · Q4_0 · Q4_K · Q5_K · Q6_K) — 파일 이름으로 먼저 어림한다(확인은 GGUF 머리로)
const quantFromName = n => { const m = /(IQ\d_[A-Z]+|Q\d_K(?:_[SML])?|Q\d_\d|BF16|F16|F32)/i.exec(n || ''); return m ? m[1].toUpperCase() : ''; };
const quantOk = q => /^(Q8_0|Q4_0|Q4_K(_[SM])?|Q5_K(_[SM])?|Q6_K)$/.test(q);
const hfName = id => String(id).split('/').pop().replace(/[-_.]?GGUF$/i, '');
const archOk = a => a === 'qwen2' || a === 'qwen3' || a === 'llama';      // 39단계 — llama(Llama 3.1 · 3.2 · Mistral)
// 크기로 어림하는 지원 여부(확인은 받기 전에 GGUF 머리로) — 매개변수 수(미세조정도 바탕과 같고, 출력을 따로 둔 변환은 낱말표만큼 크다).
// Qwen3: 1.7B · 8B · 14B 는 실행(질의 폭 = 너비), 0.6B · 4B · 32B 는 아직 아님(질의 폭 ≠ 너비), 그 밖의 크기는 받기 전 확인.
// Qwen2: 헤드 크기 64 인 작은 것(0.5B 등)은 아직 아님
const Q3OK = [[1.6e9, 2.1e9], [7.9e9, 8.4e9], [14.5e9, 15.0e9]];
const Q3BAD = [[0.55e9, 0.8e9], [3.9e9, 4.5e9], [32e9, 33.5e9]];
// 39단계 — llama 구조: Llama 3.2 3B · Mistral 7B · Llama 3.1 8B · 70B 는 실행(머리 128 · 질의 폭 = 너비), 헤드 64 인 작은 것(1B 등)은 아직 아님,
// 그 밖의 크기(Mistral NeMo 12B 등)는 받기 전 확인(채팅 틀도 그때 본다 — ChatML 미세조정 등은 거른다)
const L3OK = [[3.1e9, 3.3e9], [7.1e9, 7.4e9], [7.9e9, 8.1e9], [70.4e9, 70.7e9]];
const inRange = (p, rs) => rs.some(([lo, hi]) => p >= lo && p <= hi);
// 대화용이 아닌 모델(이미지 생성의 텍스트 인코더 · 임베딩 · 순위 매기기 · 음성) — 같은 구조라도 채팅에 쓰지 못한다(꼬리표 · 용도 · 이름으로)
const NONCHAT = ['text-to-image', 'image-to-image', 'feature-extraction', 'sentence-similarity', 'text-classification', 'text-ranking',
  'text-to-speech', 'automatic-speech-recognition', 'text-to-audio', 'audio-to-audio'];
const fdNonChat = m => {
  const pipe = m && m.cardData && m.cardData.pipeline_tag;
  return (pipe && NONCHAT.includes(pipe)) || ((m && m.tags) || []).some(t => NONCHAT.includes(t)) || /text[-_ ]?encoder|embedding|reranker|[-_]tts\b/i.test((m && m.id) || '');
};
function fdSupport(m) {
  const g = (m && m.gguf) || {};
  const a = g.architecture, p = g.total || 0;
  if (!a) return { k: 'unknown', t: T('구조 정보 없음') };
  if (!archOk(a)) return { k: 'arch', t: T`${a} · 미지원 구조` };
  if (fdNonChat(m)) return { k: 'task', t: T('대화용 모델 아님'), why: T('이미지 생성용 텍스트 인코더 · 임베딩 · 음성처럼 채팅에 쓰는 모델이 아닙니다.') };
  if (a === 'llama') {
    if (inRange(p, L3OK)) return { k: 'ok', t: T('GeulPU에서 실행됨') };
    if (p > 0 && p < 2.5e9) return { k: 'size', t: T('이 크기는 아직 미지원'), why: T('헤드 크기가 64인 작은 llama 구조(Llama 3.2 1B · TinyLlama 등)는 GeulPU가 아직 실행하지 않습니다.') };
    return { k: 'check', t: T('크기 확인 필요'), why: T('흔한 Llama 3 · Mistral 크기(3B · 7~8B · 70B)가 아닙니다. 받기 전에 파일 앞부분을 받아 실행되는지 확인합니다.') };
  }
  if (a === 'qwen3') {
    if (inRange(p, Q3OK)) return { k: 'ok', t: T('GeulPU에서 실행됨') };
    if (inRange(p, Q3BAD)) return { k: 'size', t: T('이 크기는 아직 미지원'), why: T('Qwen3의 0.6B · 4B · 32B는 질의 폭이 너비와 달라 GeulPU가 아직 실행하지 않습니다.') };
    return { k: 'check', t: T('크기 확인 필요'), why: T('흔한 Qwen3 크기(1.7B · 8B · 14B)가 아닙니다. 받기 전에 파일 앞부분을 받아 실행되는지 확인합니다.') };
  }
  if (p > 0 && p < 1.4e9) return { k: 'size', t: T('이 크기는 아직 미지원'), why: T('헤드 크기가 64인 작은 Qwen2(0.5B 등)는 GeulPU가 아직 실행하지 않습니다.') };
  return { k: 'ok', t: T('GeulPU에서 실행됨') };
}
const fmtParams = p => (p >= 1e9 ? (p / 1e9).toFixed(p >= 1e10 ? 0 : 1) + 'B' : Math.round(p / 1e6) + 'M');
// 수는 만 · 억으로(55.5만 · 1.2억)
function fmtCount(n) {
  n = n || 0;
  if (EN) {   // 영어 화면: 1.2K · 55K · 1.2M · 1.2B
    if (n >= 1e9) return (n / 1e9).toFixed(1).replace(/\.0$/, '') + 'B';
    if (n >= 1e6) return (n / 1e6).toFixed(1).replace(/\.0$/, '') + 'M';
    if (n >= 1e3) return (n >= 1e5 ? String(Math.round(n / 1e3)) : (n / 1e3).toFixed(1).replace(/\.0$/, '')) + 'K';
    return fmtNum(n);
  }
  if (n >= 1e8) return (n / 1e8).toFixed(1).replace(/\.0$/, '') + '억';
  if (n >= 1e4) return (n >= 1e6 ? String(Math.round(n / 1e4)) : (n / 1e4).toFixed(1).replace(/\.0$/, '')) + '만';
  return fmtNum(n);
}
function fmtAgo(iso) {
  const t = Date.parse(iso || '');
  if (!(t > 0)) return '-';
  const d = Math.floor((Date.now() - t) / 86400000);
  if (d < 1) return T('오늘');
  if (d < 2) return T('어제');
  if (d < 7) return T`${d}일 전`;
  if (d < 31) return T`${Math.floor(d / 7)}주 전`;
  if (d < 365) return T`${Math.floor(d / 30)}개월 전`;
  return T`${Math.floor(d / 365)}년 전`;
}
const fmtDate = iso => { const t = Date.parse(iso || ''); return t > 0 ? new Date(t).toLocaleDateString(EN ? 'en-US' : 'ko-KR', { year: 'numeric', month: 'long', day: 'numeric' }) : '-'; };
const asList = v => (Array.isArray(v) ? v : v ? [v] : []).map(String);
// 저장소의 라이선스 — cardData.license, 없으면 꼬리표(license:…)
const fdLicense = m => (m && m.cardData && m.cardData.license) || ((m && m.tags) || []).map(String).filter(t => t.startsWith('license:')).map(t => t.slice(8))[0] || '';
// 목록(앱에 든 받을 수 있는 모델 · Hugging Face 에서 더한 것)의 항목이 어느 저장소의 것인가
const fdRepoOf = c => { const m = /huggingface\.co\/([^/]+\/[^/]+)\/resolve\//.exec(c.url || ''); return m ? m[1] : String(c.source || '').replace(/^huggingface\.co\//, ''); };
const fdCat = repo => ((S.st && S.st.catalog) || []).filter(c => fdRepoOf(c).toLowerCase() === String(repo).toLowerCase());
const fdChecked = repo => fdCat(repo).some(c => c.check === 'tested' || c.check === 'crosscheck');
// 매개변수 수로 Q4_K_M 파일 크기를 어림한다(4.85 비트/매개변수) — '이 PC 메모리에 맞는 크기' 거르기
const fdEst = p => p * 4.85 / 8;
const fdFits = p => { const f = fitOf({ size: fdEst(p), params: fmtParams(p), id: '' }); return !!f && (f.k === 'ok' || f.k === 'tight'); };

// 기본은 '요즘 뜨는'(Hugging Face 의 트렌드 점수) — 30일 다운로드 순은 다운로드가 부풀려진 저장소가 위에 오기 쉽다
const FD_SORTS = [['trendingScore', T('요즘 뜨는'), T('Hugging Face의 트렌드 점수 순')], ['downloads', T('인기'), T('최근 30일 다운로드가 많은 순')],
  ['likes', T('좋아요'), T('좋아요가 많은 순')], ['lastModified', T('최근 업데이트'), T('최근에 고친 저장소 순')]];
const FD_SIZES = [['all', T('모든 크기')], ['s', '~4B'], ['m', '4~10B'], ['l', '10~20B'], ['xl', '20B~']];
const FD_EXPAND = ['gguf', 'downloads', 'likes', 'lastModified', 'cardData', 'trendingScore', 'author', 'createdAt', 'tags'].map(x => 'expand[]=' + x).join('&');
const FD = {
  el: null, bar: null, sum: null, list: null, more: null, scroll: null, det: null, y: [0, 0],
  // only: GeulPU 에서 실행되는 것만 — 늘 켠 채로 시작한다(실행 안 되는 모델도 보기는 그때그때 켠다 — 저장하지 않음)
  sort: load('geulpu.fd.sort', 'trendingScore'), only: true, size: load('geulpu.fd.size', 'all'), fit: load('geulpu.fd.fit', false),
  checked: false, q: '', draft: '',
  items: [], seen: new Set(), srcs: [], busy: false, done: false, err: '', seq: 0, started: false, dry: 0,
  sel: null, modal: null, info: {}, chk: new Map(), readmeOpen: {},
};
// 모델 화면의 '모델 찾기' 칸으로(첫 화면의 단추 등) — 검색 칸에 초점
function openHfSearch() {
  show('models');
  setModelsTab('find');
  setTimeout(() => { const i = FD.bar && FD.bar.querySelector('input'); if (i) i.focus({ preventScroll: true }); }, 60);
}
// 찾기 칸 — 한 번 만들어 두고 다시 쓴다(위: 검색 · 정렬 · 거르기, 아래: 왼쪽 목록 · 오른쪽 자세히).
// 넓은 창에서는 화면 전체가 아니라 목록 칸과 자세히 칸이 저마다 스크롤된다 — 좁은 창은 목록 하나라 화면 전체가 스크롤되고, 자세히는 창으로 연다
function finderEl() {
  if (FD.el) return FD.el;
  FD.bar = h('div', { class: 'fd-bar' });
  FD.sum = h('div', { class: 'fd-sum', 'aria-live': 'polite' });
  FD.list = h('div', { class: 'fd-list', 'aria-label': T('모델 목록') });
  FD.more = h('div', { class: 'fd-more' });
  FD.scroll = h('div', { class: 'fd-scroll' }, FD.list, FD.more);
  FD.det = h('aside', { class: 'fd-detail', 'aria-label': T('모델 자세히') });
  FD.el = h('div', { class: 'finder' }, FD.bar, h('div', { class: 'fd-body' }, FD.sum, FD.scroll, FD.det));
  fdBar();
  // 두 칸의 스크롤 자리를 적어 둔다 — 화면을 다시 짜면(받기를 시작해 칸 수가 바뀔 때 · 다른 화면에 다녀올 때) 칸이 떨어졌다 붙어 처음으로 가므로 되돌린다
  [FD.scroll, FD.det].forEach((el, i) => el.addEventListener('scroll', () => { if (el.getClientRects().length) FD.y[i] = el.scrollTop; }, { passive: true }));
  // 목록 끝이 가까워지면 더 불러온다('더 보기' 단추도 그대로 둔다 — 키보드 · 화면 낭독기)
  FD.scroll.addEventListener('scroll', fdAutoMore, { passive: true });
  $('#view-models').addEventListener('scroll', fdAutoMore, { passive: true });
  return FD.el;
}
function finderStart() {
  if (!FD.started) { FD.started = true; fdReload(); return; }
  FD.scroll.scrollTop = FD.y[0];
  FD.det.scrollTop = FD.y[1];
  fdDrawList();
  fdRedraw();
}
function fdAutoMore() { if (FD.started && !FD.busy && !FD.done && !FD.err && fdNearEnd()) fdMore(); }
function fdBarFocus(fn) {
  const a = document.activeElement;
  const fk = a && FD.bar.contains(a) ? a.dataset.fk : null;
  fn();
  if (fk) { const b = FD.bar.querySelector(`[data-fk="${CSS.escape(fk)}"]`); if (b) b.focus({ preventScroll: true }); }
}
function fdBar() {
  fdBarFocus(() => {
    const inp = h('input', { type: 'text', placeholder: T('이름으로 좁히기 — 예: qwen3, coder, r1, 14b'), 'aria-label': T('모델 이름으로 좁히기'), 'data-fk': 'fd-q', enterkeyhint: 'search' });
    inp.value = FD.draft;
    let t = 0;
    inp.addEventListener('input', () => { FD.draft = inp.value; clearTimeout(t); t = setTimeout(() => fdSearch(inp.value.trim()), 450); fdClearBtn(); });
    inp.addEventListener('keydown', e => {
      if (e.key === 'Enter' && !e.isComposing) { e.preventDefault(); clearTimeout(t); fdSearch(inp.value.trim()); }
      if (e.key === 'Escape' && inp.value) { e.preventDefault(); inp.value = ''; FD.draft = ''; clearTimeout(t); fdSearch(''); fdClearBtn(); }
    });
    const clear = h('button', { class: 'icon-btn sm', 'aria-label': T('검색어 지우기'), 'data-fk': 'fd-qx', hidden: FD.draft ? null : true,
      onclick: () => { inp.value = ''; FD.draft = ''; fdSearch(''); fdClearBtn(); inp.focus(); } }, I('x', 's'));
    const radio = (list, cur, label, fk, set) => h('div', { class: 'seg-ctl fd-seg', role: 'radiogroup', 'aria-label': label, onkeydown: radioKeys }, list.map(([k, text, tip]) =>
      h('button', {
        role: 'radio', class: cur === k ? 'on' : '', 'aria-checked': cur === k ? 'true' : 'false', tabindex: cur === k ? '0' : '-1', 'data-fk': fk + ':' + k,
        'data-tip': tip || null, 'data-tip-pos': 'bottom', onclick: () => { if (cur !== k) set(k); },
      }, text)));
    const chip = (on, text, tip, fk, flip) => h('button', { class: 'fchip' + (on ? ' on' : ''), 'aria-pressed': on ? 'true' : 'false', 'data-fk': fk, 'data-tip': tip, 'data-tip-pos': 'bottom', onclick: flip },
      on ? I('check', 's') : null, text);
    // 기본은 GeulPU 에서 실행되는 것만 — 켜면 실행 안 되는 모델(다른 구조 · 아직 미지원 크기)도 흐리게 함께 보인다
    const all = h('input', { type: 'checkbox', role: 'switch', 'data-fk': 'fd-all', 'aria-label': T('실행 안 되는 모델도 보기'),
      onchange: () => { FD.only = !all.checked; fdBar(); fdReload(); } });
    all.checked = !FD.only;
    const allSw = h('label', { class: 'fd-toggle', 'data-tip': T('끄면 GeulPU에서 실행되는 구조(qwen2 · qwen3 · llama)와 크기만 보입니다'), 'data-tip-pos': 'bottom' },
      h('span', { class: 'switch' }, all, h('i')), h('span', null, T('실행 안 되는 모델도 보기')));
    put(FD.bar,
      h('div', { class: 'fd-line' },
        h('div', { class: 'fd-search' }, I('search', 's'), inp, clear),
        radio(FD_SORTS, FD.sort, T('정렬'), 'fd-sort', k => { FD.sort = k; store('geulpu.fd.sort', k); fdBar(); fdReload(); })),
      h('div', { class: 'fd-line' },
        allSw,
        h('span', { class: 'fd-div', 'aria-hidden': 'true' }),
        chip(FD.checked, T('GeulPU 확인됨'), T('GeulPU에서 실제로 돌려 본 파일이 있는 저장소만'), 'fd-checked', () => { FD.checked = !FD.checked; fdBar(); fdReload(); }),
        chip(FD.fit, T('이 PC에 맞는 크기'), T('매개변수 수로 어림한 Q4_K_M 파일이 이 PC의 그래픽 메모리에 들어가는 것만'), 'fd-fit', () => { FD.fit = !FD.fit; store('geulpu.fd.fit', FD.fit); fdBar(); fdReload(); }),
        h('span', { class: 'grow' }),
        radio(FD_SIZES, FD.size, T('크기'), 'fd-size', k => { FD.size = k; store('geulpu.fd.size', k); fdBar(); fdReload(); })));
  });
}
function fdClearBtn() { const b = FD.bar.querySelector('[data-fk="fd-qx"]'); if (b) b.hidden = !FD.draft; }
function fdSearch(q) { if (q === FD.q) return; FD.q = q; fdReload(); }
// 정렬 · 거르기 · 검색어가 바뀌면 처음부터 — '실행되는 것만'은 꼬리표 qwen2 · qwen3 · llama · mistral 의 목록을 정렬 열쇠로 맞물려 하나로
// (39단계 — llama 구조의 GGUF 저장소는 꼬리표가 llama · mistral 이거나 없다: 꼬리표 없는 것은 '실행되는 것만'을 끄고 찾는다)
function fdReload() {
  FD.seq++;
  FD.items = []; FD.seen = new Set(); FD.done = false; FD.err = ''; FD.busy = false; FD.dry = 0;
  const base = `${HFHOST}/api/models?filter=gguf&sort=${FD.sort}&direction=-1&limit=100&${FD_EXPAND}${FD.q ? '&search=' + encodeURIComponent(FD.q) : ''}`;
  FD.srcs = FD.checked ? [] : (FD.only ? ['qwen2', 'qwen3', 'llama', 'mistral'].map(t => base + '&filter=' + t) : [base]).map(u => ({ next: u, buf: [] }));
  fdDrawList();
  FD.scroll.scrollTop = FD.y[0] = 0;
  fdMore();
}
const fdKey = m => (FD.sort === 'lastModified' ? Date.parse(m.lastModified || '') || 0 : Number(m[FD.sort]) || 0);
function fdPass(m) {
  const sup = fdSupport(m);
  if (FD.only && sup.k !== 'ok') return false;
  const p = (m.gguf && m.gguf.total) || 0;
  if (FD.size !== 'all') {
    if (!p) return false;
    const b = p / 1e9;
    if (FD.size === 's' && b > 4.5) return false;
    if (FD.size === 'm' && (b <= 4.5 || b > 10.5)) return false;
    if (FD.size === 'l' && (b <= 10.5 || b > 20.5)) return false;
    if (FD.size === 'xl' && b <= 20.5) return false;
  }
  if (FD.fit && !(p && fdFits(p))) return false;
  if (FD.q && FD.checked && !m.id.toLowerCase().includes(FD.q.toLowerCase())) return false;
  return true;
}
async function fdFetch(s) {
  const r = await fetch(s.next);
  if (!r.ok) throw new Error('HTTP ' + r.status);
  const j = await r.json();
  if (Array.isArray(j)) s.buf.push(...j);
  const m = /<([^>]+)>;\s*rel="next"/.exec(r.headers.get('Link') || '');
  s.next = m ? m[1] : null;
}
// 다음 n 개(거른 뒤) — 목록마다 정렬된 차례라 머리끼리 견주어 큰 것부터(같은 저장소는 한 번만). 한 번에 받는 쪽수는 여덟까지
async function fdMore(n = 30) {
  if (FD.busy || FD.done) return;
  const my = FD.seq;
  const before = FD.items.length;
  FD.busy = true; FD.err = '';
  fdDrawMore();
  try {
    if (FD.checked) await fdLoadChecked(my);
    else {
      let added = 0, pages = 0;
      while (added < n && pages < 8) {
        for (const s of FD.srcs) if (!s.buf.length && s.next) { pages++; await fdFetch(s); if (my !== FD.seq) return; }
        const live = FD.srcs.filter(s => s.buf.length);
        if (!live.length) { FD.done = true; break; }
        let best = live[0];
        for (const s of live) if (fdKey(s.buf[0]) > fdKey(best.buf[0])) best = s;
        const m = best.buf.shift();
        if (!m || FD.seen.has(m.id)) continue;
        FD.seen.add(m.id);
        if (fdPass(m)) { FD.items.push(m); added++; }
      }
      if (!FD.srcs.some(s => s.buf.length || s.next)) FD.done = true;
    }
  } catch (_) { if (my === FD.seq) FD.err = T('Hugging Face에 연결하지 못했습니다. 인터넷 연결을 확인한 뒤 다시 시도하세요.'); }
  if (my !== FD.seq) return;
  FD.busy = false;
  FD.dry = FD.items.length === before ? FD.dry + 1 : 0;
  fdDrawList();
  // 넓은 창에서는 첫 모델을 옆에 — 처음 열 때, 그리고 새로 불러온 목록에 고른 것이 없을 때
  if (FD.items.length && !fdNarrow() && (!FD.sel || (before === 0 && !FD.items.some(i => i.id === FD.sel)))) fdSelect(FD.items[0].id, true);
  // 목록 끝이 아직 화면 가까이에 있으면 더 — 거르기가 빡빡해 세 번 잇달아 하나도 못 얻으면 멈춘다('더 보기'로 잇는다)
  requestAnimationFrame(() => { if (!FD.done && !FD.err && !FD.busy && FD.dry < 3 && fdNearEnd()) fdMore(); });
}
function fdNearEnd() {
  if (!FD.more || !FD.more.isConnected || !FD.more.getClientRects().length) return false;
  // 넓은 창은 목록 칸이, 좁은 창은 화면 전체가 스크롤된다
  const sc = fdNarrow() ? $('#view-models') : FD.scroll;
  return FD.more.getBoundingClientRect().top < sc.getBoundingClientRect().bottom + 600;
}
// 'GeulPU 확인됨' — 앱의 목록에서 실제로 돌려 본 파일이 있는 저장소(몇 개라 한 번에), 정보는 Hugging Face 에서
async function fdLoadChecked(my) {
  const repos = [...new Set(((S.st && S.st.catalog) || []).filter(c => c.check === 'tested' || c.check === 'crosscheck').map(fdRepoOf).filter(Boolean))];
  const got = await Promise.all(repos.map(r => fetch(`${HFHOST}/api/models/${r}?${FD_EXPAND}`).then(x => (x.ok ? x.json() : null)).catch(() => null)));
  if (my !== FD.seq) return;
  const list = got.map((m, i) => m || { id: repos[i] }).filter(fdPass);
  list.sort((a, b) => fdKey(b) - fdKey(a));
  FD.items = list;
  FD.done = true;
}
const fdNarrow = () => window.matchMedia('(max-width: 960px)').matches;
function fdDrawList() {
  if (!FD.list) return;
  put(FD.list, FD.items.map(fdRow));
  const sortName = (FD_SORTS.find(s => s[0] === FD.sort) || FD_SORTS[0])[1];
  const conds = [FD.only ? T('GeulPU에서 실행되는 것') : T('실행 안 되는 것 포함'), FD.checked ? T('GeulPU 확인됨') : null, FD.fit ? T('이 PC에 맞는 크기') : null,
    FD.size !== 'all' ? (FD_SIZES.find(s => s[0] === FD.size) || [])[1] : null, FD.q ? `“${FD.q}”` : null].filter(Boolean);
  put(FD.sum, h('span', { class: 'ink' }, FD.items.length ? T`${fmtNum(FD.items.length)}개${FD.done ? '' : T(' 이상')}` : FD.busy ? T('불러오는 중') : T('0개')),
    h('span', null, T`${NB}${sortName}순${NB}${conds.join(' · ')}`),
    h('span', { class: 'grow' }),
    h('span', { class: 'faint' }, T('Hugging Face의 GGUF 저장소')));
  fdDrawMore();
}
function fdDrawMore() {
  if (!FD.more) return;
  let body = null;
  if (FD.busy) body = h('div', { class: 'row center small muted' }, h('i', { class: 'dot busy' }), T('불러오는 중…'));
  else if (FD.err) body = h('div', { class: 'col center g12' }, h('div', { class: 'note-line err' }, I('alert', 's'), h('span', null, FD.err)),
    h('button', { class: 'btn sm needs-conn', onclick: () => { FD.err = ''; fdMore(); } }, I('refresh', 's'), T('다시 시도')));
  else if (!FD.items.length && FD.done) body = h('div', { class: 'empty-box' }, h('div', { class: 'big' }, T('조건에 맞는 모델이 없습니다')),
    h('div', null, FD.only ? T("거르기를 줄이거나, '실행 안 되는 모델도 보기'를 켜면 Hugging Face의 모든 GGUF 모델이 보입니다(실행되지 않는 것은 흐리게).") : T('검색어나 거르기를 바꿔 보세요.')));
  else if (!FD.done) body = h('button', { class: 'btn ghost w100 needs-conn', onclick: () => fdMore() }, I('chevD', 's'), T('더 보기'));
  put(FD.more, body);
}
function fdStat(icon, text, label) { return h('span', { class: 'st', 'aria-label': label ? `${label} ${text}` : null }, I(icon, 's'), h('span', { 'aria-hidden': label ? 'true' : null }, text)); }
function fdRow(m) {
  const g = m.gguf || {};
  const sup = fdSupport(m);
  const cat = fdCat(m.id);
  const st = S.st;
  const rec = st && cat.some(c => c.id === recommendId(st));
  const base = asList(m.cardData && m.cardData.base_model);
  const by = [m.author || m.id.split('/')[0], base.length ? T`${base[0].split('/').pop()} 기반` : null].filter(Boolean).join(NB);
  const badge = fdChecked(m.id) ? h('span', { class: 'tag', 'data-tip': T('GeulPU에서 실제로 돌려 본 파일이 있습니다'), 'data-tip-pos': 'top' }, I('shield'), T('GeulPU 확인됨'))
    : sup.k === 'ok' ? (FD.only ? null : h('span', { class: 'tag' }, I('check'), T('실행됨')))
      : h('span', { class: 'tag dim', 'data-tip': sup.why || null, 'data-tip-pos': 'top' }, sup.k === 'unknown' ? T('구조 알 수 없음') : sup.t);
  return h('button', {
    class: 'fd-row' + (FD.sel === m.id ? ' on' : '') + (sup.k === 'arch' || sup.k === 'size' || sup.k === 'task' ? ' dim' : ''), 'data-repo': m.id, 'data-fk': 'fd:' + m.id,
    'aria-current': FD.sel === m.id ? 'true' : null, onclick: () => fdSelect(m.id),
  },
    h('span', { class: 'main' },
      h('span', { class: 'top' }, h('span', { class: 'nm' }, hfName(m.id)), g.total ? h('span', { class: 'qlab' }, fmtParams(g.total)) : null),
      h('span', { class: 'by' }, by),
      h('span', { class: 'foot' },
        fdStat('download', fmtCount(m.downloads), T('최근 30일 다운로드')), fdStat('heart', fmtCount(m.likes), T('좋아요')), fdStat('clock', fmtAgo(m.lastModified), T('업데이트')),
        h('span', { class: 'grow' }), rec ? h('span', { class: 'tag' }, I('star'), T('이 PC 추천')) : null, badge)));
}
function fdSelect(repo, auto) {
  FD.sel = repo;
  for (const b of FD.list.querySelectorAll('.fd-row')) {
    const on = b.dataset.repo === repo;
    b.classList.toggle('on', on);
    if (on) b.setAttribute('aria-current', 'true'); else b.removeAttribute('aria-current');
  }
  fdLoadInfo(repo);
  if (fdNarrow() && !auto) {
    FD.modal = repo;
    modal(h('div', { class: 'fd-d-wrap' }, fdDetail(repo)), [h('button', { class: 'btn', onclick: () => closeModal() }, T('닫기'))], true, '.foot .btn', () => { FD.modal = null; });
  }
  fdRedraw();
  if (!fdNarrow()) FD.det.scrollTop = FD.y[1] = 0;
}
// 저장소 하나의 정보(자세한 정보 · 파일 목록 · 모델 설명) — 셋을 함께 묻고 오는 대로 그린다
function fdLoadInfo(repo) {
  if (FD.info[repo] && !FD.info[repo].err) return;
  const x = FD.info[repo] = { busy: true };
  const done = () => { if (FD.sel === repo || FD.modal === repo) fdRedraw(); };
  const enc = repo.split('/').map(encodeURIComponent).join('/');
  fetch(`${HFHOST}/api/models/${enc}?${FD_EXPAND}&expand[]=downloadsAllTime`).then(r => (r.ok ? r.json() : Promise.reject(new Error('HTTP ' + r.status))))
    .then(m => { x.m = m; }).catch(() => { x.err = true; }).finally(() => { x.busy = !(x.files || x.ferr); done(); });
  fetch(`${HFHOST}/api/models/${enc}/tree/main?recursive=true`).then(r => (r.ok ? r.json() : Promise.reject(new Error('HTTP ' + r.status))))
    .then(j => { x.files = (Array.isArray(j) ? j : []).filter(f => f.type === 'file' && /\.gguf$/i.test(f.path)); })
    .catch(() => { x.ferr = true; x.err = true; }).finally(() => { x.busy = !x.m && !x.err; done(); });
  fetch(`${HFHOST}/${enc}/resolve/main/README.md`).then(r => (r.ok ? r.text() : '')).then(t => { x.readme = t || ''; }).catch(() => { x.readme = ''; }).finally(done);
}
function fdRedraw() {
  if (FD.det && FD.el && FD.el.isConnected) {
    const top = FD.det.scrollTop;
    put(FD.det, FD.sel ? fdDetail(FD.sel) : fdDetailEmpty());
    FD.det.scrollTop = top;
  }
  if (FD.modal && !$('#modal').hidden) { const mb = $('#modalbox .mb .fd-d-wrap'); if (mb) put(mb, fdDetail(FD.modal)); }
  updateDlBars();
}
function fdDetailEmpty() {
  return h('div', { class: 'fd-empty' }, I('box', 'l'), h('div', { class: 'big' }, T('모델을 고르면 자세한 정보가 여기에 보입니다')),
    h('div', { class: 'small muted' }, T`크기${NB}구조${NB}다운로드 수${NB}라이선스${NB}파일마다 이 PC에 맞는지${NB}모델 설명`));
}
// README — 앞머리(YAML) · 그림 · HTML 꼬리표를 떼고 마크다운만(md 는 글을 늘 탈출해서 넣고 http(s) 링크만 만든다)
function fdReadme(text) {
  let s = String(text || '').replace(/\r\n/g, '\n');
  s = s.replace(/^---\n[\s\S]*?\n---\n/, '');
  s = s.replace(/<!--[\s\S]*?-->/g, '');
  s = s.replace(/!\[[^\]]*\]\([^)]*\)/g, '');
  s = s.replace(/<\/?[a-zA-Z][a-zA-Z0-9-]*(?:\s[^<>]*)?\/?>/g, '');
  s = s.replace(/\n{3,}/g, '\n\n').trim();
  return s.length > 16000 ? s.slice(0, 16000) + '\n\n…' : s;
}
function fdDetail(repo) {
  const st = S.st;
  const x = FD.info[repo] || {};
  const m = Object.assign({ id: repo }, FD.items.find(i => i.id === repo) || {}, x.m || {});
  const g = m.gguf || {};
  const sup = fdSupport(m);
  const cat = fdCat(repo);
  const rec = st && cat.some(c => c.id === recommendId(st));
  const lic = fdLicense(m);
  const base = asList(m.cardData && m.cardData.base_model);
  const langs = asList(m.cardData && m.cardData.language);
  const kind = kindOf({ id: repo });
  const sub = [kind, g.architecture ? T`${g.architecture} 구조` : null, g.total ? T`매개변수 ${fmtParams(g.total)}` : null, g.context_length ? T`문맥 최대 ${fmtCtx(g.context_length)}` : null].filter(Boolean).join(NB);
  const cell = (k, v, s) => h('div', { class: 'cell' }, h('div', { class: 'k' }, k), h('div', { class: 'v' }, v), s ? h('div', { class: 's' }, s) : null);
  const head = h('div', { class: 'fd-dhead' },
    h('div', { class: 'row g8' }, h('span', { class: 'eyebrow caps' }, m.author || repo.split('/')[0]), h('span', { class: 'grow' }),
      h('a', { class: 'btn sm ghost', href: `${HFHOST}/${repo}`, target: '_blank', rel: 'noopener' }, 'Hugging Face', I('ext', 's'))),
    h('h2', { class: 'fd-title' }, hfName(repo)),
    sub ? h('div', { class: 'fd-sub' }, sub) : null,
    h('div', { class: 'fd-badges' },
      fdChecked(repo) ? h('span', { class: 'tag' }, I('shield'), T('GeulPU 확인됨')) : null,
      rec ? h('span', { class: 'tag' }, I('star'), T('이 PC 추천')) : null,
      sup.k === 'ok' ? h('span', { class: 'tag' }, I('check'), T('GeulPU에서 실행되는 구조')) : h('span', { class: 'tag dim' }, I('alert'), sup.k === 'unknown' ? T('구조 정보 없음') : sup.t),
      lic ? h('span', { class: 'tag dim caps' }, lic) : null));
  const stats = h('div', { class: 'fd-stats' },
    cell(T('다운로드', 'list'), fmtCount(m.downloads), T('최근 30일')),
    cell(T('누적 다운로드'), m.downloadsAllTime != null ? fmtCount(m.downloadsAllTime) : '-'),
    cell(T('좋아요'), fmtCount(m.likes)),
    cell(T('업데이트'), fmtAgo(m.lastModified), m.lastModified ? fmtDate(m.lastModified) : null));
  const hard = sup.k === 'arch' || sup.k === 'size' || sup.k === 'task';
  const note = sup.k === 'ok' ? null
    : h('div', { class: 'note-line' + (hard ? ' err' : '') }, I(hard ? 'alert' : 'info', 's'), h('span', null,
      sup.k === 'arch' ? T`이 저장소의 구조(${g.architecture})는 아직 GeulPU 엔진에 없습니다. 정보는 볼 수 있지만 받을 수 없습니다.`
        : sup.why || T('Hugging Face가 이 저장소의 GGUF 구조를 알려 주지 않았습니다. 파일을 받기 전에 앞부분을 받아 확인합니다.')));
  // 파일
  let files;
  if (x.files) files = fdFiles(repo, x.files, sup, m);
  else if (x.ferr) files = h('div', { class: 'note-line err' }, I('alert', 's'), h('span', null, T('파일 목록을 받지 못했습니다. 인터넷 연결을 확인한 뒤 모델을 다시 고르세요.')));
  else files = h('div', { class: 'row small muted' }, h('i', { class: 'dot busy' }), T('파일 목록을 불러오는 중…'));
  const info = h('div', { class: 'kv' },
    h('div', { class: 'k' }, T('저장소')), h('div', { class: 'v small break' }, repo),
    h('div', { class: 'k' }, T('라이선스')), h('div', { class: 'v' }, lic || T('저장소에 적혀 있지 않음')),
    base.length ? [h('div', { class: 'k' }, T('바탕 모델')), h('div', { class: 'v small' }, base.slice(0, 3).map((b, i) => [i ? ', ' : null, /^[\w.-]+\/[\w.-]+$/.test(b) ? h('a', { href: `${HFHOST}/${b}`, target: '_blank', rel: 'noopener' }, b) : b]), base.length > 3 ? T` 외 ${base.length - 3}개` : null)] : null,
    langs.length ? [h('div', { class: 'k' }, T('언어')), h('div', { class: 'v small' }, langs.slice(0, 12).join(', ') + (langs.length > 12 ? T` 외 ${langs.length - 12}개` : ''))] : null,
    m.cardData && m.cardData.pipeline_tag ? [h('div', { class: 'k' }, T('용도')), h('div', { class: 'v small' }, m.cardData.pipeline_tag)] : null,
    m.createdAt ? [h('div', { class: 'k' }, T('만든 날')), h('div', { class: 'v small' }, fmtDate(m.createdAt))] : null);
  // 모델 설명(README)
  let readme = null;
  const rtext = x.readme ? fdReadme(x.readme) : '';
  if (rtext) {
    const open = !!FD.readmeOpen[repo];
    readme = h('div', { class: 'fd-sec' }, h('div', { class: 'fd-sec-t' }, T('모델 설명'), h('span', { class: 'grow' }), h('span', { class: 'faint small' }, T('Hugging Face의 README'))),
      h('div', { class: 'md fd-readme' + (open ? ' open' : ''), html: md(rtext) }),
      h('button', { class: 'btn sm ghost', onclick: () => { FD.readmeOpen[repo] = !open; fdRedraw(); } }, I(open ? 'chevU' : 'chevD', 's'), open ? T('접기') : T('모두 보기')));
  } else if (x.readme === undefined && !x.err) readme = h('div', { class: 'row small muted' }, h('i', { class: 'dot busy' }), T('모델 설명을 불러오는 중…'));
  return h('div', { class: 'fd-d' }, head, stats, note,
    h('div', { class: 'fd-sec' }, h('div', { class: 'fd-sec-t' }, T('파일'), h('span', { class: 'grow' }), h('span', { class: 'faint small' }, T('양자화마다 하나 — 고르면 받기 전에 실행되는지 확인'))), files),
    h('div', { class: 'fd-sec' }, h('div', { class: 'fd-sec-t' }, T('정보')), info),
    readme);
}
// 이 PC 에 가장 맞는 파일 — 받는 형식 가운데 메모리에 넉넉히 맞는 것 중 Q4_K_M → 가장 큰 것, 없으면 빠듯한 것 중 가장 작은 것
function fdBest(rows, sup) {
  if (sup.k === 'arch' || sup.k === 'size' || sup.k === 'task') return null;
  const ok = rows.filter(f => { const q = quantFromName(f.path); return quantOk(q) && !/-\d{5}-of-\d{5}\.gguf$/i.test(f.path); });
  const fit = f => fitOf({ size: f.size, params: '', id: '', devices: /K/.test(quantFromName(f.path)) ? 'k' : 'all' });
  const good = ok.filter(f => { const t = fit(f); return t && t.k === 'ok'; });
  if (good.length) return good.find(f => quantFromName(f.path) === 'Q4_K_M') || good.sort((a, b) => b.size - a.size)[0];
  const tight = ok.filter(f => { const t = fit(f); return t && t.k === 'tight'; }).sort((a, b) => a.size - b.size);
  return tight[0] || null;
}
function fdFiles(repo, files, sup, m) {
  if (!files.length) return h('div', { class: 'small muted' }, T('이 저장소에는 GGUF 파일이 없습니다.'));
  // 받을 수 있는 형식(한 파일)이 먼저 — 양자화 차례 → 큰 것부터, 받지 못하는 것은 뒤에
  const can = f => quantOk(quantFromName(f.path)) && !/-\d{5}-of-\d{5}\.gguf$/i.test(f.path);
  const rows = files.slice().sort((a, b) => (can(b) - can(a)) || qRank(quantFromName(a.path)) - qRank(quantFromName(b.path)) || b.size - a.size);
  const best = fdBest(rows, sup);
  return h('div', { class: 'fd-files' }, rows.map(f => fdFileRow(repo, f, sup, f === best, m)));
}
function fdFileRow(repo, f, sup, best, m) {
  const st = S.st;
  const q = quantFromName(f.path);
  const name = f.path.split('/').pop();
  const split = /-\d{5}-of-\d{5}\.gguf$/i.test(f.path);
  const c = fdCat(repo).find(x => (x.url || '') === `${HFHOST}/${repo}/resolve/main/${f.path}` || (x.file || '').toLowerCase() === name.toLowerCase());
  const mdl = c && c.installed ? modelOf(c.installed) : st.models.find(x => x.path && x.path.toLowerCase().endsWith('\\' + name.toLowerCase()));
  const ds = c ? dlStateOf(c) : null;
  const fit = fitOf({ size: f.size, params: '', id: repo, devices: c && c.devices ? c.devices : (/K/.test(q) ? 'k' : 'all') });
  let why = '';
  if (sup.k === 'arch') why = T('지원하지 않는 구조');
  else if (sup.k === 'size') why = T('아직 미지원 크기');
  else if (sup.k === 'task') why = T('대화용 모델이 아님');
  else if (split) why = T('여러 조각으로 나뉜 파일 — 합친 파일만 받습니다');
  else if (q && !quantOk(q)) why = T`${q} 형식은 아직 미지원`;
  else if (!q) why = T('양자화 형식을 알 수 없는 파일');
  else if (!c && !(f.lfs && f.lfs.oid)) why = T('SHA-256 정보가 없는 파일');
  const k = repo + '/' + f.path;
  const chk = FD.chk.get(k);
  let act = null, extra = null;
  if (mdl) act = [h('span', { class: 'tag ok' }, I('check'), T('설치됨')), h('button', { class: 'btn sm needs-conn', onclick: () => chatWith(mdl.id) }, I('chat', 's'), T('채팅'))];
  else if (ds === 'run') extra = dlBox(c);
  else if (ds === 'pause') { act = h('button', { class: 'btn sm needs-conn', onclick: () => resumeDownload(c) }, I('download', 's'), T('이어서 다운로드')); extra = dlBox(c, true); }
  else if (ds === 'fail') { act = h('button', { class: 'btn sm needs-conn', onclick: () => startDownload(c) }, I('refresh', 's'), T('다시 다운로드')); extra = dlWhyLine(c.download.reason, c); }
  else if (why) act = h('span', { class: 'why' }, why);
  else if (chk && chk.state === 'checking') act = h('span', { class: 'why row g8 nowrap' }, h('i', { class: 'dot busy' }), T('실행되는지 확인하는 중'));
  else act = h('button', { class: 'btn sm needs-conn' + (best ? ' primary' : ''), 'data-fk': 'get:' + k, onclick: () => (c ? startDownload(c) : fdGet(repo, f, q, m)) }, I('download', 's'), T('다운로드'));
  if (!extra && chk && chk.state === 'bad') extra = h('div', { class: 'note-line err' }, I('alert', 's'), h('span', null, TM(chk.why)));
  const verified = c && (c.check === 'tested' || c.check === 'crosscheck');
  return h('div', { class: 'fd-file' + (why ? ' dim' : '') + (best && !mdl ? ' best' : '') },
    h('span', { class: 'qlab' }, q || '?'),
    h('span', { class: 'sz' }, fmtBytes(f.size)),
    h('span', { class: 'tags' },
      best && !mdl && !why ? h('span', { class: 'tag' }, I('star'), T('이 PC에 추천')) : null,
      verified ? h('span', { class: 'tag', 'data-tip': T('GeulPU에서 이 파일로 실제로 돌려 보았습니다'), 'data-tip-pos': 'top' }, I('shield'), T('실행 확인됨')) : null,
      !why && !mdl && fit && fit.k !== 'ok' ? fitTag(fit) : null),
    h('span', { class: 'grow' }),
    act ? h('span', { class: 'act' }, act) : null,
    extra ? h('div', { class: 'ext' }, extra) : null,
    h('span', { class: 'fname small faint break' }, name));
}
// 받기 — 먼저 파일 앞부분(GGUF 머리)을 받아 이 엔진 · 이 PC 에서 실행되는지 확인하고, 되면 받을지 묻는다
async function fdGet(repo, f, q, m) {
  const k = repo + '/' + f.path;
  let s = FD.chk.get(k);
  if (!s || s.state !== 'ok') {
    FD.chk.set(k, { state: 'checking' });
    fdRedraw();
    s = await hfCheckFile(repo, f);
    FD.chk.set(k, s);
    fdRedraw();
    if (s.state !== 'ok') return;
  }
  hfDownload(m || { id: repo }, f, q, s.shape);
}
async function hfCheckFile(repo, f) {
  try {
    await api('/api/hf/check', { repo, file: f.path, size: f.size });
    for (let i = 0; i < 240; i++) {
      await new Promise(r => setTimeout(r, 400));
      const r = await api('/api/hf/check');
      if (r.state === 'done' || r.state === 'failed') {
        if (r.state === 'failed') return { state: 'bad', why: checkWhy(r.reason) };
        if (!r.shape || !r.shape.ok) return { state: 'bad', why: (r.shape && r.shape.reason) || T('이 엔진이 받지 않는 파일입니다') };
        if (r.device_reason) return { state: 'bad', why: r.device_reason };
        return { state: 'ok', shape: r.shape };
      }
    }
    return { state: 'bad', why: T('확인이 너무 오래 걸립니다. 잠시 뒤 다시 누르세요.') };
  } catch (e) { return { state: 'bad', why: e.message }; }
}
async function hfDownload(x, f, q, shape) {
  const st = S.st;
  const name = f.path.split('/').pop();
  const c = { size: f.size, params: (shape.params / 1e9).toFixed(1) + 'B', id: x.id };
  const fit = fitOf(c);
  const bad = fit && (fit.k === 'no' || fit.k === 'nogpu');
  const lic = fdLicense(x);
  const body = h('div', { class: 'col g16' },
    h('p', { class: 'muted m0' }, T`${hfName(x.id)} ${q}${NB}${(shape.params / 1e9).toFixed(1)}B${NB}문맥 최대 ${fmtCtx(shape.context)}`),
    bad ? h('div', { class: 'note-line err' }, I('alert', 's'), h('span', null, T`이 PC의 그래픽 메모리(${fmtBytes(st.device.memory)})로는 실행되지 않을 수 있습니다. 그래픽 메모리 ${vramNeed(c)} GB 이상을 권장합니다.`)) : null,
    h('div', { class: 'kv' },
      h('div', { class: 'k' }, T('파일', 'k')), h('div', { class: 'v mono small break' }, name),
      h('div', { class: 'k' }, T('크기')), h('div', { class: 'v' }, T`${fmtBytes(f.size)} (${fmtNum(f.size)} 바이트)`),
      h('div', { class: 'k' }, T('저장 위치')), h('div', { class: 'v small break' }, T`${st.app.models_dir}${NB}여유 ${st.app.disk_free >= 0 ? fmtBytes(st.app.disk_free) : T('알 수 없음')}`),
      h('div', { class: 'k' }, T('출처')), h('div', { class: 'v small break' }, h('a', { href: `${HFHOST}/${x.id}`, target: '_blank', rel: 'noopener' }, `huggingface.co/${x.id}`)),
      h('div', { class: 'k' }, T('라이선스')), h('div', { class: 'v' }, lic || T('저장소에 적혀 있지 않음 — 저장소에서 확인하세요'))),
    h('details', { class: 'fold' }, h('summary', null, I('chevR', 's'), T('기술 정보')),
      h('div', { class: 'small mono muted break' }, `SHA-256 ${f.lfs.oid}`)),
    h('div', { class: 'note-line' }, I('info', 's'), h('span', null, T('이 파일의 답 품질은 GeulPU에서 확인하지 않았습니다. 구조와 형식은 받기 전에 확인했고, 다 받으면 SHA-256으로 파일이 맞는지 확인합니다.'))));
  if (!(await confirmBox(T('모델 다운로드'), body, T('다운로드'), bad ? 'quiet' : null))) return;
  try {
    const r = await api('/api/hf/add', { repo: x.id, file: f.path, size: f.size, sha256: f.lfs.oid, name: hfName(x.id), quant: q, license: lic });
    await api('/api/models/download', { id: r.id });
    toast(T("다운로드를 시작했습니다 — 진행은 상태 막대와 '내 모델'에서 볼 수 있습니다"), 'ok');
    resetSig(); refresh();
  } catch (e) { toast(e.message, 'err'); }
}
async function discardDownload(c) {
  if (!(await confirmBox(T('받다 만 파일 지우기'), h('p', { class: 'muted m0' }, T`${c.name} ${c.quant} — 지금까지 받은 ${fmtBytes((c.download && c.download.received) || 0)}를 지웁니다.${c.check === 'hf' ? T(' Hugging Face에서 더한 항목이라 목록에서도 뺍니다.') : ''}`), T('지우기'), 'danger'))) return;
  try { await api('/api/models/discard', { id: c.id }); resetSig(); refresh(); }
  catch (e) { toast(e.message, 'err'); }
}

// ---------- 벤치마크 ----------
const BENCH = [
  { kind: 'diag', title: T('하드웨어 진단'), needModel: true, time: T('1~6분'),
    desc: T('이 PC의 메모리 대역폭과 정수 연산 성능을 측정하고, 엔진의 각 연산이 하드웨어 한계에 얼마나 가까운지 보여 줍니다.') },
  { kind: 'speed', title: T('속도 측정'), needModel: true, time: T('1~3분'),
    desc: T('프롬프트 처리(512 · 2048토큰)와 답 생성 속도를 토큰/초로 세 번씩 측정합니다.') },
  { kind: 'crosscheck', title: T('재현성 검증'), needModel: false, time: T('경로마다 5~20분'),
    desc: T('DeepSeek-R1-Distill-Qwen-7B Q8_0으로 수학 문제 32개를 풀어 기준 PC의 출력과 비트 단위로 비교합니다. 빌드와 커널의 해시도 함께 확인합니다.') },
];
const KIND = { '진단': ['pulse', T('하드웨어 진단')], '속도': ['bolt', T('속도 측정')], '교차검증': ['shield', T('재현성 검증')] };
const R1_REF = 'deepseek-r1-distill-qwen-7b-q8_0';
function benchModel(runnables) {
  if (!runnables.some(m => m.id === S.benchModel)) S.benchModel = runnables.length ? runnables[0].id : '';
  return S.benchModel;
}
const isQuick = x => /빠름/.test(x.name || '') || /빠른 확인|줄인 판/.test(x.headline || '');
// 결과 요약 글에서 큰 숫자(KPI)를 — 관리자의 요약 꼴(벤치.gl)을 읽는다
function kpiFromText(kind, t) {
  const n = re => { const m = re.exec(t || ''); return m ? parseFloat(m[1]) : NaN; };
  if (kind === 'speed' || kind === '속도') {
    // 지금 꼴 "생성 60.3 토큰/초 · 프롬프트 6066 토큰/초(512토큰)", 지난 꼴 "프롬프트 512: 84.3 ms · 6070 토큰/초"
    const g = n(/생성 ([\d.]+) 토큰\/초/);
    let p = n(/프롬프트 ([\d.]+) 토큰\/초/);
    if (isNaN(p)) p = n(/프롬프트 \d+: [\d.]+ ms · ([\d.]+) 토큰\/초/);
    if (isNaN(g) && isNaN(p)) return null;
    return [isNaN(g) ? null : [fmtNum(g, 1), T('생성 토큰/초')], isNaN(p) ? null : [fmtNum(p), T('프롬프트 토큰/초')]];
  }
  if (kind === 'diag' || kind === '진단') {
    // 한계 대비 — 새 꼴 "한계의 91%", 지난 결과의 꼴 "한계의 1.10배"(잰 시간 ÷ 한계 시간)도 읽는다
    const eff = name => { const p = n(new RegExp(name + ' 한계의 (\\d+)%')); return isNaN(p) ? effOf(n(new RegExp(name + ' 한계의 ([\\d.]+)배'))) : p; };
    const bw = n(/메모리 대역폭 ([\d.]+) GB\/s/), tops = n(/정수 텐서 ([\d.]+) TOPS/), gen = eff('생성\\(문맥 1024\\)'), pr = eff('프롬프트 2048토큰');
    if (isNaN(bw)) return null;
    return [[fmtNum(bw, 1) + ' GB/s', T('메모리 대역폭')], isNaN(tops) ? null : [fmtNum(tops) + ' TOPS', T('정수 텐서')],
      isNaN(gen) ? null : [fmtEff(gen), T('생성 · 하드웨어 한계 대비'), T('한계 대비 생성')], isNaN(pr) ? null : [fmtEff(pr), T('프롬프트 · 하드웨어 한계 대비'), T('한계 대비 프롬프트')]];
  }
  return null;
}
const kpis = (items, small) => h('div', { class: 'kpis' + (small ? ' sm' : '') }, items.filter(Boolean).map(([v, l]) => h('div', { class: 'kpi' }, h('div', { class: 'v' }, v), h('div', { class: 'l' }, l))));
function renderBench() {
  const st = S.st;
  const page = $('#page-bench');
  const j0 = st.job;
  const j = j0 && (j0.state === 'running' || (['diag', 'speed', 'crosscheck'].includes(j0.kind) && jobKey(j0) !== S.jobHide)) ? j0 : null;
  const runnables = sortedModels().filter(runnable);
  const hasR1 = st.models.some(m => m.catalog === R1_REF && runnable(m));
  const sel = benchModel(runnables);
  const struct = [j && [j.id, j.state, j.kind, j.result, !!j.summary, j.reason], runnables.map(m => m.id), hasR1, sel, S.results && S.results.length];
  if (changed('bench', struct)) {
    const busy = !!(j && j.state === 'running');
    const r1c = st.catalog.find(c => c.id === R1_REF);
    rebuild(page,
      pageHead(S.results ? T`결과 ${S.results.length}개` : null, T('벤치마크'), T('한 번에 하나씩 실행됩니다. 실행하는 동안 불러온 채팅 모델은 GPU에서 내립니다. 결과는 zip으로 내보낼 수 있습니다.')),
      j ? jobCard(j) : null,
      h('div', { class: 'section bench-grid' },
        h('div', { class: 'card pick-card' },
          h('div', { class: 'pc-t' }, h('span', { class: 'lab' }, T('측정할 모델')), h('span', { class: 'small muted' }, T('하드웨어 진단 · 속도 측정에 씁니다'))),
          h('div', { class: 'pc-c' },
            h('select', {
              class: 'input', 'data-fk': 'bench-model', 'aria-label': T('하드웨어 진단 · 속도 측정에 쓸 모델'), disabled: runnables.length ? null : true,
              onchange: ev => { S.benchModel = ev.target.value; store('geulpu.benchModel', S.benchModel); },
            }, runnables.length ? runnables.map(m => h('option', { value: m.id, selected: m.id === sel ? true : null }, `${displayName(m)} · ${quantOf(m)}`)) : h('option', { value: '' }, T('실행할 수 있는 모델이 없습니다'))),
            !runnables.length ? h('button', { class: 'btn sm', onclick: () => show('models') }, I('box', 's'), T('모델 화면')) : null)),
        BENCH.map(b => benchCard(b, runnables, hasR1, busy, r1c))),
      h('div', { class: 'section' }, h('div', { class: 'section-head' }, h('span', { class: 'sec' }, T('최근 결과')), h('span', { class: 'grow' }),
        h('button', { class: 'icon-btn needs-conn', 'data-tip': T('결과 폴더 열기'), 'data-tip-pos': 'left', 'aria-label': T('결과 폴더 열기'), onclick: () => api('/api/open', { what: 'results' }).catch(e => toast(e.message, 'err')) }, I('folder'))),
        h('div', { id: 'results' }, S.results ? resultsTable(S.results) : h('div', { class: 'muted' }, T('불러오는 중…')))));
    S.log = { job: j ? j.id : 0, from: 0 };
  }
  if (j) {
    const set = (id, v) => { const e = document.getElementById(id); if (e && e.textContent !== v) e.textContent = v; };
    set('job-step', j.state === 'running' ? TM(j.step || '') : { done: T('끝났습니다'), failed: T('실패했습니다'), cancelled: T('중지했습니다') }[j.state] || '');
    set('job-time', `${{ running: T('실행 중'), done: T('완료'), failed: T('실패'), cancelled: T('중지됨') }[j.state] || j.state} · ${fmtClock(j.seconds)}`);
    const bar = document.getElementById('job-bar');
    if (bar) bar.style.width = (j.state === 'done' ? 100 : j.progress / 10) + '%';
    const tail = document.getElementById('job-tail');
    if (tail) { tail.textContent = TM(st.job_tail || ''); tail.closest('details').hidden = !st.job_tail; }
  }
  if (Date.now() - S.resultsAt > 5000) loadResults(false);
}
// 끝난 작업 카드를 닫은 기억 — 관리자를 다시 켜면 번호가 처음부터라 종류 · 결과 이름까지 묶는다
const jobKey = j => `${j.id}:${j.kind}:${j.result || ''}:${j.state}`;
const JOBSEEN = new Set();   // 실패를 이미 알린 작업
function rerun(j) {
  const id = jobModel(j);
  if ((j.kind === 'diag' || j.kind === 'speed') && id && runnable(modelOf(id))) { S.benchModel = id; store('geulpu.benchModel', id); resetSig(); renderBench(); }
  const run = document.querySelector(`.bench-card[data-kind=${j.kind}] .run`);
  if (run) run.click();
}
function jobCard(j) {
  const tone = { running: 'acc', done: 'ok', failed: 'err', cancelled: 'dim' }[j.state] || '';
  const benchDef = BENCH.find(b => b.kind === j.kind);
  const k = j.summary && j.state === 'done' ? kpiFromText(j.kind, j.summary) : null;
  const quick = j.kind === 'crosscheck' && /줄인|빠른/.test(j.title || '');
  let fail = null;
  if (j.reason && j.state !== 'done') {
    const [rt, rd] = humanJob(j.reason);
    const live = !JOBSEEN.has(jobKey(j));
    JOBSEEN.add(jobKey(j));
    fail = errorCard(rt, rd, (j.state === 'failed' || j.state === 'cancelled') && benchDef ? [h('button', { class: 'btn sm needs-conn', 'data-fk': 'rerun', onclick: () => rerun(j) }, I('refresh', 's'), T('같은 설정으로 다시 실행'))] : null, j.reason, live);
  }
  return h('div', { class: 'card job' },
    h('div', { class: 'head' },
      h('div', { class: 't' }, h('div', { class: 'ink h3' }, jobTitle(j)), h('div', { class: 'small muted', id: 'job-step' }, j.state === 'running' ? TM(j.step || '') : '')),
      h('span', { class: 'tag ' + tone, id: 'job-time' }, ''),
      j.state === 'running' ? h('button', { class: 'btn sm needs-conn', 'data-fk': 'job-stop', onclick: () => api('/api/job/cancel', {}).then(refresh).catch(e => toast(e.message, 'err')) }, I('stop', 's'), T('중지'))
        : h('button', { class: 'icon-btn', 'data-tip': T('카드 닫기'), 'data-tip-pos': 'left', 'aria-label': T('끝난 작업 카드 닫기'), onclick: () => { S.jobHide = jobKey(j); store('geulpu.jobHide', S.jobHide); resetSig(); renderBench(); } }, I('x', 's'))),
    h('div', { class: 'bar' + (j.state === 'done' ? ' done' : j.state === 'failed' ? ' fail' : '') }, h('i', { id: 'job-bar', style: `width:${j.state === 'done' ? 100 : j.progress / 10}%` })),
    k ? kpis(k, true) : j.summary && j.state === 'done' ? h('div', { class: 'verdict' }, (quick ? T('빠른 확인 — ') : '') + TM(pathLabel(headText(j.summary)))) : null,
    fail,
    j.state === 'done' && j.result ? h('div', { class: 'row' },
      h('button', { class: 'btn sm primary', onclick: () => openResult(j.result) }, T('결과 보기')),
      h('a', { class: 'btn sm', href: '/api/results/zip?name=' + encodeURIComponent(j.result) }, I('download', 's'), T('zip 내보내기'))) : null,
    h('details', { class: 'fold' }, h('summary', null, I('chevR', 's'), T('실행 기록')), h('pre', { class: 'log', id: 'joblog' })),
    h('details', { class: 'fold', hidden: true }, h('summary', null, I('chevR', 's'), T('도구 출력')), h('pre', { class: 'log', id: 'job-tail' })));
}
function benchCard(b, runnables, hasR1, busy, r1c) {
  const extra = b.kind === 'crosscheck' ? h('div', { class: 'col g12' },
    h('label', { class: 'check' }, h('input', { type: 'checkbox', class: 'cc-quick' }), T('빠른 확인 — 문제 2개 · 64토큰(전체 검증 아님)')),
    h('select', { class: 'input cc-mod', 'aria-label': T('검증할 경로') }, h('option', { value: 'both' }, T('모든 경로(권장)')), h('option', { value: 'coop' }, T('텐서 코어 경로만')), h('option', { value: 'base' }, T('기본 경로만')))) : null;
  const can = b.needModel ? runnables.length > 0 : hasR1;
  let need = null;
  if (!can && b.kind === 'crosscheck') need = h('div', { class: 'note-line' }, I('info', 's'), h('div', { class: 'col g8' }, h('span', null, T('DeepSeek-R1-Distill-Qwen-7B Q8_0 모델이 필요합니다.')),
    r1c && !r1c.installed ? h('div', null, h('button', { class: 'btn sm needs-conn', onclick: () => startDownload(r1c) }, I('download', 's'), T`다운로드(${fmtBytes(r1c.size)})`)) : null));
  else if (!can) need = h('div', { class: 'note-line' }, I('info', 's'), h('span', null, T('먼저 모델을 설치하세요.')));
  return h('div', { class: 'card bench-card', 'data-kind': b.kind },
    h('div', null, h('h2', null, b.title), h('div', { class: 'eyebrow' }, b.time)),
    h('p', null, b.desc), extra, need,
    h('div', null, h('button', {
      class: 'btn run needs-conn', disabled: (busy || !can) ? true : null, onclick: async ev => {
        const card = ev.target.closest('.card');
        const body = {};
        if (b.needModel) { body.model = S.benchModel; if (!body.model) return; }
        if (b.kind === 'crosscheck') { body.quick = card.querySelector('.cc-quick').checked; body.modules = card.querySelector('.cc-mod').value; }
        if (S.streaming) { toast(T('채팅의 답이 끝난 뒤 실행할 수 있습니다')); return; }
        if (S.st.server.state === 'ready' && !(await confirmBox(T('채팅 모델을 내립니다'), h('p', { class: 'muted m0' }, T('측정하는 동안 GPU를 혼자 쓰도록 불러온 채팅 모델을 내립니다. 계속할까요?')), T('계속')))) return;
        try { await api('/api/bench/' + b.kind, body); resetSig(); refresh(); }
        catch (e) { toast(e.message, 'err'); }
      },
    }, I('play', 's'), busy ? T('다른 작업 실행 중') : T('실행'))));
}
async function pollLog() {
  const j = S.st && S.st.job;
  if (!j || S.view !== 'bench') return;
  const pre = document.getElementById('joblog');
  if (!pre) return;
  if (S.log.job !== j.id) S.log = { job: j.id, from: 0 };
  try {
    const r = await api('/api/job/log?from=' + S.log.from);
    if (r.id !== j.id) return;
    if (r.text) {
      const near = pre.scrollHeight - pre.scrollTop - pre.clientHeight < 40;
      if (EN) { S.log.raw = (S.log.raw || '') + r.text; pre.textContent = TM(S.log.raw); }   // 줄이 조각나 오므로 모은 원문을 통째로 바꾼다
      else pre.textContent += r.text;
      if (near) pre.scrollTop = pre.scrollHeight;
    }
    S.log.from = r.size;
  } catch (_) { }
}
async function loadResults(force) {
  S.resultsAt = Date.now();
  let r;
  try { r = await api('/api/results'); } catch (_) { return; }
  if (!changed('results', r) && !force) return;
  S.results = r.results;
  put(document.getElementById('results'), resultsTable(r.results));
  if (S.view === 'bench') renderBench();
}
function stampText(s) { return String(s || '').replace(/^(\d{4})(\d{2})(\d{2})-(\d{2})(\d{2})(\d{2})$/, '$1-$2-$3 $4:$5'); }
function headText(t) { return String(t || '').split('\n')[0].replace(/^결과(\([^)]*\))?: /, ''); }
function resultVerdict(x) {
  const t = x.headline || '';
  if (/다른 출력이 있거나|끝나지 못한/.test(t)) return ['err', T('불일치')];
  if (isQuick(x) && /앞부분이 같습니다/.test(t)) return ['', T('앞부분 일치')];
  if (/똑같은 출력|앞부분이 같습니다/.test(t)) return ['ok', T('일치')];
  return null;
}
const resultModel = x => (x.kind === '교차검증' ? labelById(R1_REF) : x.model ? labelById(x.model) : '-');
function resultSummary(x) {
  const k = kpiFromText(x.kind, x.headline);
  if (k) return k.filter(Boolean).slice(0, 3).map(([v, l, s]) => `${s || l} ${v}`).join(NB);
  if (x.kind === '교차검증') {
    const t = headText(x.headline);
    const tail = t.includes(' — ') ? t.split(' — ').pop() : t;
    return (isQuick(x) ? T('빠른 확인') + NB : '') + TM(pathLabel(tail));
  }
  return TM(headText(x.headline));
}
function resultsTable(rs) {
  if (!rs.length) return h('div', { class: 'empty-box' }, h('div', { class: 'big' }, T('아직 결과가 없습니다')), h('div', null, T('위에서 측정을 하나 실행하면 여기에 쌓입니다.')));
  // 좁아지면(≤1200) 일시를 종류 밑으로, 휴대폰(≤640)에서는 모델 · 결과까지 종류 밑으로
  return h('div', { class: 'list' }, h('table', { class: 'data results' },
    h('thead', null, h('tr', null, h('th', null, T('종류')), h('th', { class: 'c-model' }, T('모델', 'th')), h('th', { class: 'c-date' }, T('일시')), h('th', { class: 'c-res' }, T('결과')), h('th', { class: 'r' }, h('span', { class: 'sr' }, T('동작'))))),
    h('tbody', null, rs.map(x => {
      const k = KIND[x.kind] || ['terminal', x.kind];
      const v = x.kind === '교차검증' ? resultVerdict(x) : null;
      const sum = resultSummary(x);
      const vt = v ? h('span', { class: 'tag ' + v[0] }, v[0] === 'err' ? I('alert') : null, v[1]) : null;
      return h('tr', { class: 'clk', onclick: e => { if (!e.target.closest('a, button')) openResult(x.name); } },
        h('td', { class: 'c-kind' }, h('span', { class: 'row nowrap g12' }, I(k[0], 's'), h('span', { class: 'ink' }, k[1])),
          h('div', { class: 'mob' }, h('span', { class: 'mob-m' }, resultModel(x) + NB), h('span', { class: 'num' }, stampText(x.stamp))),
          h('div', { class: 'mob-res' }, vt ? vt.cloneNode(true) : null, h('span', null, sum))),
        h('td', { class: 'small c-model' }, resultModel(x)),
        h('td', { class: 'num small muted c-date nw' }, stampText(x.stamp)),
        h('td', { class: 'small c-res' }, h('span', { class: 'row nowrap g8' }, vt, h('span', { class: 'ell', 'data-tip': sum, 'data-tip-pos': 'top' }, sum))),
        h('td', { class: 'r nw' },
          h('button', { class: 'icon-btn', 'data-tip': T('결과 보기'), 'data-tip-pos': 'left', 'aria-label': T('결과 보기'), onclick: () => openResult(x.name) }, I('chevR')),
          h('a', { class: 'icon-btn', 'data-tip': T('zip 내보내기'), 'data-tip-pos': 'left', 'aria-label': T('zip 내보내기'), href: '/api/results/zip?name=' + encodeURIComponent(x.name) }, I('download'))));
    }))));
}
async function getFile(name, file) {
  const r = await fetch('/api/results/file?name=' + encodeURIComponent(name) + '&file=' + encodeURIComponent(file));
  if (!r.ok) return null;
  return file.endsWith('.json') ? r.json() : r.text();
}
// 진단 도구의 칸 이름(연구 기록의 말)을 화면의 말로
function diagLabel(k) {
  let s = String(k).replace(/여덟/g, '8').replace(/넷/g, '4');
  const alt = /\(31단계 후보\)/.test(s);
  s = s.replace(/\s*\(31단계 후보\)/, '');
  let m;
  if ((m = /^슬롯 (\d+) 걸음 문맥 (\d+)$/.exec(s))) s = T`동시 ${m[1]}개 생성 · 문맥 ${m[2]}`;
  else if ((m = /^(\d+) 줄 문맥 (\d+)$/.exec(s))) s = T`${m[1]}행 일괄 처리 · 문맥 ${m[2]}`;
  else if ((m = /^생성 문맥 (\d+)$/.exec(s))) s = T`생성 · 문맥 ${m[1]}`;
  else if ((m = /^프롬프트 (\d+)$/.exec(s))) s = T`프롬프트 ${m[1]}토큰`;
  return alt ? s + T(' · 대체 커널') : s;
}
async function openResult(name) {
  modal(h('div', { class: 'muted' }, T('불러오는 중…')), null, true);
  const [txt, summary, speed, diag, env] = await Promise.all([getFile(name, '결과.txt'), getFile(name, 'summary.json'), getFile(name, 'speed.json'), getFile(name, 'diag.json'), getFile(name, 'environment.json')]);
  const kind = name.split('-')[0];
  const k = KIND[kind] || ['terminal', kind];
  const stamp = (name.match(/(\d{8}-\d{6})$/) || [])[1];
  const quick = kind === '교차검증' && (/빠름/.test(name) || /빠른 확인|줄인 판/.test(txt || ''));
  const modelTxt = kind === '교차검증' ? labelById(R1_REF) : (speed && speed.model) ? labelById(speed.model) : (env && env.model) ? labelById(env.model) : '';
  const extra = speed ? T` · 동시 ${speed.slots}개 · 문맥 ${fmtCtx(speed.context)}` : '';
  const eyebrow = [k[1] + (quick ? T(' · 빠른 확인(문제 2개 · 64토큰)') : ''), modelTxt + extra, stamp ? stampText(stamp) : ''].filter(Boolean).join(' · ');
  let title = TM(headText(txt)) || name;
  if (kind === '진단') title = T('하드웨어 진단 결과');
  if (kind === '속도') title = T('속도 측정 결과');
  const parts = [h('div', { class: 'row g8 muted' }, I(k[0], 's'), h('span', { class: 'eyebrow' }, eyebrow))];
  if (summary && summary.runs) {
    const runs = summary.runs;
    const allSame = runs.every(r => r.same === r.total && r.total > 0);
    const nProb = runs[0] ? runs[0].total : 0;
    const same = runs.reduce((a, r) => a + r.same, 0), tot = runs.reduce((a, r) => a + r.total, 0);
    title = quick ? (allSame ? T('앞 64토큰이 기준 PC와 같습니다 — 전체 검증은 아닙니다') : T('앞부분부터 기준 PC와 다릅니다'))
      : (allSame ? T`문제 ${nProb}개 모두 기준 PC와 출력이 같습니다` : T('기준 PC와 출력이 다른 문제가 있습니다'));
    parts.push(h('h2', { class: 'mt8' }, title),
      h('div', { class: 'res-head' },
        h('span', { class: 'big' }, allSame ? `${nProb}/${nProb}` : `${same}/${tot}`),
        h('span', { class: 'muted' }, quick ? T('문제의 앞 64토큰이 같음') : allSame ? (runs.length > 1 ? T`문제 · 경로 ${runs.length}개 모두 같음` : T('문제가 같은 출력')) : T`${tot}회 중 ${same}회 일치(문제 ${nProb} × 경로 ${runs.length})`),
        h('span', { class: 'grow' }), h('span', { class: 'tag ' + (quick && allSame ? '' : allSame ? 'ok' : 'err') }, allSame ? null : I('alert'), quick && allSame ? T('앞부분 일치') : allSame ? T('일치') : T('불일치'))));
    for (const run of runs) {
      const items = [...(run.items || [])].sort((a, b) => (a.result === 'same') - (b.result === 'same'));
      const bad = items.filter(it => it.result !== 'same');
      const okRun = run.same === run.total;
      parts.push(h('div', { class: 'section-head mt24' }, h('span', { class: 'sec' }, TM(pathLabel(run.label))), h('span', { class: 'grow' }),
        h('span', { class: 'tag ' + (okRun ? (quick ? '' : 'ok') : 'err') }, okRun ? null : I('alert'), `${run.same}/${run.total}`), h('span', { class: 'tag dim' }, T`동시 ${run.slots}개 · ${run.generate_minutes < 1 ? T('1분 미만') : run.generate_minutes + T('분')}`)));
      if (run.error) parts.push(h('div', { class: 'note-line err' }, I('alert', 's'), h('span', null, TM(run.error))));
      const table = rows => h('div', { class: 'list' }, h('table', { class: 'data' },
        h('thead', null, h('tr', null, h('th', null, T('문제')), h('th', null, T('판정')), h('th', { class: 'r' }, T('토큰')), h('th', { class: 'r' }, T('기준')), h('th', { 'data-tip': T('토큰마다 로짓(점수)의 SHA-256'), 'data-tip-pos': 'top' }, T('단계별 해시')))),
        h('tbody', null, rows.map(it => h('tr', null, h('td', { class: 'num', 'data-tip': T`파일 ${String(it.id).padStart(2, '0')}.json`, 'data-tip-pos': 'right' }, Number(it.id) + 1),
          h('td', null, h('span', { class: 'tag ' + (it.result === 'same' ? (quick ? '' : 'ok') : 'err') }, it.result === 'same' ? null : I('alert'), it.result === 'same' ? T('같음') : it.result === 'different' ? T`다름 · ${it.first_diff_char}번째 글자부터` : T('오류'))),
          h('td', { class: 'r num' }, fmtNum(it.n)), h('td', { class: 'r num muted' }, fmtNum(it.expected_n)), h('td', { class: 'mono small muted' }, (it.logit_hashes_sha256 || '').slice(0, 16)))))));
      if (bad.length) parts.push(table(bad));
      parts.push(h('details', { class: 'fold mt12' }, h('summary', null, I('chevR', 's'), bad.length ? T`모든 문제 ${items.length}개 보기` : T`문제 ${items.length}개 보기`), table(items)));
    }
    if (summary.build) parts.push(h('details', { class: 'fold mt16' }, h('summary', null, I('chevR', 's'), T('빌드 · 커널 해시')),
      h('div', { class: 'kv' }, Object.entries(summary.build).flatMap(([kk, v]) => [h('div', { class: 'k' }, kk), h('div', { class: 'v mono small break' }, v)]))));
  } else parts.push(h('h2', { class: 'mt8' }, title));
  if (speed && speed.cells) {
    const cells = speed.cells;
    const gen = cells.find(c => c.generate > 1 && c.prompt <= 64) || cells.find(c => c.generate > 1);
    const p512 = cells.find(c => c.generate === 1 && c.prompt === 512), p2048 = cells.find(c => c.generate === 1 && c.prompt === 2048);
    parts.push(kpis([gen ? [fmtNum(gen.tokens_per_s, 1), T('생성 토큰/초')] : null, p512 ? [fmtNum(p512.tokens_per_s), T('프롬프트 토큰/초(512토큰)')] : null, p2048 ? [fmtNum(p2048.tokens_per_s), T('프롬프트 토큰/초(2048토큰)')] : null]),
      h('div', { class: 'section-head mt24' }, h('span', { class: 'sec' }, T('측정값')), h('span', { class: 'grow' }), h('span', { class: 'small muted' }, T('세 번 측정한 가운데 값'))),
      h('div', { class: 'list' }, h('table', { class: 'data' }, h('thead', null, h('tr', null, h('th', null, T('항목')), h('th', { class: 'r' }, T('프롬프트 처리')), h('th', { class: 'r' }, T('토큰당 생성')), h('th', { class: 'r' }, T('토큰/초')))),
        h('tbody', null, cells.map(c => h('tr', null, h('td', null, TM(c.name)),
          h('td', { class: 'r num' }, c.prompt_ms !== undefined ? fmtNum(c.prompt_ms, 1) + ' ms' : '–'),
          h('td', { class: 'r num' }, c.ms_per_token !== undefined ? fmtNum(c.ms_per_token, 2) + ' ms' : '–'),
          h('td', { class: 'r num ink' }, fmtNum(c.tokens_per_s, c.tokens_per_s < 1000 ? 1 : 0))))))));
  }
  if (diag && diag['커널']) {
    const floor = diag['바닥'] || {};
    const tops = floor['협력_TOPS'] > 0 ? floor['협력_TOPS'] : floor['정수텐서_TOPS'];
    const kk = diag['커널'];
    const gen = kk['생성 문맥 1024'], pr = kk['프롬프트 2048'];
    parts.push(kpis([floor['읽기대역_GBs'] ? [fmtNum(floor['읽기대역_GBs'], 1) + ' GB/s', T('메모리 대역폭')] : null, tops ? [fmtNum(tops) + ' TOPS', T('정수 텐서')] : null,
      gen ? [fmtEff(effOf(gen['배'])), T('생성 · 하드웨어 한계 대비')] : null, pr ? [fmtEff(effOf(pr['배'])), T('프롬프트 · 하드웨어 한계 대비')] : null]),
      h('div', { class: 'section-head mt24' }, h('span', { class: 'sec' }, T('연산별 시간')), h('span', { class: 'grow' }),
        h('span', { class: 'small muted' }, T('100% = 하드웨어 한계 · 높을수록 좋습니다'))),
      h('div', { class: 'list' }, h('table', { class: 'data' }, h('thead', null, h('tr', null, h('th', null, T('연산')), h('th', { class: 'r' }, T('합계')), h('th', { class: 'r' }, T('한계 대비')))),
        h('tbody', null, Object.entries(kk).map(([name, v]) => h('tr', null, h('td', { 'data-tip': /\(31단계 후보\)/.test(name) ? T('같은 계산을 다른 방식으로 하는 후보 커널 — 비교하려고 함께 잽니다') : name !== diagLabel(name) ? T('진단 도구의 이름: ') + name : null, 'data-tip-pos': 'top' }, TM(diagLabel(name))), h('td', { class: 'r num' }, fmtNum(v['합_ms'] || 0, 2) + ' ms'),
          h('td', { class: 'r num ink nw' }, v['배'] > 0 ? [h('span', { class: 'ratio' }, h('i', { style: `width:${Math.min(100, effOf(v['배']))}%` })), fmtEff(effOf(v['배']))] : '-')))))));
  }
  if (txt && txt.split('\n').filter(Boolean).length > 1) parts.push(h('details', { class: 'fold mt16' }, h('summary', null, I('chevR', 's'), T('결과.txt 원문')), h('pre', { class: 'log' }, txt)));
  modal(h('div', null, parts), [
    h('button', { class: 'btn ghost needs-conn', onclick: () => api('/api/open', { what: 'result', name }).catch(e => toast(e.message, 'err')) }, I('folder', 's'), T('폴더 열기')),
    h('a', { class: 'btn', href: '/api/results/zip?name=' + encodeURIComponent(name) }, I('download', 's'), T('zip 내보내기')),
    h('button', { class: 'btn', onclick: () => closeModal() }, T('닫기'))], true, '.foot .btn:last-child');
}

// ---------- 하드웨어 ----------
async function renderDevice() {
  const d = S.st.device;
  if (!changed('device', d)) return;
  const page = $('#page-device');
  const cell = (k, v, s) => h('div', { class: 'cell' }, h('div', { class: 'eyebrow' }, k), h('div', { class: 'v' }, v), s ? h('div', { class: 's' }, s) : null);
  const vk = d.layer === 'vulkan';
  put(page,
    pageHead(d.layer ? `${LAYER[d.layer]} · ${fmtBytes(d.memory)}` : null, T('하드웨어'), null,
      [h('button', { class: 'btn needs-conn', onclick: () => api('/api/device/check', {}).then(() => { resetSig(); refresh(); }).catch(e => toast(e.message, 'err')) }, I('refresh', 's'), T('다시 확인'))]),
    !d.checked ? h('div', { class: 'empty-box' }, h('div', { class: 'big' }, T('확인하는 중…')))
      : !d.layer ? h('div', { class: 'empty-box' }, h('div', { class: 'big' }, T('쓸 수 있는 GPU를 찾지 못했습니다')), h('div', null, T('NVIDIA RTX 30 시리즈 이상(CUDA) 또는 Vulkan 1.3을 지원하는 GPU가 필요합니다.')),
        d.reason ? h('details', { class: 'fold mt16 tl' }, h('summary', null, I('chevR', 's'), T('확인한 내용')), h('pre', { class: 'log sans' }, TM(d.reason))) : null)
        : h('div', { class: 'spec' },
          cell('GPU', d.name || '-', d.vendor || ''),
          cell(T('백엔드'), LAYER[d.layer], d.layer === 'cuda' ? T('NVIDIA 드라이버(nvcuda.dll)') : T('Vulkan 드라이버(vulkan-1.dll)')),
          cell(T('그래픽 메모리'), fmtBytes(d.memory), 'VRAM'),
          cell(T('드라이버'), d.driver || '-', ''),
          cell(T('커널'), vk ? (d.variant === '공유32' ? T('공유 메모리 32 KB용') : T('표준')) : T('CUDA용'), T('처음 한 번 이 GPU에 맞춰 만듭니다')),
          cell(T('행렬 가속'), vk ? (d.coop ? T`텐서 코어(${d.coop})` : T('없음 — 일반 연산')) : T('텐서 코어(정수)'), vk && !d.coop ? T('텐서 코어를 쓸 수 없어 일반 연산으로 대신합니다') : '')),
    h('div', { class: 'section' }, h('div', { class: 'section-head' }, h('span', { class: 'sec' }, T('Vulkan 장치'))), h('div', { id: 'dev-vk', class: 'col' })),
    h('div', { class: 'section' }, h('div', { class: 'section-head' }, h('span', { class: 'sec' }, T('CUDA 장치'))), h('div', { id: 'dev-cu', class: 'col' })));
  try {
    const raw = await api('/api/device');
    const vl = raw.vulkan && raw.vulkan['장치들'] ? raw.vulkan['장치들'] : [];
    S.vk = vl.map(v => v['이름']);
    put($('#dev-vk'), vl.length ? h('div', { class: 'list' }, vl.map(v => {
      const integrated = v['종류'] === 1;
      return h('div', { class: 'item' },
        h('div', { class: 'main' }, h('div', { class: 'title' }, h('span', { class: 'nm' }, v['이름']), integrated ? h('span', { class: 'tag dim' }, T('내장')) : null),
          h('div', { class: 'sub' }, T`Vulkan ${v['API']}${NB}드라이버 ${v['드라이버'] ? v['드라이버']['정보'] : '-'}${NB}공유 메모리 ${(v['공유메모리'] / 1024) | 0}\u00a0KB${NB}텐서 코어 ${v['권하는_협력'] || T('없음')}`),
          ...(v['막는것'] || []).map(x => h('div', { class: 'sub bad' }, TM(x)))),
        h('div', { class: 'meta' }, v['고름'] && vk ? h('span', { class: 'tag' }, h('i', { class: 'dot ok' }), T('사용 중')) : null,
          v['돌릴수있음'] ? h('span', { class: 'tag' }, I('check'), T('지원')) : h('span', { class: 'tag err' }, I('alert'), T('지원 안 함')),
          h('span', { class: 'num muted memv', 'data-tip': integrated ? T('내장 GPU는 시스템 메모리를 나눠 씁니다') : T('Vulkan이 알려 준 장치 메모리 — 드라이버가 예약한 만큼 작게 보일 수 있습니다'), 'data-tip-pos': 'left' }, (integrated ? T('공유 ') : '') + fmtBytes(v['장치메모리']))));
    })) : h('div', { class: 'muted' }, T('없음', 'empty')));
    const cl = raw.cuda && raw.cuda.devices ? raw.cuda.devices : [];
    put($('#dev-cu'), cl.length ? h('div', { class: 'list' }, cl.map(c => h('div', { class: 'item' },
      h('div', { class: 'main' }, h('div', { class: 'title' }, h('span', { class: 'nm' }, c.name)), h('div', { class: 'sub' }, T`Compute Capability ${c.cc_major}.${c.cc_minor}${NB}SM ${c.sm}개${NB}CUDA ${(raw.cuda.driver_version / 1000) | 0}.${(raw.cuda.driver_version % 1000) / 10}`)),
      h('div', { class: 'meta' }, d.layer === 'cuda' && c.ok ? h('span', { class: 'tag' }, h('i', { class: 'dot ok' }), T('사용 중')) : null, c.ok ? h('span', { class: 'tag' }, I('check'), T('지원')) : h('span', { class: 'tag err' }, I('alert'), T('8.0 미만 · 지원 안 함')),
        h('span', { class: 'num muted memv' }, fmtBytes(c.memory)))))) :
      h('div', { class: 'muted' }, raw.cuda && raw.cuda.error ? TM(raw.cuda.error) : T('CUDA 없음 — NVIDIA 드라이버(nvcuda.dll)가 필요합니다')));
  } catch (_) { }
}

// ---------- 설정(바꾸면 바로 저장) ----------
async function loadVk() {
  if (S.vk) return;
  try { const raw = await api('/api/device'); S.vk = (raw.vulkan && raw.vulkan['장치들'] ? raw.vulkan['장치들'] : []).map(v => v['이름']); resetSig(); renderSettings(); } catch (_) { }
}
let setTimer = null;
// 저장되면 그 줄에 잠깐 '저장됨'(알림 창을 쌓지 않는다). 화면 읽기 프로그램에는 무엇이 언제 적용되는지까지 알린다
const SAVED = { row: '', t: 0 };
function markSaved(row, note) {
  SAVED.row = row; SAVED.t = Date.now();
  for (const e of document.querySelectorAll('#page-settings [data-saved]')) e.hidden = e.dataset.saved !== row;
  announce(note);
  setTimeout(() => { if (Date.now() - SAVED.t >= 2400) for (const e of document.querySelectorAll('#page-settings [data-saved]')) e.hidden = true; }, 2500);
}
// withDir: 모델 폴더 칸이 바뀐 때만 그 값을 보낸다(다른 칸을 바꿀 때 덜 쓴 경로가 따라가지 않게) · row: 저장됨을 보일 줄
function saveSettings(note, recheck, delay, withDir, row) {
  S.setDirty = true;
  clearTimeout(setTimer);
  setTimer = setTimeout(async () => {
    const page = $('#page-settings');
    const on = page.querySelector('.seg-ctl button.on');
    const num = (id, lo, hi, dflt) => { const v = parseInt(page.querySelector(id).value, 10); return isNaN(v) ? dflt : Math.min(hi, Math.max(lo, v)); };
    const vkSel = page.querySelector('#set-vk');
    const body = {
      gpu_layer: on ? on.dataset.v : 'auto', vulkan_device: vkSel ? vkSel.value : S.st.settings.vulkan_device || '', slots: num('#set-slots', 1, 8, 2), context: num('#set-ctx', 1024, 131072, 8192),
      update_check: page.querySelector('#set-upd').checked,
    };
    if (withDir) body.models_dir = page.querySelector('#set-mdir').value.trim();
    // 범위에 맞춘 값을 칸에도 다시 적는다(1~8 · 1024~131072)
    page.querySelector('#set-slots').value = body.slots;
    page.querySelector('#set-ctx').value = body.context;
    const errEl = page.querySelector('#set-mdir-err');
    try {
      await api('/api/settings', body);
      if (withDir && errEl) errEl.hidden = true;
      markSaved(row || '', note);
      if (recheck) await api('/api/device/check', {}).catch(e => toast(e.message, 'err'));
    } catch (e) {
      if (withDir && errEl) { put(errEl, I('alert', 's'), h('span', null, e.message)); errEl.hidden = false; }
      else toast(e.message, 'err');
    }
    S.setDirty = false;
    await refresh();   // 새 설정을 받아 온 뒤에 다시 그린다
    if (recheck) { resetSig(); renderSettings(true); }
    else if (!page.contains(document.activeElement)) { resetSig(); renderSettings(); }
  }, delay || 0);
}
function renderSettings(force) {
  const st = S.st;
  const page = $('#page-settings');
  if (!force && (S.setDirty || (page.contains(document.activeElement) && S.view === 'settings'))) return;   // 고치는 중에는 다시 그리지 않는다
  if (!changed('settings', [st.settings, st.app.version, st.app.commit, st.app.built, st.app.data_dir, st.app.models_dir, st.app.port, S.update && S.update.tag, S.vk, st.device.layer]) && !force) return;
  const focusV = force && document.activeElement && document.activeElement.dataset ? document.activeElement.dataset.v : null;
  const s = st.settings, a = st.app;
  const NEXT = T('저장했습니다 · 다음에 모델을 불러올 때 적용됩니다');
  const seg = h('div', {
    class: 'seg-ctl', role: 'radiogroup', 'aria-label': T('GPU 백엔드'), onkeydown: ev => {
      if (ev.key !== 'ArrowLeft' && ev.key !== 'ArrowRight') return;
      const bs = [...ev.currentTarget.querySelectorAll('button')];
      const i = bs.findIndex(b => b.classList.contains('on'));
      const nx = bs[(i + (ev.key === 'ArrowRight' ? 1 : bs.length - 1)) % bs.length];
      ev.preventDefault();
      nx.focus();
      nx.click();
    },
  }, [['auto', T('자동')], ['cuda', 'CUDA'], ['vulkan', 'Vulkan']].map(([v, t]) =>
    h('button', {
      class: v === s.gpu_layer ? 'on' : '', 'data-v': v, role: 'radio', 'aria-checked': v === s.gpu_layer ? 'true' : 'false', tabindex: v === s.gpu_layer ? '0' : '-1', onclick: ev => {
        const me = ev.currentTarget;
        if (me.classList.contains('on')) return;
        me.parentElement.querySelectorAll('button').forEach(b => { const on = b === me; b.classList.toggle('on', on); b.setAttribute('aria-checked', on ? 'true' : 'false'); b.tabIndex = on ? 0 : -1; });
        saveSettings(T('저장했습니다 · 하드웨어를 다시 확인합니다'), true, 0, false, 'layer');
      },
    }, t)));
  const names = (S.vk || []).slice();
  if (s.vulkan_device && !names.includes(s.vulkan_device)) names.push(s.vulkan_device);
  const showVk = s.gpu_layer === 'vulkan' || (s.gpu_layer === 'auto' && st.device.layer === 'vulkan');
  const vkSel = h('select', { class: 'input', id: 'set-vk', 'aria-label': T('Vulkan 장치', 'one'), onchange: () => saveSettings(T('저장했습니다 · 하드웨어를 다시 확인합니다'), true, 0, false, 'vk') },
    h('option', { value: '' }, T('자동 — 첫 번째 외장 GPU')), names.map(n => h('option', { value: n, selected: n === s.vulkan_device ? true : null }, n)));
  // 숫자 칸은 다 쓴 뒤(칸을 떠날 때 · Enter)에만 저장한다 — 쓰는 도중의 값(1 → 16 → 163 …)을 저장하지 않게
  const numIn = (id, lo, hi, step, v, label) => {
    const e = h('input', { type: 'number', id, min: String(lo), max: String(hi), step: step ? String(step) : null, value: v, 'aria-label': label });
    e.addEventListener('change', () => saveSettings(NEXT, false, 0, false, id));
    e.addEventListener('keydown', ev => { if (ev.key === 'Enter') { ev.preventDefault(); e.blur(); } });
    return e;
  };
  const slots = numIn('set-slots', 1, 8, 0, s.slots, T('동시 처리 수'));
  const ctx = numIn('set-ctx', 1024, 131072, 1024, s.context, T('채팅마다 문맥 길이'));
  const defDir = (a.data_dir || '') + '모델\\';
  const mdir = h('input', { type: 'text', id: 'set-mdir', value: s.models_dir || '', placeholder: T('기본 폴더 사용'), 'aria-label': T('모델 폴더') });
  const commitDir = () => { if (mdir.value.trim() !== (s.models_dir || '')) saveSettings(T('저장했습니다 · 다음 실행부터 적용됩니다'), false, 0, true, 'set-mdir'); };
  mdir.addEventListener('change', commitDir);
  mdir.addEventListener('keydown', ev => { if (ev.key === 'Enter') { ev.preventDefault(); mdir.blur(); } });
  const pickDir = h('button', { class: 'btn needs-conn', 'aria-label': T('모델 폴더 찾아보기'), onclick: () => browse(mdir.value.trim() || defDir, p => { mdir.value = p; commitDir(); }) }, I('folder', 's'), T('찾아보기'));
  const upd = h('input', { type: 'checkbox', id: 'set-upd', 'aria-label': T('새 버전 알림'), onchange: () => saveSettings(T('저장했습니다'), false, 0, false, 'set-upd') });
  upd.checked = !!s.update_check;
  const savedMark = id => h('span', { class: 'saved', 'data-saved': id, hidden: !(SAVED.row === id && Date.now() - SAVED.t < 2400) }, I('check', 's'), T('저장됨'));
  const row = (t, d, c, extra, id) => h('div', { class: 'set-row-wrap' }, h('div', { class: 'set-row' }, h('div', { class: 'l' }, h('div', { class: 't' }, t, id ? savedMark(id) : null), d ? h('div', { class: 'd' }, d) : null),
    h('div', { class: 'c' }, c)), extra || null);
  const langSel = h('select', { class: 'input', id: 'set-lang', 'aria-label': EN ? 'Language' : '언어', onchange: ev => { store('geulpu.lang', ev.target.value); location.reload(); } },
    [['auto', T('자동 — 브라우저 언어')], ['ko', '한국어'], ['en', 'English']].map(([v, t]) => h('option', { value: v, selected: v === LANGPREF ? true : null }, t)));
  const upTxt = S.update && S.update.tag ? (newer(S.update.tag, a.version) ? T` · 새 버전 ${S.update.tag} 있음` : T(' · 최신 버전')) : '';
  put(page,
    pageHead(null, T('설정'), T('바꾼 값은 바로 저장됩니다(숫자 · 글 칸은 다 쓰고 칸을 떠날 때).')),
    h('div', { class: 'settings' },
      h('div', { class: 'sec' }, T('실행', 'sec')),
      h('div', { class: 'set-group' },
        row(T('GPU 백엔드'), T('자동은 NVIDIA GPU면 CUDA, 그 밖에는 Vulkan을 씁니다.'), seg, null, 'layer'),
        showVk ? row(T('Vulkan 장치', 'one'), T('GPU가 여러 개일 때 Vulkan으로 쓸 장치'), vkSel, null, 'vk') : null,
        row(T('동시 처리 수'), T('동시에 답할 수 있는 채팅 수(1~8) · 다음에 모델을 불러올 때 적용'), slots, null, 'set-slots'),
        row(T('채팅마다 문맥 길이'), T('토큰 · 그래픽 메모리에 맞지 않으면 엔진이 자동으로 줄입니다. 다음에 모델을 불러올 때 적용'), ctx, null, 'set-ctx')),
      h('div', { class: 'sec mt16' }, T('저장 위치')),
      h('div', { class: 'set-group' },
        row(T('모델 폴더'), T`다음 실행부터 적용됩니다. 기본: ${defDir}`, h('div', { class: 'row nowrap w100' }, mdir, pickDir), h('div', { class: 'set-err', id: 'set-mdir-err', hidden: true }), 'set-mdir'),
        row(T('데이터 폴더'), h('span', { class: 'break' }, a.data_dir), h('button', { class: 'btn sm needs-conn', onclick: () => api('/api/open', { what: 'data' }).catch(e => toast(e.message, 'err')) }, I('folder', 's'), T('열기')))),
      h('div', { class: 'sec mt16' }, T('정보', 'about')),
      h('div', { class: 'set-group' },
        row(EN ? 'Language · 언어' : '언어 · Language', T('화면에 쓰는 말입니다. 바꾸면 화면을 다시 불러옵니다.'), langSel),
        row(T('새 버전 알림'), T('앱을 열 때 GitHub에서 최신 버전 번호만 확인합니다.'), h('label', { class: 'switch' }, upd, h('i')), null, 'set-upd'),
        row(`GeulPU v${a.version}`, `${a.commit ? T('커밋 ') + a.commit.slice(0, 7) : T('개발 빌드')}${a.built ? ' · ' + a.built.replace('T', ' ').replace('Z', ' UTC') : ''}${upTxt}`,
          h('div', { class: 'row' }, h('a', { class: 'btn sm', href: 'https://github.com/wwoosshh/GeulPU/releases', target: '_blank', rel: 'noopener' }, T('릴리스'), I('ext', 's')),
            h('a', { class: 'btn sm', href: 'https://github.com/wwoosshh/GeulPU', target: '_blank', rel: 'noopener' }, T('소스'), I('ext', 's')))),
        row(T('API 주소'), T('다른 프로그램은 OpenAI 호환 API로 쓸 수 있습니다(이 PC 안에서만).'), h('div', { class: 'row nowrap' }, h('span', { class: 'mono small muted' }, `127.0.0.1:${a.port}/v1`),
          h('button', { class: 'icon-btn', 'data-tip': T('주소 복사'), 'data-tip-pos': 'left', 'aria-label': T('API 주소 복사'), onclick: () => copyText(`http://127.0.0.1:${a.port}/v1`).then(ok => ok && toast(T('복사했습니다'), 'ok')) }, I('copy')))))));
  if (focusV) { const b = page.querySelector(`.seg-ctl button[data-v="${focusV}"]`); if (b) b.focus(); }
}

// ---------- 연구 패널(38단계) — 잡음 없는 관찰 · 개입 · 비교 · 기록 ----------
// 같은 글을 다시 셈하면 생성할 때와 같은 확률이 나온다(계약 5.2 — 생성 길과 프롬프트 길이 같은 비트, 38b). 그래서 채팅 중에는 아무것도
// 더 셈하지 않고, 들여다볼 때만 그 답을 점수 매기기(echo)로 다시 셈한다. 두 실행의 차이는 바꾼 것에서만 나온다(배치 · 캐시 · GPU ·
// 백엔드와 상관없다). 연구 기록(답마다 토큰 · 씨앗의 원문 · 요청의 해시)은 이 브라우저에만 — 최근 60 개 · 1.5 MB 까지
const TRACE_KEY = 'geulpu.traces';
let TRACES = load(TRACE_KEY, {});
if (!TRACES || typeof TRACES !== 'object' || Array.isArray(TRACES)) TRACES = {};
const newTid = () => Date.now().toString(36) + Math.random().toString(36).slice(2, 8);
function saveTraces() {
  const ids = Object.keys(TRACES).sort((a, b) => TRACES[a].at - TRACES[b].at);
  let txt = JSON.stringify(TRACES);
  while (ids.length > 1 && (ids.length > 60 || txt.length > 1.5e6)) { delete TRACES[ids.shift()]; txt = JSON.stringify(TRACES); }
  try { localStorage.setItem(TRACE_KEY, txt); }
  catch (_) {                                          // 자리가 모자라면 오래된 반을 지운다 — 채팅 기록이 먼저다
    ids.splice(0, Math.ceil(ids.length / 2)).forEach(i => delete TRACES[i]);
    try { localStorage.setItem(TRACE_KEY, JSON.stringify(TRACES)); } catch (_) { }
  }
}
async function sha256Hex(s) {
  const d = await crypto.subtle.digest('SHA-256', new TextEncoder().encode(s));
  return [...new Uint8Array(d)].map(b => b.toString(16).padStart(2, '0')).join('');
}
// 씨앗을 원문 숫자 그대로 넣은 JSON(2⁵³ 위의 정수는 JS 의 수로 오가면 바뀐다) — 객체의 끝에 붙인다
function seedBody(obj, seed) {
  const s = JSON.stringify(obj);
  return seed ? s.slice(0, -1) + (s.length > 2 ? ',' : '') + '"seed":' + seed + '}' : s;
}
// 답이 끝나면 — 연구 기록: 요청을 다시 지을 매개변수와 그 몸의 해시(씨앗은 빼고), 뽑은 토큰, 씨앗의 원문
function traceAfter(asst, rp, rr, body) {
  if (!rr || !rr.gp || !Array.isArray(rr.gp.tokens) || !rr.gp.tokens.length) return;
  const m = modelOf(rp.model);
  const tid = newTid();
  const tr = {
    v: 1, at: Date.now(), model: rp.model, sha256: (m && m.sha256) || '', name: m ? displayName(m) : rp.model, quant: m ? quantOf(m) : '',
    tpl: rp.tpl, req: { max: rp.max, temp: rp.temp, system: rp.system, think: rp.think }, seed: rr.seed || '', tokens: rr.gp.tokens,
    slot: rr.gp.slot, cached: rr.gp.prompt_tokens_cached || 0, first_us: rr.gp.first_token_ms, gen_us: rr.gp.generate_us, finish: rr.finish || '',
    layer: (S.st && S.st.server.layer) || '', device: (S.st && S.st.device.name) || '', bodyHash: '',
  };
  if (Array.isArray(rr.gp.prompt_tokens) && rr.gp.prompt_tokens.length) tr.promptIds = rr.gp.prompt_tokens;   // 서버가 쓴 그대로
  TRACES[tid] = tr;
  asst.tid = tid;
  saveTraces();
  sha256Hex(JSON.stringify(body)).then(x => { tr.bodyHash = x; saveTraces(); }).catch(() => { });
}

const LAB_TABS = [['obs', '관찰'], ['act', '개입'], ['cmp', '비교'], ['score', '점수'], ['repro', '재현'], ['out', '내보내기']];
const LAB = {
  open: false, tab: 'obs', tid: null, an: {}, fpc: {}, busy: '', err: '', sel: -1, delta: load('geulpu.lab.delta', 0.05),
  cmpTid: '', cmpModel: '', cmpRes: {}, cross: {}, scoreText: '', scoreCtx: 'answer', scoreModel: '', scoreRes: null, scoreSel: -1,
  prov: null, replay: {}, capsule: null, texts: {}, regen: null,
};
{ const t = load('geulpu.lab.tab', 'obs'); if (LAB_TABS.some(x => x[0] === t)) LAB.tab = t; }
const LAB_K = 10;                                   // 걸음마다 상위 몇 개를 셈하나
const labCur = () => (LAB.tid && TRACES[LAB.tid]) || null;
const trSha = tr => tr.sha256 || ((modelOf(tr.model) || {}).sha256) || '';
const labOverlay = () => innerWidth <= 1100;
// 확률은 늘 % — 0.1 % 아래는 지수(3.5e-2%). nat 은 유효 숫자 셋 이상(0.01 아래는 지수, 1 아래는 세 자리, 그 위는 소수 셋째 자리)
const pct = lp => { const p = Math.exp(lp); return p >= 0.9995 ? '100%' : p >= 0.001 ? (p * 100).toFixed(p >= 0.1 ? 1 : 2) + '%' : (p * 100).toExponential(1) + '%'; };
const nat = x => (x === null || x === undefined || !isFinite(x) ? '-' : x === 0 ? '0' : Math.abs(x) < 0.01 ? x.toExponential(2) : Math.abs(x) < 1 ? x.toPrecision(3) : x.toFixed(3));
const sgn = x => (x > 0 ? '+' : x < 0 ? '−' : '') + nat(Math.abs(x));
const fpShort = x => (x ? x.slice(0, 16).replace(/(.{4})(?=.)/g, '$1 ') : '-');
// 토큰 글 보이기 — 줄바꿈 · 탭은 기호로, 빈 토큰은 ∅. 앞 토큰과 한 글자인 바이트 조각(null)은 글 없이 좁은 칸(.cont)
const tokShow = s => (s === null ? '' : s === '' ? '∅' : String(s).replace(/\n/g, '↵\n').replace(/\t/g, '→'));

// 이 답을 만든 대화 찾기 — 지금 보이는 대화(live), 아니면 버전에 숨은 곳
function labFind(tid) {
  const deep = msgs => {
    for (const u of msgs) for (const v of u.vers || []) {
      if (!v.rest) continue;
      const x = v.rest.find(m => m.tid === tid) || deep(v.rest);
      if (x) return x;
    }
    return null;
  };
  for (const room of S.rooms) {
    const i = room.messages.findIndex(m => m.tid === tid);
    if (i >= 0) return { room, idx: i, msg: room.messages[i], live: true };
    const x = deep(room.messages);
    if (x) return { room, idx: -1, msg: x, live: false };
  }
  return null;
}
// 이 채팅에서 연구 기록이 있는 답(숨은 버전까지) — 질문 차례, 같은 차례는 만든 때의 차례로
function labAnswers(room) {
  const out = [];
  if (!room) return out;
  const prev = s => String(s || '').replace(/<\/?think>/g, ' ').replace(/\s+/g, ' ').trim().slice(0, 30);
  const walk = (msgs, n0, ver, hidden) => {
    let n = n0, u = null;
    for (const m of msgs) {
      if (m.role === 'user') {
        n++;
        u = m;
        (m.vers || []).forEach((v, vi) => { if (vi !== m.vi && v.rest) walk(v.rest, n, T`버전 ${vi + 1}/${m.vers.length}`, true); });
      } else if (m.role === 'assistant' && m.tid && TRACES[m.tid]) {
        const vl = u ? (u.vers && u.vers.length > 1 ? T`버전 ${u.vi + 1}/${u.vers.length}` : '') : ver;
        const tr = TRACES[m.tid];
        out.push({ tid: m.tid, n, at: tr.at, label: [T`질문 ${n}`, vl, tr.branch ? T`갈래 — 걸음 ${tr.branch.k + 1}` : '', hidden ? T('숨은 버전') : ''].filter(Boolean).join(' · ') + ' — ' + prev(m.content) });
      }
    }
  };
  walk(room.messages, 0, '', false);
  return out.sort((a, b) => a.n - b.n || a.at - b.at);
}
async function labApi(path, body) {
  const r = await fetch(path, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: typeof body === 'string' ? body : JSON.stringify(body) });
  let j = null;
  try { j = await r.json(); } catch (_) { }
  if (!r.ok) throw new Error((j && j.error && j.error.message) || ('HTTP ' + r.status));
  return j;
}
// 셈은 그 답의 모델로 — 다른 모델이 올라 있으면 바꿔 불러온다(답이 흐르는 동안 · 벤치마크 중에는 하지 않는다)
async function labEnsureModel(id) {
  if (S.st && S.st.server.state === 'ready' && S.st.server.model === id) return;
  if (S.streaming) throw new Error(T('답이 흐르는 동안에는 모델을 바꿔 불러올 수 없습니다 — 끝난 뒤에 다시 누르세요'));
  if (benchBusy()) throw new Error(T('벤치마크가 끝나면 셈할 수 있습니다'));
  const m = modelOf(id);
  if (!m) throw new Error(T('이 답을 만든 모델이 설치되어 있지 않습니다'));
  if (!runnable(m)) throw new Error(T('이 모델은 이 PC에서 실행할 수 없습니다 — ') + TM(m.device_reason || (m.shape && m.shape.reason) || ''));
  LAB.busy = T`${labelById(id)} 불러오는 중`;
  labRender();
  await ensureServer(id, null);
}
// 이 답을 만든 프롬프트의 토큰 — 요청을 보낼 때와 같은 함수로 다시 지어 해시로 확인하고, 서버의 채팅 틀로 나눈다(한 번 지으면 기록에 둔다)
async function labPrompt(tid) {
  const tr = TRACES[tid];
  if (tr.promptIds) return tr.promptIds;
  const f = labFind(tid);
  if (!f) throw new Error(T('이 답이 있던 채팅을 찾지 못했습니다'));
  if (!f.live) throw new Error(T('숨은 버전의 답입니다 — 채팅에서 그 버전을 연 뒤에 셈하세요'));
  const { body } = replyBody(f.room, f.idx, { tpl: tr.tpl, max: tr.req.max, temp: tr.req.temp, system: tr.req.system, think: tr.req.think });
  if (tr.bodyHash && (await sha256Hex(JSON.stringify(body))) !== tr.bodyHash) throw new Error(T('이 답을 만든 요청을 똑같이 다시 지을 수 없습니다 — 앞 대화가 바뀌었습니다'));
  const tk = tr.tpl === 'deepseek-r1' ? { prompt: body.prompt } : { messages: body.messages, chat_template_kwargs: body.chat_template_kwargs };
  tr.promptIds = (await labApi('/tokenize', tk)).tokens;
  saveTraces();
  return tr.promptIds;
}
// 토큰마다 보이는 글 — 한 글자의 UTF-8 이 토큰 둘 이상에 걸치면(드문 한글 · 이모지) 첫 토큰에 그 글자를, 뒤 토큰은 null(이어짐)로.
// text_offset 은 돌려준 글의 바이트 자리다. 어긋나 보이면(틀린 바이트가 U+FFFD 로 바뀐 글) 서버가 준 토큰마다의 글을 그대로 쓴다
function tokTexts(text, offs, strs) {
  const enc = new TextEncoder().encode(text || '');
  const dec = new TextDecoder('utf-8', { fatal: true });
  const n = offs.length, out = new Array(n);
  const end = i => (i < n ? offs[i] : enc.length);
  for (let i = 0; i < n;) {
    let j = i + 1, t = null;
    for (; j <= n && j - i <= 4; j++) {
      if (!(offs[i] <= end(j) && end(j) <= enc.length)) break;
      try { t = dec.decode(enc.subarray(offs[i], end(j))); break; } catch (_) { }
    }
    if (t === null || (j === i + 1 && !String(strs[i]).includes('�') && t !== strs[i])) return strs.slice();
    out[i] = t;
    for (let q = i + 1; q < j; q++) out[q] = null;
    i = j;
  }
  return out;
}
// 점수 매기기 — prompt 뒤의 tokens 를 echo 로 다시 셈한다: 걸음마다 로그확률 · 상위 k · 엔트로피 · 로짓 해시(생성할 때와 같은 비트)
async function labScore(modelId, prompt, tokens, k) {
  await labEnsureModel(modelId);
  const P = prompt.length;
  const j = await labApi('/v1/completions', {
    prompt: prompt.concat(tokens), echo: true, max_tokens: 0, logprobs: k, temperature: 0, geulpu: { echo_from: P, logit_hash: true },
  });
  const c = j.choices[0], lp = c.logprobs, g = j.geulpu;
  return {
    raw: lp.tokens.slice(P), strs: tokTexts(c.text, lp.text_offset.slice(P), lp.tokens.slice(P)), lp: lp.token_logprobs.slice(P),
    ent: g.prompt_entropy.slice(P), topIds: g.prompt_top_ids.slice(P), topLp: g.prompt_top_logprobs.slice(P), hashes: g.prompt_logit_hashes, k,
  };
}
// 답 하나의 분석(한 번 셈해 둔다)
async function labAnalyze(tid, quiet) {
  if (LAB.an[tid]) return LAB.an[tid];
  const tr = TRACES[tid];
  if (!tr) throw new Error(T('이 답의 연구 기록이 없습니다'));
  await labEnsureModel(tr.model);
  if (!quiet) { LAB.busy = T('프롬프트를 다시 짓는 중'); labRender(); }
  const prompt = await labPrompt(tid);
  if (!quiet) { LAB.busy = T`토큰 ${fmtNum(tr.tokens.length)}개를 다시 셈하는 중(점수 매기기)`; labRender(); }
  const sc = await labScore(tr.model, prompt, tr.tokens, LAB_K);
  LAB.an[tid] = Object.assign({ prompt }, sc);
  return LAB.an[tid];
}
// 갈래의 답은 '프롬프트 + 앞 토큰'을 프롬프트로, 이어 생성한 토큰을 결과로 적는다(그 요청 하나로 다시 만들어지는 꼴)
function labParts(tr, prompt) {
  const off = tr.branch ? tr.branch.k + 1 : 0;
  return { off, prompt: prompt.concat(tr.tokens.slice(0, off)), tokens: tr.tokens.slice(off) };
}
const labMax = tr => (tr.branch ? tr.branch.max : tr.req.max);
// 지문(문서/38단계.md): 토큰 지문 = SHA-256("geulpu-fp/1\nmodel <모델 파일 SHA-256>\nprompt <번호,…>\ntokens <번호,…>\n"),
// 로짓 지문 = SHA-256("<토큰 지문>\n<걸음마다 로짓 해시,…>\n"). 장치 · 백엔드는 넣지 않는다 — 어디서 셈해도 같아야 하므로
async function labFp(sha, prompt, tokens, hashes) {
  const fp = await sha256Hex(`geulpu-fp/1\nmodel ${sha}\nprompt ${prompt.join(',')}\ntokens ${tokens.join(',')}\n`);
  return { fp, fpLogit: await sha256Hex(`${fp}\n${hashes.join(',')}\n`) };
}
function labFpOf(tid) {
  const tr = TRACES[tid], an = LAB.an[tid];
  if (!tr || !an) return null;
  const sha = trSha(tr), c = LAB.fpc[tid];
  if (c && c.sha === sha) return c.fp ? c : null;
  const cp = labParts(tr, an.prompt);
  LAB.fpc[tid] = { sha };
  labFp(sha, cp.prompt, cp.tokens, an.hashes.slice(cp.off)).then(v => { LAB.fpc[tid] = Object.assign({ sha }, v); labRender(); }).catch(() => { });
  return null;
}
// 걸음 i 의 여유(1위와 2위의 로그확률 차)와 고른 토큰의 순위(상위 k 밖이면 k + 1)
function labStep(an, toks, i) {
  const top = an.topLp[i] || [], ids = an.topIds[i] || [];
  const r = ids.indexOf(toks[i]);
  return { margin: top.length > 1 ? top[0] - top[1] : Infinity, rank: r >= 0 ? r + 1 : ids.length + 1 };
}
function labSummary(an, toks) {
  const n = an.lp.length;
  let sum = 0, ent = 0, minM = Infinity, minAt = 0, frag = 0, notTop = 0;
  for (let i = 0; i < n; i++) {
    sum += an.lp[i];
    ent += an.ent[i] || 0;
    const st = labStep(an, toks, i);
    if (st.margin < minM) { minM = st.margin; minAt = i; }
    if (st.margin < LAB.delta) frag++;
    if (st.rank > 1) notTop++;
  }
  return { n, sum, mean: n ? sum / n : 0, ppl: n ? Math.exp(-sum / n) : 0, ent: n ? ent / n : 0, minM, minAt, frag, notTop };
}
// 토큰 번호의 글(대안 토큰 — 한 번 물어 둔다)
async function labTexts(ids) {
  const need = [...new Set(ids)].filter(id => id >= 0 && LAB.texts[id] === undefined);
  await Promise.all(need.map(id => labApi('/detokenize', { tokens: [id] }).then(j => { LAB.texts[id] = j.prompt; }).catch(() => { LAB.texts[id] = '?'; })));
}
const labKpi = (v, l, tip) => h('div', { class: 'kpi', 'data-tip': tip || null, 'data-tip-pos': 'top' }, h('div', { class: 'v' }, v), h('div', { class: 'l' }, l));
const labSec = t => h('div', { class: 'sec lab-sec' }, t);
const labNote = t => h('p', { class: 'small muted m0' }, t);
const labCol = (...k) => h('div', { class: 'col g12' }, ...k);
function labRun(fn) {
  LAB.err = '';
  Promise.resolve().then(fn).catch(e => { LAB.err = e.message; }).finally(() => { LAB.busy = ''; labRender(); updateSend(); });
}
const labAnalyzeBtn = (tid, label) => h('button', {
  class: 'btn primary needs-conn', disabled: (LAB.busy || S.streaming) ? true : null, onclick: () => labRun(() => labAnalyze(tid)),
}, I('play', 's'), label || T('토큰마다 확률 셈하기'));

// ---- 틀 ----
$('#labbtn').innerHTML = svg('flask', 's');
$('#labbtn').addEventListener('click', () => labToggle());
function labToggle(open) {
  LAB.open = open === undefined ? !LAB.open : open;
  $('#lab').hidden = !LAB.open;
  $('#labbtn').setAttribute('aria-expanded', LAB.open ? 'true' : 'false');
  if (!labOverlay()) store('geulpu.lab.open', LAB.open);
  if (LAB.open) {
    labSync(true);
    if (labOverlay()) setTimeout(() => { const t = $('#lab .lab-tabs button.on'); if (t) t.focus(); }, 0);
  } else if (labOverlay()) $('#labbtn').focus();
}
function labPick(tid) {
  if (LAB.tid !== tid) { LAB.tid = tid; LAB.sel = -1; LAB.regen = null; LAB.err = ''; LAB.scoreModel = ''; }
}
// 답 아래의 플라스크 — 그 답을 패널에서. 모델이 올라 있으면 바로 셈한다
function labOpen(tid) {
  labPick(tid);
  if (LAB.tab === 'score' || LAB.tab === 'out') LAB.tab = 'obs';
  if (!LAB.open) labToggle(true); else labRender();
  const tr = TRACES[tid];
  if (tr && !LAB.an[tid] && !LAB.busy && !S.streaming && S.st && S.st.server.state === 'ready' && S.st.server.model === tr.model) labRun(() => labAnalyze(tid));
}
// 채팅을 바꾸거나 답 · 상태가 바뀌면 — 고른 답이 이 채팅에 없으면 이 채팅의 마지막 답으로. 바뀐 것이 있을 때만 다시 그린다
function labSync(force) {
  if (!LAB.open) return;
  const list = labAnswers(curRoom());
  if (!LAB.tid || !list.some(x => x.tid === LAB.tid)) labPick(list.length ? list[list.length - 1].tid : null);
  const tr = labCur();
  const st = S.st ? [S.st.server.state, S.st.server.model, S.conn, tr ? trSha(tr) : '', tr && modelOf(tr.model) ? !!modelOf(tr.model).verifying : false] : null;
  if (changed('lab-list', [S.roomId, LAB.tid, list.map(x => x.label), st, !!S.streaming]) || force) labRender();
}
function labRender() {
  const box = $('#lab');
  if (!box || !LAB.open) return;
  const tabs = h('div', { class: 'seg-ctl lab-tabs', role: 'tablist', 'aria-label': T('연구 패널') }, LAB_TABS.map(([k, t]) => h('button', {
    role: 'tab', class: LAB.tab === k ? 'on' : '', 'aria-selected': LAB.tab === k ? 'true' : 'false', 'data-fk': 'lab:' + k,
    onclick: () => { LAB.tab = k; LAB.err = ''; store('geulpu.lab.tab', k); labRender(); },
  }, T(t))));
  const head = h('div', { class: 'lab-head' },
    h('div', { class: 'row g8 nowrap' }, h('span', { class: 'lab-title' }, I('flask', 's'), T('연구 패널')), h('span', { class: 'grow' }),
      h('button', { class: 'icon-btn', 'aria-label': T('연구 패널 닫기'), 'data-tip': T('닫기'), 'data-tip-pos': 'left', 'data-fk': 'lab:close', onclick: () => labToggle(false) }, I('x', 's'))),
    tabs);
  const body = h('div', { class: 'lab-body', role: 'tabpanel', 'aria-label': T(LAB_TABS.find(x => x[0] === LAB.tab)[1]) });
  if (LAB.busy) body.append(h('div', { class: 'lab-busy', role: 'status' }, h('i', { class: 'dot busy' }), h('span', null, LAB.busy)));
  if (LAB.err) body.append(h('div', { class: 'note-line err lab-err', role: 'alert' }, I('alert', 's'), h('span', null, TM(LAB.err))));
  const fn = { obs: labObs, act: labAct, cmp: labCmp, score: labScoreView, repro: labRepro, out: labOut }[LAB.tab] || labObs;
  try { body.append(fn()); }
  catch (e) { body.append(h('div', { class: 'note-line err' }, I('alert', 's'), h('span', null, String(e && e.message)))); console.error(e); }
  // 같은 탭 · 같은 답을 다시 그리면 스크롤 · 입력 칸의 커서는 그 자리에(탭이나 답을 바꾸면 맨 위부터)
  const same = box.dataset.tab === LAB.tab && box.dataset.tid === String(LAB.tid);
  const ob = box.querySelector('.lab-body'), os = box.querySelector('.tokstrip');
  const y = same && ob ? ob.scrollTop : 0, sy = same && os ? os.scrollTop : 0;
  const a = document.activeElement, sel = a && box.contains(a) && 'selectionStart' in a && a.type !== 'number' ? [a.selectionStart, a.selectionEnd] : null;
  rebuild(box, head, body);
  const on = box.querySelector('.lab-tabs button.on');             // 좁은 폭(휴대폰 · 영어)에서는 탭 줄이 옆으로 흐른다 — 고른 탭이 보이게
  if (on) { const tl = on.parentElement, d = on.getBoundingClientRect().right - tl.getBoundingClientRect().right; if (d > 0) tl.scrollLeft += d + 8; }
  box.dataset.tab = LAB.tab;
  box.dataset.tid = String(LAB.tid);
  body.scrollTop = y;
  const ns = box.querySelector('.tokstrip');
  if (ns) ns.scrollTop = sy;
  const b = document.activeElement;
  if (sel && b && box.contains(b) && 'setSelectionRange' in b) { try { b.setSelectionRange(sel[0], sel[1]); } catch (_) { } }
}
// 고른 답의 머리 — 모델 · 매개변수 · 시간
function labAnswerHead(tr) {
  const tps = tr.gen_us > 0 && tr.tokens.length > 1 ? (tr.tokens.length - 1) / (tr.gen_us / 1e6) : 0;
  return h('div', { class: 'lab-ans' },
    h('div', { class: 'row g8' }, h('span', { class: 'ink' }, tr.name), tr.quant ? h('span', { class: 'qlab' }, tr.quant) : null),
    h('div', { class: 'small muted' }, [
      tr.req.temp > 0 ? T`온도 ${tr.req.temp} · 씨앗 ${tr.seed || '?'}` : T('탐욕(온도 0)'),
      tr.finish === 'stop' ? T`토큰 ${fmtNum(tr.tokens.length)}개(끝 토큰 포함)` : T`토큰 ${fmtNum(tr.tokens.length)}개`,
      tps ? T`${tps.toFixed(1)}토큰/초` : null, tr.branch ? T`갈래 — 걸음 ${tr.branch.k + 1}부터` : null,
      `${tr.device || ''}${tr.layer ? ' · ' + (LAYER[tr.layer] || tr.layer) : ''}`,
    ].filter(Boolean).join(NB)));
}
function labPicker() {
  const list = labAnswers(curRoom());
  if (!list.length) return h('div', { class: 'lab-empty' }, I('flask', 'l'), h('div', { class: 'ink' }, T('분석할 답이 없습니다')),
    h('div', { class: 'small muted' }, T('이 판에서 받은 답부터 연구 기록이 남습니다. 질문을 보내 답을 받은 뒤 답 아래의 플라스크 단추를 누르세요.')));
  return h('select', { class: 'input w100', 'aria-label': T('분석할 답'), 'data-fk': 'lab:pick', onchange: ev => { labPick(ev.target.value); labRender(); } },
    list.map(x => h('option', { value: x.tid, selected: x.tid === LAB.tid ? true : null }, x.label)));
}

// ---- 토큰 띠 — 확률이 낮을수록 짙게, 아슬아슬한 걸음(여유 < δ)은 밑줄, 1위가 아닌 토큰을 뽑은 걸음은 테두리.
//      띠 하나가 Tab 정거장 하나 — 화살표 · Home · End 로 걸음을 옮긴다 ----
function labStrip(an, toks, sel, onSel, label) {
  const box = h('div', { class: 'tokstrip', role: 'listbox', tabindex: '0', 'aria-label': label });
  const els = an.lp.map((lp, i) => {
    const p = Math.exp(lp), st = labStep(an, toks, i), s = an.strs[i];
    const b = p >= 0.9 ? 0 : p >= 0.5 ? 1 : p >= 0.2 ? 2 : p >= 0.05 ? 3 : 4;
    return h('span', {
      id: 'tk' + i, role: 'option', 'aria-selected': sel === i ? 'true' : 'false', 'data-i': i,
      class: `tok b${b}${st.margin < LAB.delta ? ' frag' : ''}${st.rank > 1 ? ' pick' : ''}${s === null ? ' cont' : ''}${sel === i ? ' sel' : ''}`,
      'data-tip': T`걸음 ${i + 1} · 확률 ${pct(lp)} · 여유 ${nat(st.margin)}`, 'data-tip-pos': 'top',
    }, tokShow(s));
  });
  box.append(...els);
  if (sel >= 0 && sel < els.length) box.setAttribute('aria-activedescendant', 'tk' + sel);
  const go = i => {
    if (!(i >= 0 && i < els.length)) return;
    const old = box.querySelector('.tok.sel');
    if (old) { old.classList.remove('sel'); old.setAttribute('aria-selected', 'false'); }
    els[i].classList.add('sel');
    els[i].setAttribute('aria-selected', 'true');
    box.setAttribute('aria-activedescendant', els[i].id);
    els[i].scrollIntoView({ block: 'nearest' });
    onSel(i);
  };
  box.addEventListener('click', ev => { const t = ev.target.closest('[data-i]'); if (t) go(+t.dataset.i); });
  box.addEventListener('keydown', ev => {
    const cur = box.querySelector('.tok.sel'), c = cur ? +cur.dataset.i : -1, k = ev.key;
    const n = k === 'ArrowRight' || k === 'ArrowDown' ? c + 1 : k === 'ArrowLeft' || k === 'ArrowUp' ? Math.max(0, c - 1) : k === 'Home' ? 0 : k === 'End' ? els.length - 1 : null;
    if (n === null) return;
    ev.preventDefault();
    go(Math.min(els.length - 1, n));
  });
  box._go = go;
  return box;
}
// 걸음 하나 — 확률 · 순위 · 엔트로피 · 여유 · 로짓 해시, 대안 상위 k(막대). 갈래를 낼 수 있으면 대안마다 '거기서 갈래'
function labDetail(an, toks, i, canBranch) {
  const ids = an.topIds[i] || [], lps = an.topLp[i] || [];
  if (ids.some(id => LAB.texts[id] === undefined)) labTexts(ids).then(() => document.querySelectorAll('#lab .lab-detail-holder').forEach(x => x._render && x._render()));
  const st = labStep(an, toks, i);
  const txt = id => (LAB.texts[id] === undefined ? '…' : tokShow(LAB.texts[id]));
  const rows = ids.map((id, q) => h('div', { class: 'alt' + (id === toks[i] ? ' chosen' : '') },
    h('span', { class: 'alt-t' }, txt(id)),
    h('span', { class: 'alt-bar', 'aria-hidden': 'true' }, h('i', { style: `width:${Math.max(1, Math.round(Math.exp(lps[q]) * 100))}%` })),
    h('span', { class: 'alt-p' }, pct(lps[q])),
    canBranch && id !== toks[i] ? h('button', {
      class: 'icon-btn sm needs-conn', 'aria-label': T`이 토큰으로 갈래: ${LAB.texts[id] === undefined ? id : LAB.texts[id]}`,
      'data-tip': T('거기서 갈래 — 이 토큰을 고른 뒤 같은 설정으로 이어 생성(지금 답은 버전으로 남음)'), 'data-tip-pos': 'left',
      disabled: (S.streaming || LAB.busy) ? true : null, onclick: () => labBranch(i, id),
    }, I('forward', 's')) : h('span', { class: 'alt-go' }, id === toks[i] ? h('span', { role: 'img', 'aria-label': T('고른 토큰'), 'data-tip': T('고른 토큰'), 'data-tip-pos': 'left' }, I('check', 's')) : '')));
  return h('div', { class: 'card lab-detail' },
    h('div', { class: 'row g8 nowrap' }, h('span', { class: 'eyebrow' }, T`걸음 ${i + 1}`), h('span', { class: 'grow' }), h('span', { class: 'eyebrow mono' }, `#${toks[i]}`)),
    h('div', { class: 'lab-tokbig' }, tokShow(an.raw[i])),
    an.strs[i] === null || String(an.raw[i]).includes('�') ? labNote(T('UTF-8 바이트 조각인 토큰입니다 — 이웃 토큰과 합쳐 한 글자가 됩니다.')) : null,
    h('div', { class: 'kv lab-kv' },
      h('div', { class: 'k' }, T('확률')), h('div', { class: 'v' }, `${pct(an.lp[i])}${NB}${T('로그확률')} ${nat(an.lp[i])}`),
      h('div', { class: 'k' }, T('순위')), h('div', { class: 'v' }, st.rank > ids.length ? T`${ids.length}위 밖` : T`${st.rank}위`),
      h('div', { class: 'k' }, T('엔트로피')), h('div', { class: 'v' }, nat(an.ent[i]) + ' nat'),
      h('div', { class: 'k' }, T('여유')), h('div', { class: 'v' }, isFinite(st.margin) ? nat(st.margin) + ' nat' : '-'),
      h('div', { class: 'k' }, T('로짓 해시')), h('div', { class: 'v mono small' }, an.hashes[i] || '-')),
    h('div', { class: 'eyebrow' }, T`대안 상위 ${ids.length}개`),
    h('div', { class: 'alts' }, rows));
}
function labDetailHolder(get) {
  const holder = h('div', { class: 'lab-detail-holder' });
  holder._render = () => put(holder, get());
  holder._render();
  return holder;
}

// ---- 관찰 ----
function labObs() {
  const tr = labCur();
  const out = [labPicker()];
  if (!tr) return labCol(out);
  out.push(labAnswerHead(tr));
  const an = LAB.an[LAB.tid];
  if (!an) {
    out.push(labNote(T('토큰마다 확률 · 순위 · 대안 · 엔트로피를 보려면 이 답을 다시 셈합니다(점수 매기기). 생성할 때와 같은 비트의 값이고, GPU를 잠깐 씁니다.')),
      S.st && S.st.server.model !== tr.model ? labNote(T`이 답의 모델(${labelById(tr.model)})을 불러옵니다.`) : null,
      h('div', null, labAnalyzeBtn(LAB.tid)));
    return labCol(out);
  }
  const toks = tr.tokens, sm = labSummary(an, toks);
  out.push(h('div', { class: 'kpis sm lab-kpis' },
    labKpi(nat(sm.mean), T('평균 로그확률'), T('토큰마다 로그확률의 평균(nat)')),
    labKpi(sm.ppl.toFixed(3), T('퍼플렉시티'), T('exp(−평균 로그확률)')),
    labKpi(nat(sm.ent), T('평균 엔트로피'), T('걸음마다 모델 분포의 엔트로피 평균(nat)')),
    labKpi(nat(sm.minM), T`가장 작은 여유 (걸음 ${sm.minAt + 1})`, T('1위와 2위의 로그확률 차가 가장 작은 걸음')),
    labKpi(fmtNum(sm.frag), T`아슬아슬 (여유 < ${LAB.delta})`, T('1위와 2위의 차가 δ보다 작은 걸음 — 셈이나 설정이 조금만 달라도 다른 토큰이 될 자리')),
    labKpi(fmtNum(sm.notTop), T('1위가 아닌 토큰'), T('뽑기(온도 > 0)가 1위가 아닌 토큰을 고른 걸음'))));
  out.push(labNote(T('확률은 모델의 분포(온도 1)입니다 — 뽑기에 쓴 온도 · top-k · top-p로 거른 분포가 아닙니다. 값은 계약 5.2로 정해져 어느 PC · GPU에서 셈해도 같습니다.')));
  const dIn = h('input', { type: 'number', min: '0.001', max: '5', step: '0.01', value: LAB.delta, class: 'lab-num', 'aria-label': T('아슬아슬의 기준 δ(nat)'), 'data-fk': 'lab:delta' });
  dIn.addEventListener('change', () => { const v = parseFloat(dIn.value); if (v > 0 && v <= 5) { LAB.delta = v; store('geulpu.lab.delta', v); labRender(); } });
  const fragAt = [];
  for (let i = 0; i < an.lp.length; i++) if (labStep(an, toks, i).margin < LAB.delta) fragAt.push(i);
  const f = labFind(LAB.tid);
  const holder = labDetailHolder(() => (LAB.sel >= 0 && LAB.sel < an.lp.length ? labDetail(an, toks, LAB.sel, !!(f && f.live))
    : labNote(T('토큰을 누르면 그 걸음의 대안과 수가 보입니다. 띠에서 화살표 키로 걸음을 옮깁니다.'))));
  const strip = labStrip(an, toks, LAB.sel, i => { LAB.sel = i; holder._render(); }, T('답의 토큰'));
  const jump = dir => {
    const c = LAB.sel;
    const n = dir > 0 ? fragAt.find(x => x > c) : [...fragAt].reverse().find(x => x < c);
    if (n !== undefined) strip._go(n);
  };
  out.push(h('div', { class: 'row g8 small muted' },
    h('label', { class: 'row g8' }, h('span', null, T('아슬아슬의 기준 δ')), dIn, h('span', null, 'nat')), h('span', { class: 'grow' }),
    h('span', { class: 'row g0' },
      h('button', { class: 'icon-btn sm', 'aria-label': T('앞의 아슬아슬한 걸음'), 'data-tip': T('앞의 아슬아슬한 걸음'), 'data-tip-pos': 'top', disabled: fragAt.length ? null : true, onclick: () => jump(-1) }, I('chevL', 's')),
      h('button', { class: 'icon-btn sm', 'aria-label': T('다음 아슬아슬한 걸음'), 'data-tip': T('다음 아슬아슬한 걸음'), 'data-tip-pos': 'top', disabled: fragAt.length ? null : true, onclick: () => jump(1) }, I('chevR', 's')))));
  out.push(strip, h('div', { class: 'lab-legend small muted' },
    h('span', null, h('i', { class: 'lg b3' }), T('짙을수록 확률이 낮음')), h('span', null, h('i', { class: 'lg frag' }), T('밑줄 = 아슬아슬')), h('span', null, h('i', { class: 'lg pick' }), T('테두리 = 1위가 아닌 토큰'))));
  out.push(holder);
  return labCol(out);
}

// ---- 개입 ----
function labAct() {
  const tr = labCur();
  const out = [labPicker()];
  if (!tr) return labCol(out);
  const f = labFind(LAB.tid), live = !!(f && f.live), an = LAB.an[LAB.tid];
  out.push(labAnswerHead(tr));
  if (!live) out.push(h('div', { class: 'note-line' }, I('info', 's'), h('span', null, T('이 답은 지금 보이지 않는 버전에 있습니다 — 채팅에서 그 버전을 연 뒤에 개입할 수 있습니다.'))));
  // 거기서 갈래
  out.push(labSec(T('거기서 갈래')), labNote(T('어느 걸음에서 다른 토큰을 고르고 같은 설정으로 이어 생성합니다. 뽑기(온도 > 0)면 같은 씨앗 · 같은 걸음 번호의 난수를 써서 바뀐 것은 고른 토큰 하나뿐입니다. 새 답은 그 질문의 새 버전으로 남고, 지금 답은 앞 버전으로 남습니다.')));
  if (!an) out.push(h('div', null, labAnalyzeBtn(LAB.tid)));
  else if (live) {
    const sm = labSummary(an, tr.tokens);
    if (LAB.sel < 0 || LAB.sel >= an.lp.length) out.push(h('div', { class: 'row g8' }, labNote(T('관찰에서 걸음을 고르거나')),
      h('button', { class: 'btn sm', onclick: () => { LAB.sel = sm.minAt; labRender(); } }, I('target', 's'), T`가장 아슬아슬한 걸음 ${sm.minAt + 1} 고르기`)));
    else out.push(labDetailHolder(() => labDetail(an, tr.tokens, LAB.sel, true)));
  }
  // 설정을 바꿔 다시
  if (!LAB.regen) LAB.regen = { temp: String(tr.req.temp), seed: tr.req.temp > 0 ? tr.seed : '', max: String(labMax(tr)) };
  const rg = LAB.regen;
  const inp = (k, attrs) => { const e = h('input', Object.assign({ value: rg[k], 'data-fk': 'lab:rg:' + k }, attrs)); e.addEventListener('input', () => { rg[k] = e.value; }); return e; };
  out.push(labSec(T('설정을 바꿔 다시 생성')), labNote(T('같은 질문 · 같은 대화 · 같은 모델로 온도 · 씨앗 · 최대 길이만 바꿔 새 버전을 만듭니다. 그다음 비교에서 처음 갈린 걸음을 봅니다. 씨앗을 비우면 서버가 고릅니다(기록에 남음).')),
    h('div', { class: 'lab-form' },
      h('label', null, h('span', null, T('온도')), inp('temp', { type: 'number', min: '0', max: '2', step: '0.05', class: 'lab-num' })),
      h('label', null, h('span', null, T('씨앗')), inp('seed', { type: 'text', inputmode: 'numeric', placeholder: T('무작위'), class: 'lab-seed', spellcheck: 'false' })),
      h('label', null, h('span', null, T('최대 길이')), inp('max', { type: 'number', min: '1', max: '32768', step: '1', class: 'lab-num' }))),
    h('div', null, h('button', { class: 'btn needs-conn', disabled: (!live || S.streaming || LAB.busy) ? true : null, onclick: () => labRun(labRegen) }, I('refresh', 's'), T('새 버전 생성'))));
  // 질문을 바꿔
  out.push(labSec(T('질문을 바꿔')), labNote(T('채팅에서 질문의 연필 단추로 고쳐 보내면 새 버전이 생깁니다. 비교의 \'A의 답을 B의 프롬프트로 점수\'가 바꾼 곳 하나의 효과를 토큰마다 잽니다 — 같은 답 토큰의 확률이 어떻게 바뀌는가.')));
  return labCol(out);
}
// 같은 차례의 사용자 메시지 자리
function labTurn(room, idx) {
  let uk = idx - 1;
  while (uk >= 0 && room.messages[uk].role !== 'user') uk--;
  return uk;
}
// 거기서 갈래 — 같은 차례의 새 버전: 프롬프트 + 앞 i 토큰 + 고른 대안으로 이어 생성. 뽑기면 같은 씨앗 · 걸음 i + 1 부터(원래 답과 같은 난수)
async function labBranch(i, alt) {
  const tid = LAB.tid, f = labFind(tid), tr = TRACES[tid], an = LAB.an[tid];
  if (!f || !f.live || !tr || !an || S.streaming || LAB.busy) return;
  const { room, idx } = f;
  const uk = labTurn(room, idx);
  if (uk < 0) return;
  const pre = tr.tokens.slice(0, i).concat([alt]);
  const max = Math.max(1, tr.req.max - pre.length);
  let preText = '';
  try {
    await labEnsureModel(tr.model);
    preText = (await labApi('/detokenize', { tokens: pre })).prompt;
  } catch (e) { LAB.err = e.message; LAB.busy = ''; labRender(); return; }
  LAB.busy = '';
  LAB.err = '';
  if (S.roomId !== room.id) pickRoom(room.id);
  branch(room, uk, room.messages[uk].content);
  const asst = { role: 'assistant', content: preText, template: tr.tpl, model: tr.model, max: tr.req.max, partial: true };
  room.messages.push(asst);
  room.updated = Date.now();
  stickNext = true;
  S.streaming = { aborted: false, ctrl: null, loading: false, roomId: room.id, msg: asst, el: null, notice: null };
  saveRooms(); renderRooms(); renderMsgs(); renderChatTop(true);
  labRender();
  try {
    const body = { prompt: an.prompt.concat(pre), max_tokens: max, temperature: tr.req.temp, stream: true, geulpu: { step_offset: pre.length } };
    if (tr.tpl === 'qwen3') body.geulpu.special = true;
    const rr = await streamReply(asst, room, '/v1/completions', body, { text: true, bodyStr: seedBody(body, tr.req.temp > 0 ? tr.seed : '') });
    asst.stats = rr.stats;
    if (rr.gp && Array.isArray(rr.gp.tokens)) {
      const nt = newTid();
      TRACES[nt] = Object.assign({}, tr, {
        at: Date.now(), tokens: pre.concat(rr.gp.tokens), promptIds: an.prompt, branch: { from: tid, k: i, alt, max }, seed: tr.req.temp > 0 ? tr.seed : (rr.seed || ''),
        slot: rr.gp.slot, cached: rr.gp.prompt_tokens_cached || 0, first_us: rr.gp.first_token_ms, gen_us: rr.gp.generate_us, finish: rr.finish || '', bodyHash: '',
        layer: (S.st && S.st.server.layer) || '', device: (S.st && S.st.device.name) || '',
      });
      asst.tid = nt;
      saveTraces();
      labPick(nt);
      LAB.sel = i;
    }
    room.updated = Date.now();
    saveRooms();
    announce(T('갈래의 답이 끝났습니다'));
  } catch (e) {
    const aborted = S.streaming && S.streaming.aborted;
    delete asst.partial;
    if (aborted) { asst.stats = '중지됨'; asst.stopped = true; }
    else { asst.stats = '오류'; const em = { role: 'error', content: e.message }; FRESH.add(em); room.messages.push(em); }
    saveRooms();
  } finally {
    S.streaming = null;
    renderRooms(); renderMsgs(); renderChatTop(true);
    if (LAB.tid && TRACES[LAB.tid] && !LAB.an[LAB.tid]) labRun(() => labAnalyze(LAB.tid, true));
    else labRender();
  }
}
// 설정을 바꿔 다시 — 같은 차례의 새 버전(같은 모델 · 대화 · 시스템 프롬프트 · 생각 켬끔, 온도 · 씨앗 · 최대 길이만 바꿈)
async function labRegen() {
  const tid = LAB.tid, f = labFind(tid), tr = TRACES[tid];
  if (!f || !f.live || !tr || S.streaming) return;
  const rg = LAB.regen || {};
  const temp = parseFloat(rg.temp), max = parseInt(rg.max, 10), seed = String(rg.seed || '').trim();
  if (!(temp >= 0 && temp <= 2)) throw new Error(T('온도는 0 ~ 2 사이의 수여야 합니다'));
  if (!(max >= 1 && max <= 32768)) throw new Error(T('최대 길이는 1 ~ 32768 사이의 정수여야 합니다'));
  if (seed && !(/^\d{1,20}$/.test(seed) && BigInt(seed) < 2n ** 64n)) throw new Error(T('씨앗은 0 ~ 2⁶⁴−1 사이의 정수여야 합니다'));
  if (!S.conn) throw new Error(T('GeulPU.exe와 연결이 끊겨 보낼 수 없습니다'));
  if (benchBusy()) throw new Error(T('벤치마크가 끝나면 보낼 수 있습니다'));
  if (!runnable(modelOf(tr.model))) throw new Error(T('이 답을 만든 모델이 설치되어 있지 않습니다'));
  const { room, idx } = f;
  const uk = labTurn(room, idx);
  if (uk < 0) return;
  if (S.roomId !== room.id) pickRoom(room.id);
  branch(room, uk, room.messages[uk].content);
  await replyTo(room, max, { temp, seed, model: tr.model, system: tr.req.system, think: tr.req.think });
  const last = room.messages[room.messages.length - 1];
  if (last && last.tid) { labPick(last.tid); LAB.regen = null; }
}

// ---- 비교 ----
function labCmp() {
  const tr = labCur();
  const out = [labPicker()];
  if (!tr) return labCol(out);
  const an = LAB.an[LAB.tid];
  if (!an) { out.push(labNote(T('비교하려면 먼저 이 답(A)을 셈합니다.')), h('div', null, labAnalyzeBtn(LAB.tid))); return labCol(out); }
  // 두 답
  out.push(labSec(T('두 답 맞대기')), labNote(T('버전 · 갈래끼리 견줍니다. 프롬프트가 같으면 처음 갈린 걸음을, 다르면(질문을 고친 버전) 한쪽 답을 다른 쪽 프롬프트 뒤에서 점수 매겨 바꾼 곳 하나의 효과를 토큰마다 잽니다.')));
  const others = labAnswers(curRoom()).filter(x => x.tid !== LAB.tid);
  if (LAB.cmpTid && !others.some(x => x.tid === LAB.cmpTid)) LAB.cmpTid = '';
  if (!others.length) out.push(labNote(T('견줄 다른 답이 없습니다 — 개입에서 갈래를 내거나 설정을 바꿔 다시 생성하면 버전이 생깁니다.')));
  else {
    out.push(h('select', { class: 'input w100', 'aria-label': T('견줄 답(B)'), 'data-fk': 'lab:cmpB', onchange: ev => { LAB.cmpTid = ev.target.value; labRender(); } },
      h('option', { value: '' }, T('견줄 답(B) 고르기')), others.map(x => h('option', { value: x.tid, selected: x.tid === LAB.cmpTid ? true : null }, x.label))));
    const trB = LAB.cmpTid && TRACES[LAB.cmpTid], anB = trB && LAB.an[LAB.cmpTid];
    if (trB && !anB) out.push(h('div', null, labAnalyzeBtn(LAB.cmpTid, T('B도 셈하기'))));
    if (trB && anB) out.push(labPair(an, tr, anB, trB));
  }
  // 다른 모델로
  out.push(labSec(T('같은 글을 다른 모델로 점수')), labNote(T('이 답의 토큰을 다른 모델(예: 같은 모델의 다른 양자화)로 다시 셈합니다. 셈의 잡음이 없으므로 토큰마다의 차이는 모델 파일 하나의 효과입니다. 모델을 바꿔 불러옵니다(수십 초).')));
  const mA = modelOf(tr.model) || {};
  const cand = S.st ? sortedModels().filter(m => m.id !== tr.model && runnable(m) && m.shape && mA.shape && m.shape.vocab === mA.shape.vocab && m.template === mA.template) : [];
  if (!cand.length) out.push(labNote(T('어휘와 채팅 틀이 같은 다른 모델이 설치되어 있지 않습니다.')));
  else {
    if (LAB.cmpModel && !cand.some(m => m.id === LAB.cmpModel)) LAB.cmpModel = '';
    out.push(h('div', { class: 'row g8 nowrap' },
      h('select', { class: 'input grow', 'aria-label': T('견줄 모델'), 'data-fk': 'lab:cmpM', onchange: ev => { LAB.cmpModel = ev.target.value; labRender(); } },
        h('option', { value: '' }, T('모델 고르기')), cand.map(m => h('option', { value: m.id, selected: m.id === LAB.cmpModel ? true : null }, `${displayName(m)} · ${quantOf(m)}`))),
      h('button', {
        class: 'btn needs-conn', disabled: (!LAB.cmpModel || LAB.busy || S.streaming) ? true : null, onclick: () => labRun(async () => {
          const mid = LAB.cmpModel;
          LAB.cmpRes[LAB.tid + '|' + mid] = Object.assign({ model: mid }, await labScore(mid, an.prompt, tr.tokens, LAB_K));
        }),
      }, I('play', 's'), T('점수'))));
    const res = LAB.cmpModel && LAB.cmpRes[LAB.tid + '|' + LAB.cmpModel];
    if (res) out.push(labDelta(an, tr.tokens, res, T`${labelById(res.model)}에서`));
  }
  return labCol(out);
}
function labPair(an, tr, anB, trB) {
  const sameModel = tr.model === trB.model;
  const same = sameModel && an.prompt.length === anB.prompt.length && an.prompt.every((x, i) => x === anB.prompt[i]);
  const sA = labSummary(an, tr.tokens), sB = labSummary(anB, trB.tokens);
  const out = [];
  if (same) {
    let d = 0;
    while (d < tr.tokens.length && d < trB.tokens.length && tr.tokens[d] === trB.tokens[d]) d++;
    if (d === tr.tokens.length && d === trB.tokens.length) out.push(h('div', { class: 'note-line' }, I('check', 's'), h('span', null, T`프롬프트가 같고 두 답의 토큰 ${fmtNum(d)}개가 모두 같습니다.`)));
    else {
      const side = (x, a, s) => h('div', null, h('div', { class: 'eyebrow' }, x), h('div', { class: 'lab-tokbig' }, d < a.lp.length ? tokShow(a.strs[d] === null ? a.raw[d] : a.strs[d]) : T('(끝)')),
        h('div', { class: 'small muted' }, d < a.lp.length ? T`확률 ${pct(a.lp[d])} · ${labStep(a, s, d).rank}위` : ''));
      out.push(h('div', { class: 'card lab-detail' },
        h('div', { class: 'row g8' }, h('span', { class: 'eyebrow' }, T`처음 갈린 걸음 ${d + 1}`), h('span', { class: 'grow' }),
          d < an.lp.length ? h('button', { class: 'btn sm ghost', onclick: () => { LAB.tab = 'obs'; LAB.sel = d; labRender(); } }, T('A의 이 걸음 보기')) : null),
        labNote(T('프롬프트와 앞 토큰이 모두 같으므로 이 걸음의 분포는 두 답에서 같은 비트입니다 — 고른 토큰만 다릅니다.')),
        h('div', { class: 'lab-vs' }, side('A', an, tr.tokens), side('B', anB, trB.tokens))));
    }
  } else if (sameModel) {
    const key = LAB.tid + '|' + LAB.cmpTid;
    out.push(h('div', { class: 'note-line' }, I('info', 's'), h('span', null, T('두 답의 프롬프트가 다릅니다(질문 · 앞 대화 · 시스템 프롬프트 · 생각 켬끔).'))),
      h('div', null, h('button', {
        class: 'btn needs-conn', disabled: (LAB.busy || S.streaming) ? true : null, onclick: () => labRun(async () => { LAB.cross[key] = await labScore(tr.model, anB.prompt, tr.tokens, LAB_K); }),
      }, I('play', 's'), T('A의 답을 B의 프롬프트로 점수'))));
    if (LAB.cross[key]) out.push(labDelta(an, tr.tokens, LAB.cross[key], T('B의 프롬프트에서')));
  } else out.push(h('div', { class: 'note-line' }, I('info', 's'), h('span', null, T('두 답의 모델이 다릅니다 — 아래 \'같은 글을 다른 모델로 점수\'로 같은 토큰을 견주세요.'))));
  const row = (l, x, y) => h('tr', null, h('td', null, l), h('td', { class: 'r num' }, x), h('td', { class: 'r num' }, y));
  out.push(h('div', { class: 'list' }, h('table', { class: 'data lab-table' },
    h('thead', null, h('tr', null, h('th', null, ''), h('th', { class: 'r' }, 'A'), h('th', { class: 'r' }, 'B'))),
    h('tbody', null, row(T('토큰'), fmtNum(sA.n), fmtNum(sB.n)), row(T('평균 로그확률'), nat(sA.mean), nat(sB.mean)), row(T('퍼플렉시티'), sA.ppl.toFixed(3), sB.ppl.toFixed(3)),
      row(T`아슬아슬 (여유 < ${LAB.delta})`, fmtNum(sA.frag), fmtNum(sB.frag)), row(T('1위가 아닌 토큰'), fmtNum(sA.notTop), fmtNum(sB.notTop))))));
  return labCol(out);
}
// 같은 토큰을 다른 조건(모델 · 프롬프트)에서 점수 매긴 값과 견준다 — Δ = 그쪽 − 이 답. 셈의 잡음이 없으므로 Δ 는 바꾼 것 하나의 효과다
function labDelta(an, toks, res, where) {
  const n = Math.min(an.lp.length, res.lp.length);
  let sum = 0, maxAbs = -1, maxAt = 0, top1 = 0, zero = 0;
  const box = h('div', { class: 'tokstrip delta', role: 'img', 'aria-label': T`토큰마다 Δ로그확률 — ${where}` });
  for (let i = 0; i < n; i++) {
    const d = res.lp[i] - an.lp[i], a = Math.abs(d);
    sum += d;
    if (a > maxAbs) { maxAbs = a; maxAt = i; }
    if ((res.topIds[i] || [])[0] !== (an.topIds[i] || [])[0]) top1++;
    if (d === 0) zero++;
    const b = a === 0 ? 0 : a < 0.01 ? 1 : a < 0.1 ? 2 : a < 0.5 ? 3 : a < 2 ? 4 : 5;
    box.append(h('span', { class: `tok d${b}${d < 0 ? ' neg' : d > 0 ? ' pos' : ''}${an.strs[i] === null ? ' cont' : ''}`, 'data-tip': T`걸음 ${i + 1} · Δ로그확률 ${sgn(d)}`, 'data-tip-pos': 'top' }, tokShow(an.strs[i])));
  }
  return h('div', { class: 'col g8' },
    h('div', { class: 'eyebrow' }, T`같은 토큰 ${fmtNum(n)}개 — ${where}`),
    h('div', { class: 'kpis sm lab-kpis' },
      labKpi(sgn(sum), T('합 Δ로그확률'), T('답 전체의 로그 확률비 — log P(그쪽) − log P(이 답)')),
      labKpi(sgn(n ? sum / n : 0), T('평균 Δ로그확률')),
      labKpi(nat(maxAbs), T`가장 큰 |Δ| (걸음 ${maxAt + 1})`),
      labKpi(fmtNum(top1), T('1위가 바뀐 걸음'), T('두 조건에서 가장 그럴듯한 토큰이 다른 걸음 수'))),
    labNote(T`초록 = 그쪽에서 더 높은 확률, 분홍 = 더 낮은 확률, 짙을수록 큰 차이. 차이 0(비트까지 같음) ${fmtNum(zero)}걸음.`),
    box);
}

// ---- 점수 ----
function labScoreView() {
  const tr = labCur(), an = tr && LAB.an[LAB.tid];
  const ta = h('textarea', { class: 'w100 lab-ta', rows: '5', placeholder: T('점수 매길 글을 붙여 넣으세요'), 'aria-label': T('점수 매길 글'), 'data-fk': 'lab:text' });
  ta.value = LAB.scoreText;
  ta.addEventListener('input', () => { LAB.scoreText = ta.value; });
  if (!an && LAB.scoreCtx === 'answer') LAB.scoreCtx = 'none';
  const ctx = h('select', { class: 'input w100', 'aria-label': T('맥락'), 'data-fk': 'lab:ctx', onchange: ev => { LAB.scoreCtx = ev.target.value; labRender(); } },
    h('option', { value: 'answer', selected: LAB.scoreCtx === 'answer' ? true : null, disabled: an ? null : true }, an ? T('고른 답의 자리에(같은 질문 · 대화 뒤 — 그 답의 모델)') : T('고른 답의 자리에(먼저 관찰에서 셈하세요)')),
    h('option', { value: 'none', selected: LAB.scoreCtx === 'none' ? true : null }, T('맥락 없이(글만 — 첫 토큰은 확률 없음)')));
  const runs = S.st ? sortedModels().filter(runnable) : [];
  if (!LAB.scoreModel || !runs.some(m => m.id === LAB.scoreModel)) LAB.scoreModel = (tr && runs.some(m => m.id === tr.model) && tr.model) || defaultModel();
  const useAns = !!(LAB.scoreCtx === 'answer' && an);
  const model = useAns ? tr.model : LAB.scoreModel;
  const msel = h('select', { class: 'input w100', 'aria-label': T('점수 매길 모델'), 'data-fk': 'lab:smodel', disabled: useAns ? true : null, onchange: ev => { LAB.scoreModel = ev.target.value; } },
    runs.map(m => h('option', { value: m.id, selected: m.id === model ? true : null }, `${displayName(m)} · ${quantOf(m)}`)));
  const out = [labNote(T('글의 토큰마다 로그확률과 퍼플렉시티를 셉니다 — 모델이 이 글을 얼마나 그럴듯하게 보는가. 같은 글 · 같은 모델이면 어느 PC에서든 같은 값입니다.')),
    ta, ctx, msel,
    h('div', null, h('button', {
      class: 'btn primary needs-conn', disabled: (LAB.busy || S.streaming || !model) ? true : null, onclick: () => labRun(async () => {
        const text = LAB.scoreText;
        if (!text.trim()) throw new Error(T('점수 매길 글을 넣으세요'));
        await labEnsureModel(model);
        const ids = (await labApi('/tokenize', { prompt: text })).tokens;
        const pre = useAns ? an.prompt : ids.slice(0, 1), rest = useAns ? ids : ids.slice(1);
        if (!rest.length) throw new Error(T('토큰이 둘 이상인 글을 넣으세요(맥락 없이는 첫 토큰의 확률이 없습니다)'));
        LAB.busy = T`토큰 ${fmtNum(rest.length)}개를 셈하는 중`;
        labRender();
        const sc = await labScore(model, pre, rest, LAB_K);
        LAB.scoreRes = { model, ids: rest, pre, sc, ctx: useAns, text };
        LAB.scoreSel = -1;
      }),
    }, I('play', 's'), T('점수')))];
  const r = LAB.scoreRes;
  if (r) {
    const sm = labSummary(r.sc, r.ids);
    let lo = 0;
    r.sc.lp.forEach((x, i) => { if (x < r.sc.lp[lo]) lo = i; });
    const holder = labDetailHolder(() => (LAB.scoreSel >= 0 ? labDetail(r.sc, r.ids, LAB.scoreSel, false) : null));
    out.push(h('div', { class: 'eyebrow mt8' }, `${labelById(r.model)}${NB}${r.ctx ? T('고른 답의 자리에') : T('맥락 없이')}${NB}${T`토큰 ${fmtNum(sm.n)}개`}`),
      h('div', { class: 'kpis sm lab-kpis' }, labKpi(nat(sm.sum), T('합 로그확률')), labKpi(nat(sm.mean), T('평균 로그확률')), labKpi(sm.ppl.toFixed(3), T('퍼플렉시티')),
        labKpi(pct(r.sc.lp[lo]), T`가장 낮은 확률 (걸음 ${lo + 1})`)),
      labStrip(r.sc, r.ids, LAB.scoreSel, i => { LAB.scoreSel = i; holder._render(); }, T('점수 매긴 글의 토큰')), holder);
  }
  return labCol(out);
}

// ---- 재현 ----
function labRepro() {
  const tr = labCur();
  const out = [labPicker()];
  if (tr) {
    const an = LAB.an[LAB.tid], sha = trSha(tr), fx = an ? labFpOf(LAB.tid) : null, f = labFind(LAB.tid);
    out.push(labAnswerHead(tr), labSec(T('지문')),
      labNote(T('같은 모델 파일 · 같은 프롬프트 · 같은 결과면 어느 GPU · 백엔드에서 셈해도 같은 지문입니다(계약). 장치는 지문에 넣지 않고 만든 곳으로 따로 적습니다.')),
      h('div', { class: 'kv' },
        h('div', { class: 'k' }, T('토큰 지문')), h('div', { class: 'v mono small' }, !an ? T('셈하면 보입니다') : fx ? fpShort(fx.fp) : '…'),
        h('div', { class: 'k' }, T('로짓 지문')), h('div', { class: 'v mono small' }, !an ? T('셈하면 보입니다') : fx ? fpShort(fx.fpLogit) : '…'),
        h('div', { class: 'k' }, T('모델 파일')), h('div', { class: 'v mono small break' }, sha ? 'SHA-256 ' + sha.slice(0, 16) + '…' : T('SHA-256 모름')),
        h('div', { class: 'k' }, T('만든 곳')), h('div', { class: 'v small' }, `${tr.device || '-'}${NB}${LAYER[tr.layer] || tr.layer || '-'}`)));
    if (!sha) {
      const m = modelOf(tr.model);
      out.push(h('div', { class: 'note-line' }, I('info', 's'), h('span', null, T('직접 추가한 모델 파일은 SHA-256을 아직 모릅니다 — 지문 · 캡슐에는 모델 파일의 해시가 필요합니다.'))),
        m ? h('div', null, h('button', { class: 'btn sm needs-conn', disabled: m.verifying ? true : null, onclick: () => verifyModel(m.id) }, I('hash', 's'), m.verifying ? T('SHA-256 셈하는 중') : T('SHA-256 셈하기'))) : null);
    }
    if (!an) out.push(h('div', null, labAnalyzeBtn(LAB.tid, T('셈하기'))));
    // 다시 돌려 확인
    const rp = LAB.replay[LAB.tid];
    out.push(labSec(T('다시 돌려 확인')), labNote(T('같은 요청(같은 프롬프트 토큰 · 매개변수 · 씨앗)을 캐시 없이 다시 생성하고 걸음마다 토큰과 로짓 해시를 견줍니다.')),
      h('div', null, h('button', {
        class: 'btn needs-conn', disabled: (LAB.busy || S.streaming || (!tr.promptIds && !(f && f.live))) ? true : null, onclick: () => labRun(() => labReplay(LAB.tid)),
      }, I('refresh', 's'), T('다시 돌려 확인'))));
    if (rp) out.push(h('div', { class: 'note-line' + (rp.ok ? '' : ' err') }, I(rp.ok ? 'check' : 'alert', 's'), h('span', null, rp.text)));
    // 캡슐 내보내기
    out.push(labSec(T('재현 캡슐')), labNote(T('요청 · 결과 · 로그확률 · 로짓 해시 · 만든 곳을 JSON 파일 하나로. 받은 사람이 자기 GPU에서(아래 \'캡슐 불러와 확인\') 또는 GPU 없이 파이썬(도구/캡슐확인.py)으로 비트까지 같은지 확인합니다.')),
      h('div', null, h('button', { class: 'btn needs-conn', disabled: (!an || !sha || LAB.busy) ? true : null, onclick: () => labCapsuleSave(LAB.tid) }, I('download', 's'), T('캡슐 내보내기'))));
  }
  const file = h('input', { type: 'file', accept: '.json,application/json', class: 'sr', 'aria-label': T('캡슐 파일'), tabindex: '-1' });
  file.addEventListener('change', () => { const fl = file.files && file.files[0]; if (fl) labRun(() => labCapsuleCheck(fl)); });
  out.push(h('div', null, h('button', { class: 'btn needs-conn', 'data-fk': 'lab:capin', disabled: (LAB.busy || S.streaming) ? true : null, onclick: () => file.click() }, I('file', 's'), T('캡슐 불러와 확인')), file));
  if (LAB.capsule) out.push(h('div', { class: 'note-line' + (LAB.capsule.ok ? '' : ' err') }, I(LAB.capsule.ok ? 'check' : 'alert', 's'), h('span', null, LAB.capsule.text)));
  // 출처 · 기준 PC · 하드웨어
  out.push(labSec(T('출처')));
  if (!LAB.prov) out.push(h('div', null, h('button', { class: 'btn sm needs-conn', disabled: LAB.busy ? true : null, onclick: () => labRun(async () => { LAB.busy = T('해시를 셈하는 중'); labRender(); LAB.prov = await api('/api/provenance'); }) }, I('hash', 's'), T('엔진 · 커널 해시 보기'))));
  else out.push(labProvView(LAB.prov));
  out.push(labRefView(tr));
  return labCol(out);
}
async function labReplay(tid) {
  const tr = TRACES[tid];
  const an = await labAnalyze(tid);
  const cp = labParts(tr, an.prompt);
  LAB.busy = T('같은 요청을 캐시 없이 다시 생성하는 중');
  labRender();
  const body = { prompt: cp.prompt, max_tokens: labMax(tr), temperature: tr.req.temp, cache_prompt: false, geulpu: { logit_hash: true } };
  if (cp.off) body.geulpu.step_offset = cp.off;
  const j = await labApi('/v1/completions', seedBody(body, tr.req.temp > 0 ? tr.seed : ''));
  const toks = j.geulpu.tokens || [], hs = j.geulpu.logit_hashes || [], want = cp.tokens, wh = an.hashes.slice(cp.off);
  let d = 0, hd = 0;
  while (d < toks.length && d < want.length && toks[d] === want[d]) d++;
  while (hd < hs.length && hd < wh.length && hs[hd] === wh[hd]) hd++;
  const ok = d === toks.length && d === want.length && hd === hs.length && hd === wh.length;
  LAB.replay[tid] = {
    ok, text: ok ? T`${fmtNum(d)}걸음 모두 같음 — 토큰 · 로짓 해시(슬롯 ${j.geulpu.slot} · 캐시 없이)`
      : T`걸음 ${Math.min(d, hd) + 1}에서 다름 — 같은 토큰 ${d}/${want.length} · 같은 로짓 해시 ${hd}/${wh.length}`,
  };
}
function labCapsuleText(tid) {
  const tr = TRACES[tid], an = LAB.an[tid], m = modelOf(tr.model), cp = labParts(tr, an.prompt), o = cp.off, fx = LAB.fpc[tid];
  const req = { prompt_tokens: cp.prompt, max_tokens: labMax(tr), temperature: tr.req.temp };
  if (tr.req.temp > 0) Object.assign(req, { top_k: 40, top_p: 0.95, min_p: 0.05, seed: '__SEED__' });
  if (o) req.step_offset = o;
  const cap = {
    format: 'geulpu-capsule/1', created: new Date().toISOString(), contract: '5.2',
    app: { version: S.st ? S.st.app.version : '', commit: S.st ? S.st.app.commit : '' },
    model: { name: tr.name, quant: tr.quant, file: m ? String(m.path || '').split(/[\\/]/).pop() : '', size: m ? m.size : null, sha256: trSha(tr) },
    made_on: { device: tr.device, layer: tr.layer },
    request: req,
    result: {
      tokens: cp.tokens, token_logprobs: an.lp.slice(o), entropy: an.ent.slice(o), top_ids: an.topIds.slice(o), top_logprobs: an.topLp.slice(o),
      logit_hashes: an.hashes.slice(o),
    },
    fingerprint: fx && fx.fp ? { token: fx.fp, logit: fx.fpLogit } : null,
  };
  const txt = '{\n' + Object.keys(cap).map(k => JSON.stringify(k) + ':' + JSON.stringify(cap[k])).join(',\n') + '\n}\n';
  return tr.req.temp > 0 ? txt.replace('"seed":"__SEED__"', '"seed":' + (tr.seed || '0')) : txt;
}
function labDownload(name, text, type) {
  const a = h('a', { href: URL.createObjectURL(new Blob([text], { type })), download: name });
  document.body.append(a);
  a.click();
  setTimeout(() => { URL.revokeObjectURL(a.href); a.remove(); }, 1000);
}
function labCapsuleSave(tid) {
  if (!LAB.an[tid]) return;
  const fx = LAB.fpc[tid];
  labDownload(`geulpu-capsule-${fx && fx.fp ? fx.fp.slice(0, 12) : tid}.json`, labCapsuleText(tid), 'application/json');
}
// 캡슐 확인 — 같은 모델 파일(SHA-256)로 이 PC 에서 점수 매기기와 다시 생성. 모든 수가 비트까지 같아야 한다
async function labCapsuleCheck(file) {
  LAB.capsule = null;
  const raw = await file.text();
  let cap;
  try { cap = JSON.parse(raw); } catch (_) { throw new Error(T('JSON 파일이 아닙니다')); }
  if (!cap || cap.format !== 'geulpu-capsule/1' || !cap.request || !cap.result || !cap.model) throw new Error(T('GeulPU 재현 캡슐이 아닙니다'));
  const rq = cap.request, rs = cap.result;
  // 씨앗은 64비트 정수라 JSON.parse 가 자리를 잃는다 — 원문에서 읽는다(파이썬 json.dump 처럼 ':' 뒤에 빈칸이 있어도). 못 읽으면 다른 씨앗으로
  // 다시 생성해 '다름'이 되므로 멈춘다
  const ri = raw.indexOf('"request"');
  const sm = ri >= 0 ? /"seed"\s*:\s*(\d+)\s*[,}]/.exec(raw.slice(ri)) : null;
  if (rq.temperature > 0 && !sm) throw new Error(T('캡슐의 씨앗(seed)을 정수로 읽지 못했습니다 — 정밀도를 잃은 사본일 수 있습니다'));
  const models = S.st ? S.st.models : [];
  const m = models.find(x => x.sha256 && x.sha256 === cap.model.sha256);
  if (!m) {
    const same = models.find(x => !x.sha256 && cap.model.size && x.size === cap.model.size);
    throw new Error(same ? T`크기가 같은 모델(${displayName(same)})이 있지만 SHA-256을 모릅니다 — 모델 화면에서 무결성을 확인한 뒤 다시 불러오세요`
      : T`같은 모델 파일(SHA-256 ${String(cap.model.sha256 || '').slice(0, 12)}…)이 설치되어 있지 않습니다 — ${cap.model.name || ''} ${cap.model.quant || ''}`);
  }
  LAB.busy = T('캡슐의 토큰을 이 PC에서 다시 셈하는 중');
  labRender();
  const k = (rs.top_ids[0] || []).length || 1;
  const sc = await labScore(m.id, rq.prompt_tokens, rs.tokens, k);
  const n = rs.tokens.length;
  const eqA = (x, y) => x.length === y.length && x.every((v, i) => v === y[i]);
  let d = 0;
  for (; d < n; d++) {
    if (sc.lp[d] !== rs.token_logprobs[d] || sc.hashes[d] !== rs.logit_hashes[d] || sc.ent[d] !== rs.entropy[d]
      || !eqA(sc.topIds[d] || [], rs.top_ids[d] || []) || !eqA(sc.topLp[d] || [], rs.top_logprobs[d] || [])) break;
  }
  const scoreOk = d === n && sc.lp.length === n;
  LAB.busy = T('캡슐의 요청으로 다시 생성하는 중');
  labRender();
  const body = { prompt: rq.prompt_tokens, max_tokens: rq.max_tokens, temperature: rq.temperature, cache_prompt: false, geulpu: { logit_hash: true } };
  if (rq.temperature > 0) Object.assign(body, { top_k: rq.top_k, top_p: rq.top_p, min_p: rq.min_p });
  if (rq.step_offset) body.geulpu.step_offset = rq.step_offset;
  const j = await labApi('/v1/completions', seedBody(body, rq.temperature > 0 && sm ? sm[1] : ''));
  const genOk = eqA(j.geulpu.tokens || [], rs.tokens) && eqA(j.geulpu.logit_hashes || [], rs.logit_hashes);
  let fpOk = true;
  if (cap.fingerprint) {
    const v = await labFp(cap.model.sha256, rq.prompt_tokens, rs.tokens, rs.logit_hashes);
    fpOk = v.fp === cap.fingerprint.token && v.fpLogit === cap.fingerprint.logit;
  }
  const md = cap.made_on || {};
  const here = `${(S.st && S.st.device.name) || '-'} · ${LAYER[S.st && S.st.server.layer] || '-'}`;
  const where = T` (캡슐에 적힌 만든 곳 ${md.device || '-'} · ${LAYER[md.layer] || md.layer || '-'} → 여기 ${here})`;
  const ok = scoreOk && genOk && fpOk;
  LAB.capsule = {
    ok, text: ok ? T`비트까지 같음 — ${fmtNum(n)}걸음의 로그확률 · 엔트로피 · 상위 ${k} · 로짓 해시, 다시 생성한 토큰, 지문` + where
      : [scoreOk ? null : T`점수 매기기가 걸음 ${d + 1}에서 다름`, genOk ? null : T('다시 생성한 토큰 · 로짓 해시가 다름'),
        !genOk && scoreOk && rq.temperature > 0 ? T('점수는 같으니 씨앗을 확인하세요 — 다른 도구로 다시 쓴 캡슐은 씨앗(64비트 정수)이 자리를 잃었을 수 있습니다') : null,
        fpOk ? null : T('지문이 다름')].filter(Boolean).join(NB) + where,
  };
}
function labProvView(p) {
  const rows = (p.binaries || []).map(b => h('tr', null, h('td', { class: 'small' }, String(b.file).replace(/^bin[\\/]/, '')),
    h('td', { class: 'mono small' }, b.sha256 ? b.sha256.slice(0, 12) + '…' : '-'),
    h('td', null, b.reference ? (b.same ? h('span', { class: 'tag ok' }, I('check'), T('기준과 같음')) : h('span', { class: 'tag err' }, I('alert'), T('기준과 다름'))) : h('span', { class: 'tag dim' }, T('기준 없음')))));
  return h('div', { class: 'col g8' },
    labNote(T('엔진 바이너리의 SHA-256을 재현성 검증의 기준 빌드와 견줍니다. 직접 빌드한 판은 다를 수 있습니다.')),
    h('div', { class: 'list' }, h('table', { class: 'data lab-table' }, h('tbody', null, rows))),
    p.module ? h('div', { class: 'kv' }, h('div', { class: 'k' }, T('커널 모듈')), h('div', { class: 'v mono small break' }, `${p.module.file}${NB}${LAYER[p.module.layer] || p.module.layer}${NB}${p.module.sha256 ? p.module.sha256.slice(0, 16) + '…' : '-'}`)) : null,
    h('div', { class: 'small muted' }, `GeulPU v${p.app.version}${p.app.commit ? NB + T('커밋') + ' ' + p.app.commit.slice(0, 7) : ''}${NB}${T('계약')} ${p.contract}`));
}
// 기준 PC 대조 · 하드웨어 한계 — 지난 결과에서(벤치마크 화면)
function labRefView(tr) {
  if (!S.results && Date.now() - S.resultsAt > 30000) loadResults(false).then(() => { if (LAB.open && LAB.tab === 'repro') labRender(); });
  const rs = S.results || [];
  const cc = rs.find(x => x.kind === '교차검증');
  const dg = tr && rs.find(x => x.kind === '진단' && x.model === tr.model);
  const out = [labSec(T('기준 PC · 하드웨어'))];
  if (cc) {
    const v = resultVerdict(cc);
    out.push(h('div', { class: 'note-line' + (v && v[0] === 'err' ? ' err' : '') }, I(v && v[0] === 'err' ? 'alert' : 'shield', 's'),
      h('span', null, T`재현성 검증 ${stampText(cc.stamp)}: ${v ? v[1] : '-'} — ${resultSummary(cc)}`)));
  } else out.push(h('div', { class: 'note-line' }, I('shield', 's'), h('span', null, T('재현성 검증을 아직 하지 않았습니다 — 이 PC의 출력이 기준 PC와 비트까지 같은지 벤치마크에서 확인합니다.'))));
  if (dg) out.push(h('div', { class: 'note-line' }, I('pulse', 's'), h('span', null, T`하드웨어 진단 ${stampText(dg.stamp)}: ${resultSummary(dg)}`)));
  else if (tr) out.push(h('div', { class: 'note-line' }, I('pulse', 's'), h('span', null, T('이 모델의 하드웨어 진단이 없습니다 — 생성이 하드웨어 한계에 얼마나 가까운지 벤치마크에서 잽니다.'))));
  out.push(h('div', null, h('button', { class: 'btn sm ghost', onclick: () => show('bench') }, I('gauge', 's'), T('벤치마크로'))));
  return h('div', { class: 'col g8' }, out);
}

// ---- 내보내기 — 보이는 것 모두 JSON · CSV, 같은 값을 받는 파이썬 코드 ----
function labRows(an, toks) {
  return an.lp.map((lp, i) => {
    const st = labStep(an, toks, i);
    return {
      step: i + 1, id: toks[i], text: an.raw[i], logprob: lp, prob: Math.exp(lp), rank: st.rank > (an.topIds[i] || []).length ? null : st.rank, entropy: an.ent[i],
      margin: isFinite(st.margin) ? st.margin : null, logit_hash: an.hashes[i] || null, top: (an.topIds[i] || []).map((id, q) => ({ id, logprob: an.topLp[i][q] })),
    };
  });
}
const CSV_HEAD = 'step,id,text,logprob,prob,rank,entropy,margin,logit_hash';
const csvLine = r => [r.step, r.id, JSON.stringify(r.text), r.logprob, r.prob, r.rank === null ? '' : r.rank, r.entropy, r.margin === null ? '' : r.margin, r.logit_hash || ''].join(',');
const csvRows = (rows, extra) => [CSV_HEAD + (extra ? ',' + extra.join(',') : '')].concat(rows.map(r => csvLine(r) + (extra ? ',' + extra.map(k => r[k]).join(',') : ''))).join('\n') + '\n';
function labOut() {
  const tr = labCur(), an = tr && LAB.an[LAB.tid];
  const out = [labPicker()];
  const stamp = () => new Date().toISOString().slice(0, 19).replace(/[-:T]/g, '');
  const base = tr ? `geulpu-${String(tr.name || 'model').replace(/[^\w.-]+/g, '_')}-${tr.quant || ''}` : 'geulpu';
  const item = (title, desc, btns) => h('div', { class: 'lab-out' }, h('div', { class: 'ink' }, title), desc ? h('div', { class: 'small muted' }, desc) : null, h('div', { class: 'row g8' }, btns));
  const dl = (label, fn) => h('button', { class: 'btn sm', onclick: () => { const [n, t, ty] = fn(); labDownload(n, t, ty); } }, I('download', 's'), label);
  const head = (what, extra) => Object.assign({ format: 'geulpu-tokens/1', contract: '5.2', what, logprob: 'natural log, model distribution (temperature 1), contract 5.2' }, extra);
  const json = o => JSON.stringify(o) + '\n';
  if (tr && an) {
    const fx = labFpOf(LAB.tid);
    const meta = {
      model: { name: tr.name, quant: tr.quant, sha256: trSha(tr) }, params: { temperature: tr.req.temp, max_tokens: labMax(tr), seed: tr.req.temp > 0 ? tr.seed : null, branch: tr.branch || null },
      prompt_tokens: an.prompt, fingerprint: fx && fx.fp ? { token: fx.fp, logit: fx.fpLogit } : null, made_on: { device: tr.device, layer: tr.layer },
    };
    out.push(item(T('이 답의 토큰 표'), T`걸음마다 토큰 · 로그확률 · 확률 · 순위 · 엔트로피 · 여유 · 로짓 해시 · 상위 ${LAB_K}`, [
      dl('JSON', () => [base + '-answer-' + stamp() + '.json', json(head('answer', Object.assign({}, meta, { steps: labRows(an, tr.tokens) }))), 'application/json']),
      dl('CSV', () => [base + '-answer-' + stamp() + '.csv', csvRows(labRows(an, tr.tokens)), 'text/csv'])]));
    out.push(item(T('재현 캡슐'), T('요청 · 결과 · 만든 곳 — 재현에서 확인하는 파일'), [h('button', { class: 'btn sm', disabled: trSha(tr) ? null : true, onclick: () => labCapsuleSave(LAB.tid) }, I('download', 's'), 'JSON')]));
    const cm = LAB.cmpModel && LAB.cmpRes[LAB.tid + '|' + LAB.cmpModel];
    if (cm) {
      const rows = () => labRows(an, tr.tokens).map((r, i) => Object.assign(r, { other_logprob: cm.lp[i], delta: cm.lp[i] - r.logprob }));
      out.push(item(T`다른 모델로 점수 — ${labelById(cm.model)}`, T('같은 토큰의 두 모델 로그확률과 차(Δ = 그 모델 − 이 답)'), [
        dl('JSON', () => [base + '-vs-model-' + stamp() + '.json', json(head('other-model', Object.assign({}, meta, { other_model: labelById(cm.model), other_sha256: (modelOf(cm.model) || {}).sha256 || '', steps: rows() }))), 'application/json']),
        dl('CSV', () => [base + '-vs-model-' + stamp() + '.csv', csvRows(rows(), ['other_logprob', 'delta']), 'text/csv'])]));
    }
    const cr = LAB.cmpTid && LAB.cross[LAB.tid + '|' + LAB.cmpTid];
    if (cr) {
      const anB = LAB.an[LAB.cmpTid];
      const rows = () => labRows(an, tr.tokens).map((r, i) => Object.assign(r, { other_logprob: cr.lp[i], delta: cr.lp[i] - r.logprob }));
      out.push(item(T('A의 답을 B의 프롬프트로 점수'), T('같은 답 토큰의 두 프롬프트 로그확률과 차(Δ = B의 프롬프트 − A)'), [
        dl('JSON', () => [base + '-vs-prompt-' + stamp() + '.json', json(head('other-prompt', Object.assign({}, meta, { other_prompt_tokens: anB ? anB.prompt : null, steps: rows() }))), 'application/json']),
        dl('CSV', () => [base + '-vs-prompt-' + stamp() + '.csv', csvRows(rows(), ['other_logprob', 'delta']), 'text/csv'])]));
    }
  } else if (tr) out.push(labNote(T('이 답을 셈하면 토큰 표 · 캡슐을 내보낼 수 있습니다.')), h('div', null, labAnalyzeBtn(LAB.tid)));
  const r = LAB.scoreRes;
  if (r) out.push(item(T('점수 매긴 글'), `${labelById(r.model)}${NB}${r.ctx ? T('고른 답의 자리에') : T('맥락 없이')}`, [
    dl('JSON', () => ['geulpu-score-' + stamp() + '.json', json(head('score', { model: { name: labelById(r.model), sha256: (modelOf(r.model) || {}).sha256 || '' }, context_tokens: r.pre, text: r.text, steps: labRows(r.sc, r.ids) })), 'application/json']),
    dl('CSV', () => ['geulpu-score-' + stamp() + '.csv', csvRows(labRows(r.sc, r.ids)), 'text/csv'])]));
  // 파이썬
  const port = (S.st && S.st.app.port) || 44544;
  const pre = `from openai import OpenAI   # pip install openai — ${T('앱에서 모델을 불러온 상태에서')}\nc = OpenAI(base_url="http://127.0.0.1:${port}/v1", api_key="geulpu")\n`;
  const pyScore = an ? pre
    + `prompt = ${JSON.stringify(an.prompt)}\nanswer = ${JSON.stringify(tr.tokens)}\n`
    + `r = c.completions.create(model="geulpu", prompt=prompt + answer, echo=True, max_tokens=0, logprobs=${LAB_K}, temperature=0,\n`
    + `                         extra_body={"geulpu": {"echo_from": len(prompt), "logit_hash": True}})\n`
    + `lp = r.choices[0].logprobs.token_logprobs[len(prompt):]   # ${T('이 패널의 값과 비트까지 같다')}\n` : null;
  const pyGen = pre
    + `r = c.chat.completions.create(model="geulpu", messages=[{"role": "user", "content": "${T('하늘은 왜 파란가?')}"}], max_tokens=128,\n`
    + `                              temperature=0, logprobs=True, top_logprobs=5, extra_body={"geulpu": {"logit_hash": True}})\n`
    + `for t in r.choices[0].logprobs.content: print(t.token, t.logprob)\n`;
  const code = (title, py) => h('div', { class: 'lab-out' }, h('div', { class: 'row g8 nowrap' }, h('span', { class: 'ink grow' }, title),
    h('button', { class: 'btn sm ghost', onclick: () => copyText(py).then(ok => { if (ok) toast(T('복사했습니다'), 'ok'); }) }, I('copy', 's'), T('복사'))),
    h('pre', { class: 'log lab-code' }, py));
  out.push(labSec(T('파이썬')), labNote(T('패널의 값은 모두 OpenAI 꼴 API로도 받습니다 — 어느 PC · GPU에서든 같은 비트입니다.')));
  if (pyScore) out.push(code(T('이 답을 점수 매기기(echo)'), pyScore));
  out.push(code(T('생성하며 로그확률 받기'), pyGen));
  return labCol(out);
}
if (load('geulpu.lab.open', false) && !labOverlay()) setTimeout(() => labToggle(true), 0);

// ---------- 시작 ----------
(async function start() {
  S.rooms = load('geulpu.rooms', []).map(r => Object.assign({ updated: r.created }, r));
  for (const r of S.rooms) for (const m of r.messages) if (m.partial) { delete m.partial; m.stopped = true; m.stats = m.stats || '중지됨'; }   // 지난번에 흐르다 끊긴 답
  S.roomId = load('geulpu.room', null);
  S.rooms = S.rooms.filter(r => r.messages.length || r.id === S.roomId);
  setConvos(overlayConvos() ? false : load('geulpu.convos', true), false);
  updateSend();
  await refresh();
  if (!curRoom()) { S.roomId = null; newRoom(); }
  renderRooms(); renderMsgs();
  renderChatTop(true);
  let v = null;
  try { v = sessionStorage.getItem('geulpu.view'); } catch (_) { }
  show(v || 'chat');
  checkUpdate();
})();
