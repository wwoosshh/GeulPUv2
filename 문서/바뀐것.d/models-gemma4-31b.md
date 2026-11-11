갈래: 더함

- **Gemma 4 31B 가 앱의 모델 목록에 들었다**(0.24.0): Q4_K_M 파일(18.3 GB)은 개발 PC(12 GB)에 들지 않아 원격 서버의 H100(Linux)에서
  확인했다 — 다섯 실행 경로가 같은 비트이고, 전체 파일의 모든 행 로짓이 CPU 참조와 같고, GeulPU 서버로 채팅과 도구 호출을 확인했다. 채팅 답은 질문 셋 가운데 둘이 llama.cpp 와
  토큰까지 같고, 하나는 13번째 토큰에서 갈린다 — 그 자리에서 fp64 수학 참조의 1위는 엔진이 고른 토큰이다. 채팅 템플릿 · 토크나이저는
  12B 와 같다. 앱 화면에서는 돌려 보지 않았고, 목록의 설명에 그렇게 적었다. 그래픽 메모리를 문맥 4,096 에서 약 20 GB 써서 24 GB 이상의 GPU 가 필요하다.
  모델 찾기도 31B 저장소를 '확인 필요' 대신 '실행 가능'으로 표시한다. 기록: [구조/gemma4/기록.md](../../구조/gemma4/기록.md).

---

- **Gemma 4 31B is in the app's model list.** The Q4_K_M file (18.3 GB) does not fit on the development PC (12 GB), so it was
  checked on the remote H100 server (Linux): five bit-identical execution paths, all-row logits of the full file equal to the
  CPU reference, and chat and tool calls with the GeulPU server.
  Two of three chat answers equal llama.cpp's token for token; the third parts from it at the 13th token, where the top token
  of an fp64 math reference is the one the engine chose.
  Its chat template and tokenizer are those of the 12B. It was not run in the app's screens, and the list entry says so. It uses
  about 20 GB of graphics memory at a context of 4,096, so it needs a GPU with 24 GB or more. The model search now shows 31B
  repositories as runnable instead of "needs a check". Record: [구조/gemma4/기록.md](../../구조/gemma4/기록.md).
