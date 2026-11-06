# 파이썬 API — 파이썬에서 GeulPU 를 부리기

*영어판: [docs/python-api.md](../docs/python-api.md).*

`import geulpu` 는 GeulPU 앱을 부르는 작은 클라이언트를 준다. 앱을 따로 띄우거나 떠 있는 앱에 붙어, GGUF 모델을 불러오고, 스텝마다
로짓 해시와 함께 생성하고, 토큰에 점수를 매기고, 한 토큰에서 분기하고, 요청을 다시 보내고, 두 실행을 비트까지 비교한다. 파이썬
표준 라이브러리만 쓰고(3.9 이상), 앱의 HTTP API — 연구 패널과 디버거가 쓰는 그것([연구자 안내](../docs/research-guide.md#5-from-scripts--the-api) ·
[디버거 안내](../docs/debugger-guide.md#8-from-scripts--debuggerpy--api--run-log)) — 로 묻기 때문에 모든 값은 엔진의 것이고, 이 모듈은
스스로 셈하지 않는다. 소스는 [`도구/pypi/geulpu/api.py`](../도구/pypi/geulpu/api.py).

```python
import geulpu

with geulpu.start() as g:                                   # 창 없는 GeulPU 를 따로 띄움
    g.load("~/models/Qwen3-8B-Q4_K_M.gguf", precision="exact")
    a = g.chat([{"role": "user", "content": "Name one planet."}], max_tokens=32, seed=1)
    b = g.rerun(a)                                          # 같은 요청 · 같은 시드
    print(a.text, geulpu.compare(a, b))                     # Comparison(same=True, …)
    s = g.score(a)                                          # 답의 토큰마다 로그확률을 다시 셈
    f = g.branch(a, step=3)                                 # 스텝 3 에서 2위 토큰으로 바꿔 이어감
    print(g.info())                                         # 판 · GPU · 드라이버 · 모델 SHA-256 · 계산 길
```

## 받기

- 0.22.0 부터 PyPI 의 `geulpu` 꾸러미에 들어 있다: `pip install geulpu` 뒤 `import geulpu`.
- GeulPU 0.21.0(`pip install geulpu==0.21.0`)에서는 [`api.py`](../도구/pypi/geulpu/api.py) 를 스크립트 옆에 `geulpu_api.py` 로
  저장하고 `import geulpu_api as geulpu`. 설치된 꾸러미의 앱을 같은 방법으로 찾는다.

## 띄우기 · 붙기

| 부름 | 하는 일 |
|---|---|
| `geulpu.start(data=None, port=None, *, exe=None, timeout=180, wait_device=True)` | 창 없이 GeulPU 앱을 띄우고, 답하고 GPU 확인을 마칠 때까지 기다린다. `data` 는 그 자료 폴더(기본 `~/GeulPU-api` — 평소 쓰는 앱의 폴더 `~/GeulPU` 와 따로 둔다. 두 앱이 한 폴더를 함께 쓰면 안 된다), `port` 는 포트(기본: 빈 포트), `exe` 는 실행 파일(기본: 꾸러미 안의 것). 콘솔 출력은 자료 폴더의 `geulpu-console.log` 로 간다. |
| `geulpu.connect(url=None)` | 떠 있는 앱의 클라이언트 — `url` 의 기본은 `$GEULPU_URL`, 없으면 `http://127.0.0.1:44544`. 앱은 `geulpu --no-open` 으로 띄운다. |
| `g.close()` | 이 클라이언트가 바꾼 앱 설정을 되돌린다. `start()` 가 띄운 앱이면 서버와 앱도 끈다(서버 프로세스는 앱의 일 객체 · SIGTERM 으로 함께 끝난다). `with` 블록의 끝과 파이썬 프로세스의 끝이 이것을 부른다. |

## 모델 불러오기

`g.load(model, *, precision=None, kv8=None, fast_attention=None, speculative=None, context=None, slots=None, layer=None)`

- `model` 은 `.gguf` 경로(필요하면 앱의 목록에 더한다), `g.models()` 의 모델 아이디, 또는 파일 SHA-256 의 앞 8 자 이상. 앱이 파일의
  SHA-256 을 가질 때까지(앱이 스스로 세지 않는 파일이면 세라고 청한다), 서버가 준비될 때까지 기다리고 서버 상태를 돌려준다.
- 키워드 인자는 먼저 앱의 설정을 바꾼다: `precision` `"exact"` · `"fast"`, `kv8`(KV 캐시 8비트), `fast_attention`, `speculative`(비트가
  같은 추측 디코딩), `context`, `slots`(동시 요청 1–8), `layer`(`"cuda"` · `"vulkan"` · `"auto"`). 주지 않은 것은 앱의 설정 그대로다.
  같은 모델을 같은 설정으로 다시 부르면 아무것도 하지 않는다.
- 몇 설정은 앱이 모델과 장치마다 정한다. 듣지 않은 설정이 있으면 `load()` 가 까닭과 함께 경고한다 — 8비트 KV 캐시는 빠른 계산 ·
  CUDA · 그것을 받는 모델(모델 칸의 `shape.kv8_ok`)이어야 하고, 빠른 어텐션은 그 위에서만 돈다. 이를테면 Qwen3-0.6B 는 받지 않는다 —
  숨은 폭(1,024)이 128 × 헤드 16 이 아니다.
- `g.add_model(path)` 는 불러오지 않고 더하기만, `g.models()` 는 앱의 모델 목록, `g.unload()` 는 서버를 끈다.

## 생성

`g.chat(messages, **params)` 는 `/v1/chat/completions`, `g.complete(prompt, **params)` 는 `/v1/completions`(`prompt` 는 채팅 틀 없는 글
또는 토큰 번호의 목록)로 보낸다. `params` 는 요청의 칸 — `max_tokens`, `temperature`(API 의 기본은 0.8), `seed`, `top_p`, `top_k`,
`min_p`, `stop`, `tools`, `tool_choice`, `logprobs`, `top_logprobs`(0–20) 들. 흘림(stream)은 끈다. 요청은 늘 프롬프트 토큰과 스텝마다
로짓 해시를 묻고, `geulpu={…}` 로 다른 확장 칸을 더한다([사용 안내](../docs/user-guide.md#use-it-from-other-programs-openai-compatible-api)).
`model` 은 빼 둔다 — 불러온 모델이 답한다.

둘 다 `Generation` 을 돌려준다:

| 칸 | |
|---|---|
| `text` · `reasoning` · `tool_calls` · `finish_reason` | API 가 돌려준 답. |
| `tokens` · `logit_hashes` | 생성한 토큰 번호와 스텝마다 로짓 해시(16 자리 16진 글 — 로짓 행 전체의 64비트 해시). 둘 다 끝 토큰을 뽑은 스텝을 넣는다. |
| `prompt_tokens` · `seed` | 채팅 틀을 씌운 프롬프트, 뽑기가 쓴 시드. |
| `logprobs` · `token_texts` · `entropy` · `top_ids` · `top_logprobs` | 요청에 `logprobs` · `top_logprobs` 가 있으면 답의 토큰마다 — OpenAI 처럼 끝 토큰으로 끝나면 `tokens` 보다 하나 짧다. 없으면 `None`. |
| `request` · `response` · `seconds` · `model` · `usage` · `path` | 보낸 본문, 앱의 JSON 그대로, 그 밖. |

`g.rerun(gen, **changes)` 는 `gen` 의 요청을 그 시드로 다시 보낸다 — 재생이고, 바꾼 것이 없으면 같다. 키워드 인자는 먼저 요청의 칸을
바꾸므로 `g.rerun(a, messages=다른것)` 이 요청 분기다.

## 점수

`g.score(gen)` 은 생성의 토큰을 그 프롬프트 뒤에서 생성 없이 다시 센다(echo). 같은 모델 · 같은 계산 방식이면 로짓 해시가 생성의 것과
같다. `g.score(prompt, answer, top=2)` 는 아무 토큰이나 센다: `prompt` 는 토큰 번호, 글(채팅 틀 없이) 또는 채팅 메시지(모델의 틀로, 답
앞까지), `answer` 는 토큰 번호, `Generation`, 또는 — 글 프롬프트일 때 — 글(프롬프트 + 답의 토큰에서 프롬프트 몫을 뗀 것. 토큰 하나가
이음매에 걸치면 번호로 준다). `Score` 에는 `tokens` · `logprobs` · `entropy` · `top_ids` · `top_logprobs` · `token_texts` ·
`logit_hashes` 와 `total`(로그확률의 합)이 있다. 값은 모델의 분포다 — 온도 1, 거르기 없음([계약 5.2](../계약/계약.md)).

## 분기

`g.branch(gen, step, token=None, top=2)` 는 `gen` 을 `step` 에서 다른 토큰으로 잇는다 — 디버거의 토큰 분기. `token=None` 은 원래와 다른
가장 높은 토큰(요청이 묻었으면 `gen.top_ids` 에서, 아니면 그 스텝의 점수에서), `token="same"` 은 원래 토큰 — 원래 답의 나머지가 그대로
나와야 하는 자기 확인 — 이고, 토큰 번호를 줘도 된다. 분기는 프롬프트 토큰, `step` 앞의 답, 시드, 샘플링 칸, `max_tokens` 의 남은 몫을
그대로 쓰고 원래의 난수로 잇는다(`step_offset`). `Branch` 는 분기한 답 전체의 `Generation` 에 `step` · `token` · `original_token` ·
`new_tokens` · `prefix_same`(스텝 0 … `step` 의 로짓 해시를 원래와 같게 다시 셌는가)을 더한 것이다. `/v1/completions` 로 돌아 도구 호출은
글 그대로이고, 끝 토큰 · `max_tokens` · `stop` 에서만 끝난다. `text` 는 특수 토큰을 글로 푼 답이다. `g.rerun(branch)` 는 같은 분기를
다시 돌린다.

## 비교 · 기록

- `geulpu.compare(a, b)` 는 두 결과(`Generation` · `Branch` · `Score`)를 견준다: 프롬프트 토큰, 그다음 스텝마다 로짓 해시와 토큰.
  `Comparison` 은 모두 같으면 참이고, 아니면 `why` · `first_diff_step` 이 갈라진 자리를 말한다 — `"logits differ from step 3"`, 분기라면
  `"tokens differ from step 24 (same logits there)"`.
- `g.info(provenance=False)` 는 논문의 방법 절에 드는 것을 돌려준다: GeulPU 판과 커밋, GPU 와 드라이버, 불러온 모델 파일과 SHA-256,
  서버의 계산 길(계산 방식 · 8비트 KV 캐시 · 빠른 어텐션 · 추측 · 문맥 · YaRN). `provenance=True` 면 서버 실행 파일과 불러온 커널
  모듈의 SHA-256 을 더한다.
- `g.tokenize(글_또는_메시지)` · `g.detokenize(번호)` 는 모델의 토크나이저를 쓴다. `g.get(path)` · `g.post(path, body)` 로 다른 길에도
  닿는다. 오류는 `geulpu.GeulPUError`(앱이 HTTP 오류로 답했으면 `status` · `body`).
- 여러 스레드에서 불러도 된다. 서버는 슬롯 수만큼 함께 돌리고 비트는 같다.

## 확인한 것

RTX 4070 Ti · Windows 11 · PyPI 의 GeulPU 0.21.0 앱 · Qwen3-0.6B Q4_K_M(`시험/파이썬API.py --모델`, GPU 줄로):

- 따로 띄운 앱이 모델을 불러오고(10.9 초 — SHA-256 은 앱이 셈) 답했다.
- 온도 0.8 의 48 토큰 채팅이 `rerun` 으로 같게 돌아왔고, 넷을 한꺼번에 보낸 `rerun` 도 같았다.
- `score(a)` 의 로짓 해시가 생성의 것과 같았다.
- `branch(a, 24, "same")` 은 원래 답을 돌려주었고, `branch(a, 24)`(토큰 17330 → 1156)는 스텝 0–24 를 원래 해시로 다시 세고 스텝 24 에서
  갈라졌다.
- `precision="fast", kv8=True, fast_attention=True` 로 불러오자 이 모델이 8비트 KV 캐시를 받지 않는다고 경고했고, 빠른 계산의 답은
  다시 보내도 같았다(로짓은 정밀 계산과 스텝 0 부터 다르다).
- `close()` 가 앱과 앱이 띄운 프로세스 둘을 끝냈다.

빠른 점검은 이 모듈을 가짜 앱으로 돌리고(`시험/파이썬API.py`, 2f), 빌드한 앱을 GPU 없이 띄우고 끈다(4t).
