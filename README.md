# GeulPU — 내 PC의 그래픽카드로 돌리는 AI 채팅

*English: [README.en.md](README.en.md)*

> 이 저장소(GeulPUv2)는 연구 저장소 [GeulPU](https://github.com/wwoosshh/GeulPU)(2026-10-16 연구 동결, 앱 0.5.0)의 엔진과 앱을 그대로 이어
> 받아 **실사용**(성능 · 안정성 · 기능)을 목적으로 이어 가는 곳입니다. 기능은 모두 그대로이고(연구 패널 · 교차검증 포함), 연구 기록(단계
> 문서 · 논문)은 GeulPU 에 남아 있습니다. 옮긴 범위는 [문서/이관범위.md](문서/이관범위.md), 판마다 바뀐 것은 [문서/바뀐것.md](문서/바뀐것.md).

GeulPU는 Qwen3 · Qwen2.5 · DeepSeek-R1 · Llama 3 · Mistral 같은 언어 모델을 **내 PC의 그래픽카드(GPU)에서** 돌리는 Windows 앱입니다. 설치 프로그램 없이
zip을 풀고 `GeulPU.exe`를 실행하면 브라우저에 채팅 화면이 열립니다. 모델은 앱 안에서 Hugging Face로부터 받고, 대화는 이 PC 밖으로
나가지 않습니다.

엔진은 한국어 프로그래밍 언어 **글(geul)** 로 썼습니다 — GPU 커널부터 추론 서버 · 앱까지. CUDA 툴킷 · Python · PyTorch 없이 그래픽
드라이버만으로 돌고, 계산 결과가 CPU 참조 구현과 **비트까지 같도록** 정밀도를 계약으로 정해 둡니다([무엇이 다른가](#무엇이-다른가)).
그래서 같은 글을 다시 셈하면 어느 PC · GPU에서든 같은 확률이 나오고, **연구 패널**에서 토큰마다의 확률을 들여다보고 · 한 토큰을 바꿔
갈래를 내고 · 다른 모델이나 프롬프트와 견주고 · 남의 결과를 비트까지 확인합니다([연구 패널](#5-연구-패널--토큰마다-확률--개입--비교--재현)).

> 적힌 대로만 잃는다. 덜 잃는 데 드는 셈은 숨기지 않는다.

**[⬇ 최신 버전 받기 — Windows 64비트 · zip 약 1.4 MB](https://github.com/wwoosshh/GeulPUv2/releases/latest)** · 파이썬이 있다면 `pipx install geulpu` 한 줄로([설치](#파이썬이-있다면--명령-한-줄로))

[설치](#설치) · [처음 쓰기](#처음-쓰기) · [지원하는 모델](#지원하는-모델) · [실행 방식](#실행-방식) · [문제 해결](#문제-해결) ·
[업데이트 · 삭제](#업데이트--삭제) · [무엇이 다른가](#무엇이-다른가) · [개발 · 연구](#개발--연구)

## 설치

### 필요한 것

| | |
|---|---|
| 운영체제 | Windows 10 · 11 (64비트) |
| 그래픽카드 | NVIDIA RTX 30 시리즈 이상은 CUDA로, 그 밖의 GPU(AMD Radeon 등)는 Vulkan 1.3으로 돕니다 |
| 그래픽 메모리 | 6 GB부터(작은 모델) — 7B · 8B는 8 GB(Q4_K_M) · 12 GB(Q8_0), 14B는 12 GB 이상([아래 표](#내-pc에-맞는-모델)) |
| 디스크 | 앱 약 5 MB + 모델 하나에 2 ~ 9 GB |
| 인터넷 | 모델을 찾고 받을 때와 새 버전을 확인할 때만 |

그래픽 드라이버는 최신으로 두세요. RTX 20 · GTX 16 이하의 NVIDIA 카드는 CUDA 경로를 쓰지 못합니다(계산 능력 8.0 미만). 이때 앱은 Vulkan으로
돌 수 있는지 스스로 확인합니다. AMD는 Radeon RX 9060 XT에서 확인했습니다.

### 파이썬이 있다면 — 명령 한 줄로

zip을 받아 풀지 않고, 파이썬의 설치 도구로 받아 곧바로 실행할 수 있습니다([PyPI의 geulpu](https://pypi.org/project/geulpu/) — 릴리스
zip과 같은 앱).

```powershell
pipx install geulpu        # 또는 uv tool install geulpu · pip install geulpu
geulpu                     # 브라우저에 http://127.0.0.1:44544 가 열립니다
```

- 업데이트는 앱을 끈 뒤 `pipx upgrade geulpu`(uv는 `uv tool upgrade geulpu`, pip는 `pip install -U geulpu`)입니다. 새 판이 나오면 앱이
  이 명령을 알려 줍니다.
- 모델 · 설정 · 결과는 zip 판과 같은 `%USERPROFILE%\GeulPU`에 있어, 꾸러미를 지워도(`pipx uninstall geulpu`) 남습니다.
- `geulpu` 명령을 찾지 못하면 `python -m geulpu`로 실행합니다(pip의 사용자 설치는 명령 폴더가 PATH에 없을 수 있습니다). pipx · uv를 권합니다.

아래 1 ~ 3은 zip으로 설치하는 길입니다.

### 1. 받기

[릴리스 페이지](https://github.com/wwoosshh/GeulPUv2/releases/latest)에서 **`GeulPU-<버전>-windows-x64.zip`** 을 받습니다. 모델은 들어 있지 않아
작습니다(모델은 앱 안에서 받습니다).

받은 파일이 그대로인지 보려면(선택) 같은 페이지의 `SHA256SUMS.txt` 와 견줍니다. PowerShell에서:

```powershell
Get-FileHash .\GeulPU-0.5.1-windows-x64.zip
```

### 2. 압축 풀기

zip을 마우스 오른쪽 단추로 누르고 **모두 압축 풀기** → 원하는 폴더(예: `C:\GeulPU`)에 풉니다. 경로에 한글이 있어도 됩니다.
zip을 연 채로 그 안의 `GeulPU.exe` 를 바로 실행하지 말고, 꼭 풀어서 실행하세요.

### 3. 실행

`GeulPU.exe` 를 두 번 누릅니다.

1. **"Windows의 PC 보호"** 창이 뜨면 **추가 정보 → 실행**을 누릅니다. 코드 서명을 하지 않은 실행 파일이라 나타납니다. 릴리스 zip은
   GitHub Actions가 이 저장소의 소스로 빌드합니다([실행 방식](#실행-방식)).
2. 검은 창(콘솔)이 하나 뜹니다. 이것이 GeulPU 본체입니다 — **이 창을 닫으면 앱이 꺼집니다.**
3. 브라우저에 `http://127.0.0.1:44544` 가 열립니다. 열리지 않으면 이 주소를 주소창에 직접 입력하세요.

## 처음 쓰기

화면 왼쪽 막대의 아이콘은 위에서부터 **채팅 · 모델 · 벤치마크 · 하드웨어**, 맨 아래가 **설정**입니다. 모델이 하나도 없으면 처음에 이런
화면이 나옵니다. **추천 모델 다운로드**는 이 PC의 그래픽 메모리에 맞춰 고른 모델을 바로 받고, **모델 둘러보기**는 아래의 모델 찾기로 갑니다.

![처음 화면 — 먼저 모델이 필요합니다](docs/img/first-run.png)

### 1. 모델 받기 — 모델 → 모델 찾기

- Hugging Face의 GGUF 모델이 **요즘 뜨는** 순으로 보입니다. **인기 · 좋아요 · 최근 업데이트** 순으로 바꾸거나, 위 칸에 이름(예: `qwen3`,
  `coder`, `14b`)을 넣어 좁힙니다.
- 기본으로 **GeulPU에서 실행되는 모델만** 보입니다. **실행 안 되는 모델도 보기**를 켜면 모두 보이고, 실행되지 않는 것은 흐리게 까닭과
  함께 보입니다. **이 PC에 맞는 크기**를 켜면 그래픽 메모리에 들어가는 것만 남습니다.
- 처음이라면 **GeulPU 확인됨**을 켜 보세요. GeulPU가 직접 돌려 본 파일이 있는 저장소만 남습니다(아래 그림) —
  [내 PC에 맞는 모델](#내-pc에-맞는-모델).

![모델 찾기 — 왼쪽은 목록, 오른쪽은 고른 모델의 자세한 정보](docs/img/find-models.png)

- 모델을 누르면 오른쪽에 설명 · 다운로드 수 · 라이선스와 **파일 목록**이 나옵니다. 이 PC에 가장 알맞은 파일에는 **이 PC에 추천**,
  GeulPU가 직접 돌려 본 파일에는 **실행 확인됨**이 붙고, 그래픽 메모리에 빠듯하거나 모자라는 파일은 그렇게 표시됩니다. 고른 파일의
  **다운로드**를 누릅니다.

![파일 목록 — 이 PC에 추천 · 실행 확인됨 표시와 다운로드 단추](docs/img/model-files.png)

- 받기 전에 파일 앞부분만 읽어 실행되는지 먼저 확인하고, 끊기면 이어서 받고, 다 받으면 SHA-256으로 확인합니다. 받는 중인 파일은
  **내 모델** 칸과 아래 상태 막대에서 보고 멈추거나 이을 수 있습니다.
- 이미 가진 GGUF 파일은 위쪽의 **GGUF 파일 추가**로 넣습니다(파일은 그 자리에 둔 채 씁니다).

### 2. 채팅하기

- 위쪽 모델 단추에서 모델을 고르고 메시지를 보냅니다(**Enter** 보내기 · **Shift+Enter** 줄바꿈).
- **처음 한 번**은 이 GPU와 모델에 맞는 커널을 만들고(빠른 CPU에서 10초 안팎, 느리면 1 ~ 3분) 모델을 그래픽 메모리에 올립니다(7B 약
  20초). 다음부터는 올리기만 합니다.
- **추론 모델**(Qwen3 · DeepSeek-R1)은 답하기 전에 생각을 씁니다. 생각은 'N초 동안 생각함'으로 접혀 있고, 눌러서 펼쳐 봅니다. 답 아래에는
  토큰 수 · 속도(토큰/초)가 나옵니다.

![채팅 — Qwen3-8B Q4_K_M, 생각은 접혀 있고 답 아래에 속도가 보입니다](docs/img/chat-ko.png)

- **Qwen3의 생각 끄기**: 오른쪽 위 **생성 설정**(슬라이더 아이콘) → **생각 과정 쓰기**를 끄면 생각 없이 바로 답합니다.
- **생성 설정**에서 온도(0이면 같은 질문에 늘 같은 답) · 최대 답변 길이 · 시스템 프롬프트도 정합니다(이 브라우저에 저장).

![생성 설정 — 온도 · 최대 답변 길이 · 생각 과정 쓰기 · 시스템 프롬프트](docs/img/generation-settings.png)

- 답을 다시 생성하거나 내 메시지를 고쳐 보낼 수 있고, 앞의 답은 버전으로 남아 ‹ 1 / 2 › 로 넘겨 봅니다. 코드 · 표 · 수식(LaTeX)을
  그려 줍니다.
- **긴 대화**: 대화가 채팅마다 문맥 길이에 가까워지면 앞의 대화를 같은 모델로 요약해(최근 차례는 그대로) 오류 없이 이어 갑니다. 요약은
  대화 안에 카드로 남아 펼쳐 볼 수 있고, 요약된 앞 메시지는 흐리게 남습니다. 답 아래의 '문맥 N / L'이 쓴 양입니다. 생성 설정의
  **긴 대화 자동 요약**으로 끕니다.
- 대화는 이 브라우저에만 저장됩니다.

### 3. 벤치마크 (선택)

- **하드웨어 진단**: 메모리 대역 · 텐서 코어 성능을 재고, 커널마다 하드웨어 한계에 얼마나 가까운지 보여 줍니다.
- **속도 측정**: 프롬프트(512 · 2048 토큰)와 생성 속도(토큰/초).
- **재현성 검증**: DeepSeek-R1-Distill-Qwen-7B Q8_0으로 수학 문제 32개를 풀어 기준 PC의 출력과 비트까지 같은지 견줍니다(빠른 확인은
  문제 2개 · 64토큰).
- 결과는 zip으로 내보낼 수 있습니다(사용자 경로는 지웁니다).

### 4. 설정

GPU 백엔드(자동 · CUDA · Vulkan), Vulkan 장치, 동시 처리 수(동시에 답하는 채팅 수 1 ~ 8)와 채팅마다 문맥 길이, **계산 방식**,
**추측 디코딩**, 모델 폴더, 언어, 새 버전 알림. 바꾸면 바로 저장됩니다.

**계산 방식** — **빠름**(기본)과 **정밀** 가운데 고릅니다. 다음에 모델을 불러올 때 적용됩니다.
- **빠름**: 행렬곱에 들어가는 활성값을 8비트(llama.cpp와 같은 급)로 셉니다. 프롬프트 처리가 정밀보다 1.3 ~ 1.6배 빠릅니다(CUDA · 7B ·
  8B). 다음 토큰 확률의 차이는 llama.cpp와 같은 급입니다([문서/2단계.md](문서/2단계.md)).
- **정밀**: 0.5.0까지의 계약 그대로(활성값 16비트 — 연구용 정밀도)입니다.
- 두 방식 모두 결과가 계약의 CPU 참조와 비트까지 같고, 같은 방식끼리는 어느 길로 계산해도 같은 비트입니다. 재현성 검증과 Vulkan
  장치는 늘 정밀로 셉니다.

**추측 디코딩** — **켬**(기본) · 끔. 프롬프트와 앞의 답에서 이어질 토큰을 찾아(초안 모델 없음) 한 걸음에 여럿을 검증합니다. 이 엔진은
걸음의 행 수와 상관없이 같은 비트를 내므로 **답 · 로그확률 · 연구 패널의 값이 끈 것과 비트까지 같고**, 코드 고치기 · 문서 고쳐 쓰기 ·
요약처럼 프롬프트의 조각을 되풀이하는 답의 생성이 3.5 ~ 4 배 빨라집니다(보통 채팅 1.0 ~ 1.5 배 — [문서/3단계.md](문서/3단계.md)).
다음에 모델을 불러올 때 적용됩니다.

**언어** — 화면은 브라우저 언어를 따릅니다(한국어가 아니면 영어). **설정 → 언어 · Language**에서 자동 · 한국어 · English를 고를 수 있고,
고른 값은 이 브라우저에 저장됩니다.

### 5. 연구 패널 — 토큰마다 확률 · 개입 · 비교 · 재현

GeulPU의 계산은 계약으로 정해져 있어서, 같은 글을 다시 셈하면 생성할 때와 **같은 비트**의 확률이 나옵니다 — 함께 도는 요청 수 · 캐시 ·
GPU · 백엔드(CUDA · Vulkan)와 상관없이(같은 계산 방식에서 — Vulkan은 늘 정밀이라, CUDA와 견줄 때는 설정의 계산 방식을 정밀로). 그래서 두
실행의 차이는 바꾼 것에서만 나옵니다. 채팅 위쪽의 플라스크 단추나 답 아래의 플라스크로
엽니다(이 판에서 받은 답부터 — 연구 기록은 이 브라우저에만).

**처음이면 [연구 패널 안내서](연구자안내.md)부터** — 실험 일곱(망설인 걸음 · 거기서 갈래 · 두 답 비교 · 양자화의 효과 · 낱말 하나의
효과 · 글의 점수 · 재현 캡슐)을 화면 그림과 함께 따라 하고, 숫자 읽는 법과 API 예시까지 담았습니다.

- **관찰** — 답을 토큰으로: 토큰마다 확률 · 순위 · 대안 상위 10 · 엔트로피 · 1위와 2위의 차(여유). 아슬아슬한 걸음(여유 < δ — 셈이나
  설정이 조금만 달라도 다른 토큰이 될 자리)과 1위가 아닌 토큰을 뽑은 걸음을 표시합니다.
- **개입** — 어느 걸음에서 대안 토큰을 골라 **거기서 갈래**: 같은 설정 · 같은 씨앗 · 같은 걸음 번호의 난수로 이어 생성해서 바뀐 것은 고른
  토큰 하나뿐입니다. 온도 · 씨앗 · 최대 길이만 바꿔 다시 생성할 수도 있습니다. 새 답은 버전으로 남습니다.
- **비교** — 두 답(버전 · 갈래)의 처음 갈린 걸음. 질문을 고친 버전이면 같은 답 토큰을 다른 프롬프트로 점수 매겨 토큰마다의 차를,
  같은 글을 다른 모델(예: 같은 모델의 Q8_0 ↔ Q4_K_M)로 점수 매겨 양자화 하나의 효과를 봅니다.
- **점수** — 글을 붙여 넣으면 토큰마다 로그확률과 퍼플렉시티.
- **재현** — 지문(SHA-256 — 셈법은 [계약](계약/계약.md) 5.2), 캐시 없이 다시 돌려 확인, **재현 캡슐**(요청 · 결과 · 로그확률 · 로짓 해시를 JSON 하나로) 내보내기와 불러와
  확인, 엔진 바이너리 · 커널의 해시.
- **내보내기** — 보이는 값 모두 JSON · CSV, 같은 값을 받는 파이썬 코드.

확률은 모델의 분포(온도 1)입니다 — 뽑기에 쓴 온도 · top-k · top-p로 거른 분포가 아닙니다. 패널은 들여다볼 때만 셈하므로 채팅 속도는
그대로입니다.

![연구 패널 — 관찰: 토큰 띠에서 걸음 하나를 골라 대안 상위 10과 수를 봅니다](docs/img/research-observe-ko.png)

![연구 패널 — 비교: 같은 답을 같은 모델의 Q4_K_M 파일로 다시 점수 매겨 양자화 하나의 효과를 토큰마다 봅니다](docs/img/research-compare-ko.png)

받은 캡슐은 GPU 없이도 확인합니다 — 저장소를 받아(`pip install numpy`) `python -I 도구/캡슐확인.py <캡슐.json> <모델.gguf>` 를 실행하면
계약의 CPU 참조 구현이 같은 토큰 · 로짓 해시 · 로그확률을 내는지 봅니다(7B는 몇 분). 캡슐에는 계산 방식이 적혀 있어 그 방식으로
확인합니다.

### 다른 프로그램에서 쓰기 (OpenAI 호환 API)

앱에서 모델을 불러온 상태(채팅을 한 번 보내면 불러옵니다)에서 `http://127.0.0.1:44544/v1` 을 OpenAI 호환 주소로 씁니다. `model` 값은
아무 이름이나 됩니다(지금 불러온 모델이 답합니다). 응답의 `model` 도 보낸 이름 그대로이니, 어느 모델이 답했는지는 `/v1/models` 로 봅니다.

PowerShell 창에 붙여 넣어 실행합니다(한글이 깨지지 않게 응답을 UTF-8로 읽습니다 — `.ps1` 파일로 저장하려면 'UTF-8(BOM)'으로).

```powershell
$body = @{ messages = @(@{ role = 'user'; content = '안녕하세요' }) } | ConvertTo-Json -Depth 4
$r = Invoke-WebRequest -Uri http://127.0.0.1:44544/v1/chat/completions -Method Post -ContentType 'application/json; charset=utf-8' -Body ([Text.Encoding]::UTF8.GetBytes($body)) -UseBasicParsing
([Text.Encoding]::UTF8.GetString($r.RawContentStream.ToArray()) | ConvertFrom-Json).choices[0].message.content
```

```python
from openai import OpenAI   # pip install openai
client = OpenAI(base_url="http://127.0.0.1:44544/v1", api_key="geulpu")   # 키는 아무 글자나
r = client.chat.completions.create(model="geulpu", messages=[{"role": "user", "content": "안녕하세요"}])
print(r.choices[0].message.content)
```

- Qwen3의 생각을 끄려면 요청에 `"chat_template_kwargs": {"enable_thinking": false}` 를 넣습니다(Python은 `extra_body=` 로).
- Qwen3 · DeepSeek-R1의 생각은 기본으로 태그 없이 답 앞에 붙어 옵니다. 요청에 `"geulpu": {"special": true}` 를 넣으면 `<think>` ·
  `</think>` 가 글로 와서 생각과 답을 나눌 수 있습니다.
- 이 PC 안에서만 접속되고 인증은 없습니다. 다른 웹사이트가 브라우저를 통해 보내는 요청은 막습니다.
- `/v1/chat/completions` 는 Qwen2.5 · Qwen3 · Llama 3(3.1 · 3.2) · Mistral(v0.3)의 대화 틀을 씁니다. DeepSeek-R1은 아직 `/v1/completions` 에
  R1의 대화 틀을 직접 넣어 씁니다.
- Llama 3의 틀은 시스템 머리에 날짜를 넣습니다 — 같은 질문이 날마다 다른 프롬프트가 되지 않게 `26 Jul 2024`로 고정하고, 요청의
  `"chat_template_kwargs": {"date_string": "16 Oct 2026"}` 로 바꿉니다. Mistral의 틀에는 시스템 역할이 없어 시스템 말은 첫 질문 앞에
  `{시스템}\n\n` 으로 붙입니다(llama.cpp와 같은 꼴).
- Llama 3 · Mistral은 글 프롬프트(`/v1/completions` 의 글 · `/tokenize` 의 `prompt`)에 BOS를 앞에 붙입니다(llama.cpp · vLLM과 같이 —
  `/tokenize` 는 `"add_special_tokens": false` 면 붙이지 않음). 토큰 번호 프롬프트는 그대로 씁니다.

**로그확률 · 점수 매기기 · 토큰화(연구용)** — 연구 패널의 값은 모두 API로도 받습니다. 값은 계약 5.2([계약](계약/계약.md))로 정해져 어느
PC · GPU에서든 같은 비트입니다.

- `/v1/chat/completions`: `"logprobs": true, "top_logprobs": 5`(0 ~ 20) — OpenAI 꼴(`token` · `logprob` · `bytes` · `top_logprobs`).
- `/v1/completions`: `"logprobs": 5`, `"echo": true`(프롬프트 토큰의 로그확률까지), `"max_tokens": 0`(생성 없이 점수만). `prompt` 는 글이나
  토큰 번호 배열.
- `"geulpu"` 덧붙임: `"logit_hash": true`(걸음마다 로짓 행의 64비트 해시), `"prompt_tokens": true`(채팅 틀을 씌운 프롬프트 토큰),
  `"echo_from": n`(echo 의 로그확률을 번호 n 인 토큰부터 — 0부터 센다, 앞은 null), 응답의 `entropy` · `top_ids` · `top_logprobs`(번호와 나란히) · `seed`.
- `/tokenize` · `/detokenize` · `/tokenizer_info` — vLLM과 같은 꼴.
- `geulpu` 의 `tokens` · `logit_hashes` 는 멈춘 토큰(EOS)을 뽑은 걸음까지 담고, 로그확률 배열(`logprobs` · `entropy` · `top_ids` ·
  `top_logprobs`)은 OpenAI처럼 답 글의 토큰만 담습니다 — 답이 EOS로 끝나면 하나 짧습니다(EOS 걸음의 확률은 `tokens` 를 echo로 점수
  매겨 받습니다). `first_token_ms` 는 이름과 달리 µs입니다(호환을 위해 그대로).
- 생성 중 로그확률을 물으면 토큰당 시간이 약 2% 늘어납니다(7B · 8B — 셈을 다음 걸음의 GPU 셈 뒤에 숨긴다). 묻지 않으면 그대로입니다.

```python
# 글의 토큰마다 로그확률(점수 매기기) — 첫 토큰은 앞 맥락이 없어 null
r = client.completions.create(model="geulpu", prompt="The capital of France is Paris.", echo=True, max_tokens=0, logprobs=5)
print(list(zip(r.choices[0].logprobs.tokens, r.choices[0].logprobs.token_logprobs)))
```

**lm-evaluation-harness** — `local-completions` 로 그대로 붙습니다. 토크나이저는 서버의 것을 써서(`tokenizer_backend=remote`) Hugging Face
토크나이저를 받지 않습니다. 문항마다의 결과는 `num_concurrent` · 동시 처리 수 · CUDA ↔ Vulkan과 상관없이 같은 비트입니다
([38단계](https://github.com/wwoosshh/GeulPU/blob/main/문서/38단계.md)). 동시 처리 수는 **설정**에서 올립니다(최대 8).

```bash
pip install "lm-eval[api]"
lm_eval --model local-completions --tasks arc_easy,gsm8k --model_args model=geulpu,base_url=http://127.0.0.1:44544/v1/completions,tokenizer_backend=remote,tokenized_requests=True,num_concurrent=8
```

## 지원하는 모델

GGUF 파일 하나를 받습니다(여러 조각으로 나뉜 파일은 llama.cpp의 `llama-gguf-split --merge` 로 합쳐서).

| 구조 | 모델 | 크기 |
|---|---|---|
| qwen2 | Qwen2.5-Instruct · Qwen2.5-Coder · DeepSeek-R1-Distill-Qwen과 그 미세조정 모델 | 1.5B 이상(헤드 크기 128) — 7B · 14B를 돌려 봄 |
| qwen3 | Qwen3와 그 미세조정 모델 | 1.7B · 8B · 14B |
| llama | Llama 3.1 · 3.2 Instruct, Mistral 7B Instruct v0.3, 같은 대화 틀을 쓰는 그 미세조정 모델 | 3B · 7B · 8B — 70B는 모양을 합성 모델로 |

- **형식**: Q8_0 · Q4_K_M · Q5_K_M · Q6_K, 그리고 모든 텐서가 Q4_0인 Q4_0 파일.
- **라이선스**: 모델마다 다릅니다. 앱 목록의 Llama 3.1 · 3.2는 Meta의 커뮤니티 라이선스(사용 조건 · 이용 정책 — 월 사용자 7억 명이
  넘는 서비스는 Meta의 허가가 필요), Mistral 7B v0.3 · Qwen2.5 · Qwen3 파일은 Apache 2.0, DeepSeek-R1-Distill은 MIT입니다(앱의 모델 설명에도).
- **아직 안 되는 것**: Gemma · Phi 같은 다른 구조, 질의 폭이 너비와 다른 모델(Qwen3의 0.6B · 4B · 32B, Mistral NeMo), 헤드 크기가 64인
  작은 모델(Qwen2.5-0.5B · Llama 3.2 1B), MoE(Mixtral · Qwen3-30B-A3B 등), Llama 3 · Mistral과 다른 대화 틀(ChatML 등)을 쓰는 llama
  구조 미세조정, 이미지 입력, Q2_K · Q3_K · IQ · Q5_0 · F16 · BF16. 모델 찾기는 이런 파일을 기본으로 숨깁니다.
- **GPU에 따라**: k-양자화(Q4_K_M 등)는 CUDA와, 공유 메모리가 48 KB 이상인 Vulkan GPU에서 돕니다. 윈도의 AMD 드라이버(공유 메모리
  32 KB)에서는 지금 Q8_0(과 Q4_0) 파일만 돕니다.

### 내 PC에 맞는 모델

GeulPU가 직접 돌려 본 파일입니다(앱의 **GeulPU 확인됨**). 크기는 윈도 탐색기와 같은 1024 단위입니다.

| 그래픽 메모리 | 권하는 파일 |
|---|---|
| 6 GB | Qwen3-1.7B Q8_0 (1.7 GB) · Llama-3.2-3B-Instruct Q8_0 (3.2 GB) |
| 8 GB | Qwen3-8B Q4_K_M (4.7 GB) · Qwen2.5-7B-Instruct Q4_K_M (4.4 GB) · DeepSeek-R1-Distill-Qwen-7B Q4_K_M (4.4 GB) · Llama-3.1-8B-Instruct Q4_K_M (4.6 GB) · Mistral-7B-Instruct-v0.3 Q4_K_M (4.1 GB) |
| 12 GB | Qwen3-8B Q8_0 (8.1 GB) · Qwen2.5-7B-Instruct Q8_0 · Qwen2.5-Coder-7B-Instruct Q8_0 · DeepSeek-R1-Distill-Qwen-7B Q8_0 (7.5 GB) · Llama-3.1-8B-Instruct Q8_0 (8.0 GB) · Mistral-7B-Instruct-v0.3 Q8_0 (7.2 GB) — 14B Q4_K_M도 문맥을 줄여 돕니다 |
| 16 GB 이상 | Qwen2.5-14B-Instruct Q4_K_M · DeepSeek-R1-Distill-Qwen-14B Q4_K_M (8.4 GB) |

Qwen3-14B Q4_K_M은 같은 모양의 합성 모델로 출력이 비트까지 맞는지만 확인했고 아직 돌려 보지 않았습니다. 앱은 파일마다 이 PC의 그래픽
메모리에 맞는지 미리 알려 줍니다.

## 실행 방식

```mermaid
flowchart LR
    B["브라우저<br/>127.0.0.1:44544"] -- "화면 · /api · /v1" --> M["GeulPU.exe<br/>관리자"]
    M -- "모델을 처음 쓸 때" --> G["kernel-gen.exe<br/>ptx-backend.exe · spirv-backend.exe"]
    G -- "이 GPU용 커널" --> K[("모듈 폴더")]
    M -- "띄우고 /v1 을 넘김" --> S["server-cuda.exe<br/>server-vulkan.exe"]
    K --> S
    S -- "nvcuda.dll · vulkan-1.dll" --> GPU[("GPU")]
```

1. **관리자**(`GeulPU.exe`)가 `127.0.0.1:44544` 에서 화면을 열고 브라우저를 띄웁니다. 이 PC 안에서만 들리고, 같은 포트에 이미 떠 있으면
   그 화면을 엽니다. 44544는 '글'의 유니코드 번호(U+AE00)입니다.
2. 처음에 GPU를 확인합니다 — NVIDIA RTX 30 이상이면 CUDA(`nvcuda.dll`), 아니면 Vulkan(`vulkan-1.dll`). 설정에서 바꿀 수 있습니다.
3. 모델을 처음 쓸 때 **커널을 만듭니다**. 커널 생성기(`kernel-gen.exe`)가 그 모델의 구조 · 모양 · 형식에 맞는 GPU 커널 소스를 글(geul)로
   쓰고, 글 컴파일러의 뒷단(`ptx-backend.exe` · `spirv-backend.exe`)이 그것을 PTX(NVIDIA) · SPIR-V(Vulkan)로 바꿔 자료 폴더에 둡니다.
   앱 버전마다 한 번입니다.
4. **추론 서버**(`server-cuda.exe` · `server-vulkan.exe`)가 GGUF를 그래픽 메모리에 올리고, 그 커널을 그래픽 드라이버로 바로 돌립니다.
   CUDA 툴킷 · 런타임 · Python이 필요 없습니다.
5. 채팅은 브라우저 → 관리자 → 추론 서버(OpenAI 꼴 API)로 가고, 답은 토큰마다 흘러 돌아옵니다.
6. 검은 창을 닫으면 관리자와 그 아래 프로세스(서버 · 도구)가 함께 꺼집니다.

관리자 · 추론 서버 · 커널 생성기 · 뒷단 모두 글(geul)로 썼고, 가져오는 것은 윈도의 DLL(관리자: kernel32 · ws2_32 · shell32 · winhttp ·
bcrypt · advapi32)과 그래픽 드라이버의 DLL뿐입니다.

**zip 안**

| 파일 | 하는 일 |
|---|---|
| `GeulPU.exe` | 관리자 — 화면, 모델 받기 · 확인, 서버와 도구 띄우기 |
| `bin\server-cuda.exe` · `bin\server-vulkan.exe` | 추론 서버(OpenAI 꼴 HTTP API) |
| `bin\kernel-gen.exe` | 커널 생성기 — 모델 모양에 맞는 커널 소스(글) |
| `bin\ptx-backend.exe` · `bin\spirv-backend.exe` | 글 컴파일러의 뒷단 — 커널을 PTX · SPIR-V로 |
| `bin\device-*.exe` · `bin\diag-*.exe` · `kernels\` | GPU 확인 · 하드웨어 진단 |
| `web\` | 화면(HTML · CSS · JavaScript) |
| `data\` | 받을 수 있는 모델 목록 · 재현성 검증 기준 |
| `version.json` | 버전 · 커밋 · 파일마다 SHA-256 |

**자료 폴더** — `%USERPROFILE%\GeulPU` (버전을 바꿔도 그대로 남습니다)

| | |
|---|---|
| `모델\` | 받은 모델(GGUF) |
| `모듈\<버전>\` | 이 GPU에 맞춰 만든 커널 |
| `결과\` | 벤치마크 결과 |
| `기록\` | 서버 · 작업 기록 — 문제가 생기면 여기를 봅니다 |
| `설정.json` · `모델목록.json` · `허깅목록.json` | 설정 · 내 모델 목록 · Hugging Face에서 고른 파일 |

자료 폴더를 옮기려면 환경 변수 `GEULPU_HOME` 을 두거나 `GeulPU.exe --자료 <폴더>` 로 실행합니다. 그 밖의 실행 옵션은
`--포트 <번호>`(다른 포트) · `--열지않음`(브라우저를 띄우지 않음)입니다.

**네트워크 · 개인정보**

- 관리자와 추론 서버는 `127.0.0.1` 에서만 듣습니다 — 다른 기기에서는 접속할 수 없습니다.
- 인터넷은 세 곳에만 씁니다: 모델 찾기(화면이 Hugging Face API를 직접 읽음), 모델 받기(huggingface.co), 새 버전 확인(GitHub — 설정의 새 버전
  알림으로 끔).
- 사용 통계를 보내지 않습니다. 대화는 브라우저의 저장소에만 남습니다.
- 설치 프로그램이 없고 레지스트리에 쓰지 않습니다.

**빌드 · 신뢰**: 릴리스 zip은 태그를 밀면 GitHub Actions가 이 저장소의 소스로 빌드합니다(`앱/꾸리기.ps1` — 글 컴파일러를 고정점까지 빌드한
뒤 뒷단 · 서버 · 도구 · 관리자). 서버 · 생성기 · 뒷단은 재현성 검증 기준(`교차검증/expected.json`)의 빌드 해시와 같은 바이트인지 확인한 뒤에야
꾸립니다.

## 문제 해결

| 증상 | 할 일 |
|---|---|
| 브라우저가 열리지 않음 | 주소창에 `http://127.0.0.1:44544` 를 직접 입력합니다. |
| "Windows의 PC 보호" | **추가 정보 → 실행**(서명하지 않은 실행 파일). |
| 백신이 실행을 막음 | 풀어 둔 GeulPU 폴더를 예외로 둡니다. 소스와 빌드 과정은 이 저장소에 공개되어 있습니다. |
| 포트 44544를 다른 프로그램이 씀 | `GeulPU.exe --포트 44600` 처럼 다른 포트로 실행합니다. |
| "쓸 수 있는 GPU 없음" | 그래픽 드라이버를 최신으로 올린 뒤 **하드웨어 → 다시 확인**. |
| "이 GPU 미지원" · "그래픽 메모리 부족" | 더 작은 파일(Q4_K_M)이나 작은 모델을 고릅니다. 설정에서 채팅마다 문맥 길이나 동시 처리 수를 줄이면 메모리가 덜 듭니다. |
| 첫 메시지가 오래 걸림 | 처음 한 번은 커널을 만듭니다(1 ~ 3분). 다음부터는 모델을 올리는 시간(수십 초)만 듭니다. |
| 다운로드가 멈춤 | **내 모델** 또는 상태 막대에서 **이어서 다운로드** — 받은 데부터 잇습니다. |
| 그 밖 | `%USERPROFILE%\GeulPU\기록\` 의 기록과 함께 [이슈](https://github.com/wwoosshh/GeulPUv2/issues)를 남겨 주세요. |

## 업데이트 · 삭제

- **새 버전**: 앱 화면이 GitHub의 새 릴리스를 알려 줍니다. 앱을 끄고 새 zip을 풀어 그 `GeulPU.exe` 를 실행하면 됩니다. 모델 · 설정 · 결과는
  자료 폴더에 그대로 있어 다시 받지 않습니다(커널만 새 버전에 맞춰 한 번 다시 만듭니다). 바뀐 것은 [바뀐것.md](문서/바뀐것.md).
- **삭제**: 앱을 끄고 GeulPU 폴더를 지웁니다. 받은 모델까지 지우려면 `%USERPROFILE%\GeulPU` 도 지웁니다.
- **파이썬 설치 도구로 깐 경우**: 업데이트는 앱을 끈 뒤 `pipx upgrade geulpu`(`uv tool upgrade geulpu` · `pip install -U geulpu`),
  삭제는 `pipx uninstall geulpu`(`uv tool uninstall geulpu` · `pip uninstall geulpu`) — 자료 폴더는 남습니다.

## 무엇이 다른가

- **정밀도를 계약으로 밝힌다.** 활성값은 32 개 블록마다 정수 — 정밀 모드는 자리 둘(±2¹⁴), 빠른 모드(기본 — [2단계](문서/2단계.md))는
  행렬곱에 들어가는 활성값만 자리 하나(8 비트) —, KV 캐시는 자리 둘 블록 정수, 곱의 합은 정수 텐서 코어의 정확한 합, 지수 함수는 적어 둔
  다항식(적힌지수). 그래서 두 모드 모두 GPU 의 결과가 CPU 참조 구현([계약/정수계약.py](계약/정수계약.py))과 **비트까지** 같다. 손실은
  모델 파일의 형식(Q8_0 · Q4_0 · k-양자화)과 이 계약이 밝힌 만큼뿐이다.
- **어느 길로 계산해도 같은 비트.** 프롬프트를 한 번에 넣든, 나눠 넣든, 하나씩 넣든, 대화로 이어 가든, 여러 문장을 묶든 결과가 같다.
  흔한 엔진(llama.cpp · PyTorch)은 함께 계산하는 양에 따라 결과 비트가 바뀐다. 퍼플렉시티(wikitext-2)도 llama.cpp 는 배치 설정에 따라
  15.8171 ~ 15.8200 으로 바뀌고, 글은 어느 방식으로 넣어도 마지막 비트까지 같다([20단계](https://github.com/wwoosshh/GeulPU/blob/main/문서/20단계.md)). GPU 가 달라도 같다:
  RTX 3080 Ti(Ampere) · RTX 4070 Ti(Ada) · RTX 5070(Blackwell)이 DeepSeek-R1-Distill-Qwen-7B 로 푼 수학 문제 32 개에서 출력과 82,673 걸음의
  로짓까지 비트가 같았고, 회사가 다른 AMD Radeon RX 9060 XT(RDNA4 · Vulkan — 두 PC, 드라이버 두 판)도 NVIDIA 의 CUDA 기준과 82,673 걸음 모두 같았다(같은 조건에서 llama.cpp 는 한 PC 안에서 함께 도는 요청 수만 바꿔도 32 문제의 출력이 모두 갈렸다 — [교차검증](교차검증/)).
  평가 도구도 같다: lm-evaluation-harness 의 문항마다 결과(arc_easy 500 · gsm8k 100)가 동시 요청 1 ↔ 8 · CUDA ↔ Vulkan 에서 모두 같았다
  (같은 설정에서 llama.cpp b11643 은 gsm8k 100 문항 중 41 문항의 생성 글이 동시 요청 수에 따라 달랐다 — [38단계](https://github.com/wwoosshh/GeulPU/blob/main/문서/38단계.md)).
- **추측 디코딩을 켜도 같은 답.** 위의 성질 덕에 검증 걸음(행 여럿)의 로짓이 하나씩 셈한 로짓과 비트까지 같아, 탐욕은 물론 뽑기
  (걸음 번호로 정한 난수)도 같은 토큰을 낸다 — 추측을 켜도 끈 것과 토큰 · 걸음마다 로짓 해시 · 로그확률이 같다(시험: 요청 열일곱 × 맞는 ·
  틀린 · 반쯤 틀린 초안, 슬롯 넷 동시, CUDA · Vulkan — [3단계](문서/3단계.md)). 같은 측정에서 llama.cpp b11721 의 llama-server 는
  `--spec-type ngram-simple` 을 켜자 같은 탐욕 요청 여덟 중 둘(요약 · 수학)의 답이 추측 없음과 달랐다.
- **정밀 모드의 숨은 손실이 작다 — 0.5B 에서 잰 값.** Qwen2.5-0.5B-Instruct(Q8_0)의 숨은 손실(같은 가중치의 fp64 참값 대비 KL)이 llama.cpp 의
  1/615(평가글 461 토큰) · 1/385(wikitext-2 148,920 토큰)다. 퍼플렉시티로 보면 글은 참값과 구별되지 않고(상대 −6.6 × 10⁻⁶), llama.cpp 는
  0.34% 높다 — 가중치를 Q8_0 으로 줄인 손실(0.24%)과 같은 급이라 체감할 만큼은 아니다([20단계](https://github.com/wwoosshh/GeulPU/blob/main/문서/20단계.md)). 7B · 14B 는 실제 엔진의
  출력으로 재지 않았다(반올림 자리만 흉내 낸 검토: llama.cpp 식의 1/3805 · 1/4467 — [계약손실검토](https://github.com/wwoosshh/GeulPU/blob/main/문서/계약손실검토.md)).
  빠른 모드는 숨은 손실을 llama.cpp 와 같은 급으로 둔다 — 정밀 모드 대비 다음 토큰 분포의 KL 이 k-양자화 7B · 8B 에서 llama.cpp 의
  0.65 ~ 0.98 배, Q8_0 에서 2.2 ~ 2.4 배다([2단계](문서/2단계.md)).
- **속도 — 빠른 모드의 프롬프트는 4096 토큰까지 llama.cpp 보다 빠르고, 생성은 같은 급이다.** (RTX 4070 Ti)
  - 빠른 모드(4단계 — llama.cpp 최신 공식 판 b11721 · 같은 GGUF · 번갈아 세 바퀴의 가운데 · llama.cpp 는 길이마다 가장 빠른 설정)의
    프롬프트 시간: Qwen2.5-7B Q4_K_M 0.89 · 0.90 · 0.95 · 1.04배, Q8_0 0.74 · 0.76 · 0.81 · 0.92배, Llama-3.1-8B Q4_K_M 0.92 · 0.93 ·
    0.99 · 1.11배(512 · 2048 · 4096 · 8192 토큰 — [4단계](문서/4단계.md)). 긴 문맥에서 늦어지는 것은 어텐션(아직 자리 둘)의 몫이다.
    같은 측정에서 정밀 모드는 1.16 ~ 1.47배.
  - 긴 세션([긴문맥재기](문서/긴문맥재기.md) — 빠른 모드): 빈 캐시에서 16k · 32k 토큰을 넣는 시간이 llama.cpp 의 1.19 · 1.40배(Qwen2.5-7B
    Q4_K_M) · 1.29 · 1.40배(Llama-3.1-8B Q4_K_M), 세션 8k · 16k · 31k 에 새 입력 1,024 토큰이 1.18 · 1.40 · 1.57배 · 1.29 · 1.44 · 1.08배 —
    깊이당 어텐션 비용이 llama.cpp 의 1.9 ~ 2.4배다. 생성은 같은 깊이에서 1.01 ~ 1.06배. 문맥은 Qwen2.5 가 32,768 까지(YaRN 없음),
    Llama-3.1-8B 는 12 GB 에서 33k 남짓.
  - 추측 디코딩(3단계 — 기본 켬, Qwen2.5-7B Q4_K_M · 빠른 모드 · 슬롯 1 · 세 바퀴의 가운데, 생성 토큰/초): 코드 고치기 329 · 356, 문서 고쳐
    쓰기 389, 요약 347 — 추측을 끈 95 의 3.5 ~ 4.1 배이고 llama.cpp b11721 의 가장 빠른 추측(ngram-simple 266 · 245 · 188 · 139)보다
    1.2 ~ 2.5 배 빠르다. 보통 채팅 탐욕 112 · 뽑기 97 · 수학 144, 번역 96(추측 없음 95 — llama.cpp 도 95).
  - 생성은 llama.cpp 와 같은 급이다: 7B · 14B 가 0.970 ~ 1.012배 시간(Q8_0 · Q4_0 · k-양자화 — [23](https://github.com/wwoosshh/GeulPU/blob/main/문서/23단계.md) ·
    [24](https://github.com/wwoosshh/GeulPU/blob/main/문서/24단계.md) · [26단계](https://github.com/wwoosshh/GeulPU/blob/main/문서/26단계.md)). 생성은 가중치를 읽는 시간이 정하므로 더 든 셈이 그 뒤에 숨는다.
    빠른 모드의 생성은 정밀 모드의 0.993 ~ 0.998배 시간이다(2단계 — 문맥 짧을 때 · 2048 · 8192).
  - 정밀 모드의 프롬프트는 짧을 때만 빨랐다(30단계): 7B Q4_0 이 llama.cpp 최신 공식 릴리스(b11567, 같은 GGUF)의 0.85배(32 토큰) · 1.03배(128) ·
    1.11배(2048) · 1.27배(8192) 시간이고, Q8_0 은 0.85 ~ 1.30배, k-양자화는 0.95 ~ 1.56배다(같은 실행 — [30단계](https://github.com/wwoosshh/GeulPU/blob/main/문서/30단계.md)). 활성값을 자리
    둘(16비트)로 두는 계약이라 같은 최적화라면 행렬곱의 정수 텐서 일이 활성값 8비트인 llama.cpp 의 두 배이고, 어텐션의 정확한 합 ·
    적힌 지수도 셈을 더한다. 길수록 비가 커지는 것은 어텐션의 몫이다.
  - 0.5B(프롬프트 128 ~ 8192 토큰 0.55 ~ 0.82배, 생성 0.63 ~ 0.65배 — [14](https://github.com/wwoosshh/GeulPU/blob/main/문서/14단계.md) · [15단계](https://github.com/wwoosshh/GeulPU/blob/main/문서/15단계.md))와 GPT-2 XL
    (프롬프트 0.78 ~ 0.87배)에서 llama.cpp 보다 빨랐던 것은 커널을 합치고 메가커널로 짠 솜씨의 몫이지, 계약이 공짜라서가 아니다.
  - 같은 최적화에서 계약의 값은 2026-10-13 에 같은 엔진에서 제약을 끄고 쟀다(작업 폴더에서 — 원자료는 저장소에 없다). 타일 판의 활성값
    자리 둘 → 하나: 7B Q4_0 프롬프트 512 · 2048 · 8192 가 1.38 · 1.35 · 1.26 배(하한 — 끝손질과 어텐션의 q 는 자리 둘 그대로). 적힌 지수 →
    근사 지수: 거의 0. 생성은 약 1.0 배(가중치를 읽는 시간이 정한다). 결정성만의 값은 따로 떼어 재지 못했다.
  - 숫자와 재는 법은 [문서/단계기록.md](https://github.com/wwoosshh/GeulPU/blob/main/문서/단계기록.md).

## 개발 · 연구

아래는 엔진을 소스에서 빌드하려는 분을 위한 내용입니다. 앱만 쓰려면 위의 [설치](#설치)로 충분합니다. 연구 기록(단계 문서 · 논문)은
[GeulPU 저장소](https://github.com/wwoosshh/GeulPU)에 있고, 아래 표의 'N단계' 링크는 그곳으로 갑니다.

### 지금 되는 것 · 안 되는 것 (2026-10)

| | |
|---|---|
| 모델 | **Llama-3.1-8B · Llama-3.2-3B · Mistral-7B v0.3**(39단계 — 구조 llama: q · k 줄 되돌리기 · RoPE 늘이기 인자를 계약 4.7 로, 다섯 파일이 CPU 참조와 비트까지 같고 llama.cpp 와 같은 글의 모든 토큰에서 평균 KL 4 × 10⁻⁴ 이하 · 70B 는 모양을 합성 모델로 — [39단계](https://github.com/wwoosshh/GeulPU/blob/main/문서/39단계.md)) · **Qwen3-1.7B · 8B · 14B**(37단계 — 머리마다 Q · K 정규화를 계약 4.6 과 판 하나로, CPU 참조와 비트까지 같음 · 14B 는 모양을 합성 모델로 — [37단계](https://github.com/wwoosshh/GeulPU/blob/main/문서/37단계.md)) · **Qwen2.5-0.5B-Instruct** · **Qwen2.5-7B-Instruct** · **Qwen2.5-14B-Instruct** · DeepSeek-R1-Distill-Qwen (채팅 — Llama 계열 블록: RMSNorm · RoPE · SwiGLU · GQA, 머리 크기 64 · 128) — GGUF Q8_0 · Q4_0 · **k-양자화(Q4_K · Q5_K · Q6_K — 텐서마다 형식이 다른 Q4_K_M · Q5_K_M · Q6_K 섞음 그대로, 26단계 — 이 PC 에서 다시 양자화한 파일로 시험)**. k-양자화도 CPU 참조와 비트까지 같고, 7B 생성은 llama.cpp 의 0.977 ~ 1.009배 시간(아홉 칸) — 14B Q4_K_M 생성 1.009 · 1.012배([26단계](https://github.com/wwoosshh/GeulPU/blob/main/문서/26단계.md)). 27단계에서 프롬프트의 타일 판을 고쳐(값은 비트까지 그대로) 7B 프롬프트가 llama.cpp 의 1.25 · 1.39배(Q4_K_M) · 1.37 · 1.49배(Q5_K_M) · 1.45 · 1.56배(Q6_K) · 1.13 · 1.29배(Q4_0) · 1.14 · 1.30배(Q8_0)다(2048 · 8192, [27단계](https://github.com/wwoosshh/GeulPU/blob/main/문서/27단계.md)). 24단계부터 머리 크기 128 의 Qwen2 는 모양(너비 · 머리 · KV 머리 · 확장폭 · 낱말)을 GGUF 의 설정에서 받는다 — 14B 는 실제로(llama.cpp 대비 생성 0.982 · 0.986 · 0.994배, 프롬프트 1.22 · 1.34배), 이 PC 에 들지 않는 32B · 72B 모양은 층을 줄인 합성 모델로 비트까지 시험했다([24단계](https://github.com/wwoosshh/GeulPU/blob/main/문서/24단계.md)). 7B 는 22단계부터, Q4_0 은 23단계부터: CPU 참조와 비트까지 같고, llama.cpp(같은 GGUF) 대비 생성 0.97 ~ 0.99배 시간(Q8_0 0.980 ~ 0.990 · Q4_0 0.970 ~ 0.985 — 문맥 짧을 때 · 2048 · 8192), 프롬프트는 23단계에 1.15 · 1.30배(Q8_0) · 1.23 · 1.38배(Q4_0, 2048 · 8192)였다 ([22단계](https://github.com/wwoosshh/GeulPU/blob/main/문서/22단계.md) · [23단계](https://github.com/wwoosshh/GeulPU/blob/main/문서/23단계.md)). GPT-2 (small 124M · XL 1.5B) — GGUF Q8_0 · Q4_0, 시험용 합성 긴 모델(위치 102400) |
| 터미널 실행기 | `빌드/글/실행기.exe 모델.gguf` — 글(geul) 언어 프로그램(18단계). 채팅(Qwen 의 ChatML · Qwen3 · Llama 3 · Mistral 의 틀), 답을 흘려 보이고, 차례마다 KV 캐시를 이어 쓴다. 탐욕(같은 대화면 같은 답). 프롬프트까지 약 0.3 초(일꾼 스레드로 겹쳐 올린다 — 파이썬판은 2.9 초). 파이썬판 `python 실행.py 모델.gguf` 도 같은 답 |
| 서버 | `빌드/글/서버.exe 모델.gguf` — 글(geul) 언어 프로그램(28단계). OpenAI 꼴 HTTP API(채팅 · completions · SSE 흘려보내기), 슬롯 · 연속 배치 · 프롬프트 캐시, 뽑기 계약(씨앗이 같으면 같은 토큰). 같은 요청은 함께 도는 요청과 상관없이 같은 비트다 — 네 모델 × 요청 열둘 × 차례 넷에서 다른 것 0 (llama-server 는 함께 돈 요청에 따라 12 경우 중 11 에서 탐욕 답이 바뀌었다). 머리 128 의 Qwen2(7B · 14B). 7B Q4_0 합 처리량은 넷 스트림 302 · 여덟 508 토큰/초(llama-server 360 · 632 — 행 여럿 줄 판이 다음 일), 한 스트림은 llama-server 보다 4 % 짧다([28단계](https://github.com/wwoosshh/GeulPU/blob/main/문서/28단계.md)). 캐시를 쓴 첫 토큰은 13.8 ms(llama-server 13.0 ms — 30단계부터 남은 행을 생성 걸음으로, [30단계](https://github.com/wwoosshh/GeulPU/blob/main/문서/30단계.md)) |
| 실행 환경 | Windows 또는 Linux x86-64(glibc) + NVIDIA Ampere(RTX 30) 이상 + 드라이버 (이 저장소의 숫자는 RTX 4070 Ti 에서 잰 것). 리눅스는 29단계에 WSL2 Ubuntu 24.04 에서 시험했다 — 같은 GPU 에서 윈도와 같은 비트, 속도는 0.5 % 안팎([29단계](https://github.com/wwoosshh/GeulPU/blob/main/문서/29단계.md)). 실행기는 글 실행 파일 하나 + nvcuda.dll(리눅스는 libc.so.6 · libcuda.so.1) — 파이썬이 들지 않는다. 빌드(생성기 → 커널 소스 → PTX → 실행기)도 글 컴파일러만으로(19단계). 시험 · 재기 · 파이썬 연구 엔진(GPT-2)은 Python 3.12 + numpy. **Vulkan(31단계)**: 글 호스트의 Vulkan 판(`시험_엔진V` · `시험_7BV` · `재기_7BV` — 가져오는 것은 kernel32 · shell32 · vulkan-1.dll, 커널은 글 컴파일러의 SPIR-V 뒷단)이 NVIDIA(0.5B · 7B 다섯 형식 · 14B)와 Intel 내장 그래픽(0.5B)에서 CUDA 의 기록과 바이트까지 같은 출력을 낸다. 속도는 아직 못 미친다 — 34단계에서 텐서곱의 흉내를 협력 행렬로 바꾸고(`--협력` 모듈) 뒷단 · Vulkan 층을 고쳐, 7B 가 llama.cpp Vulkan 대비 프롬프트 1.15 ~ 1.41배(31단계 4.4 ~ 6.1배), 생성 1.00 ~ 1.04배(31단계 1.09 ~ 1.24배)다 — 기록이고 아직 판정은 아니다. 실행기 · 서버의 Vulkan 판은 아직 |
| 앱 | 36 ~ 39단계 — `GeulPU.exe`(글로 쓴 관리자) + 웹 화면: 모델 찾기(Hugging Face 의 GGUF 전체를 정렬해 둘러보기) · 받기 · 더하기 · 확인, 채팅(Qwen2.5 의 ChatML · Qwen3 의 틀과 생각 켬 · 끔 · R1 · Llama 3 · Mistral 의 틀), 자기 진단 · 속도 · 교차검증, 결과 zip, 새 판 알림. **연구 패널**(38단계 — 토큰마다 확률 · 갈래 · 비교 · 점수 · 재현 캡슐, 로그확률은 계약 5.2) · 영어 화면. 릴리스 zip(1.4 MB)은 태그를 밀면 CI 가 소스에서 꾸린다(초안 — 공개는 사람이). Windows x64 만 — 쓰는 법은 위 [설치](#설치) |
| 층 나누기 | 25단계 — 한 GPU 에 들지 않는 모델을 층으로 나눠 여러 GPU 에(`--나누기 P` · `--장치 a,b,…` · `--예산 GB`). 이 PC 에는 GPU 가 한 장이라 한 장 안의 여러 쿠다 문맥으로 흉내 냈다: 나눠 돌린 길 16 개가 한 조각과 바이트까지 같고, 7B 두 조각의 생성은 한 조각의 1.025 ~ 1.029 배(조각마다 문맥 — 그 차이의 대부분은 한 GPU 에서 문맥을 오가는 흉내의 값) · 1.002 ~ 1.003 배(문맥 하나). 14B 두 조각은 12 GB 한 장에 문맥 8192 로 들지 않아 그 칸을 재지 못했다 — 판정 25c 는 통과하지 못함([25단계](https://github.com/wwoosshh/GeulPU/blob/main/문서/25단계.md)) |
| 아직 없는 것 | 실제 여러 GPU 에서의 층 나누기(이 PC 에서는 흉내뿐), 계약의 손실을 7B 이상의 실제 엔진 출력으로 fp64 참값과 견주기(실제로 잰 것은 0.5B — 7B · 14B 는 반올림 자리만 흉내 낸 fp64 계산으로 검토했다: llama.cpp 식의 1/3805 · 1/4467, [계약손실검토](https://github.com/wwoosshh/GeulPU/blob/main/문서/계약손실검토.md)), k-양자화의 Q2_K · Q3_K · IQ 계열 · Q5_0(실제 72B 의 k-양자화 파일은 ffn_down 이 Q5_0 · Q8_0 이라 아직 돌지 않는다), MoE, 네이티브 리눅스(실제 리눅스 컴퓨터)에서의 GPU 실행(WSL2 에서만 시험), 애플 GPU 에서의 실행과 재현(NVIDIA 세 세대와 AMD RX 9060 XT 가 비트까지 같다 — [교차검증](교차검증/), Intel 내장 그래픽은 Vulkan 으로 0.5B 가 같다 — [31단계](https://github.com/wwoosshh/GeulPU/blob/main/문서/31단계.md)), AMD 에서의 속도(RX 9060 XT 의 R1 생성이 RTX 4070 Ti 의 3 배 남짓 — 어텐션이 느리다), 공유 메모리 32 KB 장치(윈도의 AMD)의 k-양자화, 질의 폭이 너비와 다른 모델(Qwen3-0.6B · 4B · 32B · Mistral NeMo), 머리 64 의 작은 llama(Llama-3.2-1B), Gemma · Phi 등 다른 구조, 서버의 tools · 벌점 · 인증 |

### 구조

| 폴더 | 무엇 |
|---|---|
| [엔진/](엔진/) | 글(geul) 언어 엔진 — `gguf.gl`(GGUF 읽기) · `가중치.gl` · `가중치K.gl`(가중치 깔기, k-양자화) · `로프.gl`(RoPE 표) · `토크나이저.gl`(BPE · SentencePiece · 유니코드 15.0 표) · `엔진.gl` · `엔진K.gl` · `엔진표.gl`(판 고르기 · CUDA 그래프 · 생성 · 슬롯) · `조각.gl`(층 나누기) · `뽑기.gl` · `로그확률.gl`(계약 5 · 5.2) · `서버.gl` · `서버V.gl` · `서버몸.gl`(OpenAI 꼴 HTTP 서버 — CUDA · Vulkan) · `장치C.gl` · `장치V.gl`(장치 확인) · `진단.gl` · `진단V.gl` · `진단몸.gl`(자기 진단) · `실행기.gl`(터미널 실행기) · `쿠다*.gl` · `벌칸*.gl`(GPU 층). `리눅스/` 는 리눅스 대상의 덧판 |
| [커널/](커널/) | `커널생성.gl` · `생성*.gl` — 모델의 구조 · 모양 · 형식 · 장치마다 커널 모듈(글 소스)을 내는 생성기. `진단*.gl` — 진단 커널 |
| [앱/](앱/) | `관리자.gl`(GeulPU.exe)과 그 모듈(웹 · 받기 · 벤치 · 모델보기 · 상태 · 일 · 묶기 · 윈도 · 위 · 판), `web/`(화면 — 바깥 자원 없는 HTML · JS · CSS, 한국어 · 영어), `data/models.json`(받을 모델 목록), `꾸리기.ps1`(릴리스 꾸리기), `읽어보기.txt`(zip 의 README), `DESIGN.md`(디자인 시스템) |
| [계약/](계약/) | [계약.md](계약/계약.md) · [contract.md](계약/contract.md) — 정밀도 계약. `정수계약.py` — 계약의 CPU 참조 구현(모든 비트 대조의 기준), `로그확률.py` — 로그확률(계약 5.2)의 파이썬판, `gguf읽기.py` |
| [도구/캡슐확인.py](도구/캡슐확인.py) | 연구 패널의 재현 캡슐을 GPU 없이 CPU 참조로 확인한다 |
| [시험/](시험/) | `빠른점검.py`(커밋마다의 CI), `계약자체.py`(계약의 CPU 참조 자체 시험), `참조7B.py`(7B 의 CPU 참조 — 끝에서 끝 비트 대조의 기준), `판23.py` · `판26.py` · `머리128.py` · `줄128.py`(커널 판 하나하나 대 CPU 참조 — GPU 에서, `쿠다드라이버.py`), `품질.py`(정밀 모드 대비 숨은 손실을 llama.cpp 와 견줌), `뽑기.py` · `로그확률38.py`(뽑기 · 로그확률 계약의 파이썬판 대 글판), `서버28.py` · `로그확률서버38.py`(서버의 비트 · API · 로그확률 회귀 시험), `모듈해시.txt` · `SASS넘침.txt` · `글저장소커밋.txt` |
| [재기/](재기/) | 속도 · 품질 재기 — `견주기2.sh`(정밀 · 빠른 · llama.cpp 의 프롬프트 시간, 바퀴마다 차례를 바꿔) · `견주기요약.py`, `생성2.sh`(생성 ms/토큰), `품질2.sh`, `라마.py`(llama.cpp C API). 글 쪽 재기 · 시험 프로그램은 `엔진/` 의 `시험_7B.gl` · `재기_7B.gl` · `재기_프롬프트.gl` · `재기_커널.gl` · `퍼플렉시티.gl` |
| [교차검증/](교차검증/) | 앱의 재현성 검증(벤치마크 → 교차검증)의 기준 — `expected.json`(기준 PC 의 출력 · 빌드 해시) · `requests.json`(문제 32 개). 꾸리기가 빌드 해시를 여기와 견준다 |
| [문서/](문서/) | [이관범위.md](문서/이관범위.md) · [바뀐것.md](문서/바뀐것.md) · 단계 기록([2단계.md](문서/2단계.md) — 빠른 계약 F1 · [2b단계.md](문서/2b단계.md) — 빠른 어텐션 계획 · [2c단계.md](문서/2c단계.md) — 행렬곱 판 · [3단계.md](문서/3단계.md) — 추측 디코딩 · [4단계.md](문서/4단계.md) — 긴 프롬프트의 어텐션 판 · [4b단계.md](문서/4b단계.md) — 긴 문맥 생성의 접기 고침), [긴문맥재기.md](문서/긴문맥재기.md)(긴 세션의 프롬프트 · 생성 — llama.cpp 와) |
| [docs/](docs/) | 영어 문서 — [research-guide.md](docs/research-guide.md)(연구 패널 안내서) · [glossary.md](docs/glossary.md)(글 코드 · 한국어 문서를 읽는 용어집) · `img/`(화면 그림) |
| [연구자안내.md](연구자안내.md) | 연구 패널 안내서(한국어) — 화면 그림과 함께 따라 하는 실험 일곱 |

### 소스에서 빌드하기

1. **준비물**: Windows(또는 Linux x86-64 — 아래 4), NVIDIA 드라이버(Ampere 이상) 또는 Vulkan 드라이버. 앱 · 서버는 이것과 글 컴파일러로
   충분하다. 시험은 Python 3.12 + `pip install -r requirements.txt`(numpy).
2. **글 컴파일러와 뒷단**: 글 저장소의 `research/gpu` 가지를 이 저장소 옆 `../geul-lang-v2` 에 받아 고정 커밋([시험/글저장소커밋.txt](시험/글저장소커밋.txt)
   — CI · 꾸리기 · 교차검증 기준이 모두 이 커밋이다)으로 맞추고, 자체 호스팅 컴파일러와 PTX · SPIR-V 뒷단을 만든다.
   ```
   git clone -b research/gpu https://github.com/wwoosshh/geul-lang ../geul-lang-v2
   cd ../geul-lang-v2
   git checkout $(tr -d '\r\n ' < ../GeulPUv2/시험/글저장소커밋.txt)
   python build.py selfhost 컴파일러
   GEUL_ROOT=. build/self_컴파일러.exe research/gpu/자체PTX/PTX.gl -o build/자체PTX.exe
   GEUL_ROOT=. build/self_컴파일러.exe research/gpu/자체SPIRV/SPIRV.gl -o build/자체SPIRV.exe
   ```
3. **앱 꾸리기**(릴리스와 같은 것): `앱\꾸리기.ps1 -GeulRoot ..\geul-lang-v2 [-GeulC ..\geul-lang-v2\build\self_컴파일러.exe] -NoZip` 이
   `빌드\앱\GeulPU-<판>\` 에 `GeulPU.exe` · `bin\`(server-cuda · server-vulkan · device-cuda · device-vulkan · diag-cuda · diag-vulkan ·
   kernel-gen · ptx-backend · spirv-backend) · `kernels\` · `web\` · `data\` 를 만들고, 서버 · 생성기 · 뒷단의 SHA-256 을 교차검증 기준과
   견준다. 모델의 커널 모듈은 앱이 처음 그 모델을 올릴 때 이 GPU 에 맞춰 만든다(자료 폴더의 `모듈\<판>\`).

   **서버만 띄우기**: 생성기와 서버를 빌드하고, 모델의 모양 · 형식에 맞는 모듈을 만들어 PTX 로 바꿔 서버 옆에 두고 띄운다(모듈이 없으면
   서버가 만들 명령을 알린다). 터미널 실행기는 `엔진/실행기.gl`(`빌드/글/실행기.exe 모델.gguf` — 같은 모듈).
   ```
   mkdir -p 빌드/글
   GEUL_ROOT=../geul-lang-v2 ../geul-lang-v2/build/self_컴파일러.exe 커널/커널생성.gl -o 빌드/글/커널생성.exe
   GEUL_ROOT=../geul-lang-v2 ../geul-lang-v2/build/self_컴파일러.exe 엔진/서버.gl -o 빌드/글/서버.exe
   빌드/글/커널생성.exe qwen2큰 K 3584 28 4 18944 152064 빌드/글/qwen2K_3584x28x4x18944x152064.gl     # Qwen2.5-7B 의 k-양자화 모듈
   ../geul-lang-v2/build/자체PTX.exe 빌드/글/qwen2K_3584x28x4x18944x152064.gl -o 빌드/글/qwen2K_3584x28x4x18944x152064.ptx
   빌드/글/서버.exe 모델.gguf --슬롯 4 --문맥 4096 --포트 8080
   ```
4. **리눅스**: 같은 소스를 글 컴파일러의 리눅스 대상으로 빌드한다(ELF — 가져오는 것은 `libc.so.6` · `libcuda.so.1` 뿐). WSL2 Ubuntu 24.04
   에서 시험했고, 네이티브 리눅스에서 GPU 로는 아직 돌려 보지 않았다.

### 점검 — 커밋마다의 빠른 점검(CI)

[![빠른 점검](https://github.com/wwoosshh/GeulPUv2/actions/workflows/ci.yml/badge.svg)](https://github.com/wwoosshh/GeulPUv2/actions/workflows/ci.yml)

- **빠른 점검**(커밋마다, GPU · 모델 파일 없이 — GitHub Actions 의 Windows · Ubuntu 러너, 5 분 안팎, [시험/빠른점검.py](시험/빠른점검.py)).
  글 컴파일러는 CPU 프로그램이라 커널 소스 → PTX · SPIR-V 까지는 GPU 없이 만든다: 글 저장소의 고정 커밋에서 글 컴파일러와 뒷단 둘을
  빌드하고, 생성기와 엔진 프로그램 모두를 빌드하고, 모듈 열여덟(Qwen2 · Qwen3 · Llama 의 모양 · 형식 · 공유32 · 자리하나)을 처음부터
  만들어 소스 · PTX · SPIR-V 를 [시험/모듈해시.txt](시험/모듈해시.txt) 와 바이트까지 견준다(커널이 뜻하지 않게 바뀌면 걸린다 — 일부러
  바꾼 커밋은 `--해시갱신`). NVIDIA ptxas(검사에만 — 엔진은 쓰지 않는다)로 SASS(sm_89)까지 만들어 커널마다 레지스터 · 넘침(spill)을
  적는다([시험/SASS넘침.txt](시험/SASS넘침.txt)). 그 밖에 계약의 CPU 참조 자체 시험([시험/계약자체.py](시험/계약자체.py)), 판 표가
  생성기 · 엔진에서 같은가, 실행기 · 서버 · 관리자의 가져오기 표(DLL), 파이썬 문법. 이 PC 에서도 같은 것을 돈다:
  `python -X utf8 시험/빠른점검.py --글 ../geul-lang-v2`.
- **GPU 시험**(손으로): 끝에서 끝의 비트 대조는 계약의 CPU 참조([시험/참조7B.py](시험/참조7B.py)), 서버의 회귀 시험은
  [시험/서버28.py](시험/서버28.py) · [시험/로그확률서버38.py](시험/로그확률서버38.py). 연구 단계마다 돌린 공식 실행 스크립트는
  [GeulPU 의 공식실행/](https://github.com/wwoosshh/GeulPU/tree/main/공식실행)에 있다.

### 연구 기록

이 엔진은 2026-10-08 ~ 10-18 의 연구(0 ~ 40단계)에서 나왔다. 단계마다 판정 기준을 실험보다 먼저 적고 모든 변형을 비트 대조한 뒤에
시간을 잰 기록은 GeulPU 저장소에 있다 — [단계 기록](https://github.com/wwoosshh/GeulPU/blob/main/문서/단계기록.md) ·
[영어 요약](https://github.com/wwoosshh/GeulPU/blob/main/docs/research-log.md) · [논문 초안](https://github.com/wwoosshh/GeulPU/tree/main/docs/paper).
연구는 2026-10-16 에 동결했다([방향 전환](https://github.com/wwoosshh/GeulPU/blob/main/문서/방향전환.md)). 이 저장소는 그 엔진을 실사용으로 이어 간다.

## 라이선스

MIT — [LICENSE](LICENSE).
