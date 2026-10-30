# Privacy policy

*한국어는 아래에 있습니다.*

GeulPU runs on your own computer. It has no accounts, no usage statistics and no crash reporting. This page lists everything it
sends over the network and everything it stores.

## What leaves your computer

| When | Where | What is sent |
|---|---|---|
| **Find models** (the app's model browser) | `huggingface.co` — your browser calls the Hugging Face API directly | your search text, sort and filter choices, and which repository you open |
| **Downloading a model** | `huggingface.co` (the app) | the repository and file name; a few MB of the file's header first, to check that it runs |
| **New-version notice** (on by default, switch it off in Settings) | `github.com` (the app), `pypi.org` for installs made with pip, pipx or uv | the request for the latest version number — nothing about you |
| **Coding agents started from the Agents page** | whatever the agent itself contacts | GeulPU only passes the agent the local server address; the agent's own network use is governed by the agent's policy |

Nothing else is sent. The manager and the inference server listen only on `127.0.0.1`, refuse requests from web pages and from other
host names, and other devices cannot connect unless you forward the port yourself.

## What stays on your computer

Everything below is in the data folder (`%USERPROFILE%\GeulPU` on Windows, `~/GeulPU` on Linux, or the folder given with
`--data` / `GEULPU_HOME`) and is yours to delete:

- model files and the kernel modules built for your GPU;
- the prompt cache on disk (the context store), which holds the KV cache of your recent prompts;
- logs of the app and the server (request counts and timings; no message text, except the error texts the server prints);
- debugger run logs and reproduction files — **these contain your prompts and the answers**, and are written only while you record
  or export them;
- settings. Conversations of the chat screen stay in your browser's storage.

Benchmark results and reproducibility-check zips are created only when you run them, and are shared only if you upload them.

## Changes

This policy changes only through a pull request to this repository, so its history is public.

---

# 개인정보 처리 방침

GeulPU는 사용자의 컴퓨터에서 돕니다. 계정도, 사용 통계도, 오류 보고도 없습니다. 이 문서는 망으로 보내는 것과 저장하는 것 전부를 적습니다.

## 컴퓨터 밖으로 나가는 것

| 언제 | 어디로 | 무엇을 |
|---|---|---|
| **모델 찾기**(앱의 모델 둘러보기) | `huggingface.co` — 브라우저가 Hugging Face API를 직접 부릅니다 | 검색어, 정렬 · 거르기 선택, 연 저장소 |
| **모델 받기** | `huggingface.co`(앱) | 저장소와 파일 이름. 먼저 파일 앞 몇 MB를 받아 실행되는지 확인합니다 |
| **새 판 알림**(기본 켬, 설정에서 끔) | `github.com`(앱), pip · pipx · uv로 설치했으면 `pypi.org` | 최신 판 번호를 묻는 요청 — 사용자에 관한 것은 없습니다 |
| **에이전트 페이지에서 띄운 코딩 에이전트** | 그 에이전트가 쓰는 곳 | GeulPU는 로컬 서버 주소만 넘깁니다. 에이전트의 망 사용은 그 에이전트의 정책을 따릅니다 |

그 밖에는 아무것도 보내지 않습니다. 관리자와 추론 서버는 `127.0.0.1`에서만 듣고, 웹 페이지와 다른 호스트 이름의 요청은 거부하며, 사용자가
포트를 직접 열어 주지 않는 한 다른 기기는 접속할 수 없습니다.

## 컴퓨터에 남는 것

아래는 모두 데이터 폴더(Windows `%USERPROFILE%\GeulPU`, 리눅스 `~/GeulPU`, 또는 `--자료` · `GEULPU_HOME`으로 준 폴더)에 있고 언제든 지울 수
있습니다:

- 모델 파일과 이 GPU용으로 만든 커널 모듈;
- 디스크의 프롬프트 캐시(문맥 저장소) — 최근 프롬프트의 KV 캐시;
- 앱과 서버의 기록(요청 수와 시간; 메시지 본문은 없고, 서버가 찍는 오류 글만);
- 디버거의 실행 기록과 재현 파일 — **프롬프트와 답이 들어 있으며**, 기록을 켜거나 내보낼 때만 쓰입니다;
- 설정. 채팅 화면의 대화는 브라우저 저장소에 남습니다.

벤치마크 결과와 재현성 검증 zip은 사용자가 돌릴 때만 만들어지고, 올려야만 공유됩니다.

## 바뀔 때

이 방침은 이 저장소의 풀 리퀘스트로만 바뀌므로 이력이 공개됩니다.
