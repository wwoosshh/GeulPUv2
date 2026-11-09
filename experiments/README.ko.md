# 실험 기록

여기 있는 폴더 하나가 GeulPU 로 한 실험 하나다 — 연구 질문, 실행 전에 적은 판정 기준, 모델(SHA-256 과 출처), 실행, 그리고 다시
실행하는 데 필요한 기록. GeulPU 는 언제 실행해도 같은 비트를 계산하므로, 여기 기록한 실험은 다시 실행해 같은 토큰과 로짓을 내야 한다.

[English](README.md) · [실험 목록](index.md) · [형식(JSON Schema)](experiment.schema.json) · [템플릿](_template/)

## 실험 하나 = 폴더 하나

```
experiments/<YYYY-MM-DD>-<이름>/
  experiment.json   프로그램이 읽는 기록(형식 geulpu-experiment/1)
  plan.md           연구 질문 · 가설 · 방법 · 판정 기준 — 실행 전에 커밋한다
  results.md        답과, 판정 기준마다의 판정
  records/          다시 실행하는 데 필요한 것 — 실행 로그(.jsonl.gz) · 영수증(.json.gz) · 재현 파일
  expected.json     실행마다 다시 내야 할 것: {"runs": {"<실행 id>": {...}}}
  scripts/          쓴 스크립트(GeulPU 의 기능이 되기 전까지)
```

꼭 있어야 하는 것은 `experiment.json` 과 `plan.md` 이고, 상태가 `done` 이면 `results.md` 도 있어야 한다.

## experiment.json

| 필드 | 뜻 |
|---|---|
| `format` | `geulpu-experiment/1` |
| `id` | 폴더 이름, `YYYY-MM-DD-이름` |
| `title` · `question` | 짧은 제목과, 실험이 답하는 연구 질문 하나 |
| `status` | `planned` · `running` · `done` |
| `created` · `authors` | 날짜(`YYYY-MM-DD`)와 저자 |
| `geulpu` | `version`(필수) · `commit` · `mode`(`exact` · `fast`) · `provenance`(파이썬 API `info(provenance=True)` 가 돌려주는 것) |
| `devices` · `engines` | GPU 와 드라이버; 비교하거나 재현하는 실험이면 다른 엔진(예: llama.cpp 빌드) |
| `models` | 모델마다 이름 · SHA-256 · 출처(`{"hf_repo", "revision", "file"}` 또는 만든 방법) |
| `runs` | 실행마다 `id` · `kind` · GeulPU 의 기능(`operation`) · 기록 파일(`record`) · `settings` · `seed`. 분기면 원래 실행(`base`)과 개입(`intervention` — `type` · `step` · `value`), 점수 · 렌즈면 원래 실행(스텝부터 다른 토큰을 넣었으면 개입도) |
| `criteria` | 하나 이상: `id` · `text`(실행 전에 적음) · `result`(`pending` · `pass` · `fail`) |
| `files` | 폴더 안의 파일 중 `experiment.json` · `plan.md` · `results.md` 를 뺀 모두: `path` · `sha256` · `bytes` |
| `reproduce` · `links` | 실험을 다시 실행하는 명령; 관련 실험 · 문서 · 논문 |

기준은 [experiment.schema.json](experiment.schema.json) 이고, 이 표는 그 요약이다.

## 무엇을 저장하나

- 모델 파일은 넣지 않는다. SHA-256 과 출처로 가리킨다.
- 다시 계산할 수 있는 값(스텝마다 확률 · 상위 k · 엔트로피)은 작게 둔다. 앱이 남기는 실행의 기록 파일은 GeulPU 가 준 답을 그대로 담고, 상위
  후보는 요청한 수만큼이다(실행 화면은 마진에 2 개가 필요하다) — 실험에 필요할 때만 더 묻는다. 큰 값은 해시를 넣고, 다시 실행할 때 계산해서
  비교한다.
- 실험 하나는 압축(zip)해서 5 MB 이하. 더 큰 원자료는 저장소 밖(릴리스 첨부나 다른 저장소)에 두고 SHA-256 으로 가리킨다. Git LFS 는
  쓰지 않는다.
- 판정 기준이 먼저다. `plan.md` 와 `criteria` 를 첫 실행 전에 커밋하면, git 이력이 판정 기준을 결과보다 먼저 적었음을 보여 준다.

## 실험 더하기

1. `_template/` 을 `experiments/<YYYY-MM-DD>-<이름>/` 으로 복사하고 `id` 를 폴더 이름으로 바꾼다.
2. `question` · `plan.md` · `criteria` 를 쓰고 커밋한다.
3. 실행한다. 기록은 `records/` 에 두고, 실행마다 `runs` 에 적는다.
4. `python 도구/experiment_check.py --files experiments/<폴더>` 가 `experiment.json` 에 붙일 `files` 목록을 출력한다.
5. `results.md` 를 쓰고, 판정 기준마다 `result` 와 `"status": "done"` 을 적고, [index.md](index.md) 에 한 줄을 더한다.
6. `python 도구/experiment_check.py experiments/<폴더>` 가 통과해야 한다.

GeulPU 앱이 떠 있으면 CLI(와 앱의 실험 화면)가 1 ~ 3 단계와 5 · 6 단계를 한다: `GeulPU experiment create --name <이름> --root experiments` 가
이 템플릿에서 폴더를 만들고 `index.md` 에 더한다. `GeulPU experiment update` 가 계획과 판정 기준(첫 실행 전에만), 그 뒤의 판정 · `results.md` ·
상태를 쓴다. 어느 명령에든 `--experiment <폴더>` 를 붙이면 그 호출이 `records/commands.jsonl` 에 남고, 200 으로 끝난 실행은
`records/runs/<id>.json` 과 `runs` 의 항목(종류 · 원래 실행 · 개입), `models` 의 모델, 앱이 쓴 파일의 `files` 항목으로 남는다.
`GeulPU experiment check --path <폴더> --wait` 가 이 점검 도구를 돌린다([문서/명령줄.md](../문서/명령줄.md), [문서/화면구현.md](../문서/화면구현.md)
§3 · §4). 직접 더한 파일은 4 단계를 거친다.

## 점검

```
python 도구/experiment_check.py                        # 여기 있는 모든 폴더와 index.md
python 도구/experiment_check.py experiments/<폴더>     # 폴더 하나
python 도구/experiment_check.py --self-test            # 올바른 실험의 복사본을 망가뜨려, 잘못마다 잡는지 확인
```

표준 라이브러리만 쓰고 GPU 가 없어도 된다. `experiment.json` 이 스키마에 맞는지, `id` 가 폴더 이름인지, `plan.md` 가 있는지(상태가
`done` 이면 `results.md` 도), 폴더 안의 파일이 모두 `files` 에 SHA-256 · 크기와 함께 있는지, 실행의 기록 파일이 목록에 있고 분기의 바탕이
실행인지, `expected.json` 의 실행이 `runs` 에 있는지, 폴더를 압축한 크기가 5 MB 이하인지, `index.md` 가 모든 실험 폴더를 링크하는지
확인한다. GeulPU 의 CI 가 커밋마다 돌린다.

실행을 다시 돌려 `expected.json` 과 비교하는 일은 아직 자동이 아니다.
