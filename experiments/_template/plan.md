# Plan — Short title of the experiment

Write this file and the `criteria` in `experiment.json` before any run, and commit them first. The git history then shows that the
criteria came before the results.
(이 파일과 `experiment.json` 의 `criteria` 를 어떤 실행보다 먼저 쓰고 먼저 커밋한다 — git 이력이 기준이 결과보다 먼저였음을 보여 준다.)

## Question (연구 질문)

One question, the same as `question` in `experiment.json`.

## Hypotheses (가설)

What you expect, and why.

## Method (방법)

Models (file, SHA-256, source), GeulPU version and computation mode, the runs and their settings, and the GeulPU operations they use
(for example `run.chat`, `run.branch`, `run.lens` — `GeulPU ops` lists them all).

## Criteria (판정 기준)

The same list as `criteria` in `experiment.json`: what has to hold, with numbers where possible.

## Not done (하지 않는 것)

What this experiment does not try to show.
