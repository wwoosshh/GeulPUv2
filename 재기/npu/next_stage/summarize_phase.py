import json
from pathlib import Path
from statistics import mean

here=Path(__file__).resolve().parent
d=json.loads((here/'results/phase-bench.json').read_text())
assert len(d['runs'])>=5
lines=['# GeulPU 단계별 CPU/NPU 비교','',
       'Ryzen AI 7 350에서 Qwen3-0.6B-Q8_0의 가중치, 입력, KV 캐시 형식, FP32 후처리 계약을 고정했다. CPU 경로는 네이티브 AVX2 4워커다. 단계 분담 경로는 프롬프트 선형층 전체(출력층 포함)에 4행 GL NPU 커널과 CPU 후처리를 사용하고 이후 생성에는 CPU를 쓴다. 어텐션과 KV는 처음부터 호스트에 있어 전환 시 KV 복사는 없다. 원래 컨텍스트 기본값은 256이며 이번 시험만 257로 설정해 256토큰 입력 뒤 한 단계를 추가했다.','',
       '| 입력 토큰 | CPU 첫 출력(초) | 분담 첫 출력(초) | 분담/CPU 시간 | CPU 다음 토큰(초) | 분담 다음 토큰(초) |',
       '|---:|---:|---:|---:|---:|---:|']
for n in (16,64):
    cpu,phase=[next(r for r in d['runs'] if r['length']==n and r['backend']==b) for b in ('cpu','phase')]
    assert all(cpu[k]==phase[k] for k in ('prefill_hash','decode_hash','first_token','second_token'))
    lines.append(f"| {n} | {cpu['prefill']:.3f} | {phase['prefill']:.3f} | {phase['prefill']/cpu['prefill']:.2f} | {cpu['next_token']:.3f} | {phase['next_token']:.3f} |")
lines+=['','각 입력 길이에서 첫 출력과 전환 직후 다음 출력의 전체 FP32 로짓 배열을 uint32로 직접 비교했다. 서로 다른 백엔드 간 16·64토큰에서 4쌍 모두 비트 동일. 출력은 고정 난수 토큰 입력에 대한 greedy 결과다. 모델 로딩 및 워밍업 제외. 전체 모델은 조건별 1회이며 p95나 안정적인 작은 차이를 추정할 표본 수는 아니다. 64토큰에서는 실행 순서를 역전했다. 16토큰 결과는 이전 세션에서 완료했고, 64·256토큰은 재개 세션에서 측정했다. 기존 256토큰 NPU 전체 실행은 의도적으로 중단하고 묶음 전송 개선 경로로 대체했다. 256토큰 CPU 기준은 첫 출력 125.964초, 다음 토큰 8.233초였다.','',
        '| 실제 첫 층 FFN up projection | CPU(ms) | NPU+CPU(ms) |','|---:|---:|---:|']
for n in (16,64,256):
    cpu,npu=[mean(r['seconds'] for r in d['linear'] if r['length']==n and r['backend']==b)*1000 for b in ('cpu','npu')]
    lines.append(f'| {n}행 | {cpu:.1f} | {npu:.1f} |')
lines+=['','선형층은 워밍 후 CPU/NPU/NPU/CPU 순서로 각 두 번 재고 모두 비트를 비교했다. 단일 작업자·4행 단위 재전송 구조에 한정한 관측이다. [실행 코드](phase_bench.py) · [원자료 및 시간 분해](results/phase-bench.json)']
(here/'PHASE-BENCH.md').write_text('\n'.join(lines)+'\n',encoding='utf8')

joined_path=here/'results/joined-prefill.json'
paired_path=here/'results/joined-paired.json'
if joined_path.exists() and paired_path.exists():
    joined=json.loads(joined_path.read_text())
    paired=json.loads(paired_path.read_text())
    full=next(r for r in joined if 'full_length' in r)
    cpu_full=next(r for r in d['runs'] if r['backend']=='cpu' and r['length']==256)
    assert (full['first_hash'],full['next_hash'])==(cpu_full['prefill_hash'],cpu_full['decode_hash'])
    times={key:mean(r['seconds'] for r in paired if r['backend']==key) for key in ('cpu','old','new')}
    extra=[
      '',
      '## 묶음 전송 개선',
      '',
      '4행마다 따로 제출하던 GL 타일을 다음 행 묶음과 연결해 프레임을 채운다. 정수 내적과 블록 누산 순서는 유지하며, 호스트의 타일 배치와 결과 복원 순서를 변경했다. 입력 256행, 출력 3072열, 깊이 1024인 실제 FFN up projection에서 기존 형상의 제출 횟수는 768회이고 새 경로의 실제 제출 횟수는 96회다. 전송 프레임은 1,572,864개에서 196,608개로 감소한다. 여기서 프레임 수는 타일 단위이며 제출 한 번은 2048프레임이다. 새 경로의 해당 형상에서는 패딩 프레임이 없다.',
      '',
      '| 같은 세션의 교차 측정 | 평균 시간(초) |',
      '|---|---:|',
      f"| CPU | {times['cpu']:.3f} |",
      f"| 기존 NPU + CPU | {times['old']:.3f} |",
      f"| 묶음 전송 NPU + CPU | {times['new']:.3f} |",
      '',
      f"워밍업 제외 CPU/기존/새경로/새경로/기존/CPU 순서로 각 두 번 측정했다. 기존 대비 {times['old']/times['new']:.2f}배이며 CPU 대비 걸린 시간은 {times['new']/times['cpu']:.2f}배다. 모두 출력 uint32 배열 일치. 묶음 전송 후처리는 순차 실행하며 기존 경로에는 CPU 후처리 중첩이 켜져 있다.",
      '',
      '| 256토큰 전체 모델 | 첫 출력(초) | 다음 출력(초) |',
      '|---|---:|---:|',
      f"| CPU | {cpu_full['prefill']:.3f} | {cpu_full['next_token']:.3f} |",
      f"| 묶음 전송 프롬프트 + CPU 생성 | {full['prefill']:.3f} | {full['next_token']:.3f} |",
      '',
      '전체 모델은 각각 1회 측정이며 같은 난수 토큰 입력, 동일 Q8_0 가중치를 사용했다. 출력층의 1행 계산도 CPU에 배정했다. 두 단계 전체 로짓 SHA-256이 CPU 기준과 일치했다. 16/64/256행 및 5행·55열의 비정렬 꼬리 장치 시험 12회도 CPU와 비트 일치했다. 추가로 CPU 타일 에뮬레이터에서 5×55×96, 16×95×160, 33×130×64를 검사했다.',
      '',
      '검증한 단계 전환은 호스트 KV를 유지하므로 장치 간 KV 복사 비용이 없다. 64토큰 초과에서는 기존 CPU 어텐션 경로를 사용하므로 양자화 KV 재사용 최적화의 이점을 주장하지 않는다. 비교한 CPU와 NPU 모두 한 제품의 성능 한계를 대표하지 않는다. 이번 결과로 확인한 것은 현재 실행 경로의 비용과 변경 효과다.',
      '',
      '적용 판단은 CPU를 기본 실행 경로로 두고 묶음 전송은 연구용 선택 경로로 유지하는 것이다. 이 데이터에서는 입력 길이에 따른 NPU 우위 전환점이 관측되지 않았다. 추가 NPU 개발의 우선순위는 타일 내부 성능과 다중 작업자 확장, 가중치 재사용을 검증하는 것이다. 장치 이용률과 소비 전력을 측정하지 않았으므로 하드웨어 전체의 한계나 에너지 효율을 결론내리지 않는다.',
      '',
      '[묶음 전송 구현](joined_prefill.py) · [장치 및 모델 원자료](results/joined-prefill.json) · [동일 세션 교차 원자료](results/joined-paired.json)',
      '',
      '재현 명령은 PowerShell에서 ./재기/npu/next_stage/viability.ps1 phase_bench, bench_joined, paired_joined 순서다. phase_bench는 저장된 결과를 재사용한다. 별도 새 측정을 원하면 기존 결과를 다른 이름으로 보관한 뒤 시작해야 한다. 보고서 생성은 summarize_phase.py를 사용한다.'
    ]
    with (here/'PHASE-BENCH.md').open('a',encoding='utf8') as f:
        f.write('\n'.join(extra)+'\n')
