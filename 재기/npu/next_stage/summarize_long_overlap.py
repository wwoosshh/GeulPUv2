import json
import hashlib
from pathlib import Path
from statistics import median
from collections import defaultdict

here = Path(__file__).resolve().parent
d = json.loads((here/'results/long-overlap.json').read_text())
assert d['complete'] and len(d['runs']) == 12
lines = ['# 64·256토큰 모델 중첩·재사용 비교', '',
         'Qwen3-0.6B-Q8_0, NPU 작업자 4개와 CPU 후처리 스레드 4개를 고정했다. 기본과 결합 경로를 A/B/B/A/A/B 순서로 길이별 3회씩 실행했다. 모델 로딩은 제외하고 공통 3토큰 워밍업 후 측정했다. 출력층과 다음 토큰은 공통 CPU 경로다. 모든 실행의 첫·다음 출력 전체 로짓 SHA256을 기존 동일 입력 CPU 기준과 비교했다.', '',
         '| 입력 토큰 | 경로 | 첫 출력 중앙값 초 | 최소–최대 초 | 다음 출력 중앙값 초 |', '|---:|---|---:|---:|---:|']
for length in (64, 256):
    for mode in ('baseline', 'combined'):
        rows = [r for r in d['runs'] if r['length']==length and r['mode']==mode]
        vals = [r['prefill'] for r in rows]
        lines.append(f"| {length} | {mode} | {median(vals):.3f} | {min(vals):.3f}–{max(vals):.3f} | {median(r['decode'] for r in rows):.3f} |")
lines += ['', '## 절전으로 중단된 표본', '',
          '마지막 256토큰 combined 실행은 28,798.402초(약 8시간)로 기록됐다. Windows Kernel-Power 이벤트에서 2026-11-02 06:11:04 Idle Timeout에 따른 Modern Standby 진입, 06:28:04 대기 전환, 13:47:25 덮개 관련 전환, 14:08:28 덮개 관련 복귀가 확인됐다(한국 시간). 층 11 ffn_up 1,020.501초, ffn_down 26,361.184초, 층 12 qkv 1,263.167초에 지연이 집중돼 있다. 전원 상태 전환에 영향을 받은 마지막 표본은 정상 실행 성능 집계에서 제외한다. 원자료와 위 전체 표에는 그대로 보존하며 출력 정확성 검사에는 포함한다. 절전 시간을 단순 차감하여 복원하지 않는다. 재측정하지 않았다.', '',
          '| 토큰 | 정상 기본 표본 | 정상 결합 표본 | 기본 중앙값 초 | 결합 중앙값 초 |', '|---:|---:|---:|---:|---:|']
assert d['runs'][-1]['length'] == 256 and d['runs'][-1]['mode'] == 'combined' and d['runs'][-1]['prefill'] > 28000
d['runs'] = d['runs'][:-1]
for length in (64,256):
    base = [r['prefill'] for r in d['runs'] if r['length']==length and r['mode']=='baseline']
    combined = [r['prefill'] for r in d['runs'] if r['length']==length and r['mode']=='combined']
    lines.append(f'| {length} | {len(base)} | {len(combined)} | {median(base):.3f} | {median(combined):.3f} |')
lines += ['', '이하 성능 집계는 정상 실행 11개만 사용한다. 256토큰 결합 경로는 2표본뿐이므로 반복 안정성 판단은 제한적이다.', '', '## 선형 연산별 집계', '',
          '각 실행에서 같은 종류의 모든 층 시간을 합한 뒤 실행 간 중앙값을 계산했다. 각 층의 개별 M/N/K와 시간은 원자료에 저장했다. 장치 시간에는 전송과 재사용 경로의 내부 패킹이 포함된다. 후처리 작업 시간은 장치 시간과 겹치므로 두 값을 합하면 안 된다. 대기는 호출자에서 관찰한 후처리 제출·대기 시간이다.', '',
          '| 토큰 | 경로 | 연산 | M/N/K | 전체 ms | 장치 ms | 후처리 작업 ms | 제출·대기 ms | 호스트 비대기 ms |', '|---:|---|---|---|---:|---:|---:|---:|---:|']
for length in (64, 256):
    for mode in ('baseline', 'combined'):
        groups = defaultdict(list)
        shapes = {}
        for run in d['runs']:
            if run['length'] != length or run['mode'] != mode:
                continue
            totals = defaultdict(lambda: [0., 0., 0., 0., 0.])
            for r in run['linears']:
                name = r['operation']
                shapes[name] = f"{r['m']}/{r['n']}/{r['k']}"
                p = r['profile'] if r['backend'] != 'cpu' else {}
                values = [r['seconds'], p.get('device_and_transfer_seconds', 0), p.get('finalization_seconds', 0), p.get('post_wait_seconds', 0), p.get('packing_seconds', 0)]
                totals[name] = [a+b for a,b in zip(totals[name], values)]
            for name, values in totals.items():
                groups[name].append(values)
        for name, samples in groups.items():
            values = [median(s[i] for s in samples)*1000 for i in range(5)]
            lines.append(f'| {length} | {mode} | {name} | {shapes[name]} | '+' | '.join(f'{v:.1f}' for v in values)+' |')
lines += ['', '## 모델 시간 분해', '', '| 토큰 | 경로 | 선형층 초 | 어텐션 초 | 나머지 초 |', '|---:|---|---:|---:|---:|']
for length in (64,256):
    for mode in ('baseline','combined'):
        runs = [r for r in d['runs'] if r['length']==length and r['mode']==mode]
        values = [median(r['profile']['linear_seconds'] for r in runs), median(r['profile']['attention_seconds'] for r in runs), median(r['prefill']-r['profile']['linear_seconds']-r['profile']['attention_seconds'] for r in runs)]
        lines.append(f'| {length} | {mode} | '+' | '.join(f'{v:.3f}' for v in values)+' |')
lines += ['', '## 적용 판단', '']
for length in (64,256):
    base = median(r['prefill'] for r in d['runs'] if r['length']==length and r['mode']=='baseline')
    new = median(r['prefill'] for r in d['runs'] if r['length']==length and r['mode']=='combined')
    lines.append(f'- {length}토큰: 기본 대비 {base/new:.3f}배 속도, 시간 {(1-new/base)*100:.2f}% 단축.')
lines += ['', '12회 모델 실행의 24개 로짓 해시가 CPU 기준과 일치했다. 이번 비교는 동일 NPU 경로 간 최적화 비교이며 CPU 단독보다 빠름을 입증하는 시험은 아니다. 조건별 정상 2~3표본으로 다른 모델·입력 전체나 전력 효율을 일반화하지 않는다. 기본 실행 경로는 변경하지 않았다.', '',
          '이번 Qwen 모델의 64·256행 선형층에는 결합 경로를 적용 후보로 삼을 근거가 있다. 16행은 앞선 시험에서 전체 효과가 불명확했고 1행은 기존 CPU 경로를 유지한다. 64라는 보편적 경계가 증명된 것은 아니다. 다음 구현 우선순위는 장치 제출 묶기 및 부분합 반환량 축소다. 두 변경은 기존 ordered FMA 순서를 보존하도록 설계해야 한다. CPU 어텐션과 정규화 등 나머지 비용도 전체 속도의 한계로 남는다.', '',
          '재현: `viability.ps1 bench_long_overlap` 실행 후 `summarize_long_overlap.py`. [전체 원자료](results/long-overlap.json).']
lines += ['', '호스트 비대기 열은 기존 packing_seconds 필드다. 재사용 경로의 내부 패킹은 장치 시간에 포함되므로, 이 열만으로 두 경로의 전체 패킹 비용을 비교할 수 없다. 원자료는 반복·층별 수치를 보존한다.']
(here/'LONG-OVERLAP.md').write_text('\n'.join(lines)+'\n', encoding='utf8', newline='\n')
sources = ('bench_long_overlap.py', 'joined_prefill.py', 'reuse_device.py', 'multi_device.py', 'parallel_post.py', 'cached_qwen.py')
(here/'results/long-overlap-sources.json').write_text(json.dumps({name:hashlib.sha256((here/name).read_bytes()).hexdigest() for name in sources}, indent=2), encoding='utf8', newline='\n')
print('\n'.join(lines[:12]+lines[-8:]))
