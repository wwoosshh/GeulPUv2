import json
from statistics import median
from pathlib import Path

here=Path(__file__).resolve().parent
data=json.loads((here/'results/parallel-post.json').read_text())
assert data['complete'] and data['checks']==111 and len(data['model'])==4
lines=['# CPU 후처리 병렬화 시험', '',
       'NPU는 4작업자로 고정하고, 출력 열을 나누어 CPU 후처리 1·2·4스레드를 비교했다. 각 출력값의 블록별 FMA 순서를 유지한다. 입력 부분합은 한 번 연속 배치로 변환하고 원래 stride로 읽으므로 작업자마다 입력 전체를 복제하지 않는다. 출력 조각만 합친다. 기존 C DLL의 엄격 FP32 후처리를 재사용했다.', '',
       '| 측정 | 1스레드 ms | 2스레드 ms | 4스레드 ms |', '|---|---:|---:|---:|']
for label,rows in [('후처리 단독 64×32×1024',data['post'])]+[(f'선형층 {m}행',[r for r in data['linear'] if r['rows']==m]) for m in (16,64,256)]:
    values=[median(r['seconds'] for r in rows if r['workers']==w)*1000 for w in (1,2,4)]
    lines.append('| '+label+' | '+' | '.join(f'{v:.2f}' for v in values)+' |')
lines+=['','후처리 단독은 6회, 선형층은 4회 교대 실행 중앙값이다. 후처리 단독의 모양은 [행, 깊이 블록, 출력 열]이다. 선형층은 실제 Qwen3-0.6B-Q8_0 FFN up 가중치(K=1024,N=3072)를 사용했다.','',
        '| 모델 후처리 스레드 | 첫 출력 초 | 다음 출력 초 |','|---:|---:|---:|']
for workers in sorted(set(r['workers'] for r in data['model'])):
    rows=[r for r in data['model'] if r['workers']==workers]
    first=median(r['prefill'] for r in rows)
    second=median(r['decode'] for r in rows)
    lines.append(f'| {workers} | {first:.3f} | {second:.3f} |')
lines+=['','모델은 고정 16토큰 입력에서 1/선택/선택/1 순서로 실행했다. 선택 스레드 수는 256행 선형층 실측으로 정했다. 1행 출력층과 다음 토큰 생성은 모든 조건에서 CPU 단독이다. 각 경로 2표본으로 모델 로딩 제외, 계획 워밍업 후 시간이다. CPU 시간 및 전력은 측정하지 않았다.', '',
        '비연속 입력, 출력 열 1·55·256·1024, 자리수 1·2, 비정규수 처리 두 모드와 실제 선형층을 포함한 111회 비교가 비트 일치했다. 4회 모델 실행의 첫·다음 출력 8개 전체 로짓 SHA-256도 기존 CPU 기준과 일치했다.', '',
        'ParallelPost는 선택 가능한 후처리 구현이다. CPU 후처리 스레드와 NPU 실행의 중첩을 동시에 바꾼 시험은 수행하지 않았다. 다음 단계에서 중첩까지 적용하면 스레드 경쟁과 메모리 사용량을 다시 확인해야 한다.', '',
        '[후처리 구현](parallel_post.py) · [원자료](results/parallel-post.json) · [시험 코드](bench_parallel_post.py)', '',
        '재현은 viability.ps1 bench_parallel_post 실행 후 summarize_parallel_post.py로 보고서를 만든다. 결과 파일은 덮어쓰므로 이전 측정은 먼저 보관한다.']
(here/'PARALLEL-POST.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
print('\n'.join(lines[4:]))
