import hashlib
import json
from statistics import median
from pathlib import Path

here=Path(__file__).resolve().parent
d=json.loads((here/'results/reuse-overlap.json').read_text())
assert d['complete'] and d['checks']==72 and len(d['model'])==4
lines=['# CPU 후처리 중첩과 NPU 가중치 재사용', '',
       'NPU 4작업자와 CPU 후처리 4스레드를 고정했다. baseline은 순차 처리, overlap은 앞 출력 청크의 후처리를 다음 청크의 NPU 작업과 중첩, reuse는 장치 가중치 재사용, combined는 둘을 결합한 경로다. 최종 출력의 FMA 블록 순서는 유지한다.', '',
       '재사용 그래프는 각 NPU 작업자가 32×32 INT8 가중치 FIFO 객체 하나를 acquire한 채 서로 다른 입력 4행 묶음 네 개에 같은 GL 커널을 호출한 뒤 release한다. 모델 전체 가중치가 장치에 상주하는 구현은 아니다. 그룹당 16입력 행 범위의 장치 내부 재사용이다. CPU 중첩은 단일 조정 스레드와 후처리 4스레드를 사용하며 보유 중인 후처리 청크를 최대 두 개로 제한한다. 각 청크의 부분합 저장소를 독립적으로 유지한다.', '',
       '| 실제 Qwen FFN 입력 행 | 기본 ms | 중첩 ms | 재사용 ms | 둘 다 ms |', '|---:|---:|---:|---:|---:|']
for m in (16,64,256):
    values=[median(r['seconds'] for r in d['linear'] if r['rows']==m and r['mode']==mode)*1000 for mode in ('baseline','overlap','reuse','combined')]
    lines.append(f'| {m} | '+' | '.join(f'{v:.2f}' for v in values)+' |')
lines+=['','같은 Qwen3-0.6B-Q8_0 첫 FFN up 가중치(K=1024,N=3072)를 사용했다. 워밍업 후 각 조건 4회, 순서와 역순을 교대로 실행한 중앙값이다. 재사용 경로에는 호스트에서 타일 순서를 바꾸는 비용이 포함된다. profile의 후처리 시간은 장치 시간과 겹치므로 합해서 전체 시간으로 해석하지 않는다. 기존 packing_seconds 필드는 호스트의 비대기 시간을 나타내며 두 장치 경로의 내부 패킹 포함 위치가 달라 직접 비교에 주의한다.', '',
        '| 256행의 명시적 버퍼 전송 | 기본 MiB | 재사용 MiB |','|---|---:|---:|']
for label,key in (('입력 A','input_a_bytes'),('가중치 B','weight_bytes'),('출력 부분합 C','output_bytes')):
    values=[next(r[key] for r in d['linear'] if r['rows']==256 and r['mode']==mode)/(1<<20) for mode in ('baseline','combined')]
    lines.append(f'| {label} | {values[0]:.0f} | {values[1]:.0f} |')
lines+=['','두 경로 모두 제출 96회다. 표는 실행 횟수와 실제 버퍼 크기로 집계한 명시적 업로드/읽기 바이트이며 하드웨어 버스 계측값은 아니다. 가중치만 75% 감소하며 A+B+C 전체는 336→192MiB로 약 42.9% 감소한다. 부분합의 CPU 반환량은 유지된다.','',
        '| 16토큰 모델 경로 | 첫 출력 초 | CPU 다음 출력 초 |','|---|---:|---:|']
for mode in dict.fromkeys(r['mode'] for r in d['model']):
    rows=[r for r in d['model'] if r['mode']==mode]
    lines.append(f"| {mode} | {median(r['prefill'] for r in rows):.3f} | {median(r['decode'] for r in rows):.3f} |")
lines+=['','256행 실측으로 선택한 경로와 기본 경로를 기본/선택/선택/기본 순서로 비교했다. 고정 16토큰 입력이며 1행 출력층과 후속 생성은 공통 CPU 경로다. 로딩 제외, 워밍업 후 경로별 2표본이다. 에너지 소비 및 CPU 사용 시간은 계측하지 않았다.', '',
        '홀수 행·열과 깊이 블록 꼬리, 1행 fallback, 반복 선형층을 포함한 72회 비교에서 CPU와 출력 비트가 일치했다. 모델 첫·다음 출력 8개 전체 로짓 해시도 기존 CPU 기준과 일치했다. 선택 가능한 연구 경로로 제공하며 장문이나 다른 모델에서의 성능은 이번 시험 범위에 포함하지 않았다.', '',
        '[원자료](results/reuse-overlap.json) · [장치 재사용 구현](reuse_device.py) · [중첩 구현](joined_prefill.py) · [시험](bench_reuse_overlap.py)', '',
        '재현은 viability.ps1 bench_reuse_overlap 실행 후 summarize_reuse_overlap.py로 보고서를 생성한다. 결과 파일은 재실행 시 덮어쓴다.']
base=[r['prefill'] for r in d['model'] if r['mode']=='baseline']
chosen=[r['prefill'] for r in d['model'] if r['mode']==d['winner']]
lines += ['', '## 판단과 다음 단계', '',
          f'모델 첫 출력의 중앙값 변화는 {(1-median(chosen)/median(base))*100:.2f}% 단축이다. 기본 경로 범위는 {min(base):.3f}–{max(base):.3f}초, 선택 경로는 {min(chosen):.3f}–{max(chosen):.3f}초다. 두 표본씩의 변동 폭을 고려하면 모델 전체의 속도 개선을 입증했다고 볼 수 없다. CPU 다음 출력의 차이도 공통 CPU 경로의 측정 변동이므로 이번 최적화의 효과로 해석하지 않는다.', '',
          '256행 선형층에서는 중첩과 재사용의 결합 효과가 관찰됐다. 그러나 이 결과를 16토큰 모델 전체로 일반화하지 않으며 기본 실행 경로를 일괄 변경하지 않는다. 다음 단계는 64·256토큰 모델의 반복 비교와 층별 M/N/K 및 패킹·대기 시간 계측으로 적용 조건을 정하는 것이다. 장치 재사용만으로 제출 횟수와 부분합 반환량이 줄지는 않았으므로, 이후에는 여러 타일 제출의 결합과 부분합 반환 비용을 별도로 줄여야 한다.']
(here/'REUSE-OVERLAP.md').write_text('\n'.join(lines)+'\n',encoding='utf8',newline='\n')
hashes={name:hashlib.sha256((here/name).read_bytes()).hexdigest() for name in ('reuse_device.py','joined_prefill.py','parallel_post.py','bench_reuse_overlap.py')}
(here/'results/reuse-overlap-sources.json').write_text(json.dumps(hashes,indent=2),encoding='utf8',newline='\n')
print('\n'.join(lines[-10:]))
