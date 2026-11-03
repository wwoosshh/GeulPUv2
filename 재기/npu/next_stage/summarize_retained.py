import hashlib
import json
from pathlib import Path
from statistics import median

here=Path(__file__).resolve().parent
d=json.loads((here/'results/retained-device.json').read_text())
old=json.loads((here/'results/transfer-diagnostic.json').read_text())
assert d['complete'] and d['checks']==26 and len(d['model'])==2
lines=['# 비압축 실행 핸들 재사용·복사 축소 시험', '',
       '기존 진단은 재실행하지 않았다. 새 handle 경로와 handle_lean 경로를 동일 세션에서 A/B/B/A/A/B 순서로 각 3회 측정했다. 기존 진단과 동일한 FFN up 가중치, M=256 K=1024 N=3072, 입력 seed=619, 4,096프레임, NPU 작업자 4개와 CPU 후처리 4스레드를 사용했다.', '',
       '## 구현', '',
       'handle은 첫 CallableDesign 호출이 반환한 실행 핸들을 보유하고 이후 DefaultNPURuntime.run으로 실행한다. 반복 캐시 키 생성·컴파일 객체 준비·load 단계를 건너뛰되 runtime.run의 장치 확인과 완료 검사는 유지한다. 고정 형상과 버퍼를 가진 장치 인스턴스 내에서만 재사용한다. 명시적 clear_handle 뒤 재로딩도 검증했다. 알려진 XRT stale-context 오류에서는 핸들을 버리고 기존 CallableDesign 복구 경로로 돌아가도록 했지만 실제 절전 후 복구를 이번에 시험하지는 않았다.', '',
       'handle_lean은 이에 더해 정렬된 그룹·가중치의 패딩용 복사를 생략하고, 장치 입력 버퍼를 전체 0으로 채우는 대신 사용하지 않는 꼬리만 초기화한다. 다운로드 후 중간 copy를 없애고 장치 호스트 버퍼 뷰에서 최종 청크 배열로 즉시 재배치한다. 다음 장치 실행 전에 재배치를 끝내며 CPU 후처리는 별도 소유 청크를 사용한다. 정수 연산과 FP32 FMA 순서는 바뀌지 않는다.', '',
       '## 검증', '',
       '홀수 행·열, 깊이 꼬리, 버퍼 재사용 뒤 이전 출력 보존, 명시적 핸들 해제·재로딩을 포함해 CPU 기준 비트 비교 26회가 통과했다. handle_lean을 사용한 16토큰 Qwen 모델 첫·다음 전체 로짓 해시 2개도 일치했다. 6개 측정 구간에서 핸들 재로딩은 모두 0회다.', '',
       '| 경로 | 전체 중앙값 ms | 최소–최대 ms | 호출 준비 잔여 ms | 입력 패킹 ms | 출력 중간 복사 ms | 출력 재배치 ms |', '|---|---:|---:|---:|---:|---:|---:|']
sets=[('기존 진단(이전 세션)',[r for r in old['runs'] if r['mode']=='plain'])]+[(name,[r for r in d['runs'] if r['mode']==name]) for name in ('handle','handle_lean')]
for name,rows in sets:
    times=[r['seconds']*1000 for r in rows]
    overhead=median(r['stages']['dispatch_wait']-r['stages']['runtime_wait_ns']/1e9 for r in rows)*1000
    pack=median(r['stages']['pack_copy'] for r in rows)*1000
    copy=median(r['stages']['output_copy'] for r in rows)*1000
    scatter=median(r['stages']['scatter'] for r in rows)*1000
    lines.append(f'| {name} | {median(times):.2f} | {min(times):.2f}–{max(times):.2f} | {overhead:.2f} | {pack:.2f} | {copy:.2f} | {scatter:.2f} |')
lines+=['','호출 준비 잔여는 호출 전체 시간에서 런타임 제출·완료 대기 시간을 뺀 값이다. 스케줄링 등의 영향도 있으므로 순수 API 실행 시간이라고 단정하지 않는다. 기존 진단과의 차이는 서로 다른 세션의 관찰 비교다. 두 새 경로끼리는 같은 세션이지만 각 3표본이고 실행 후반에 시간이 줄어드는 변동이 있어 작은 차이를 확정적 우열로 일반화하지 않는다.', '',
        '## 판단', '',
        '핸들 재사용은 호출 준비 잔여를 크게 줄였으며 유지할 가치가 있다. 다만 전체 모델의 성능 개선은 이번에 계측하지 않았고 기본 경로를 자동 교체하지 않았다.', '',
        '복사 축소는 출력 중간 복사를 없앴지만 입력 패킹 시간이 늘었고 전체 중앙값도 handle 단독보다 좋지 않았다. 입력은 원래 전치된 strided 배열이므로 연속 복사를 생략하면 뒤의 gather 접근이 비효율적일 수 있다. 이는 코드와 계측에 따른 원인 후보이지 독립적으로 입증된 원인은 아니다. handle_lean을 일괄 채택하지 않는다.', '',
        '다음 변경은 handle 경로의 연속 입력 준비를 유지하면서 출력 중간 복사만 제거하는 분리 실험이 적절하다. 입력 레이아웃 변경과 출력 복사 제거를 함께 적용해 이득을 상쇄하지 않도록 한다. 이후에만 실제 모델 경로에 적용할 후보를 고른다.', '',
        '[원자료](results/retained-device.json) · [핸들 재사용](retained_device.py) · [복사 축소](lean_reuse_device.py) · [시험](test_retained_device.py)']
(here/'RETAINED-DEVICE.md').write_text('\n'.join(lines)+'\n',encoding='utf8',newline='\n')
names=('retained_device.py','lean_reuse_device.py','test_retained_device.py','diagnose_transfer.py')
(here/'results/retained-device-sources.json').write_text(json.dumps({n:hashlib.sha256((here/n).read_bytes()).hexdigest() for n in names},indent=2),encoding='utf8',newline='\n')
print('\n'.join(s for s in lines if s.startswith('|')))
