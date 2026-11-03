import hashlib
import json
from pathlib import Path

here=Path(__file__).resolve().parent
d=json.loads((here/'results/adaptive-output-view.json').read_text())
assert d['complete'] and d['policy_checks']==11 and d['tile_checks']==16 and d['linear_checks']==5
assert len(d['model_hashes'])==2 and d['preserved_outputs']==16
assert d['handle_loads']=={'2048':1,'4096':1,'8192':1}
lines=['# 자동 용량 선택·핸들 재사용·출력 뷰 통합 검증', '',
       'AdaptiveOutputViewDevice를 추가했다. 기존 AdaptiveReuseDevice의 용량 선택 정책을 그대로 사용하고, 각 용량의 실행 장치를 OutputViewDevice로 생성한다. 2,048·4,096·8,192 프레임 장치는 서로 다른 버퍼와 실행 핸들을 보유한다. 연속 입력 준비를 유지하고 출력 중간 복사만 제거한다. 출력 재배치가 소유권이 독립적인 청크로 끝난 후 다음 장치 작업을 시작한다.', '',
       '기존 기본 경로나 기존 결과 파일은 바꾸지 않았다. 새 통합 경로만 정확성 검사했고 성능 벤치마크를 재실행하지 않았다.', '',
       '| 검사 | 결과 |','|---|---|',
       '| 선택 경계·잘못된 크기 | 11개 통과 |',
       '| 용량 전환 중 정수 부분합 | 16개 CPU INT64 기준과 일치 |',
       '| 단일 행·홀수 크기·64/65행 경계 FP32 | 5개 CPU 출력 비트와 일치 |',
       '| 이전 출력 수명 | 16개 결과를 후속 호출과 모델 실행 이후에도 보존 |',
       '| 장치 출력 버퍼 alias | 없음 |',
       '| 용량별 핸들 로딩 | 2,048·4,096·8,192 각각 1회 |',
       '| 실제 모델 출력 | Qwen3-0.6B-Q8_0 64토큰 첫·다음 로짓 해시 2개 일치 |', '',
       '용량 선택 순서: '+ ' → '.join(str(x) for x in d['transitions'])+'.', '',
       '선택 순서를 정방향·역방향으로 바꿔 큰 용량에서 작은 용량으로 돌아가는 경우와 동일 용량 재방문을 모두 확인했다. 단순히 결과가 같은지만 보는 데 그치지 않고, 해당 용량의 핸들이 독립적이며 반복 호출 중 다시 로드되지 않는지 검사했다. 출력 원본을 계속 보유하고 이후 다른 입력 실행이 이를 덮어쓰지 않는지도 비교했다.', '',
       '## 사용', '',
       '기존 JoinedPrefill의 device 인수에 AdaptiveOutputViewDevice()를 전달하고, CPU 후처리와 finalizer_executor는 기존과 같이 관리한다. 단일 행 모델 출력·디코드는 기존 CPU 라우팅을 유지할 수 있다. 하나의 장치 인스턴스는 순차적으로 호출해야 한다. 동시에 다른 요청에서 같은 장치 버퍼를 공유하는 서비스 동시성은 이번 검증 범위가 아니다.', '',
       '성능은 앞선 고정 용량 시험 결과만 존재한다. 이번 통합 검증을 새 모델 가속률의 증거로 해석하지 않는다. 자동 용량 정책의 임계값도 전체 모델에 최적임을 보장하지 않는다. 절전·장치 리셋 후 핸들 복구는 별도 검증이 필요하다.', '',
       '[통합 구현](adaptive_output_view.py) · [시험](check_adaptive_output_view.py) · [원자료](results/adaptive-output-view.json)', '',
       '재현: viability.ps1 check_adaptive_output_view. 보고서만 생성: summarize_adaptive_output_view.py.']
(here/'ADAPTIVE-OUTPUT-VIEW.md').write_text('\n'.join(lines)+'\n',encoding='utf8',newline='\n')
names=('adaptive_output_view.py','adaptive_reuse.py','output_view_device.py','retained_device.py','diagnose_transfer.py','check_adaptive_output_view.py')
(here/'results/adaptive-output-view-sources.json').write_text(json.dumps({n:hashlib.sha256((here/n).read_bytes()).hexdigest() for n in names},indent=2),encoding='utf8',newline='\n')
print(json.dumps({k:d[k] for k in ('complete','policy_checks','tile_checks','linear_checks','handle_loads','preserved_outputs')}))
