import hashlib
import json
from pathlib import Path

here=Path(__file__).resolve().parent
d=json.loads((here/'results/model-integration.json').read_text())
assert d['complete'] and d['checks']==17 and len(d['scenarios'])==4 and d['reset_checks']==7
lines=['# 상태를 포함한 모델 통합 정확성 검증', '',
       'AdaptiveOutputViewDevice + JoinedPrefill + CPU 정확 후처리 중첩 경로를 Qwen3-0.6B-Q8_0 모델에 연결했다. 같은 모델 인스턴스와 가중치를 계속 사용하되 CPU 기준 실행과 통합 실행 사이에는 KV 상태를 초기화했다. 새로운 고정 난수 토큰 입력(seed=734)을 사용했다. 성능 시간은 측정하지 않았고 이전 성능 벤치마크도 반복하지 않았다.', '',
       '| 시나리오 | 입력 방식 | 후속 디코드 | 일치한 단계 |','|---|---|---:|---:|']
for r in d['scenarios']:
    lines.append(f"| {r['name']} | {' + '.join(map(str,r['chunks']))}토큰 | 3회 | {len(r['steps'])} |")
lines+=['','## 확인한 계약', '',
        '총 17단계에서 전체 FP32 로짓을 uint32 비트 배열로 직접 비교했고 SHA256도 저장했다. 각 단계의 생성 토큰, 처리 위치, KV 바이트 수 및 모든 층의 FP32 K/V 캐시 해시가 CPU 기준과 일치했다. 단일 행 출력층과 연속 디코드는 공통 CPU 경로이며 여러 행의 프리필 선형층만 NPU 경로다. 따라서 이 결과를 NPU 단일 토큰 디코드 검증으로 해석하지 않는다.', '',
        '초기화 7회 모두 위치=0, 모든 FP32 KV 캐시=None, 양자화 캐시 비어 있음을 확인했다. 17토큰 요청을 65토큰 요청 및 분할 요청 이후 재실행했을 때 처음의 CPU 기준 출력과 다시 일치했다. 각 요청 안에서 이전 로짓 배열을 보유하고 다음 실행 후에도 값이 변하지 않는지 확인했다.', '',
        '7+10 분할 프리필은 동일한 분할 방식의 CPU 실행과 비교했다. 한 번에 17토큰을 넣는 방식과 분할 방식 자체의 동등성을 별도 계약으로 검증한 것은 아니다. 65토큰 요청과 후속 3회 디코드는 짧은 어텐션 캐시 경계(64)를 넘는 경로도 실행한다.', '',
        '## 자원 재사용', '',
        '관찰된 용량별 선택 횟수: '+json.dumps(d['selections'])+'.',
        '관찰된 용량별 실행 핸들 로딩 횟수: '+json.dumps(d['handle_loads'])+'.',
        '통합 경로의 CPU/NPU 선형 호출 횟수: '+json.dumps(d['routes'])+'.', '',
        '이번 실제 모델 시나리오에서 선택된 용량의 핸들은 모두 1회만 로드됐다. 2,048·4,096·8,192 전체 용량 전환과 버퍼 수명은 앞선 ADAPTIVE-OUTPUT-VIEW.md의 별도 통합 검사로 검증했다. 실제 모델에서 선택되지 않은 용량까지 이번에 실행했다고 주장하지 않는다.', '',
        '## 범위와 판단', '',
        '이번 모델·입력·CPU 기준 구현에 대해 상태 초기화와 반복 요청, 분할 프리필, 다단계 디코드가 비트 계약을 유지했다. 제품 전체의 보편적 정확성 증명은 아니며 다른 모델, 더 긴 컨텍스트, 동시 요청, 절전·장치 리셋 복구는 범위 밖이다. CPU 기준과 통합 경로는 모델 코드와 FP32 후처리를 공유하므로 공유 코드 자체의 오류까지 배제하는 독립 검증은 아니다.', '',
        '기존 기본 실행 경로는 변경하지 않았다. 다음 배포·제품 통합 판단에서는 성능 주장과 정확성 결과를 분리하고, 동시 요청 및 장치 수명 관리를 별도 검증해야 한다.', '',
        '[원자료](results/model-integration.json) · [검증 구현](check_model_integration.py) · [용량 전환 검증](ADAPTIVE-OUTPUT-VIEW.md)']
(here/'MODEL-INTEGRATION.md').write_text('\n'.join(lines)+'\n',encoding='utf8',newline='\n')
names=('check_model_integration.py','adaptive_output_view.py','adaptive_reuse.py','output_view_device.py','retained_device.py','diagnose_transfer.py','joined_prefill.py','cached_qwen.py','cpu_linear.py','parallel_post.py')
(here/'results/model-integration-sources.json').write_text(json.dumps({n:hashlib.sha256((here/n).read_bytes()).hexdigest() for n in names},indent=2),encoding='utf8',newline='\n')
print(json.dumps({k:d[k] for k in ('complete','checks','reset_checks','handle_loads','routes')}))
