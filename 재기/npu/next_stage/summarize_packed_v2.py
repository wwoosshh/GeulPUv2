import hashlib
import json
from pathlib import Path
from statistics import median

here=Path(__file__).resolve().parent
v1=json.loads((here/'results/packed-partials.json').read_text())
v2=json.loads((here/'results/packed-partials-v2.json').read_text())
plain=json.loads((here/'results/batch-capacity.json').read_text())
bound=json.loads((here/'results/unpack24-boundaries.json').read_text())
assert v2['complete'] and bound['complete']
lines=['# INT24 복원 SIMD화와 헤더 내 넘침 보정', '',
       '새 v2 경로만 실행했다. 기존 벤치마크는 반복하지 않았다. 초기 파일 생성의 인코딩 오류로 발생한 컴파일 실패는 수정한 뒤 새 검증을 완료했다. 이전 v1 구현과 원자료는 보존했다.', '',
       '## 변경', '',
       'CPU 복원은 NumPy 임시 배열 연산에서 네이티브 SSSE3 SIMD로 이동했다. 12바이트에서 4개의 INT24를 안전하게 읽고 바이트 셔플과 산술 시프트로 부호 확장한다. 마지막 원소에서도 버퍼를 초과하여 읽지 않는다. DLL은 MSVC /O2로 빌드했다.', '',
       'NPU 압축은 기존 GL 내적 객체와 C++ 래퍼를 링크한다. 기존 32바이트 헤더에 넘침 개수와 최대 7개의 보정 정보를 넣는다. 보정 정보 하나는 원소 인덱스 7비트와 원래 INT32 상위 8비트를 저장한다. 하위 24비트와 합치면 원래 INT32를 복원할 수 있다. 타일당 전송 크기는 416바이트 그대로다. 8개 이상 넘치면 헤더를 -1로 표시하고 기존 INT32 재요청 경로를 사용한다. 모든 입력에서 재요청을 없앤 설계는 아니다.', '',
       '## 검증', '',
       f"정수 부분합 검사 {v2['integer_checks']}개, FP32 비트 비교 {v2['linear_checks']}개, Qwen3-0.6B-Q8_0 16토큰 모델의 첫·다음 로짓 해시 2개가 통과했다. 정수 검사에는 타일당 넘침 1·7·8개 경계를 포함하며, 8개에서만 재요청하는 것을 확인했다. 별도 SIMD 검사 {bound['cases']}개({bound['values']}개 값)는 INT32 최소·최대 및 INT24 경계의 정확한 복원을 확인했다.", '',
       '## 성능과 반환량', '',
       '각 새 선형층 경로는 워밍업 후 3회 측정한 중앙값이다. 동일 Qwen FFN up 가중치, NPU 작업자 4개, 4,096프레임, CPU 후처리 4스레드 중첩을 사용했다. v1과 비압축 수치는 저장된 이전 세션 결과이므로 통제된 동일 세션 속도 비교는 아니다.', '',
       '| 행 | 비압축 이전 ms | v1 이전 ms | v2 신규 ms | v1 복원 ms | v2 복원 ms | v2 재요청 | v2 반환 MiB |', '|---:|---:|---:|---:|---:|---:|---:|---:|']
for m in (64,256):
    a=[r for r in v1['runs'] if r['rows']==m]
    b=[r for r in v2['runs'] if r['rows']==m]
    p=[r['seconds'] for r in plain['linear'] if r['rows']==m and r['frames']==4096]
    vals=[median(p)*1000,median(r['seconds'] for r in a)*1000,median(r['seconds'] for r in b)*1000,median(r['metrics']['decode_seconds'] for r in a)*1000,median(r['metrics']['decode_seconds'] for r in b)*1000]
    lines.append(f'| {m} | '+' | '.join(f'{v:.2f}' for v in vals)+f" | {b[0]['metrics']['replay_calls']} | {b[0]['metrics']['output_bytes']/(1<<20):.2f} |")
lines+=['','이번 대상 데이터에서는'+' 64행의 548타일, 256행의 1,995타일을 헤더로 보정했고 모든 측정에서 재요청은 0회였다. 256행 출력은 비압축 96MiB→78MiB로 18.75% 감소했다. 입력·가중치를 포함한 명시적 버퍼 바이트는 192→174MiB로 9.375% 감소한다. 하드웨어 버스 카운터 계측값은 아니다.', '',
        '전체 시간에는 NPU 압축, 반환, SIMD 복원, 기존 FP32 후처리와 중첩 비용이 포함된다. 복원 시간은 헤더 검사와 출력 배열 할당도 포함한다. NPU 압축의 독립 사이클 비용은 아직 분리 측정하지 않았으므로 남은 지연을 NPU 압축 하나로 단정하지 않는다.', '',
        '## 판단', '',
        'v1의 주요 문제였던 CPU 복원과 흔한 넘침의 재요청은 줄였다. 그러나 새 전체 시간은 기존 비압축 기록보다 여전히 길다. 기본 경로로 채택할 근거가 없으므로 비압축 자동 용량 선택을 유지한다. 다음 후보는 NPU 압축 래퍼의 벡터화와 내적 결과 저장을 합쳐 임시 INT32 배열 및 별도 순회를 줄이는 것이다. 해당 개선 전 기존 전체 모델 벤치마크를 반복하지 않는다.', '',
        '[v2 원자료](results/packed-partials-v2.json) · [SIMD 경계 검사](results/unpack24-boundaries.json) · [NPU 래퍼](pack24_v2.cc) · [SIMD 복원](unpack24.c) · [전송·재요청](packed_device_v2.py)']
(here/'PACKED-PARTIALS-V2.md').write_text('\n'.join(lines)+'\n',encoding='utf8',newline='\n')
sources=('pack24_v2.cc','unpack24.c','packed_device_v2.py','packed_design_v2.py','build_pack24_v2.py','test_packed_partials_v2.py','check_unpack24.py')
(here/'results/packed-partials-v2-sources.json').write_text(json.dumps({n:hashlib.sha256((here/n).read_bytes()).hexdigest() for n in sources},indent=2),encoding='utf8',newline='\n')
print('\n'.join(s for s in lines if s.startswith('|')))
