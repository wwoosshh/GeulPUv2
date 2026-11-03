import hashlib
import json
from pathlib import Path
from statistics import median

here=Path(__file__).resolve().parent
v2=json.loads((here/'results/packed-partials-v2.json').read_text())
v3=json.loads((here/'results/packed-partials-v3.json').read_text())
plain=json.loads((here/'results/batch-capacity.json').read_text())
assert v3['complete'] and v3['integer_checks']==7 and v3['linear_checks']==10 and len(v3['model'])==2
lines=['# NPU 내적·INT24 압축 융합 시험', '',
       '기존 v2는 GL 내적 객체가 128개 INT32 임시 배열을 채운 뒤 C++ 래퍼가 넘침 검사와 압축을 별도 순회했다. 새 v3는 내적 네 개를 계산할 때마다 넘침 정보를 헤더에 기록하고 즉시 세 개의 32비트 압축 워드로 저장한다. 소스에서 전체 128원소 임시 배열과 별도 압축 순회를 제거했다. 컴파일러의 레지스터 spill이 전혀 없다고 주장하지는 않는다.', '',
       '이번 커널은 GL에서 낮춘 AIE 정수 내적과 같은 연산을 사용하는 수동 C++ 융합 시제품이다. GL 컴파일러 자체에 융합 최적화를 구현한 것은 아니다. 기존 gl.o와 기존 경로는 수정하지 않았다. 출력 ABI는 v2와 같아 타일당 416바이트, 헤더 내 넘침 보정 최대 7개, 8개 이상 INT32 재요청을 유지한다. SIMD 복원 구현도 그대로 사용한다.', '',
       '## 검증', '',
       '새 v3 경로의 정수 부분합 검사 7개(넘침 1·7·8개 경계 포함), FP32 비트 비교 10개 및 Qwen3-0.6B-Q8_0 16토큰 첫·다음 로짓 해시 2개가 모두 통과했다. 기존 성능 벤치마크는 다시 실행하지 않았다.', '',
       '## 결과', '',
       '4,096프레임, NPU 작업자 4개, CPU 후처리 4스레드 중첩. 새 경로는 워밍업 후 3회 측정한 중앙값이다. 이전 기록은 서로 다른 세션이므로 정확한 인과적 속도 향상률을 확정할 수 없다.', '',
       '| FFN 행 | 비압축 이전 ms | v2 이전 ms | v3 신규 ms | v3 최소–최대 ms | v3 장치·전송 ms | v3 복원 ms | 재요청 | 반환 MiB |', '|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
for m in (64,256):
    old=[r for r in v2['runs'] if r['rows']==m]
    new=[r for r in v3['runs'] if r['rows']==m]
    vals=[r['seconds']*1000 for r in new]
    p=median(r['seconds'] for r in plain['linear'] if r['rows']==m and r['frames']==4096)*1000
    lines.append(f"| {m} | {p:.2f} | {median(r['seconds'] for r in old)*1000:.2f} | {median(vals):.2f} | {min(vals):.2f}–{max(vals):.2f} | {median(r['profile']['device_and_transfer_seconds'] for r in new)*1000:.2f} | {median(r['metrics']['decode_seconds'] for r in new)*1000:.2f} | {new[0]['metrics']['replay_calls']} | {new[0]['metrics']['output_bytes']/(1<<20):.2f} |")
lines+=['','장치·전송 시간은 호스트 패킹·복원 등 grouped 호출 내부 비용도 포함한다. 따라서 NPU 커널만의 시간으로 읽으면 안 된다. 전체 시간에는 NPU 압축, 전송, SIMD 복원, FP32 후처리가 포함된다. 모델 실행은 정확성 통합 검증이며 전체 모델의 성능 개선은 이번에 측정하지 않았다.', '',
        '출력 반환량은 256행 78MiB로 v2와 같고 비압축 96MiB보다 18.75% 작다. 입력·가중치까지 포함한 명시적 버퍼 바이트는 재요청이 없을 때 192→174MiB다. 이는 버퍼 집계이며 하드웨어 버스 계측값은 아니다.', '',
        '## 적용 판단', '']
for m in (64,256):
    new=median(r['seconds'] for r in v3['runs'] if r['rows']==m)
    old=median(r['seconds'] for r in v2['runs'] if r['rows']==m)
    p=median(r['seconds'] for r in plain['linear'] if r['rows']==m and r['frames']==4096)
    lines.append(f'- {m}행: v2 기록 대비 시간 {(1-new/old)*100:.1f}% 변화(양수=단축), 비압축 기록 대비 {(new/p-1)*100:.1f}% 변화(양수=증가). 서로 다른 세션의 관찰 비교다.')
lines+=['','소규모 3표본과 이전 세션 기록만으로 압축을 기본으로 채택하지 않는다. 비압축 자동 용량 선택 경로를 유지한다. 추가 최적화 전에 장치 실행 자체의 시간과 호스트 비용을 분리 계측할 필요가 있다. 출력 전송량 감소만으로 전체 가속을 보장하지 않는다.', '',
        '[원자료](results/packed-partials-v3.json) · [융합 커널](pack24_fused.cc) · [디스어셈블리](results/pack24-fused.asm) · [시험](test_packed_partials_v3.py)']
(here/'PACKED-PARTIALS-V3.md').write_text('\n'.join(lines)+'\n',encoding='utf8',newline='\n')
names=('pack24_fused.cc','build_pack24_v3.py','packed_device_v3.py','packed_design_v3.py','test_packed_partials_v3.py','unpack24.c')
(here/'results/packed-partials-v3-sources.json').write_text(json.dumps({n:hashlib.sha256((here/n).read_bytes()).hexdigest() for n in names},indent=2),encoding='utf8',newline='\n')
print('\n'.join(s for s in lines if s.startswith('|') or s.startswith('- ')))
