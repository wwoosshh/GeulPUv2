import hashlib
import json
from pathlib import Path
from statistics import median

here=Path(__file__).resolve().parent
paired=json.loads((here/'results/system-comparison.json').read_text())
assert paired['complete'] and len(paired['pairs'])==3
lines=['# CPU 단독과 최종 CPU–NPU 분담의 실용성 비교', '',
       '동일 Qwen3-0.6B-Q8_0 가중치, 동일 토큰 입력(seed=407), 동일 CPU 어텐션·정확 FP32 후처리로 비교했다. hybrid는 비압축 자동 용량 선택, NPU 작업자 4개, CPU 후처리 4스레드 중첩, 용량별 핸들 재사용, 연속 입력 준비, 출력 중간 복사 제거를 사용한다. 출력층과 디코드는 CPU다. CPU 단독은 기존 네이티브 AVX2 정수 내적과 정확 후처리 경로다.', '',
       '각 입력 길이·모드마다 별도 프로세스를 사용해 다른 모드의 가중치 계획·장치 버퍼가 메모리 수치에 섞이지 않도록 했다. 모델 로딩과 3토큰 프리필·1토큰 디코드 워밍업은 제외했다. hybrid는 모든 제출 용량 그래프를 미리 실행해 JIT를 측정에서 제외했다. 같은 프로세스에서 KV를 초기화하며 요청 2개를 연속 처리했다. 길이별 실행 순서는 CPU→hybrid, hybrid→CPU, CPU→hybrid다. 완전한 열·전력 통제 실험은 아니다.', '',
       '## 속도와 순차 처리량', '',
       '입력 길이별 요청당 출력은 첫 토큰을 포함해 4토큰이다. 첫 출력은 모델 step 시간으로 토큰화·서버 통신을 포함하지 않는다. decode tok/s는 후속 3토큰/그 시간이다. 요청/분은 두 요청의 측정 총시간으로 계산한 순차 처리량이며 동시 요청 또는 포화 서버 처리량이 아니다.', '',
       '| 입력 | 모드 | 첫 출력 초 | 요청 전체 초 | 후속 생성 tok/s | 순차 요청/분 |', '|---:|---|---:|---:|---:|---:|']
data={}
for length in (16,64,256):
    for mode in ('cpu','hybrid'):
        d=json.loads((here/f'results/system-{mode}-{length}.json').read_text())
        assert d['complete'] and len(d['runs'])==2
        data[length,mode]=d
        runs=d['runs']
        lines.append(f"| {length} | {mode} | {median(r['prefill_seconds'] for r in runs):.3f} | {median(r['resources']['wall_seconds'] for r in runs):.3f} | {6/sum(sum(r['decode_seconds']) for r in runs):.3f} | {120/sum(r['resources']['wall_seconds'] for r in runs):.3f} |")
lines+=['','## 자원 효율', '',
        'CPU 초는 프로세스의 모든 스레드가 사용한 user+kernel CPU 시간이다. 평균 코어 수는 CPU 초/벽시계 초다. 이 값이 작아도 요청이 오래 걸려 총 CPU 초가 늘면 계산 효율 개선으로 간주하지 않는다. RSS와 private 메모리는 요청 중 100ms 간격으로 샘플링한 최대값이다. 드라이버·장치 메모리 전체나 시스템 전력은 포함하지 않는다.', '',
        '| 입력 | 모드 | 요청당 CPU 초 | 평균 CPU 코어 수 | 최대 샘플 RSS MiB | 최대 private MiB |', '|---:|---|---:|---:|---:|---:|']
for (length,mode),d in data.items():
    runs=d['runs']
    lines.append(f"| {length} | {mode} | {median(r['resources']['cpu_seconds'] for r in runs):.3f} | {sum(r['resources']['cpu_seconds'] for r in runs)/sum(r['resources']['wall_seconds'] for r in runs):.3f} | {max(r['resources']['peak_sampled_rss_bytes'] for r in runs)/(1<<20):.1f} | {max(r['resources']['peak_sampled_private_bytes'] for r in runs)/(1<<20):.1f} |")
lines+=['','CPU와 hybrid의 각 길이·반복 4개 실행에서 첫 출력과 후속 3개 전체 로짓 해시가 모두 동일했다. 총 12요청, 48개 로짓 배열을 검증했다. 샘플 수는 조건별 2개로 제한적이다.', '',
        '## 전력과 NPU 단독의 범위', '',
        '현재 OS 전원 구성표는 균형 조정이다. AMD 전력 도구는 권한을 높여 실행해도 LoadLibraryA 실패가 발생해 전력·에너지를 계측하지 못했다. CPU 시간 감소나 RSS 감소를 에너지 절감으로 환산하지 않는다. 에너지 효율 판정에는 지원되는 전력 센서 또는 외부 전력계가 추가로 필요하다.', '',
        'Geul 정확성 계약을 지키는 NPU 단독 모델 실행 경로는 아직 구현되지 않았다. 따라서 NPU 단독을 0초·실패 성능으로 넣지 않고 미지원으로 표시한다. 아래 별도 FastFlowLM 실행은 NPU 기반 프리필·디코드의 실용적 참고이며 엄격 계약의 대체 경로가 아니다.']
native_path=here/'results/native-npu-system.json'
if native_path.exists():
    native=json.loads(native_path.read_text())
    if native.get('complete'):
        lines+=['','## 별도 NPU 런타임 참고', '',
                'FastFlowLM 1.0.10, Qwen3 0.6B Q4_1/model.q4nx, performance 모드, 문맥 1,024, 한국어/임의 토큰 대신 별도 영어 프롬프트, 출력 상한 64, 워밍업 후 서로 다른 프롬프트 2개를 사용했다. 같은 모델 계열이지만 가중치·양자화·프롬프트·출력 길이가 달라 위 결과와 공정한 속도비를 계산할 수 없다. prefix 캐시 재사용은 통제하지 않았다. NPU 전용 백엔드라도 서버·스케줄링에는 CPU를 사용한다.', '',
                '| 요청 | 첫 표시 출력 초 | 전체 초 | 입력/출력 토큰 | 출력/전체 tok/s | 서버 CPU 초 | 서버 RSS MiB |', '|---:|---:|---:|---|---:|---:|---:|']
        for i,r in enumerate(native['runs']):
            u=r.get('usage') or {}
            count=u.get('completion_tokens',0)
            lines.append(f"| {i+1} | {r['first_visible_seconds']:.3f} | {r['total_seconds']:.3f} | {u.get('prompt_tokens','?')}/{count} | {count/r['total_seconds']:.2f} | {r['resources']['cpu_seconds']:.3f} | {r['resources']['peak_sampled_rss_bytes']/(1<<20):.1f} |")
        lines+=['','FastFlowLM이 보고한 두 요청의 cached_tokens는 모두 0이고, 런타임 디코드 처리량은 각각 '+', '.join(f"{r['usage']['decoding_speed_tps']:.2f} tok/s" for r in native['runs'])+'였다. 이는 위 표의 첫 응답 대기까지 포함한 출력/전체 tok/s와 정의가 다르다. 로컬 서버 로그에서도 NPU 잠금과 실행을 확인했다. 서버는 시험 후 종료했다.']
    else:lines+=['','별도 FastFlowLM 시험은 완료되지 않았다. 오류 원자료를 확인해야 하며 성능 결과로 해석하지 않는다.']
lines+=['','## 동일 계약 비교의 판단', '']
for length in (16,64,256):
    cpu=data[length,'cpu']['runs'];hybrid=data[length,'hybrid']['runs']
    wall=lambda runs:median(r['resources']['wall_seconds'] for r in runs)
    cput=lambda runs:median(r['resources']['cpu_seconds'] for r in runs)
    lines.append(f'- {length}입력: hybrid/CPU 요청 시간 {wall(hybrid)/wall(cpu):.3f}배, CPU 사용 시간 {cput(hybrid)/cput(cpu):.3f}배. 1보다 작아야 각각 개선이다.')
lines+=['','현재 엄격 계약의 분담 구현은 256토큰 첫 출력에서 약 5% 단축을 보였지만 요청 전체 지연과 순차 처리량의 실질적 우위를 확보하지 못했다. 256토큰에서는 총 CPU 사용 시간이 약 16% 더 크고, 메모리도 더 크다. 디코드는 두 모드 모두 같은 CPU 코드이므로 관찰된 디코드 차이를 NPU 연산의 직접 효과라고 해석하지 않는다. 실행 순서·열·메모리 상태 등의 원인은 분리하지 않았다.', '',
        '별도 NPU 런타임에서는 빠른 생성이 가능했다. 따라서 이번 결과는 NPU 하드웨어가 본질적으로 모델 실행에 부적합하다는 근거가 아니라, 현재 엄격 계약 구현과 일반 NPU 추론의 성능 격차를 보여준다. 양자화·가중치·프롬프트·출력 길이 차이 때문에 그 격차를 특정 설계 하나의 비용으로 환산할 수는 없다.', '',
        '이 결과는 현재 프로토타입과 CPU 기준 경로의 비교다. 최적화된 모든 CPU 런타임 또는 NPU 하드웨어 자체의 한계를 뜻하지 않는다. 동시 요청 포화 처리량, 긴 생성, 전력 효율은 이번 결과로 확정하지 않는다.', '',
        '[동일 계약 비교 원자료](results/system-comparison.json) · [비교 코드](compare_systems.py) · [별도 NPU 원자료](results/native-npu-system.json)']
(here/'SYSTEM-COMPARISON.md').write_text('\n'.join(lines)+'\n',encoding='utf8',newline='\n')
names=('compare_systems.py','system_metrics.py','check_native_npu_system.py','adaptive_output_view.py','joined_prefill.py','cpu_linear.py','cached_qwen.py')
(here/'results/system-comparison-sources.json').write_text(json.dumps({n:hashlib.sha256((here/n).read_bytes()).hexdigest() for n in names},indent=2),encoding='utf8',newline='\n')
print('\n'.join(s for s in lines if s.startswith('|') or s.startswith('- ')))
