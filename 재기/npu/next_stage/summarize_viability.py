import json,statistics
from pathlib import Path
here=Path(__file__).resolve().parent
data=json.loads((here/'results/viability-compare.json').read_text())
assert len(data['rounds'])==24
table=[]
for name in ('baseline','optimized','cpu_cached'):
    rows=[r for r in data['rounds'] if r['name']==name and r['repeat']==1]
    wall=statistics.mean(r['wall'] for r in rows[1:])
    cpu=statistics.mean(r['cpu'] for r in rows[1:])
    table.append(f"| {name} | {rows[0]['wall']:.3f} | {wall:.3f} | {1/wall:.3f} | {cpu:.3f} | {cpu/wall:.2f} |")
report="""# Qwen3-0.6B-Q8_0 추론 타당성 비교

Ryzen AI 7 350에서 같은 모델, 같은 디코더, 프롬프트 토큰 [1,2,3]으로 실행했다. 매 조건 두 회, 회당 prefill 한 번과 decode 세 번을 측정했다. 표는 두 번째 회 평균이다. 모델 로딩과 커널 준비 시간은 제외했다. 모든 단계의 전체 FP32 로짓 SHA-256을 CPU 기준과 비교했고 24개가 일치했다. 생성 토큰은 [16,17,18,19]였다. 이전 전체-prefix 참조에 일치했던 마지막 로짓 해시도 재현했다.

| 실행 | prefill 초 | 초/생성 토큰 | 토큰/초 | CPU초/토큰 | 평균 사용 코어 상당량 |
|---|---:|---:|---:|---:|---:|
"""+'\n'.join(table)+"""

baseline은 GL 4행·고정 2048프레임 및 FP32 KV 캐시, optimized는 한 토큰 선형층에 GL 1행·512/1024/2048 가변 프레임과 양자화 KV 캐시를 사용한다. cpu_cached는 4개 워커의 네이티브 AVX2 정수 내적, 동일한 엄격 FP32 후처리 및 동일한 양자화 KV 캐시다. 공통 캐시 최적화는 길이 64 이하 기본 계약에서 단일 어텐션 조각을 재사용하며 기존 FP32 KV와 양자화 KV를 함께 보관한다. 각 조건을 순서대로 실행했으며 개별 최적화의 모델 수준 기여도와 장문 성능은 분리 측정하지 않았다. 고정된 짧은 요청에서의 선택 근거로 사용한다.

CPU초는 Python 프로세스 전체 스레드의 사용자·커널 시간이다. 이를 경과 시간으로 나눈 평균 코어 사용량과, 같은 토큰 하나에 소비한 CPU 시간은 서로 다른 지표다. 시스템 전체 및 드라이버 외부 작업은 포함하지 않는다. AC 전원 사용 중 배터리 방전율 0이며 조회한 에너지 카운터로 유효한 J/token을 얻지 못했다.

1행 커널은 유효 타일 128개를 512프레임에서 2.06 ms, 2048프레임에서 5.23 ms에 처리했다. 프레임 조건마다 5회 출력이 정수 참조와 일치했다. 시간은 첫 실행을 제외한 중앙값이다. 기존 4행 GL은 손으로 최적화한 AIE C++와 320바이트 기계어가 동일했다. GL 정적 검사 회귀시험 8개가 통과했고 범위 초과, 오정렬, 오버플로를 컴파일 때 거부했다. 개발 시간과 유지보수 비용은 측정하지 않았다.

[생성 및 프레임 원자료](results/viability-compare.json) · [CPU 단독 원자료](results/viability-cpu.json) · [GL 기계어 비교](../gl_aie/build-results.json) · [이전 정확성 검증](RESULTS-2026-10-31.md)
"""
(here/'VIABILITY-2026-10-31.md').write_text(report,encoding='utf8',newline='\n')
print('\n'.join(table))
