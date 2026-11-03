"""Compile existing experiment reports without running any benchmark."""
import hashlib
import json
import os
import re
import subprocess
from pathlib import Path
from urllib.parse import unquote

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
OUT=HERE/'TEST-SUMMARY-2026-11-03.md'
parts=[
('초기 실행 기준과 정확한 NPU 선형층',[
 ('외부 NPU 런타임 초기 기준','results/2026-10-27.md'),
 ('정확한 선형층 초기 검증','results/2026-10-28-exact.md'),
 ('초기 최적화','results/2026-10-28-optimized.md'),
 ('계획·가중치 준비 캐시','results/2026-10-28-planned.md'),
 ('자리 결합 및 모델 검증','FUSION-RESULTS-2026-10-30.md')]),
('GL·Native C·Python과 장치 커널 비교',[
 ('CPU 데이터 처리: GL/C/Python/NumPy','../language/RESULTS.md'),
 ('GL·C 소스 최적화 후 비교','../language/AUDIT-RESULTS-2026-10-29.md'),
 ('CPU 정수 행렬 부분합: GL/C/AVX2','../language/MATRIX-DECISION-2026-10-29.md'),
 ('실제 NPU 커널: GL/C/펼친 C/AIE C++','gl_aie/README.md')]),
('정확 후처리·단계 분담·장치 병렬화',[
 ('네이티브 후처리·FMA 적합성·초기 중첩','next_stage/RESULTS-2026-10-31.md'),
 ('초기 실용성 검증 질문','next_stage/VIABILITY-2026-10-31.md'),
 ('CPU/NPU 단계 분담 및 묶음 제출','next_stage/PHASE-BENCH.md'),
 ('NPU 작업자 1·2·4개','next_stage/MULTIWORKER-2026-11-01.md'),
 ('CPU 후처리 스레드 1·2·4개','next_stage/PARALLEL-POST.md')]),
('중첩·가중치 재사용·제출 용량',[
 ('기본/중첩/재사용/결합 4조합','next_stage/REUSE-OVERLAP.md'),
 ('64·256토큰 전체 모델 및 층별 분석','next_stage/LONG-OVERLAP.md'),
 ('제출 용량 2,048·4,096·8,192','next_stage/BATCH-CAPACITY.md'),
 ('입력 크기별 자동 용량 정책','next_stage/ADAPTIVE-REUSE.md')]),
('무손실 부분합 압축의 구현과 기각',[
 ('범위 분석·INT24 압축·넘침 재요청','next_stage/PACKED-PARTIALS.md'),
 ('SIMD 복원·헤더 내 넘침 보정','next_stage/PACKED-PARTIALS-V2.md'),
 ('NPU 내적·압축 융합','next_stage/PACKED-PARTIALS-V3.md'),
 ('실행·전송·호스트 단계별 진단','next_stage/TRANSFER-DIAGNOSTIC.md')]),
('비압축 실행 경로 최적화와 통합 정확성',[
 ('핸들 재사용과 입력·출력 복사 축소','next_stage/RETAINED-DEVICE.md'),
 ('연속 입력 유지·출력 중간 복사만 제거','next_stage/OUTPUT-VIEW.md'),
 ('자동 용량＋핸들＋출력 뷰 통합','next_stage/ADAPTIVE-OUTPUT-VIEW.md'),
 ('상태 초기화·KV·분할 입력·연속 생성','next_stage/MODEL-INTEGRATION.md')]),
('최종 속도·효율·처리량과 NPU 전용 런타임 참고',[
 ('CPU 단독 대 최종 분담 및 별도 FastFlowLM','next_stage/SYSTEM-COMPARISON.md')])]

intro='''# GeulNPU 시험·벤치마크 종합 기록

작성일: 2026-11-03. 기존 보고서와 저장된 결과를 통합한 작업 공유용 문서다. **이 문서를 만들기 위해 벤치마크를 재실행하지 않았다.** 초기 실험부터 최종 통합 경로까지 조합·조건·지연시간·정확성 결과·실패 사례를 파트별로 수록한다.

## 먼저 읽을 현재 결론

- 현재 완성된 것은 **NPU 정수 내적＋CPU의 계약 후처리·모델 연산**을 사용하는 연구용 통합 경로다. Geul 계약 전체를 NPU 내부에서 실행하는 백엔드는 아직 미완성이다.
- 비압축, 장치 작업자 4개, CPU 후처리 중첩, 가중치 타일 재사용, 제출 용량 자동 선택, 용량별 실행 핸들 재사용, 연속 입력 준비, 출력 중간 복사 제거가 통합 후보다. 전역 기본 실행 경로를 제품 NPU 백엔드로 교체한 것은 아니다.
- INT24 압축은 무손실 정확성을 확인했지만 같은 세션 진단에서 256행 FFN이 비압축 550ms 대 압축 712ms로 느려 채택하지 않았다.
- 최종 동일 Q8 계약 비교에서 16토큰 요청은 CPU 12.28초 대 분담 14.06초, 64토큰은 36.56초 대 36.27초, 256토큰은 145.75초 대 145.73초였다. **각 요청은 출력 총 4토큰**이며 조건별 2회다. 전체 지연·순차 처리량의 뚜렷한 우위를 확보하지 못했다.
- 256토큰에서 분담의 첫 출력은 약 5% 짧았지만 총 CPU 사용 시간은 약 16% 더 컸다. 메모리는 약 0.6GiB 더 사용했다. 전력 도구 실패로 에너지 효율은 미측정이다.
- 별도 FastFlowLM은 Qwen 0.6B Q4 계열에서 64토큰 생성에 1.65~1.76초, 런타임 디코드 61.5~65.7토큰/초를 기록했다. **다른 양자화·가중치·입력·계약이므로 Q8 엄격 계약과 직접적인 가속 배수로 비교하지 않는다.**
- 다음 핵심 개발은 기존 Geul 계약의 변환·반올림·FMA·규칙 R을 NPU에서 구현하고, CPU로 부분합을 반환하지 않는 완결된 NPU 선형층을 만드는 일이다. 기존 FMA 링크 실패·벡터 MAC 불일치는 시험한 구현의 실패이며 하드웨어에서 계약 구현이 불가능하다는 증명이 아니다.

## 공통 환경과 수치 읽는 법

| 항목 | 기준 |
|---|---|
| PC | AMD Ryzen AI 7 350 / XDNA2 NPU / Windows / RAM 32GB |
| 주된 모델 | Qwen3-0.6B-Q8_0; 초기 언어 시험은 합성 입력, 외부 런타임은 별도 Q4 모델 |
| 주된 실제 FFN 형상 | M=입력 행, N=3072, K=1024; 다른 표의 M/N/K는 각 원문 표기 우선 |
| 시간 단위 | 장치·선형층은 주로 ms, 모델은 주로 초; 표 머리와 셀 표기 우선 |
| 중앙값 | 각 절에 기록된 반복 수의 중앙값. 최소·최대가 있는 경우 함께 기재 |
| CPU 기준 | 계약 구현과 비교한 비트 일치 기준. 모든 CPU 제품 런타임 중 최적 성능을 뜻하지 않음 |
| 전송량 | 주로 호출 횟수×명시적 버퍼 크기. 하드웨어 버스 카운터 실측치와 구분 |
| 실행 대기 | 제출·장치 내부 DMA·FIFO·스케줄링·완료 대기를 포함할 수 있으며 순수 코어 연산 시간이 아님 |
| 정확성 | 검증한 입력에서의 비트·로짓·KV 일치. 모든 입력·모델의 증명이나 품질 평가는 아님 |

**비교 규칙:** 같은 절 안의 교대·동일 세션 비교와, 다른 날짜·세션의 이전 기록을 구분한다. 각 단계의 최고 기록을 이어 붙여 누적 가속률을 만들지 않는다. 중첩된 CPU 후처리 시간과 NPU 시간을 더해 전체 시간으로 해석하지 않는다. GPU 피크 대비 50~70%, NPU 50 TOPS, 가정한 대역폭으로 계산한 하한 등은 이번 실측으로 확인한 활용률이 아니므로 벤치마크 표에 섞지 않는다.

**원문 보존 방식:** 아래 세부 절은 해당 단계의 기존 보고서를 수록한다. 절별 결론은 당시 시점의 판단이므로 현재 채택 상태는 위 요약 및 마지막 파트를 우선한다. 예를 들어 초기 문서의 “GL NPU 백엔드 미구현”은 이후 제한된 정수 커널 변환기 구현으로 일부 진전됐으며, 현재도 일반 GL 전체와 계약 전체의 NPU 백엔드는 아니다. 원문 코드 블록의 상대 경로·재현 명령은 해당 원본 보고서의 실행 위치를 따른다.

## 목차
'''

manifest=[]
missing=[]
chunks=[intro]
for i,(title,sections) in enumerate(parts,1):
    chunks.append(f'- [파트 {i}. {title}](#part-{i})')
chunks.append('')

def rewrite(text,source):
    def link(m):
        label,target=m.group(1),m.group(2)
        raw=target.strip('<>')
        if re.match(r'^[a-zA-Z][a-zA-Z0-9+.-]*:',raw) or raw.startswith('#'):return m.group(0)
        path,sep,anchor=raw.partition('#')
        resolved=(source.parent/unquote(path)).resolve()
        if not resolved.exists():
            missing.append(dict(source=str(source.relative_to(ROOT)),target=raw))
            return f'{label} (원문 경로: `{raw}`; 현재 파일 없음)'
        dest=Path(os.path.relpath(resolved,OUT.parent)).as_posix()+(sep+anchor if sep else '')
        return f'[{label}](<{dest}>)' if ' ' in dest else f'[{label}]({dest})'
    lines=text.splitlines()
    if lines and lines[0].startswith('# '):lines=lines[1:]
    result=[]
    fenced=False
    for line in lines:
        if line.lstrip().startswith('```'):fenced=not fenced
        if not fenced:
            match=re.match(r'^(#{1,6})\s+(.*)',line)
            if match:line='#'*min(len(match[1])+2,6)+' '+match[2]
            line=re.sub(r'\[([^\]]+)\]\(([^)]+)\)',link,line)
        result.append(line)
    return '\n'.join(result).strip()

for i,(title,sections) in enumerate(parts,1):
    chunks.extend([f'\n<a id="part-{i}"></a>\n',f'## 파트 {i}. {title}\n'])
    for j,(name,path) in enumerate(sections,1):
        source=(HERE/path).resolve()
        text=source.read_text(encoding='utf-8-sig')
        manifest.append(dict(part=i,section=j,title=name,source=source.relative_to(ROOT).as_posix(),sha256=hashlib.sha256(source.read_bytes()).hexdigest(),table_header_count=len(re.findall(r'^\|\s*:?-',text,re.M))))
        relative=Path(os.path.relpath(source,HERE)).as_posix()
        chunks.extend([f'### {i}.{j}. {name}\n',f'원본 보고서: [{source.name}]({relative}). 아래 조건·결론은 해당 단계의 기록이다.\n',rewrite(text,source),''])

chunks.extend(['## 부록 A. 측정 실패·제한과 제외 기준\n',
'''| 항목 | 처리 |
|---|---|
| NPU scalar FMA 링크 실패 | 실행·성능 값 없음. 도구 체인 심벌 연결 실패와 하드웨어 불가능을 구분 |
| NPU 벡터 MAC 계약 불일치 | 256개 중 84개 불일치. 이 명령 매핑을 계약 경로에 채택하지 않음 |
| 256토큰 장기 중첩 시험의 절전 | 약 8시간인 마지막 표본은 원자료에 보존하고 정상 성능 집계에서 제외; 정상 결합 표본 2개 |
| 초기 256토큰 분담 실행 중단 | 완료한 측정으로 취급하지 않음; 이후 묶음 제출 경로로 별도 시험 |
| AMD 전력 도구 초기화 실패 | 에너지·전력 효율 미측정. CPU 초로 대신 계산하지 않음 |
| 기타 인코딩·컴파일·계측 반환형 오류 | 수정 후 완료된 결과만 해당 표의 성능 표본에 포함 |
| 후속 고정 용량 기록 간 비교 | 같은 세션의 통제 비교가 아니면 정확한 인과적 가속률 주장 금지 |
| CPU 기준 공유 | 모델·후처리의 공유 코드 자체 오류까지 배제하는 독립 검증은 아님 |
| 운영 범위 | 동시 요청, 장치 리셋·절전 복귀, 장문·다른 모델에 대한 보장은 별도 검증 필요 |
''',
'## 부록 B. 추가 설계·계약·이전 종합 문서\n',
'- [Geul 계약](../../계약/contract.md)\n- [초기 NPU 설계](ARCHITECTURE.md)\n- [정확성 설계](EXACT.md)\n- [GL 커널 설계 검토](GEUL-KERNEL-DESIGN.md)\n- [2026-10-31까지의 GL/NPU 행렬 종합 보고서](GL-NPU-MATRIX-REPORT-2026-10-31.md)\n- [업스트림 변경 분석](UPSTREAM-ANALYSIS-2026-11-01.md)\n',
'## 부록 C. 수록 자료의 추적성\n',
'수록한 보고서의 파일명·SHA256·파트·표 개수는 [자료 목록](TEST-SUMMARY-2026-11-03.sources.json)에 보존한다. 이 문서는 기존 기록의 편집본이며 새 성능 측정값을 생성하지 않았다. 각 세부 절의 원자료 링크에서 JSON 표본과 검증 결과를 확인할 수 있다.\n'])
OUT.write_text('\n'.join(chunks).rstrip()+'\n',encoding='utf8',newline='\n')
metadata=dict(created='2026-11-03',repository_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),reports=manifest,missing_historical_links=missing,output_sha256=hashlib.sha256(OUT.read_bytes()).hexdigest())
(HERE/'TEST-SUMMARY-2026-11-03.sources.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2),encoding='utf8',newline='\n')
print(json.dumps(dict(file=str(OUT),reports=len(manifest),tables=sum(m['table_header_count'] for m in manifest),lines=len(OUT.read_text(encoding='utf8').splitlines()),missing_historical_links=len(missing)),ensure_ascii=False))
