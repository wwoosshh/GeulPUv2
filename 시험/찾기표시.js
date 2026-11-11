// 모델 찾기의 저장소 표시(앱/web/app.js 의 fdSupport)를 GPU · 인터넷 없이 돌린다(시험/찾기표시.py 가 부른다 — node 가 있을 때).
// 고정 자료(시험/찾기표시경우.json)는 2026-10-30 조사(트렌딩 상위 200 저장소 — Hugging Face 메타데이터의 구조 · 매개변수 수 · 파일 이름과,
// 엔진의 모델 확인 · 분류로 낸 판정)와 엔진의 구조 목록(서버 --구조들). app.js 에서 fdSupport 와 그것이 부르는 함수만 떼어 돌린다
// (시험/디버거화면.js 와 같은 방법). 화면 문구(T)는 원문 그대로 돌려주는 대역으로.
//   node 시험/찾기표시.js <app.js> <찾기표시경우.json>
// 판정마다 허용하는 표시(k): 조사가 받는다고 본 저장소(accepted)는 ok · check(표시는 돌려 본 것이 아니라 규칙의 판단이라 check 도 된다),
// 받지 못하는 저장소는 어느 것도 ok 가 아니어야 한다 — 거짓 '실행됨'이 이 시험의 까닭(2026-10-30: 상위 50 의 '실행됨' 23 개 가운데 14 개가 거짓).
// 마지막 줄 "찾기 표시: 모두 통과 (N 저장소)"(종료 0) 또는 "찾기 표시: 틀림 N"(종료 1).
'use strict';
const fs = require('fs');
const src = fs.readFileSync(process.argv[2], 'utf8');
const fx = JSON.parse(fs.readFileSync(process.argv[3], 'utf8'));

function cut(start, endMark) {
  const a = src.indexOf(start);
  if (a < 0) throw new Error('app.js 에 없다: ' + start);
  const b = src.indexOf(endMark, a + start.length);
  if (b < 0) throw new Error('끝을 못 찾았다: ' + start);
  return src.slice(a, b + endMark.length);
}
const fn = name => cut(`function ${name}(`, '\n}\n');
const line = name => cut(`const ${name} = `, '\n');
const T = (s, ...v) => (typeof s === 'string' ? s : s.reduce((o, p, i) => o + p + (i < v.length ? v[i] : ''), ''));
const code = [line('quantFromName'), line('quantOk'), line('engArchs'), line('archOk'), line('cudaArch'), line('fdArch'), line('quantFamily'),
  cut('const quantOkFor = ', '\n};\n'), line('Q3OK'), line('L3OK'), line('inRange'), line('G3OK'), line('G3Q8'), line('fdQ8Only'), line('G4E'), line('fdKOnly'), line('fdQuantOk'), cut('const NONCHAT = ', '];\n'), cut('const fdNonChat = ', '\n};\n'),
  line('fdSplit'), line('fdGgufFiles'), line('fdUD'), line('fdHard'), fn('fdSupport'), line('isBig'), line('fmtParams')].join('\n');
const S = { st: { engine: { archs: fx.archs }, catalog: [] } };
const fdChecked = () => false;      // 앱 목록에서 돌려 본 저장소 — 고정 자료에는 없다
// eslint-disable-next-line no-new-func
const lib = new Function('T', 'S', 'fdChecked', code + '\nreturn { fdSupport, fdHard, isBig, fmtParams };')(T, S, fdChecked);

// 조사의 판정 → 허용하는 표시. accepted 만 ok 를 허용한다. 조사의 check(크기 규칙으로 모름)보다 더 좁힌 표시(받는 파일 없음 · 분할 파일만 ·
// Gemma 4 변형)는 더 정확한 것이라 허용한다. 0.24.0 — 조사 때의 gemma4-PLE(E2B · E4B)는 엔진이 k-양자화 파일을 받게 되어 없앴다
const ALLOWED = {
  accepted: ['ok', 'check'],
  check: ['check', 'quant', 'split', 'variant'],
  arch: ['arch'],
  quant: ['quant', 'check'],
  'quant(UD-only, unverified)': ['check', 'quant'],
  'split-only': ['split', 'quant'],
  'gemma4-MoE': ['variant'],
  'no-arch': ['unknown'],
  'clip-arch': ['check', 'unknown', 'arch', 'quant'],
  nonllm: ['arch', 'task', 'unknown', 'quant', 'check'],      // check: Hugging Face 가 구조를 clip 으로 읽은 저장소에 모델 파일이 따로 있을 때
};
const counts = {}, bad = [];
let shown = 0, falseOk = 0;
const rows = fx.rows.concat(fx.added || []);      // added: 조사 뒤에 더한 저장소(0.24.0 — Gemma 3)
for (const r of rows) {
  const m = { id: r.id, gguf: r.gguf, cardData: { pipeline_tag: r.pipeline_tag || null }, tags: r.tags || [], siblings: (r.files || []).map(f => ({ rfilename: f })) };
  const sup = lib.fdSupport(m);
  const c = counts[r.expect] || (counts[r.expect] = {});
  c[sup.k] = (c[sup.k] || 0) + 1;
  if (sup.k === 'ok' || sup.k === 'check') shown++;
  if (sup.k === 'ok' && r.expect !== 'accepted') falseOk++;
  const allow = ALLOWED[r.expect];
  if (!allow) { bad.push(`${r.id}: 모르는 판정 ${r.expect}`); continue; }
  if (!allow.includes(sup.k)) bad.push(`${r.id}: 판정 ${r.expect} → 표시 ${sup.k} (${sup.t})`);
}
// 0.24.0 — Gemma 3 270M · 1B 는 Q8_0 파일만 실행된다: 그 저장소에서 Q8_0 파일을 뺀 것은 '받는 형식의 파일 없음'이어야 한다(파생한 경우)
for (const r of rows.filter(x => x.gguf && x.gguf.architecture === 'gemma3' && x.gguf.total < 2e9)) {
  const files = (r.files || []).filter(f => !/Q8_0/i.test(f));
  const sup = lib.fdSupport({ id: r.id, gguf: r.gguf, cardData: { pipeline_tag: r.pipeline_tag || null }, tags: r.tags || [], siblings: files.map(f => ({ rfilename: f })) });
  if (sup.k !== 'quant') bad.push(`${r.id}: Q8_0 파일을 뺐는데 표시 ${sup.k} (${sup.t})`);
}
// 0.24.0 — Gemma 4 E2B · E4B 는 k-양자화 파일만 실행된다: 그 저장소에서 k-양자화 파일을 뺀 것(Q8_0 · BF16 · IQ4_XS …)은 '받는 형식의 파일 없음'
for (const r of rows.filter(x => x.gguf && x.gguf.architecture === 'gemma4' && x.gguf.total > 4.5e9 && x.gguf.total < 7.65e9)) {
  const files = (r.files || []).filter(f => !/Q[456]_K/i.test(f));
  const sup = lib.fdSupport({ id: r.id, gguf: r.gguf, cardData: { pipeline_tag: r.pipeline_tag || null }, tags: r.tags || [], siblings: files.map(f => ({ rfilename: f })) });
  if (files.length && sup.k !== 'quant') bad.push(`${r.id}: k-양자화 파일을 뺐는데 표시 ${sup.k} (${sup.t})`);
}
// 매개변수 수의 표시와 그것으로 하는 메모리 어림(10B 이상이면 크게): 1B 미만은 'M' 으로 적히고 큰 모델로 어림하지 않는다
for (const [p, want, big] of [[268098176, '268M', false], [999885952, '1.0B', false], [3880263168, '3.9B', false], [11766034176, '12B', true]]) {
  const t = lib.fmtParams(p);
  if (t !== want || lib.isBig({ params: t, id: '' }) !== big) bad.push(`매개변수 ${p}: 표시 ${t}(기대 ${want}) · 큰 모델 어림 ${lib.isBig({ params: t, id: '' })}(기대 ${big})`);
}
for (const [e, c] of Object.entries(counts)) console.log(`${e.padEnd(28)} ${Object.entries(c).map(([k, n]) => `${k} ${n}`).join(' · ')}`);
console.log(`기본 목록(ok · check)에 보이는 저장소 ${shown} / ${rows.length}, 받지 못하는 저장소의 거짓 '실행 가능' ${falseOk}`);
for (const b of bad) console.log('틀림  ' + b);
if (bad.length) { console.log(`찾기 표시: 틀림 ${bad.length}`); process.exit(1); }
console.log(`찾기 표시: 모두 통과 (${rows.length} 저장소)`);
