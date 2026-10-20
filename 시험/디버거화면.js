// 디버거 v1 3단계 — 앱 화면(앱/web/app.js)의 디버거 계산을 GPU 없이 돌린다(시험/디버거도구.py 가 부른다 — node 가 있을 때).
// app.js 에서 비교 · 중단점 · 큰 정수 JSON 함수만 떼어 입력 JSON 의 사례마다 결과를 JSON 으로 낸다 — 파이썬(도구/debugger.py)과 같은 값인가는
// 부른 쪽이 본다. 화면 문구(T)는 원문 그대로 돌려주는 대역으로.
//   node 시험/디버거화면.js <app.js> <입력.json>
'use strict';
const fs = require('fs');
const src = fs.readFileSync(process.argv[2], 'utf8');
const inp = JSON.parse(fs.readFileSync(process.argv[3], 'utf8'));

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
const code = [cut("const BIG = '", '\nconst bigShow').replace(/\nconst bigShow$/, ''), line('dbgG'), line('dbgFinish'), line('dbgEq'), fn('dbgFirstDiff'), fn('dbgServerOf'),
  fn('dbgDiff'), line('dbgScoreFor'), fn('dbgBreaks'), fn('dbgCompare'), fn('dbgParse'), line('dbgIsHeader')].join('\n');
// eslint-disable-next-line no-new-func
const lib = new Function('T', 'Blob', code + '\nreturn { bigParse, bigStringify, dbgCompare, dbgBreaks, dbgParse };')(T, require('buffer').Blob);

const out = { compare: [], breaks: [], big: [] };
for (const c of inp.compare || []) {
  const A = lib.dbgParse(c.a, 'A'), B = lib.dbgParse(c.b, 'B');
  const r = lib.dbgCompare(A, B);
  out.compare.push({
    rows: r.rows.map(x => ({ i: x.i, compared: x.compared, same: x.same, first_divergence: x.first_divergence, steps_a: x.steps_a, steps_b: x.steps_b,
      scored: x.scored, delta_sum: x.delta_sum, delta_mean: x.delta_mean, b_top1_differs: x.b_top1_differs, a_top1_differs: x.a_top1_differs,
      logprob_a: x.logprob_a, logprob_b: x.logprob_b, worst: x.worst ? x.worst.map(w => w.step) : undefined })),
    summary: r.summary,
  });
}
for (const c of inp.breaks || []) out.breaks.push(lib.dbgBreaks(c.tokens, c.score, c.margin, c.prob, c.tool_call).map(b => [b.step, b.kinds]));
for (const t of inp.big || []) out.big.push(lib.bigStringify(lib.bigParse(t)));
process.stdout.write(JSON.stringify(out));
