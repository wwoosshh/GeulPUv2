// 앱리디자인 E1(문서/화면구현.md §1) — 명령 창의 명령 해석(앱/web/ui/command.js)이 CLI 와 같은 요청 · 같은 오류를 내는지 본다. GPU · 앱 없이.
//   1. 예 파일(시험/명령창예.json — CLI 쪽은 시험/명령줄점검.py 가 같은 예를 확인한다)마다: 한 줄을 나눈 단어가 argv 와 같고, 해석한 결과(요청의
//      op · method · path · body, 오류의 영어 까닭과 다음에 할 일, 도움말, ops)가 CLI 가 낸 것과 같다. 앞에 GeulPU 를 붙여도 같다.
//   2. 명령 쓰기(toCommand)로 만든 명령을 다시 해석하면 같은 본문이 된다 — 예의 요청 본문과 화면이 보내는 본문(스트리밍 채팅 등).
//   3. 따옴표(quote)로 싼 단어는 단어 나누기로 그대로 돌아온다.
//   4. 화면 코드의 요청(requestFor)이 CLI 와 같은 쿼리 문자열을 만든다.
//
//   node 시험/명령창점검.mjs      → 마지막 줄 "명령 창 점검: 통과 — …"(종료 0), 틀리면 "못 미침"(종료 1)
import { mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { dirname, join } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

const ROOT = join(dirname(fileURLToPath(import.meta.url)), '..');
// command.js 는 브라우저의 ES 모듈 — node 가 어느 버전이든 모듈로 읽도록 .mjs 로 옮겨 불러온다
const tmp = mkdtempSync(join(tmpdir(), 'geulpu-console-'));
const modFile = join(tmp, 'command.mjs');
writeFileSync(modFile, readFileSync(join(ROOT, '앱', 'web', 'ui', 'command.js'), 'utf8'));
const C = await import(pathToFileURL(modFile).href);
rmSync(tmp, { recursive: true, force: true });

const reg = C.index(JSON.parse(readFileSync(join(ROOT, '앱', 'data', 'operations.json'), 'utf8')));
const cases = JSON.parse(readFileSync(join(ROOT, '시험', '명령창예.json'), 'utf8')).cases;
const bad = [];
const same = (a, b) => JSON.stringify(a) === JSON.stringify(b);
const norm = body => (body === null ? null : JSON.parse(body));

function outcome(r) {
  if (r.kind === 'request') return { request: { op: r.op, method: r.method, path: r.path, body: norm(r.body) } };
  if (r.kind === 'error') return { error: { message: r.message, next: r.next } };
  if (r.kind === 'help') return { help: r.op ? r.op.id : null };
  if (r.kind === 'ops') return { ops: true };
  return { other: r.kind };
}

// 1. 예 파일
let roundTrips = 0;
for (const c of cases) {
  let toks;
  try { toks = C.tokenize(c.line); } catch (e) { toks = String(e); }
  if (!same(toks, c.argv)) bad.push(`예 '${c.name}': 단어 ${JSON.stringify(toks)} ≠ argv ${JSON.stringify(c.argv)}`);
  const got = outcome(C.parse(reg, c.line + ' --dry-run'));
  if (!same(got, c.expect)) bad.push(`예 '${c.name}': ${JSON.stringify(got).slice(0, 220)} ≠ CLI ${JSON.stringify(c.expect).slice(0, 220)}`);
  const withName = outcome(C.parse(reg, 'GeulPU ' + c.line + ' --dry-run'));
  if (!same(withName, got)) bad.push(`예 '${c.name}': 앞에 GeulPU 를 붙이면 다름`);
  // 2. 명령 쓰기 → 다시 해석(POST 요청의 본문)
  if (c.expect.request && c.expect.request.method === 'POST') {
    const line = C.toCommand(reg, c.expect.request.op, c.expect.request.body);
    const back = C.parse(reg, line);
    if (back.kind !== 'request' || back.op !== c.expect.request.op || !same(norm(back.body), c.expect.request.body)) {
      bad.push(`예 '${c.name}': 명령 쓰기 '${line}' 를 다시 해석하면 ${JSON.stringify(outcome(back)).slice(0, 200)}`);
    }
    roundTrips++;
  }
}

// 2b. 화면이 보내는 본문 — 목록에 없는 필드는 --json, 스트리밍은 빼고, 자료형이 다른 값도 --json
const screenBodies = [
  ['run.chat', { messages: [{ role: 'user', content: "it's \"fine\"\n줄" }], stream: true, stream_options: { include_usage: true }, max_tokens: 64, geulpu: { logit_hash: true } }],
  ['run.complete', { prompt: '--not an option', max_tokens: 3, temperature: 0, seed: 1 }],
  ['run.complete', { prompt: 'x', n_probs: 5 }],
  ['run.complete', { prompt: 'x', max_tokens: '3' }],
  ['settings.set', { slots: 4, kv8_cache: false, precision: 'fast' }],
  ['run.lens', { prompt: [1, 2], geulpu: { layer_lens: { top_k: 3 } }, echo: true, max_tokens: 0, logprobs: 1 }],
  ['store.clear', {}],
  // F1(문서/사용성시험.md) — 줄바꿈이 든 글 인자(계획 · 결과)도 한 줄 명령으로(명령 창의 입력 줄은 한 줄이라 줄바꿈을 지운다)
  ['experiment.update', { path: 'E:\\x\\2026-11-09-a', plan: '# Plan\n\nline 2\r\n', results: 'one\ntwo' }],
];
for (const [op, body] of screenBodies) {
  const line = C.toCommand(reg, op, body);
  if (/[\r\n]/.test(line)) bad.push(`화면 본문 ${op}: 명령 쓰기가 여러 줄 — ${JSON.stringify(line).slice(0, 200)}`);
  const back = C.parse(reg, line);
  const want = { ...body };
  if (want.stream === true) { delete want.stream; delete want.stream_options; }
  if (back.kind !== 'request' || back.op !== op || !same(norm(back.body), want)) bad.push(`화면 본문 ${op}: '${line}' → ${JSON.stringify(outcome(back)).slice(0, 200)}`);
  roundTrips++;
}

// 3. 따옴표
const words = ['', 'plain', 'two words', "it's", '"', "'", '\\', 'E:\\research dir\\x', '--flag', '-5', '한국어 단어', '한국어', '😀 emoji', 'a\tb', 'a\nb',
  "'\\''", '$HOME `x` %PATH%', '{"a":[1,2]}', '=', '@file'];
for (let i = 0; i < 300; i++) {
  let s = '';
  const n = (i * 7919) % 12;
  for (let j = 0; j < n; j++) s += ' \'"\\a-=$가\t'[(i * 31 + j * 17) % 12];
  words.push(s);
}
for (const w of words) {
  let back;
  try { back = C.tokenize('x ' + C.quote(w) + ' y'); } catch (e) { back = String(e); }
  if (!same(back, ['x', w, 'y'])) bad.push(`따옴표: ${JSON.stringify(w)} → ${C.quote(w)} → ${JSON.stringify(back)}`);
}

// 4. 화면 코드의 요청 — 예의 GET 쿼리와 같은 길
const getCase = cases.find(c => c.name === 'GET 쿼리(퍼센트)');
const req = C.requestFor(reg, 'experiment.list', { root: getCase.argv[getCase.argv.length - 1] });
if (req.path !== getCase.expect.request.path) bad.push(`requestFor: ${req.path} ≠ ${getCase.expect.request.path}`);
const ev = C.requestFor(reg, 'system.events', { after: 12, wait: 25000 });
if (ev.method !== 'GET' || ev.path !== '/api/events?after=12&wait=25000') bad.push(`requestFor(system.events): ${ev.method} ${ev.path}`);

for (const b of bad.slice(0, 40)) console.log('틀림 —', b);
if (bad.length) {
  console.log(`명령 창 점검: 못 미침 ${bad.length}`);
  process.exit(1);
}
console.log(`명령 창 점검: 통과 — 예 ${cases.length} 개(CLI 와 같은 요청 · 오류), 명령 쓰기 → 다시 해석 ${roundTrips} 개, 따옴표 ${words.length} 개`);
