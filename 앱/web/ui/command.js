// 앱리디자인 E1(문서/화면구현.md §1) — 명령 창의 명령 해석과 명령 쓰기.
// GeulPU CLI(앱/명령줄.gl)와 같은 규칙으로 기능 목록에서 요청을 만든다: 같은 명령이면 CLI 의 --dry-run 과 같은 요청(op · method · path · body)이고,
// 오류도 같은 경우에 같은 영어 까닭으로 난다(화면에는 한국어 까닭도 함께 돌려준다). 화면(DOM)도 앱(fetch)도 쓰지 않는 모듈이라 node 로
// 시험한다(시험/명령창점검.mjs · 시험/명령창예.json — 같은 예를 시험/명령줄점검.py 가 CLI 로 확인한다).

const BOOL_WORDS = ['true', 'false', '1', '0', 'yes', 'no'];
const CLI_OPTIONS = ['port', 'app-port', '포트', 'session', '세션', 'timeout', 'json', 'out', 'experiment', '실험'];
const INT_TEXT = /^\s*-?[0-9]+\s*$/;
const SAFE_WORD = /^[A-Za-z0-9._\-/:@%+=,가-힣]+$/;

/** 기능 목록에 id 로 찾는 표를 붙인다(한 번만). */
export function index(reg) {
  if (!reg.byId) reg.byId = new Map((reg.operations || []).map(o => [o.id, o]));
  return reg;
}

/** CLI 로 부를 수 있는 기능 — 있음(exists)이고 내부용이 아님. */
export const callable = o => !!o && o.status === 'exists' && !o.internal;

/** 기능의 명령 이름(영어). */
export const commandName = o => (o.cli && o.cli.en) || o.id;

const paramOf = (o, name) => (o.params || []).find(p => p.name === name) || null;
const helpNext = name => 'GeulPU help ' + name;

function fail(message, message_ko, next, next_ko, op) {
  return { kind: 'error', message, message_ko, next: next || null, next_ko: next_ko || next || null, op: op || null, exit: 2 };
}

// ---- 단어 나누기 ----
// 공백으로 나눈다. 작은따옴표 안은 그대로, 큰따옴표 안은 \" · \\ 만 풀고, 따옴표 밖의 \ 는 ' " \ 와 공백 앞에서만 그 글자를 그대로 둔다
// (윈도 경로 E:\x 는 그대로). 따옴표(quote)로 싼 문자열은 늘 같은 단어로 돌아온다.
export function tokenize(line) {
  const s = String(line);
  const out = [];
  let cur = null;
  let i = 0;
  while (i < s.length) {
    const ch = s[i];
    if (ch === ' ' || ch === '\t' || ch === '\n' || ch === '\r') {
      if (cur !== null) { out.push(cur); cur = null; }
      i++;
      continue;
    }
    if (cur === null) cur = '';
    if (ch === "'") {
      const j = s.indexOf("'", i + 1);
      if (j < 0) throw new SyntaxError('unclosed quote');
      cur += s.slice(i + 1, j);
      i = j + 1;
    } else if (ch === '"') {
      i++;
      while (i < s.length && s[i] !== '"') {
        if (s[i] === '\\' && (s[i + 1] === '"' || s[i + 1] === '\\')) { cur += s[i + 1]; i += 2; } else { cur += s[i]; i++; }
      }
      if (i >= s.length) throw new SyntaxError('unclosed quote');
      i++;
    } else if (ch === '\\' && i + 1 < s.length && "'\"\\ ".includes(s[i + 1])) {
      cur += s[i + 1];
      i += 2;
    } else {
      cur += ch;
      i++;
    }
  }
  if (cur !== null) out.push(cur);
  return out;
}

/** 단어 하나를 명령 줄에 넣는 형식으로 — 따옴표가 필요 없으면 그대로, 아니면 작은따옴표(안의 ' 는 '\''). bash · PowerShell 에서도 같다. */
export function quote(v) {
  const s = String(v);
  if (s !== '' && SAFE_WORD.test(s)) return s;
  return "'" + s.replace(/'/g, "'\\''") + "'";
}

// ---- 값과 본문 ----

// 값 하나를 기능 목록의 자료형에 맞춰 JSON 텍스트로(CLI 의 인자값글과 같다) — 틀리면 { error }
function valueJson(type, val) {
  if (type === 'boolean') {
    if (val === null) return { json: 'true' };
    if (val === 'true' || val === '1' || val === 'yes') return { json: 'true' };
    if (val === 'false' || val === '0' || val === 'no') return { json: 'false' };
    return { error: 'needs true or false', error_ko: 'true 또는 false 여야 합니다' };
  }
  if (val === null) return { error: `needs a value (${type})`, error_ko: `값이 없습니다(${type})` };
  if (type === 'string') return { json: JSON.stringify(val) };
  let m;
  try { m = JSON.parse(val); } catch (_) { m = undefined; }
  let ok = false;
  if (m !== undefined) {
    if (type === 'integer') ok = typeof m === 'number' && INT_TEXT.test(val);
    else if (type === 'number') ok = typeof m === 'number';
    else if (type === 'array') ok = Array.isArray(m);
    else if (type === 'object') ok = m !== null && typeof m === 'object' && !Array.isArray(m);
    else ok = true;
  }
  if (!ok) return { error: `needs ${type} (JSON)`, error_ko: `${type}(JSON)이어야 합니다` };
  return { json: val };
}

// 인자들을 JSON 객체 텍스트로 — 점이 들어간 이름은 안쪽 객체(geulpu.step_offset → {"geulpu":{"step_offset":…}}), 순서는 처음 나온 위치(CLI 의 몸객체쓰기)
function objectText(args, prefix) {
  let out = '{';
  let first = true;
  for (let i = 0; i < args.length; i++) {
    const name = args[i].name;
    if (!name.startsWith(prefix)) continue;
    const rest = name.slice(prefix.length);
    const dot = rest.indexOf('.');
    if (dot < 0) {
      out += (first ? '' : ',') + JSON.stringify(rest) + ':' + args[i].json;
      first = false;
      continue;
    }
    const seg = rest.slice(0, dot);
    const inner = prefix + seg + '.';
    let written = false;
    for (let j = 0; j < i; j++) if (args[j].name.startsWith(inner)) written = true;
    if (!written) {
      out += (first ? '' : ',') + JSON.stringify(seg) + ':' + objectText(args, inner);
      first = false;
    }
  }
  return out + '}';
}

// 쿼리 문자열의 값 — UTF-8 바이트마다 [A-Za-z0-9-._~] 밖은 %XX(대문자, CLI 의 퍼센트쓰기)
export function percent(s) {
  let out = '';
  for (const b of new TextEncoder().encode(String(s))) {
    const keep = (b >= 48 && b <= 57) || (b >= 65 && b <= 90) || (b >= 97 && b <= 122) || b === 45 || b === 46 || b === 95 || b === 126;
    out += keep ? String.fromCharCode(b) : '%' + b.toString(16).toUpperCase().padStart(2, '0');
  }
  return out;
}

// ---- 명령 해석 ----

/** 명령 한 줄(앞의 GeulPU 는 있어도 되고 없어도 됨)을 요청 · 도움말 · 오류로. */
export function parse(reg, line) {
  let argv;
  try { argv = tokenize(line); } catch (_) {
    return fail('unclosed quote', '따옴표가 닫히지 않았습니다', null, null, null);
  }
  if (argv.length && /^geulpu(\.exe)?$/i.test(argv[0])) argv = argv.slice(1);
  if (!argv.length) return { kind: 'empty' };
  return parseArgv(reg, argv);
}

/** 이미 나눈 단어들(CLI 의 argv 에서 프로그램 이름을 뺀 것)을 해석한다 — CLI 의 명령줄돌리기 1 · 3 ~ 6 과 같은 차례. */
export function parseArgv(reg, argv) {
  index(reg);
  const ops = reg.operations || [];
  // 1. 명령 이름의 단어 — 첫 '--' 앞
  let k = 0;
  const words = [];
  while (k < argv.length && !argv[k].startsWith('--')) words.push(argv[k++]);
  let cmd = words.join(' ');
  // 3. 내장 명령
  if (cmd === 'ops' || cmd === '기능 목록') return { kind: 'ops' };
  if (cmd === 'guide' || cmd === '안내') return { kind: 'guide' };
  let help = cmd === 'help' || cmd === '도움말';
  if (cmd.startsWith('help ') || cmd.startsWith('도움말 ')) {
    help = true;
    cmd = cmd.slice(cmd.indexOf(' ') + 1);
    if (cmd === '') cmd = 'help';
  }
  if (help && (cmd === 'help' || cmd === '도움말')) return { kind: 'help', op: null };
  // 4. 기능 찾기
  const o = ops.find(x => x.id === cmd || (x.cli && (x.cli.en === cmd || x.cli.ko === cmd)));
  if (!o) {
    const sp = cmd.indexOf(' ');
    const first = sp < 0 ? cmd : cmd.slice(0, sp);
    const similar = [];
    for (const x of ops) {
      if (callable(x) && similar.length < 12 && x.cli && ((x.cli.en || '').startsWith(first) || (x.cli.ko || '').startsWith(first))) similar.push(x.cli.en || '');
    }
    let next = 'GeulPU ops lists every command';
    let nextKo = 'ops 로 모든 명령을 봅니다';
    if (similar.length) {
      next += ' — similar: ' + similar.join(' · ');
      nextKo += ' — 비슷한 명령: ' + similar.join(' · ');
    }
    return fail('unknown command: ' + cmd, '알 수 없는 명령입니다: ' + cmd, next, nextKo, null);
  }
  const id = o.id;
  const name = commandName(o);
  if (!callable(o)) {
    if (o.internal) return fail(id + ' is internal to the app, not a command', '앱 안에서만 쓰는 기능이라 명령이 아닙니다: ' + id, 'GeulPU ops lists every command', 'ops 로 모든 명령을 봅니다', id);
    return fail(id + ' is planned, not available yet', '아직 만들지 않은 기능입니다: ' + id, 'GeulPU ops lists every command', 'ops 로 모든 명령을 봅니다', id);
  }
  if (help) return { kind: 'help', op: o };

  // 5. 옵션과 인자 — 기능에 같은 이름의 인자가 있으면 기능의 인자다
  const options = { wait: false, dryRun: false, json: null, experiment: null, port: null, session: null, timeout: null, out: null };
  const args = [];
  for (; k < argv.length; k++) {
    const a = argv[k];
    if (!a.startsWith('--')) return fail(`the value '${a}' does not follow a --parameter`, `값 '${a}' 앞에 --인자 이름이 없습니다`, helpNext(name), null, id);
    let nm = a.slice(2);
    let val = null;
    const eq = nm.indexOf('=');
    if (eq >= 0) {
      val = nm.slice(eq + 1);
      nm = nm.slice(0, eq);
    }
    const p = paramOf(o, nm);
    let takes = eq < 0 && k + 1 < argv.length && !argv[k + 1].startsWith('--');
    // 값 없이 쓴 boolean 다음 단어는 그 값이 true · false 일 때만 가져간다
    if (p && p.type === 'boolean' && takes) takes = BOOL_WORDS.includes(argv[k + 1]);
    if (!p) {
      if (nm === 'wait' || nm === '기다림') { options.wait = true; continue; }
      if (nm === 'dry-run') { options.dryRun = true; continue; }
      if (CLI_OPTIONS.includes(nm)) {
        if (val === null && takes) { val = argv[k + 1]; k++; }
        if (val === null) return fail(`--${nm} needs a value`, `값이 필요합니다: --${nm}`, helpNext(name), null, id);
        if (nm === 'json') options.json = val;
        else if (nm === 'experiment' || nm === '실험') options.experiment = val;
        else if (nm === 'session' || nm === '세션') options.session = val;
        else if (nm === 'timeout') options.timeout = val;
        else if (nm === 'out') options.out = val;
        else options.port = val;
        continue;
      }
      const names = (o.params || []).map(x => x.name);
      const next = names.length ? 'pass fields the registry does not list with --json; parameters of this command: ' + names.join(', ') : 'this command takes no parameters';
      const nextKo = names.length ? '목록에 없는 필드는 --json 으로 넣습니다 — 이 명령의 인자: ' + names.join(', ') : '이 명령은 인자가 없습니다';
      return fail(`unknown parameter --${nm} for ${name}`, `${name} 의 인자가 아닙니다: --${nm}`, next, nextKo, id);
    }
    if (val === null && takes) { val = argv[k + 1]; k++; }
    if (args.some(x => x.name === nm)) return fail(`--${nm} is given twice`, `같은 인자를 두 번 썼습니다: --${nm}`, helpNext(name), null, id);
    const type = p.type || 'string';
    const v = valueJson(type, val);
    if (v.error) return fail(`--${nm} ${v.error}`, `--${nm} 의 값: ${v.error_ko}`, helpNext(name), null, id);
    args.push({ name: nm, raw: val, json: v.json, type });
  }
  if (options.session !== null && options.session.length === 0) options.session = null;

  // 6. 요청
  const http = o.http || {};
  const method = http.method || 'GET';
  let path = http.path || '/';
  const post = method === 'POST';
  let body = null;
  if (options.json !== null) {
    if (!post) return fail(`${name} is a GET command — it takes no --json body`, `GET 명령이라 --json 본문을 받지 않습니다: ${name}`, helpNext(name), null, id);
    if (args.length) return fail('--json gives the whole body — do not add other parameters to it', '--json 은 본문 전체입니다 — 다른 인자와 함께 쓰지 않습니다', 'put every field into the --json body', '모든 필드를 --json 본문에 넣으세요', id);
    if (options.json.startsWith('@')) return { ...fail('the command console does not read --json @file — give the body itself', '명령 창에서는 --json @파일 을 읽지 않습니다 — 본문을 그대로 넣으세요', null, null, id), console: true };
    let bo;
    try { bo = JSON.parse(options.json); } catch (_) { bo = undefined; }
    if (bo === undefined || bo === null || typeof bo !== 'object' || Array.isArray(bo)) return fail('--json must be a JSON object', '--json 은 JSON 객체여야 합니다', helpNext(name), null, id);
    if (bo.stream === true) return fail('the CLI prints whole answers — leave stream off', '명령 창은 응답 전체를 받습니다 — stream 은 끄세요', null, null, id);
    body = options.json;
  } else {
    for (const p of o.params || []) {
      if (p.required && !args.some(x => x.name === p.name)) return fail(`--${p.name} is required`, `필요한 인자가 없습니다: --${p.name}`, helpNext(name), null, id);
    }
    for (const a of args) {
      for (const b of args) {
        if (b.name.startsWith(a.name + '.')) return fail(`--${a.name} and --${b.name} cannot both be given`, `함께 쓸 수 없습니다: --${a.name} · --${b.name}`, 'give the inner fields only, or the whole value with --json', '안쪽 필드만 쓰거나 값 전체를 --json 으로 넣으세요', id);
      }
    }
    if (args.some(a => a.name === 'stream' && a.json === 'true')) return fail('the CLI prints whole answers — leave stream off', '명령 창은 응답 전체를 받습니다 — stream 은 끄세요', null, null, id);
    if (post) body = objectText(args, '');
    else if (args.length) path += '?' + args.map(a => percent(a.name) + '=' + percent(a.type === 'string' && a.raw !== null ? a.raw : a.json)).join('&');
  }
  return { kind: 'request', op: id, name, method, path, body, options };
}

// ---- 기능 id 와 인자로 요청(화면 코드가 쓴다) ----

/** 화면 코드의 호출: POST 는 인자 객체를 본문으로, GET 은 쿼리 문자열로(문자열은 그대로, 다른 자료형은 JSON). */
export function requestFor(reg, opId, params) {
  index(reg);
  const o = reg.byId.get(opId);
  if (!o || !o.http) throw new Error('not an operation with an HTTP route: ' + opId);
  const method = o.http.method || 'GET';
  let path = o.http.path;
  let body = null;
  if (method === 'POST') body = JSON.stringify(params || {});
  else {
    const q = Object.entries(params || {}).filter(([, v]) => v !== undefined && v !== null);
    if (q.length) path += '?' + q.map(([k, v]) => percent(k) + '=' + percent(typeof v === 'string' ? v : JSON.stringify(v))).join('&');
  }
  return { op: opId, method, path, body };
}

// ---- 명령 쓰기(명령 창의 줄 · 명령 복사) ----

function fits(type, v) {
  if (type === 'string') return typeof v === 'string';
  if (type === 'boolean') return typeof v === 'boolean';
  if (type === 'integer') return Number.isInteger(v);
  if (type === 'number') return typeof v === 'number' && Number.isFinite(v);
  if (type === 'array') return Array.isArray(v);
  if (type === 'object') return v !== null && typeof v === 'object' && !Array.isArray(v);
  return true;
}

const shorten = (s, n) => (n > 0 && s.length > n ? s.slice(0, n) + '…' : s);

/**
 * 기능 id 와 요청 본문(객체)을 CLI 에 그대로 입력할 수 있는 명령으로. 본문의 모든 필드가 기능 목록의 인자(점이 들어간 이름 포함)이고 자료형이
 * 맞으면 --인자 값으로, 아니면 --json '<본문>'. 스트리밍(stream: true 와 stream_options)은 빼고 적는다 — CLI 는 응답 전체를 받고, 받는 내용은
 * 같다. short > 0 이면 보이기용으로 긴 값을 그 길이에서 줄인다(복사용은 0).
 */
export function toCommand(reg, opId, params, short = 0) {
  index(reg);
  const o = reg.byId.get(opId);
  if (!o) return null;
  const out = ['GeulPU', commandName(o)];
  if (params === undefined || params === null) return out.join(' ');
  if (typeof params === 'object' && !Array.isArray(params) && params.stream === true) {
    params = Object.fromEntries(Object.entries(params).filter(([key]) => key !== 'stream' && key !== 'stream_options'));
  }
  const decl = new Map((o.params || []).map(p => [p.name, p]));
  const flat = [];
  let ok = typeof params === 'object' && !Array.isArray(params);
  const walk = (obj, pre) => {
    for (const [key, v] of Object.entries(obj)) {
      const full = pre + key;
      const p = decl.get(full);
      if (p && fits(p.type || 'string', v)) { flat.push([full, v, p.type || 'string']); continue; }
      if (v !== null && typeof v === 'object' && !Array.isArray(v) && [...decl.keys()].some(n => n.startsWith(full + '.'))) { walk(v, full + '.'); continue; }
      ok = false;
    }
  };
  if (ok) walk(params, '');
  if (!ok) {
    out.push('--json', quote(shorten(JSON.stringify(params), short)));
    return out.join(' ');
  }
  for (const [n, v, t] of flat) {
    if (t === 'boolean') { out.push(v ? '--' + n : '--' + n + ' false'); continue; }
    const text = t === 'string' ? v : JSON.stringify(v);
    if (text.startsWith('--')) out.push('--' + n + '=' + quote(shorten(text, short)));
    else out.push('--' + n + ' ' + quote(shorten(text, short)));
  }
  return out.join(' ');
}
