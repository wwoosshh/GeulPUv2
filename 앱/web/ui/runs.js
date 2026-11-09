// 앱리디자인 E3(문서/화면구현.md §3 · 화면설계.md §3 · §9) — 실행: 실험의 실행 탭(목록 · 새 실행) · 실행 화면(토큰별 출력 · 고른 스텝 · 분기 ·
// 재실행) · 비교 탭. 값은 앱이 실험 폴더에 남긴 기록 파일(records/runs/<id>.json — 받은 요청과 응답)에서 읽는다. 앱은 기능 목록
// 클라이언트(ops.js)로만 부르고, 실행은 X-GeulPU-Experiment 를 붙여 그 실험에 남긴다(앱이 응답을 받아 실행으로 남긴다 — 실험.gl).
// E4(문서/화면구현.md §4) — 스텝 · 층 화면(ui/steps.js)이 기록 읽기 · 실행 부르기 · 분기를 함께 쓴다. 점수 · 로짓 렌즈 실행의 실행 화면과 설명은 steps.js 가.
import { T, TM, el, fmtMs, samePath, sourceLabel } from './dom.js';
import { call, registry, send } from './ops.js';
import { MEASURE_OPS, measureLabel, measureView } from './steps.js';
import { compareExtras, coverTop, deltaNow, infoExtras, matchLine, summaryView, toolsView } from './research.js';

const BAR_KEY = 'geulpu.runBar';
const BARS = ['margin', 'prob', 'entropy'];
const CHAT_OPS = new Set(['run.chat', 'run.messages', 'run.responses']);
// 앱리디자인 E6 — Anthropic Messages · OpenAI Responses 형식: 서버가 시드와 GeulPU 확장 필드를 받지 않아(엔진의 형식 바꾸기) 토큰별 값이 없다
const AGENT_OPS = new Set(['run.messages', 'run.responses']);
const AGENT_KIND = { 'run.messages': 'messages', 'run.responses': 'responses' };
const MARGIN_FULL = 3;          // 막대가 꽉 차는 마진(nat)
const LIST_MAX = 200;           // 실행 탭에서 기록 파일을 읽는 실행 수(최근 것부터)

const records = new Map();      // '<실험 경로>|<기록 파일>' → {rec, text} · {error} · {loading}
const pieces = new Map();       // 토큰 번호 → 글(기록에 후보의 글이 없을 때만 /detokenize 로 — 한 번)
const ui = {
  bar: loadBar(),
  sel: new Map(),               // '<실험 경로>|<실행 id>' → 고른 스텝
  busy: null,                   // 진행 중인 실행(새 실행 · 분기 · 재실행)의 설명
  error: null,                  // {path, run, text}
  cmp: null,                    // {path, a, b}
};

function loadBar() {
  try {
    const v = localStorage.getItem(BAR_KEY);
    return BARS.includes(v) ? v : 'margin';
  } catch (_) {
    return 'margin';
  }
}
function setBar(v, ctx) {
  ui.bar = v;
  try { localStorage.setItem(BAR_KEY, v); } catch (_) { /* 기억하지 못해도 이번에는 */ }
  ctx.rerender();
}

export const appError = r => TM((r && r.json && r.json.error && r.json.error.message) || (r ? 'HTTP ' + r.status : T('앱에 닿지 못했습니다')));
export const isIds = a => Array.isArray(a) && a.every(Number.isInteger);
export const quoteTok = s => JSON.stringify(String(s));
export const fx = (x, n) => (typeof x === 'number' && Number.isFinite(x) ? x.toFixed(n) : '–');
export const pct = p => (typeof p === 'number' ? (p >= 0.1 ? (p * 100).toFixed(1) : (p * 100).toFixed(2)) + '%' : '–');

// ---- 기록 파일 ----

const recordFile = run => run.record || `records/runs/${run.id}.json`;

/** 실행의 기록 파일 — 한 번 읽어 둔다. 읽는 동안은 {loading: true}, 다 읽으면 ctx.rerender() */
export function recordOf(ctx, run) {
  const k = ctx.path + '|' + recordFile(run);
  const c = records.get(k);
  if (c) return c;
  const loading = { loading: true };
  records.set(k, loading);
  call('experiment.read', { path: ctx.path, file: recordFile(run) })
    .then(r => {
      if (r.ok && r.json) {
        try { records.set(k, { rec: JSON.parse(r.json.text), text: r.json.text }); } catch (_) { records.set(k, { error: T('기록 파일이 JSON 이 아닙니다') }); }
      } else records.set(k, { error: r.status === 404 ? T('기록 파일이 없습니다') : appError(r) });
    }, () => records.set(k, { error: T('앱에 닿지 못했습니다') }))
    .then(() => ctx.rerender());
  return loading;
}

/** 실험을 다시 읽을 때 — 읽지 못했던 기록 파일 · 분석 값은 다시 읽는다 */
export function forgetErrors(path) {
  for (const [k, v] of records) if (v.error && k.startsWith(path + '|')) records.delete(k);
  for (const [k, v] of analyses) if (v.error && k.startsWith(path + '|')) analyses.delete(k);
}

// ---- 분석 값(F3-3 — 문서/사용성시험.md §3) ----
// 요약 수치 · 비교의 요약과 처음 다른 줄 · 렌즈 문장의 값은 앱이 계산한다(experiment.summary · experiment.compare — CLI 와 같은 값). 기록
// 파일은 바뀌지 않으므로 실행 · δ · 쌍마다 한 번 부른다. 부르는 동안은 {loading: true}, 받으면 ctx.rerender()

const analyses = new Map();     // '<실험 경로>|s|<기록 파일>|<δ>' · '<실험 경로>|c|<기록 파일 a>|<기록 파일 b>' → {val} · {error} · {loading}

function analysisOf(ctx, key, opId, params) {
  const c = analyses.get(key);
  if (c) return c;
  const loading = { loading: true };
  analyses.set(key, loading);
  call(opId, params)
    .then(r => analyses.set(key, r.ok && r.json ? { val: r.json } : { error: appError(r) }), () => analyses.set(key, { error: T('앱에 닿지 못했습니다') }))
    .then(() => ctx.rerender());
  return loading;
}
/** 실행 하나의 요약(geulpu-run-summary/1) — {val} · {error} · {loading} */
export const summaryOfRun = (ctx, run, delta) =>
  analysisOf(ctx, `${ctx.path}|s|${recordFile(run)}|${delta}`, 'experiment.summary', { path: ctx.path, run: run.id, delta });
/** 두 실행의 비교(geulpu-run-compare/1) — {val} · {error} · {loading} */
export const compareOfRuns = (ctx, a, b) =>
  analysisOf(ctx, `${ctx.path}|c|${recordFile(a)}|${recordFile(b)}`, 'experiment.compare', { path: ctx.path, a: a.id, b: b.id });

export function runsOf(ctx) {
  const e = ctx.exp.experiment || {};
  return Array.isArray(e.runs) ? e.runs.filter(r => r && typeof r.id === 'string') : [];
}
export function entryOf(ctx, id) {
  return runsOf(ctx).find(r => r.id === id) || null;
}
/** 실행의 값(기록 파일을 읽었으면) — 아니면 null */
export function dataById(ctx, id) {
  const entry = entryOf(ctx, id);
  if (!entry) return null;
  const c = recordOf(ctx, entry);
  return c.rec ? dataOf(entry, c.rec, c.text) : null;
}

/** E5 — 스트리밍 실행의 기록(events — 받은 SSE 이벤트 모두)에서 스트리밍하지 않은 응답과 같은 형식을 짓는다: 채팅은 message.content 와
   logprobs.content, 텍스트 생성은 text 와 logprobs(tokens · token_logprobs · top_logprobs · text_offset). 확장 필드 geulpu 는 마지막 이벤트의 것 */
function fromStream(rec) {
  const last = rec.response && typeof rec.response === 'object' ? rec.response : null;
  if (!last || !Array.isArray(rec.events)) return last;
  let content = '';
  let text = '';
  let chat = false;
  let comp = false;
  let finish = null;
  const lc = [];
  const lt = { tokens: [], token_logprobs: [], top_logprobs: [], text_offset: [] };
  let hasLc = false;
  let hasLt = false;
  for (const e of rec.events) {
    const c = e && Array.isArray(e.choices) ? e.choices[0] : null;
    if (!c) continue;
    if (c.delta) {
      chat = true;
      if (typeof c.delta.content === 'string') content += c.delta.content;
    }
    if (typeof c.text === 'string') {
      comp = true;
      text += c.text;
    }
    if (c.finish_reason) finish = c.finish_reason;
    const lp = c.logprobs;
    if (lp && Array.isArray(lp.content)) {
      hasLc = true;
      lc.push(...lp.content);
    }
    if (lp && Array.isArray(lp.tokens)) {
      hasLt = true;
      lt.tokens.push(...lp.tokens);
      lt.token_logprobs.push(...(lp.token_logprobs || []));
      lt.top_logprobs.push(...(lp.top_logprobs || []));
      lt.text_offset.push(...(lp.text_offset || []));
    }
  }
  const out = Object.assign({}, last);
  if (chat) out.choices = [{ index: 0, message: { role: 'assistant', content }, logprobs: hasLc ? { content: lc } : null, finish_reason: finish }];
  else if (comp) out.choices = [{ index: 0, text, logprobs: hasLt ? lt : null, finish_reason: finish }];
  return out;
}

/** 기록 파일 → 화면이 쓰는 값. 스텝 k 의 값은 응답의 확장 필드 geulpu(토큰 · 상위 후보 · 엔트로피 · 로짓 해시), 글은 OpenAI 형식의 logprobs 에서 */
export function dataOf(entry, rec, text) {
  const req = rec.request && typeof rec.request === 'object' ? rec.request : {};
  const res = rec.stream && Array.isArray(rec.events) ? fromStream(rec) : rec.response && typeof rec.response === 'object' ? rec.response : null;
  const g = (res && res.geulpu) || {};
  const ch = res && Array.isArray(res.choices) ? res.choices[0] : null;
  const ids = isIds(g.tokens) ? g.tokens : [];
  const n = ids.length;
  const gq = req.geulpu && typeof req.geulpu === 'object' ? req.geulpu : {};
  const d = {
    entry, rec, req, res, g, ch, ids, n, chat: CHAT_OPS.has(rec.op),
    ptoks: isIds(g.prompt_tokens) ? g.prompt_tokens : isIds(req.prompt) ? req.prompt : null,
    off: Number.isInteger(gq.step_offset) ? gq.step_offset : 0,
    echoFrom: req.echo === true && Number.isInteger(gq.echo_from) ? gq.echo_from : null,
    // 끝 토큰(EOS)으로 멈춘 답은 그 스텝의 값이 없다(채팅의 logprobs.content 도 그렇다 — 앱리디자인 E5 의 채팅 실행에서 찾음)
    topIds: Array.isArray(g.top_ids) && g.top_ids.length >= n - 1 && n > 0 ? g.top_ids : null,
    topLp: Array.isArray(g.top_logprobs) && g.top_logprobs.length >= n - 1 && n > 0 ? g.top_logprobs : null,
    ent: Array.isArray(g.entropy) && g.entropy.length >= n - 1 && n > 0 ? g.entropy : null,
    hashes: Array.isArray(g.logit_hashes) ? g.logit_hashes : null,
    echoHashes: Array.isArray(g.prompt_logit_hashes) ? g.prompt_logit_hashes : null,
    texts: null, lps: null, alts: null, ptexts: null, seedText: null,
  };
  const lp = ch && ch.logprobs;
  if (lp && Array.isArray(lp.tokens)) {                    // 텍스트 생성 — echo 면 프롬프트 토큰이 먼저
    const m = lp.tokens.length;
    if (m >= n) {
      d.texts = lp.tokens.slice(m - n).map(String);
      d.ptexts = lp.tokens.slice(0, m - n).map(String);
      if (Array.isArray(lp.token_logprobs)) d.lps = lp.token_logprobs.slice(m - n);
      if (Array.isArray(lp.top_logprobs)) d.alts = lp.top_logprobs.slice(m - n).map(o => (o && typeof o === 'object' ? Object.entries(o) : []));
    }
  } else if (lp && Array.isArray(lp.content)) {           // 채팅
    const c = lp.content.slice(0, n);
    if (c.length === n || c.length === n - 1) {
      d.texts = c.map(x => String(x.token));
      d.lps = c.map(x => x.logprob);
      d.alts = c.map(x => (Array.isArray(x.top_logprobs) ? x.top_logprobs.map(y => [String(y.token), y.logprob]) : []));
      if (c.length === n - 1) {
        d.texts.push('');
        d.lps.push(null);
        d.alts.push([]);
      }
    }
  }
  // 토큰마다 보이는 텍스트 — 한 글자의 UTF-8 바이트가 토큰 여럿에 나뉘면 그 글자를 첫 토큰에 모으고 나머지는 null(연구 패널의 tokTexts 와 같은
  // 방식). 채팅은 content 의 bytes, 텍스트 생성은 text 와 text_offset(바이트 위치). raw 는 서버가 준 토큰 문자열 그대로(후보의 글)
  d.raw = d.texts ? d.texts.slice() : null;
  if (d.texts && lp && Array.isArray(lp.content)) {
    const parts = lp.content.slice(0, n).map(x => (Array.isArray(x.bytes) ? Uint8Array.from(x.bytes) : null));
    if (parts.length && parts.every(Boolean)) mergeBytes(parts).forEach((t, k) => { d.texts[k] = t; });
  } else if (d.texts && lp && Array.isArray(lp.text_offset) && typeof ch.text === 'string') {
    const enc = new TextEncoder().encode(ch.text);
    const offs = lp.text_offset;
    const m = lp.tokens.length;
    const ok = offs.length === m && offs.every((o, i) => Number.isInteger(o) && o >= 0 && o <= enc.length && (i === 0 || o >= offs[i - 1]));
    if (ok) {
      const merged = mergeBytes(offs.map((o, i) => enc.subarray(o, i + 1 < m ? offs[i + 1] : enc.length)));
      d.ptexts = merged.slice(0, m - n);
      d.texts = merged.slice(m - n);
    }
  }
  // 시드 — JS 의 수는 2^53 을 넘는 시드를 바꾸므로 기록 파일의 글에서 그대로 읽는다
  if (text) {
    const i = text.indexOf('"response":');
    const m = i >= 0 ? /"seed":\s*(-?\d+)/.exec(text.slice(i)) : null;
    if (m) d.seedText = m[1];
  }
  if (!d.seedText && Number.isInteger(req.seed)) d.seedText = String(req.seed);
  return d;
}

/** 토큰마다의 바이트 → 보이는 텍스트. 앞 토큰부터 1 ~ 4 토큰의 바이트를 이어 UTF-8 로 풀리는 가장 짧은 것을 그 첫 토큰에, 나머지는 null.
   네 토큰으로도 풀리지 않으면 그 토큰만 대체 문자로 */
function mergeBytes(parts) {
  const dec = new TextDecoder('utf-8', { fatal: true });
  const out = new Array(parts.length).fill(null);
  for (let i = 0; i < parts.length;) {
    let t = null;
    let j = i + 1;
    for (; j <= parts.length && j - i <= 4; j++) {
      const len = parts.slice(i, j).reduce((s, p) => s + p.length, 0);
      const buf = new Uint8Array(len);
      let o = 0;
      for (const p of parts.slice(i, j)) {
        buf.set(p, o);
        o += p.length;
      }
      try {
        t = dec.decode(buf);
        break;
      } catch (_) { /* 다음 토큰까지 */ }
    }
    if (t === null) {
      out[i] = new TextDecoder().decode(parts[i]);
      i += 1;
    } else {
      out[i] = t;
      i = j;
    }
  }
  return out;
}

/** 스텝 k 의 마진 · 샘플링된 토큰의 로그확률 · 확률 · 엔트로피 · 순위 · 로짓 해시 */
export function stepVals(d, k) {
  const top = d.topLp ? d.topLp[k] || [] : [];
  const tids = d.topIds ? d.topIds[k] || [] : [];
  const i = tids.indexOf(d.ids[k]);
  let lp = d.lps && typeof d.lps[k] === 'number' ? d.lps[k] : null;
  if (lp === null && i >= 0 && typeof top[i] === 'number') lp = top[i];
  return {
    margin: top.length > 1 ? top[0] - top[1] : null,
    lp,
    prob: lp === null ? null : Math.exp(lp),
    ent: d.ent && typeof d.ent[k] === 'number' ? d.ent[k] : null,
    rank: i >= 0 ? i + 1 : null,
    hash: d.hashes ? d.hashes[k] || null : null,
  };
}

function lowestMargin(d) {
  let at = -1;
  let best = Infinity;
  for (let k = 0; k < d.n; k++) {
    const m = stepVals(d, k).margin;
    if (m !== null && m < best) {
      best = m;
      at = k;
    }
  }
  return at;
}

/** 스텝 k 의 j 번째 후보의 글 — 샘플링된 토큰이면 그 글, 아니면 OpenAI 형식의 후보(같은 로그확률), 없으면 /detokenize 로 읽어 둔 것 */
export function candText(d, k, j) {
  const id = d.topIds[k][j];
  if (id === d.ids[k] && d.raw) return d.raw[k];
  const v = d.topLp[k][j];
  if (d.alts && d.alts[k]) {
    const hit = d.alts[k].find(a => a[1] === v);
    if (hit) return hit[0];
  }
  return pieceText(id);
}

/** /detokenize 로 읽어 둔 토큰 하나의 글 — 아직 없으면 null(fetchPieces 로 읽는다) */
export const pieceText = id => (pieces.get(id) === undefined ? null : pieces.get(id));

export function fetchPieces(ids, ctx) {
  const need = [...new Set(ids)].filter(id => !pieces.has(id));
  if (!need.length) return;
  for (const id of need) pieces.set(id, null);
  Promise.all(need.map(id => call('proxy.detokenize', { tokens: [id] }, { method: 'POST' })
    .then(r => { if (r.ok && r.json && typeof r.json.prompt === 'string') pieces.set(id, r.json.prompt); }, () => { /* 모델이 없으면 번호로 */ })))
    .then(() => ctx.rerender());
}

/** 위치(프롬프트를 포함한 토큰열의 위치) → 로짓 해시. 생성 스텝 k 는 프롬프트 길이 + k, echo 의 해시는 echo_from 부터 */
function hashMap(d) {
  const m = new Map();
  const p = d.ptoks ? d.ptoks.length : 0;
  if (d.hashes) d.hashes.forEach((h, k) => m.set(p + k, h));
  if (d.echoHashes && d.echoFrom !== null) d.echoHashes.forEach((h, j) => m.set(d.echoFrom + j, h));
  return m;
}

/** 위치 → 이 실행의 스텝 번호(분기는 원래 실행의 번호를 잇는다 — step_offset) */
const stepAt = (d, pos) => pos - (d.ptoks ? d.ptoks.length : 0) + d.off;

/** 두 실행을 토큰열(프롬프트 포함)의 위치로 맞춘다 — 같은 앞부분의 길이, 두 실행 모두 해시가 있는 위치의 같음 · 다름 */
function compareData(a, b) {
  if (!a.ptoks || !b.ptoks) return null;
  const sa = a.ptoks.concat(a.ids);
  const sb = b.ptoks.concat(b.ids);
  const m = Math.min(sa.length, sb.length);
  let L = 0;
  while (L < m && sa[L] === sb[L]) L++;
  const ha = hashMap(a);
  const hb = hashMap(b);
  const pos = [...ha.keys()].filter(p => hb.has(p)).sort((x, y) => x - y);
  let eq = 0;
  let firstDiff = -1;
  for (const p of pos) {
    if (ha.get(p) === hb.get(p)) eq++;
    else if (firstDiff < 0) firstDiff = p;
  }
  return { sa, sb, L, same: L === sa.length && L === sb.length, pos, eq, firstDiff, ha, hb };
}

// ---- 설명 ----

export function kindLabel(entry) {
  const k = entry.kind;
  if (k === 'branch') return T('분기', '종류');
  if (k === 'check') return T('재실행');
  if (k === 'score') return T('점수', '실행');
  if (k === 'lens') return T('로짓 렌즈');
  if (entry.operation === 'run.messages') return 'Anthropic Messages';
  if (entry.operation === 'run.responses') return 'OpenAI Responses';
  if (CHAT_OPS.has(entry.operation)) return T('채팅');
  if (k === 'generate') return T('텍스트 생성');
  return k || T('실행');
}

/** 실행 하나의 설명 — 종류마다(목록 · 실행 화면의 제목 아래) */
function describe(ctx, entry, d) {
  const s = entry.settings || {};
  const parts = [];
  const sampling = entry.kind !== 'score' && entry.kind !== 'lens' && !MEASURE_OPS.has(entry.operation);
  if (entry.kind === 'branch' && entry.base && entry.intervention) {
    const last = d && d.ptexts && d.ptexts.length ? d.ptexts[d.ptexts.length - 1] : null;
    const txt = last !== null ? quoteTok(last) : '#' + entry.intervention.value;
    parts.push(T`${entry.base} 의 스텝 ${entry.intervention.step} 에 ${txt}`);
  } else if (entry.kind === 'check' && entry.base) {
    const base = dataById(ctx, entry.base);
    const c = d && base ? compareData(base, d) : null;
    parts.push(c && c.pos.length ? T`${entry.base} 재실행 — 로짓 해시 ${c.eq} / ${c.pos.length} 같음` : T`${entry.base} 재실행`);
  } else if (MEASURE_OPS.has(entry.operation)) {
    const lab = measureLabel(ctx, entry);              // E4 — 'r1 의 스텝 3 · 28 층' · 'r1 의 스텝 3 에 " The" · 로그확률 −2.31'
    if (lab) return lab;
  }
  if (AGENT_OPS.has(entry.operation)) {
    if (typeof s.temperature === 'number') parts.push(T`온도 ${s.temperature}`);
    const a = d && d.res ? agentParts(entry.operation, d.res) : null;
    if (a && typeof a.outTok === 'number') parts.push(T`출력 토큰 ${a.outTok}`);
    return parts.join(' · ');
  }
  if (sampling) {
    const seed = d && d.seedText ? d.seedText : entry.seed;
    if (seed !== undefined && seed !== null) parts.push(T`시드 ${seed}`);
    if (typeof s.temperature === 'number') parts.push(T`온도 ${s.temperature}`);
    // F3-1 — 분기는 앞에서 받은 토큰(개입한 토큰까지 — step_offset)과 새로 만든 토큰을 나눠 적고, 원래 난수로 이은 위치를 적는다
    if (d && entry.kind === 'branch' && d.off > 0) parts.push(T`앞 ${d.off} 토큰 + 새 ${d.n} 토큰 = ${d.off + d.n} 토큰`, T`스텝 ${d.off} 부터 원래 난수로 이음`);
    else if (d) parts.push(T`${d.n} 토큰`);
  } else if (entry.kind !== 'check') {
    const n = d && d.ptoks ? d.ptoks.length : null;
    if (n !== null) parts.push(entry.base ? T`${entry.base} 의 토큰 ${n} 개` : T`토큰 ${n} 개`);
    else if (entry.base) parts.push(T`${entry.base} 의 토큰`);
    const L = d && d.g.layer_lens && d.g.layer_lens.layers;
    if (entry.kind === 'lens' && typeof L === 'number') parts.push(T`${L} 층`);
  }
  return parts.join(' · ');
}

function modeLabel(p) {
  return p === 'exact' ? T('정밀') : p === 'fast' ? T('빠름') : p || '–';
}

/** 실행 정보 줄 — 기록 파일의 server(모델 SHA-256 · 계산 방식) · app(GeulPU 버전) · 시드 · 출처 */
export function runInfo(ctx, d) {
  const s = d.rec.server || {};
  const models = Array.isArray((ctx.exp.experiment || {}).models) ? ctx.exp.experiment.models : [];
  const m = models.find(x => x && typeof x === 'object' && x.sha256 && x.sha256 === s.sha256);
  const name = m ? m.name || m.file : d.rec.model || '';
  const [src] = sourceLabel(d.rec.source, d.rec.session);
  return el('div', { class: 'xp-prov', 'aria-label': T('실행 정보') },
    el('span', null, T('모델', '하나'), ' ', el('b', null, name), s.sha256 ? el('code', null, ' ' + String(s.sha256).slice(0, 8)) : null),
    el('span', null, T('계산 방식'), ' ', el('b', null, modeLabel(s.precision))),
    d.rec.app ? el('span', null, 'GeulPU ', el('b', null, d.rec.app)) : null,
    d.seedText && d.entry.kind !== 'score' && d.entry.kind !== 'lens' && !MEASURE_OPS.has(d.entry.operation) ? el('span', null, T('시드'), ' ', el('b', null, d.seedText)) : null,
    el('span', null, T('출처'), ' ', el('b', null, src)),
    typeof d.rec.ms === 'number' ? el('span', null, fmtMs(d.rec.ms)) : null,
    ...infoExtras(ctx, d));
}

// ---- 실행 부르기(실험 헤더 — 앱이 실행으로 남긴다) ----

/** 이 호출의 end 이벤트가 오면 남긴 실행 id 로 — 명령 창이 넘겨 주는 이벤트(geulpu:event)에서 start 의 본문으로 이 호출을 찾는다.
   E5 — op 가 null 이면 기능은 묻지 않는다(채팅의 답 — 앱이 본문으로 기능을 가른다) */
export function awaitRun(op, exp, body) {
  let callNo = null;
  let done;
  let timer = 0;
  let canon = null;
  try { canon = JSON.stringify(JSON.parse(body)); } catch (_) { canon = null; }
  const bytes = new TextEncoder().encode(body).length;
  const promise = new Promise(res => { done = res; });
  const finish = v => {
    document.removeEventListener('geulpu:event', on);
    clearTimeout(timer);
    done(v);
  };
  const on = ev => {
    const e = ev.detail;
    if (!e) return;
    if (e.type === 'start' && callNo === null && e.source === 'gui' && (op === null || e.op === op) && samePath(e.experiment, exp)) {
      const mine = e.params !== undefined ? JSON.stringify(e.params) === canon : e.params_bytes === bytes;
      if (mine) callNo = e.call;
    } else if (e.type === 'end' && callNo !== null && e.call === callNo) finish(e.run || null);
  };
  document.addEventListener('geulpu:event', on);
  return {
    promise,
    giveUp: ms => { clearTimeout(timer); timer = setTimeout(() => finish(null), ms); },
  };
}

/** 실행 기능 하나를 실험 헤더와 함께 — 답이 오고 그 호출의 end 가 오면 {id(남긴 실행), json}. 실패하면 던진다 */
async function runCall(ctx, op, body) {
  const reg = await registry();
  const o = reg.byId.get(op);
  const wait = awaitRun(op, ctx.exp.path, body);
  let r;
  try {
    r = await send({ op, method: 'POST', path: o.http.path, body }, { experiment: ctx.exp.path });
  } catch (_) {
    wait.giveUp(0);
    throw new Error(T('앱에 닿지 못했습니다'));
  }
  if (!r.ok) {
    wait.giveUp(0);
    throw new Error(appError(r));
  }
  wait.giveUp(20000);                  // 이벤트가 늦거나 놓쳤으면 실험을 다시 읽어 가장 새 실행으로
  let id = await wait.promise;
  await ctx.reload();
  if (!id) {
    const runs = runsOf(ctx);
    id = runs.length ? runs[runs.length - 1].id : null;
  }
  return { id, json: r.json };
}

/** 시드를 글 그대로 넣은 본문(2^53 을 넘는 시드도 바뀌지 않게) */
function bodyWithSeed(obj, seedText) {
  if (!seedText) return JSON.stringify(obj);
  const mark = '__geulpu_seed__';
  return JSON.stringify(Object.assign({}, obj, { seed: mark })).replace(JSON.stringify(mark), seedText);
}

/** 기록 파일의 글에서 받은 요청을 바이트 그대로(재실행 — 앱이 같은 요청으로 알아보게) */
export function rawRequest(text) {
  const i = text.indexOf('"request":');
  if (i < 0) return null;
  let k = i + 10;
  while (k < text.length && /\s/.test(text[k])) k++;
  const start = k;
  let depth = 0;
  let inStr = false;
  for (; k < text.length; k++) {
    const c = text[k];
    if (inStr) {
      if (c === '\\') k++;
      else if (c === '"') inStr = false;
    } else if (c === '"') inStr = true;
    else if (c === '{' || c === '[') depth++;
    else if (c === '}' || c === ']') {
      depth--;
      if (depth === 0) return text.slice(start, k + 1);
    }
  }
  return null;
}

const SAMPLING = ['temperature', 'top_p', 'top_k', 'min_p', 'presence_penalty', 'frequency_penalty', 'repeat_penalty', 'ignore_eos'];

function topCount(d) {
  const gq = d.req.geulpu && typeof d.req.geulpu === 'object' ? d.req.geulpu : {};
  if (Number.isInteger(gq.top_logprobs) && gq.top_logprobs > 0) return gq.top_logprobs;
  if (Number.isInteger(d.req.logprobs) && d.req.logprobs > 0) return d.req.logprobs;
  if (Number.isInteger(d.req.top_logprobs) && d.req.top_logprobs > 0) return d.req.top_logprobs;
  return 2;
}

/** 분기 요청 — docs/cli.md 의 분기: 프롬프트 토큰 + 답의 앞 k 토큰 + 고른 후보, step_offset = k + 1(원래 실행의 난수), 같은 시드 · 샘플링 설정,
   echo 로 앞부분의 로짓 해시도 */
function branchBody(d, k, tok) {
  const K = topCount(d);
  const max = Number.isInteger(d.req.max_tokens) ? d.req.max_tokens : d.n;
  const o = { prompt: d.ptoks.concat(d.ids.slice(0, k), [tok]), max_tokens: Math.max(1, max - (k + 1)) };
  for (const f of SAMPLING) if (d.req[f] !== undefined) o[f] = d.req[f];
  o.echo = true;
  o.logprobs = K;
  o.geulpu = { echo_from: d.ptoks.length, step_offset: d.off + k + 1, logit_hash: true, top_logprobs: K, prompt_tokens: true };
  if (d.req.geulpu && d.req.geulpu.special === true) o.geulpu.special = true;     // E6-2 — 특수 토큰(생각 토큰)의 글도 받는다
  return bodyWithSeed(o, d.seedText);
}

export async function doBranch(ctx, d, k, tok) {
  if (ui.busy) return;
  ui.busy = T`${d.entry.id} 의 스텝 ${d.off + k} 에서 분기하는 중`;
  ui.error = null;
  ctx.rerender();
  try {
    const { id } = await runCall(ctx, 'run.branch', branchBody(d, k, tok));
    ui.busy = null;
    if (id) ctx.openCompare(d.entry.id, id);
    else ctx.rerender();
  } catch (e) {
    ui.busy = null;
    ui.error = { path: ctx.path, run: d.entry.id, text: e.message };
    ctx.rerender();
  }
}

/** E4 — 스텝 · 층 화면의 계산(점수 · 로짓 렌즈) 하나를 실험의 실행으로. 진행 중 표시 · 실패는 분기와 같은 줄에. 남긴 실행 id(실패면 null) */
export async function measure(ctx, op, body, label, runId) {
  if (ui.busy) return null;
  ui.busy = label;
  ui.error = null;
  ctx.rerender();
  try {
    const { id } = await runCall(ctx, op, body);
    ui.busy = null;
    ctx.rerender();
    return id;
  } catch (e) {
    ui.busy = null;
    ui.error = { path: ctx.path, run: runId, text: e.message };
    ctx.rerender();
    return null;
  }
}

/** 진행 중인 실행이 있는가(스텝 · 층 화면의 버튼) */
export const isBusy = () => !!ui.busy;

/** 실행 화면에서 고른 스텝(스텝 · 층 화면에서 돌아갈 때 그 스텝으로) */
export function selectStep(ctx, id, k) {
  ui.sel.set(ctx.path + '|' + id, k);
}

async function doRerun(ctx, d, text) {
  if (ui.busy) return;
  const body = rawRequest(text);
  if (!body) return;
  ui.busy = T`재실행하는 중 — ${d.entry.id}`;
  ui.error = null;
  ctx.rerender();
  try {
    const { id } = await runCall(ctx, d.rec.op, body);
    ui.busy = null;
    if (id) ctx.openCompare(d.entry.id, id);
    else ctx.rerender();
  } catch (e) {
    ui.busy = null;
    ui.error = { path: ctx.path, run: d.entry.id, text: e.message };
    ctx.rerender();
  }
}

// ---- 새 실행 ----

function field(label, input, hint) {
  // F3-1 — 라벨로 감싼 입력의 접근성 이름에 입력의 값이 섞이지 않게 라벨 글을 이름으로
  if (typeof label === 'string' && input && input.setAttribute && !input.getAttribute('aria-label')) input.setAttribute('aria-label', label);
  return el('label', { class: 'run-field' }, el('span', null, label), input, hint ? el('small', { class: 'muted' }, hint) : null);
}
const num = (v, int) => {
  const s = String(v).trim();
  if (!s) return undefined;
  const x = Number(s);
  if (!Number.isFinite(x) || (int && !Number.isInteger(x))) return null;
  return x;
};

/** 새 실행 창 — prefill 은 '이 설정으로 새 실행'의 실행 값 */
export async function newRun(ctx, prefill) {
  const req = prefill ? prefill.req : {};
  const kind0 = prefill ? AGENT_KIND[prefill.entry.operation] || (prefill.chat ? 'chat' : 'complete') : 'complete';
  const chatInit = kind0 !== 'complete';
  const msgs = Array.isArray(req.messages) ? req.messages : [];
  const textOf = m => (typeof m.content === 'string' ? m.content : Array.isArray(m.content) ? m.content.map(x => (x && x.text) || '').join('') : '');
  const sys0 = typeof req.system === 'string' ? req.system : typeof req.instructions === 'string' ? req.instructions
    : msgs.filter(m => m && m.role === 'system').map(textOf).join('\n');
  const user0 = typeof req.input === 'string' ? req.input : msgs.filter(m => m && m.role === 'user').map(textOf).pop() || '';
  // E6-2 — 채팅 실행의 앞의 대화(시스템 메시지와 마지막 사용자 메시지 밖)와 생각 설정 · 특수 토큰의 글을 이어받는다(연구 패널의 '설정을 바꿔 다시')
  const lastUser = msgs.map(m => m && m.role).lastIndexOf('user');
  const earlier = kind0 === 'chat' && lastUser > 0 ? msgs.slice(0, lastUser).filter(m => m && m.role !== 'system') : [];
  const ctk = req.chat_template_kwargs && typeof req.chat_template_kwargs === 'object' ? req.chat_template_kwargs : null;
  const think0 = ctk && typeof ctk.enable_thinking === 'boolean' ? (ctk.enable_thinking ? 'on' : 'off') : 'default';
  const special0 = !!(req.geulpu && typeof req.geulpu === 'object' && req.geulpu.special === true);
  const keepEarlier = el('input', { type: 'checkbox', id: 'nr-earlier', checked: earlier.length ? true : null });
  const earlierRow = earlier.length ? el('label', { class: 'run-check' }, keepEarlier, ' ', T`앞의 대화 ${earlier.length} 메시지를 함께 보냅니다`) : null;
  const think = el('select', { id: 'nr-think', class: 'input' },
    el('option', { value: 'default', selected: think0 === 'default' ? true : null }, T('모델 기본')),
    el('option', { value: 'on', selected: think0 === 'on' ? true : null }, T('켬')),
    el('option', { value: 'off', selected: think0 === 'off' ? true : null }, T('끔')));
  const kind = el('select', { id: 'nr-kind', class: 'input' },
    el('option', { value: 'complete', selected: kind0 === 'complete' ? true : null }, T('텍스트 생성')),
    el('option', { value: 'chat', selected: kind0 === 'chat' ? true : null }, T('채팅')),
    el('option', { value: 'messages', selected: kind0 === 'messages' ? true : null }, 'Anthropic Messages'),
    el('option', { value: 'responses', selected: kind0 === 'responses' ? true : null }, 'OpenAI Responses'));
  const prompt = el('textarea', { id: 'nr-prompt', rows: '4', spellcheck: 'false' });
  prompt.value = typeof req.prompt === 'string' ? req.prompt : '';
  const system = el('textarea', { id: 'nr-system', rows: '2', spellcheck: 'false' });
  system.value = sys0;
  const user = el('textarea', { id: 'nr-user', rows: '4', spellcheck: 'false' });
  user.value = user0;
  const v = (k, dflt) => (req[k] !== undefined ? String(req[k]) : dflt);
  const maxT = el('input', { type: 'number', min: '1', step: '1', value: req.max_output_tokens !== undefined ? String(req.max_output_tokens) : v('max_tokens', '64') });
  const temp = el('input', { type: 'number', min: '0', step: '0.05', value: v('temperature', '0.8') });
  const seed = el('input', { type: 'text', inputmode: 'numeric', value: prefill && prefill.seedText ? prefill.seedText : String(1 + Math.floor(Math.random() * 2147483646)) });
  const topP = el('input', { type: 'number', min: '0', max: '1', step: '0.05', value: v('top_p', ''), placeholder: T('기본') });
  const topK = el('input', { type: 'number', min: '0', step: '1', value: v('top_k', ''), placeholder: T('기본') });
  const minP = el('input', { type: 'number', min: '0', max: '1', step: '0.01', value: v('min_p', ''), placeholder: T('기본') });
  const cands = el('input', { type: 'number', min: '2', max: '20', step: '1', value: prefill ? String(topCount(prefill)) : '2' });
  const err = el('p', { class: 'xp-err', role: 'alert' });
  const note = el('p', { class: 'muted run-note' });
  const promptBox = el('div', { class: 'run-form-part' }, field(T('프롬프트'), prompt));
  // Messages 의 토큰 수 세기(run.messages.count_tokens — 실험의 명령 기록에 남는다)
  const countOut = el('span', { class: 'muted small' });
  const countRow = el('div', { class: 'row g8' }, el('button', {
    class: 'btn ghost sm', type: 'button',
    onclick: async () => {
      const o = { messages: [{ role: 'user', content: user.value }] };
      if (system.value.trim()) o.system = system.value;
      countOut.textContent = T('세는 중…');
      try {
        const r = await call('run.messages.count_tokens', o, { experiment: ctx.exp.path });
        countOut.textContent = r.ok && r.json && typeof r.json.input_tokens === 'number' ? T`입력 토큰 ${r.json.input_tokens}` : appError(r);
      } catch (_) { countOut.textContent = T('앱에 닿지 못했습니다'); }
    },
  }, T('토큰 수 세기')), countOut);
  const chatBox = el('div', { class: 'run-form-part' }, field(T('시스템 프롬프트'), system, T('비우면 넣지 않습니다')), earlierRow, field(T('사용자 메시지'), user),
    field(T('생각(thinking)'), think, T('채팅 템플릿의 생각 켬 · 끔 — 모델 기본이면 보내지 않습니다')), countRow);
  const fSeed = field(T('시드'), seed), fTopK = field('top-k', topK), fMinP = field('min-p', minP), fCands = field(T('상위 후보 수'), cands, T('마진에는 2 가 필요합니다'));
  const noteGeulpu = el('p', { class: 'muted' }, T('요청에는 늘 logprobs · geulpu.top_logprobs · geulpu.logit_hash · geulpu.prompt_tokens 를 넣습니다. 결과는 이 실험의 실행으로 남습니다.'));
  const noteAgent = el('p', { class: 'muted' }, T('이 형식은 서버가 시드와 GeulPU 확장 필드를 받지 않습니다 — 결과는 이 실험의 실행으로 남지만 답의 텍스트 · 도구 호출 · 사용량만 있고 토큰별 값은 없습니다. 온도 0 이면 같은 요청에 같은 답입니다.'));
  const sync = () => {
    const k = kind.value, agent = k === 'messages' || k === 'responses';
    promptBox.hidden = k !== 'complete';
    chatBox.hidden = k === 'complete';
    countRow.hidden = k !== 'messages';
    for (const f of [fSeed, fTopK, fMinP, fCands]) f.hidden = agent;
    noteGeulpu.hidden = agent;
    noteAgent.hidden = !agent;
  };
  kind.addEventListener('change', sync);
  sync();
  const form = el('div', { class: 'xp-form run-form' },
    el('h2', null, T('새 실행')),
    note,
    field(T('종류'), kind),
    promptBox, chatBox,
    el('div', { class: 'run-form-grid' },
      field(T('최대 토큰'), maxT), field(T('온도'), temp), fSeed,
      field('top-p', topP), fTopK, fMinP, fCands),
    noteGeulpu, noteAgent,
    err);
  const ok = el('button', {
    class: 'btn primary', type: 'button',
    onclick: async () => {
      const chat = kind.value === 'chat';
      const agent = kind.value === 'messages' || kind.value === 'responses';
      const vals = agent ? { max_tokens: num(maxT.value, true), temperature: num(temp.value, false), top_p: num(topP.value, false) }
        : { max_tokens: num(maxT.value, true), temperature: num(temp.value, false), top_p: num(topP.value, false), top_k: num(topK.value, true), min_p: num(minP.value, false) };
      const K = num(cands.value, true);
      const bad = Object.entries(vals).find(([, x]) => x === null);
      if (bad || !vals.max_tokens || vals.max_tokens < 1 || vals.temperature === undefined || (!agent && (!K || K < 2 || K > 20))) {
        err.textContent = T('숫자를 확인하세요 — 최대 토큰은 1 이상, 상위 후보 수는 2 ~ 20');
        return;
      }
      if (agent) {
        if (!user.value.trim()) {
          err.textContent = T('사용자 메시지를 적으세요');
          return;
        }
        const a = {};
        if (kind.value === 'messages') {
          a.max_tokens = vals.max_tokens;
          if (system.value.trim()) a.system = system.value;
          a.messages = [{ role: 'user', content: user.value }];
        } else {
          if (system.value.trim()) a.instructions = system.value;
          a.input = user.value;
          a.max_output_tokens = vals.max_tokens;
        }
        if (think.value === 'off') {
          if (kind.value === 'messages') a.thinking = { type: 'disabled' };
          else a.reasoning = { effort: 'none' };
        }
        a.temperature = vals.temperature;
        if (vals.top_p !== undefined) a.top_p = vals.top_p;
        ok.disabled = true;
        err.textContent = '';
        note.textContent = T('실행하는 중…');
        try {
          const { id } = await runCall(ctx, kind.value === 'messages' ? 'run.messages' : 'run.responses', JSON.stringify(a));
          globalThis.closeModal();
          if (id) ctx.openRun(id);
        } catch (e) {
          note.textContent = '';
          err.textContent = e.message;
          ok.disabled = false;
        }
        return;
      }
      const sd = seed.value.trim();
      if (sd && !/^\d{1,20}$/.test(sd)) {
        err.textContent = T('시드는 0 이상의 정수입니다');
        return;
      }
      const o = {};
      if (chat) {
        if (!user.value.trim()) {
          err.textContent = T('사용자 메시지를 적으세요');
          return;
        }
        o.messages = (system.value.trim() ? [{ role: 'system', content: system.value }] : []).concat(keepEarlier.checked ? earlier : [],
          [{ role: 'user', content: user.value }]);
      } else {
        if (!prompt.value) {
          err.textContent = T('프롬프트를 적으세요');
          return;
        }
        o.prompt = prompt.value;
      }
      for (const [k2, x] of Object.entries(vals)) if (x !== undefined) o[k2] = x;
      if (chat) {
        o.logprobs = true;
        o.top_logprobs = K;
      } else o.logprobs = K;
      o.geulpu = { top_logprobs: K, logit_hash: true, prompt_tokens: true };
      if (chat && special0) o.geulpu.special = true;
      if (chat && think.value !== 'default') o.chat_template_kwargs = Object.assign({}, ctk || {}, { enable_thinking: think.value === 'on' });
      else if (chat && ctk) o.chat_template_kwargs = ctk;
      const body = bodyWithSeed(o, sd || null);
      ok.disabled = true;
      err.textContent = '';
      note.textContent = T('실행하는 중…');
      try {
        const { id } = await runCall(ctx, chat ? 'run.chat' : 'run.complete', body);
        globalThis.closeModal();
        if (id) ctx.openRun(id);
      } catch (e) {
        note.textContent = '';
        err.textContent = e.message;
        ok.disabled = false;
      }
    },
  }, T('실행'));
  const cancel = el('button', { class: 'btn ghost', type: 'button', onclick: () => globalThis.closeModal() }, T('취소'));
  globalThis.modal(form, [cancel, ok], true, chatInit ? '#nr-user' : '#nr-prompt');
  // 불러온 모델 — 없으면 모델 화면으로 가는 안내(화면설계.md §8)
  try {
    const s = await call('system.state');
    const sv = s.json && s.json.server;
    if (!(sv && sv.state === 'ready' && sv.model)) {
      note.replaceChildren(T('불러온 모델이 없습니다 — 모델 화면에서 모델을 불러온 뒤 실행하세요. '),
        el('button', { type: 'button', class: 'linkish', onclick: () => { globalThis.closeModal(); if (typeof globalThis.show === 'function') globalThis.show('models'); } }, T('모델 화면으로')));
      ok.disabled = true;
    } else note.textContent = T`불러온 모델: ${sv.model}`;
  } catch (_) { /* 앱에 닿지 못하면 실행할 때 까닭이 나온다 */ }
}

// ---- 실행 탭 ----

/** F3-1 — 생성 실행의 프롬프트(채팅은 마지막 사용자 메시지) 앞부분 — 실행 목록에서 같은 설정의 실행끼리 가려 보게 */
function promptHead(entry, d) {
  if (!d || !d.req || (entry.kind !== 'generate' && entry.kind !== 'check')) return null;
  const q = d.req;
  let s = typeof q.prompt === 'string' ? q.prompt : null;
  const msgs = Array.isArray(q.messages) ? q.messages : null;
  if (s === null && msgs) {
    const u = [...msgs].reverse().find(m => m && m.role === 'user');
    const c = u ? u.content : null;
    s = typeof c === 'string' ? c : Array.isArray(c) ? c.map(x => (x && typeof x.text === 'string' ? x.text : '')).join('') : null;
  }
  if (s === null && typeof q.input === 'string') s = q.input;
  if (!s) return null;
  s = s.replace(/\s+/g, ' ').trim();
  return s.length > 56 ? s.slice(0, 55) + '…' : s;
}

function runItem(ctx, entry) {
  const c = recordOf(ctx, entry);
  const d = c.rec ? dataOf(entry, c.rec, c.text) : null;
  const [src, kindCls] = d ? sourceLabel(d.rec.source, d.rec.session) : ['', ''];
  const head = c.error ? null : promptHead(entry, d);
  return el('button', { class: 'xp-item run-item', role: 'listitem', type: 'button', onclick: () => ctx.openRun(entry.id) },
    el('span', { class: 'rid' }, entry.id),
    el('span', { class: 't' }, el('span', { class: 'k' }, kindLabel(entry)), ' ', el('span', null, c.error ? c.error : describe(ctx, entry, d)),
      head ? el('span', { class: 'run-prompt muted' }, ' · “' + head + '”') : null),
    el('span', { class: 'meta' },
      d && typeof d.rec.ms === 'number' ? el('span', null, fmtMs(d.rec.ms)) : null,
      src ? el('span', { class: 'src ' + kindCls }, src) : null));
}

export function runsTab(ctx) {
  const runs = runsOf(ctx);
  const out = [
    el('div', { class: 'xp-actions' },
      el('button', { class: 'btn primary', type: 'button', onclick: () => newRun(ctx, null) }, T('새 실행')),
      runs.length >= 2 ? el('button', { class: 'btn', type: 'button', onclick: () => ctx.openCompare(null, null) }, T('실행 둘 비교')) : null,
      el('span', { class: 'muted' }, T`CLI 에서는 실행 명령에 이 옵션을 붙이면 여기에 남습니다: --experiment ${(ctx.exp.experiment || {}).id || ''}`)),
  ];
  if (!runs.length) {
    out.push(el('div', { class: 'empty-box' }, el('div', { class: 'big' }, T('아직 실행이 없습니다')),
      el('div', null, T('새 실행을 하면 결과가 이 실험의 records/runs 에 남습니다.'))));
    return out;
  }
  const recent = runs.slice(-LIST_MAX);
  out.push(el('div', { class: 'xp-list', role: 'list', 'aria-label': T('실행 목록') }, recent.map(r => runItem(ctx, r))));
  if (runs.length > LIST_MAX) out.push(el('p', { class: 'muted' }, T`최근 실행 ${LIST_MAX} 개만 보입니다(모두 ${runs.length} 개).`));
  return out;
}

// ---- 실행 화면 ----

function barWidth(d, v, maxEnt) {
  let x = null;
  if (ui.bar === 'margin') x = v.margin === null ? null : Math.min(1, v.margin / MARGIN_FULL);
  else if (ui.bar === 'prob') x = v.prob;
  else x = v.ent === null ? null : v.ent / (maxEnt || 1);
  return x === null ? 0 : Math.max(0.06, Math.min(1, x));
}

function legend() {
  const t = { margin: T('막대 길이 = 마진(1 위와 2 위 로그확률의 차) — 짧을수록 마진이 작은 스텝'), prob: T('막대 길이 = 샘플링된 토큰의 확률'), entropy: T('막대 길이 = 엔트로피(이 실행에서 가장 큰 값에 맞춤)') }[ui.bar];
  return t;
}

function barSwitch(ctx) {
  const names = { margin: T('마진'), prob: T('확률'), entropy: T('엔트로피') };
  return el('div', { class: 'run-bars' },
    el('span', { class: 'muted', 'aria-hidden': 'true' }, T('막대')),
    el('div', { class: 'seg-ctl', role: 'radiogroup', 'aria-label': T('토큰 아래 막대') },
      BARS.map(b => el('button', {
        type: 'button', role: 'radio', class: ui.bar === b ? 'on' : null, 'aria-checked': String(ui.bar === b),
        onclick: () => setBar(b, ctx),
      }, names[b]))));
}

/** 프롬프트 부분(흐리게) — 문자열 프롬프트 · 채팅 메시지 · echo 로 받은 토큰(분기는 바꾼 토큰을 표시) */
function promptPart(ctx, d) {
  const out = [];
  if (typeof d.req.prompt === 'string') out.push(el('span', { class: 'run-p' }, d.req.prompt));
  else if (Array.isArray(d.req.messages)) {
    for (const m of d.req.messages) {
      if (!m || typeof m !== 'object') continue;
      const c = typeof m.content === 'string' ? m.content : Array.isArray(m.content) ? m.content.map(x => (x && x.text) || '').join('') : '';
      out.push(el('span', { class: 'run-p run-msg' }, `${m.role || ''}: ${c}`));
    }
  } else if (d.ptexts && d.ptexts.length) {
    const changed = d.entry.kind === 'branch' ? d.ptexts.length - 1 : -1;
    d.ptexts.forEach((t, i) => out.push(el('span', {
      class: 'run-p' + (i === changed ? ' changed' : ''), title: i === changed ? T('바꾼 토큰') : null,
    }, t)));
  } else if (d.ptoks) out.push(el('span', { class: 'run-p' }, T`(프롬프트 토큰 ${d.ptoks.length} 개)`));
  return out;
}

function stripView(ctx, d, sel, low, onSel) {
  const dlt = deltaNow();
  let maxEnt = 0;
  if (d.ent) for (let k = 0; k < d.n; k++) maxEnt = Math.max(maxEnt, d.ent[k] || 0);
  const toks = [];
  for (let k = 0; k < d.n; k++) {
    const v = stepVals(d, k);
    const w = barWidth(d, v, maxEnt);
    const text = d.texts ? d.texts[k] : '#' + d.ids[k];
    const raw = d.raw ? d.raw[k] : text;
    const label = T`스텝 ${d.off + k}: ${quoteTok(text === null ? raw : text)}, 마진 ${fx(v.margin, 3)}, 확률 ${pct(v.prob)}`;
    toks.push(el('span', {
      class: 'run-tok' + (text === null ? ' cont' : '') + (k === sel ? ' sel' : '') + (k === low ? ' low' : '') + (v.rank !== null && v.rank > 1 ? ' notop' : '')
        + (v.margin !== null && v.margin < dlt ? ' dlt' : ''),
      'data-k': String(k), role: 'button', tabindex: k === sel ? '0' : '-1', 'aria-label': label, 'aria-pressed': String(k === sel),
      'data-note': k === low ? T`마진이 가장 작은 스텝 · ${fx(v.margin, 3)}` : null,
    }, text === null ? '' : text, el('span', { class: 'bar', style: `width:${(w * 100).toFixed(1)}%`, 'aria-hidden': 'true' })));
  }
  const box = el('div', { class: 'run-strip', role: 'group', 'aria-label': T('토큰별 출력') }, promptPart(ctx, d), toks);
  box.addEventListener('click', ev => {
    const t = ev.target.closest('.run-tok');
    if (t) onSel(Number(t.dataset.k), true);
  });
  box.addEventListener('keydown', ev => {
    if (ev.key === 'ArrowRight' || ev.key === 'ArrowLeft') {
      ev.preventDefault();
      onSel(Math.max(0, Math.min(d.n - 1, sel + (ev.key === 'ArrowRight' ? 1 : -1))), true);
    } else if (ev.key === 'Home' || ev.key === 'End') {
      ev.preventDefault();
      onSel(ev.key === 'Home' ? 0 : d.n - 1, true);
    } else if (ev.key === 'Enter' && ev.target.closest('.run-tok')) {      // E4 — 스텝 · 층 화면
      ev.preventDefault();
      ctx.openStep(d.entry.id, sel);
    } else if (ev.key === 'b' && !ev.ctrlKey && !ev.metaKey && !ev.altKey) {
      const alt = bestAlt(d, sel);
      if (alt !== null && d.ptoks) {
        ev.preventDefault();
        doBranch(ctx, d, sel, alt);
      }
    }
  });
  return box;
}

/** 샘플링되지 않은 후보 가운데 가장 높은 것(키 b) */
function bestAlt(d, k) {
  if (!d.topIds) return null;
  const c = (d.topIds[k] || []).find(id => id !== d.ids[k]);
  return c === undefined ? null : c;
}

function stepPanel(ctx, d, k, text) {
  const v = stepVals(d, k);
  const kids = [el('h3', null, T`스텝 ${d.off + k}`)];
  const cov = coverTop(ctx, d, k);
  if (cov && (!d.topIds || !Array.isArray(d.topIds[k]) || cov.ids.length > d.topIds[k].length)) {
    const missing = [];
    const rows = cov.ids.map((id, j) => {
      const lp = cov.lps[j];
      const t = id === d.ids[k] && d.texts ? d.texts[k] : pieceText(id);
      if (t === null) missing.push(id);
      const sampled = id === d.ids[k];
      return el('div', { class: 'run-cand' + (sampled ? ' sampled' : '') },
        el('span', { class: 'w' }, t === null ? '#' + id : quoteTok(t), sampled ? el('span', { class: 'tag dim' }, T('샘플링됨')) : null),
        el('span', { class: 'lp' }, fx(lp, 3)),
        el('div', { class: 'pb', 'aria-hidden': 'true' }, el('span', { style: `width:${Math.max(1, Math.exp(lp) * 100).toFixed(1)}%` })),
        sampled ? null : el('button', {
          type: 'button', class: 'btn sm', disabled: ui.busy || !d.ptoks ? true : null, onclick: () => doBranch(ctx, d, k, id),
        }, T('이 토큰으로 분기')));
    });
    if (missing.length) fetchPieces(missing, ctx);
    kids.push(el('div', { class: 'muted run-sub' }, T`상위 후보 ${cov.ids.length} 개 · 로그확률 — 점수 실행 ${cov.run}`), el('div', { class: 'run-cands' }, rows));
  } else if (d.topIds && d.topLp) {
    const tids = d.topIds[k] || [];
    const missing = [];
    const rows = tids.map((id, j) => {
      const lp = d.topLp[k][j];
      const t = candText(d, k, j);
      if (t === null) missing.push(id);
      const sampled = id === d.ids[k];
      return el('div', { class: 'run-cand' + (sampled ? ' sampled' : '') },
        el('span', { class: 'w' }, t === null ? '#' + id : quoteTok(t), sampled ? el('span', { class: 'tag dim' }, T('샘플링됨')) : null),
        el('span', { class: 'lp' }, fx(lp, 3)),
        el('div', { class: 'pb', 'aria-hidden': 'true' }, el('span', { style: `width:${Math.max(1, Math.exp(lp) * 100).toFixed(1)}%` })),
        sampled ? null : el('button', {
          type: 'button', class: 'btn sm', disabled: ui.busy || !d.ptoks ? true : null,
          title: d.ptoks ? null : T('이 실행은 프롬프트 토큰을 기록하지 않아 분기할 수 없습니다(geulpu.prompt_tokens)'),
          onclick: () => doBranch(ctx, d, k, id),
        }, T('이 토큰으로 분기')));
    });
    if (missing.length) fetchPieces(missing, ctx);
    kids.push(el('div', { class: 'muted run-sub' }, T('상위 후보 · 로그확률(샘플링 설정 전 — 모델의 분포)')), el('div', { class: 'run-cands' }, rows));
  } else {
    kids.push(el('p', { class: 'muted' }, T('이 실행에는 스텝마다의 후보가 없습니다 — logprobs 와 geulpu.top_logprobs 를 넣어 실행하면 보입니다.')));
  }
  kids.push(el('dl', { class: 'run-kv' },
    el('dt', null, T('마진')), el('dd', null, v.margin === null ? '–' : fx(v.margin, 4) + ' nat'),
    el('dt', null, T('샘플링된 토큰의 확률')), el('dd', null, pct(v.prob) + (v.rank ? T` · ${v.rank} 위` : '')),
    el('dt', null, T('엔트로피')), el('dd', null, v.ent === null ? '–' : fx(v.ent, 3) + ' nat'),
    el('dt', null, T('로짓 해시')), el('dd', null, v.hash ? el('code', { class: 'hash', title: v.hash }, String(v.hash).slice(0, 8)) : '–')));
  kids.push(el('div', { class: 'xp-actions' },
    el('button', { type: 'button', class: 'btn primary', onclick: () => ctx.openStep(d.entry.id, k) }, T('스텝 · 층 화면')),
    el('button', { type: 'button', class: 'btn', disabled: ui.busy || !rawRequest(text) ? true : null, onclick: () => doRerun(ctx, d, text) }, T('재실행')),
    el('button', { type: 'button', class: 'btn ghost', onclick: () => newRun(ctx, d) }, T('이 설정으로 새 실행'))));
  return el('aside', { class: 'run-step', 'aria-live': 'polite', 'aria-label': T('고른 스텝') }, kids);
}

export function busyLine(ctx, id) {
  const out = [];
  if (ui.busy) out.push(el('p', { class: 'xp-check' }, el('span', { class: 'tag acc' }, T('실행 중')), ui.busy));
  if (ui.error && ui.error.path === ctx.path && (!id || ui.error.run === id)) {
    out.push(el('p', { class: 'xp-check' }, el('span', { class: 'tag err' }, T('실패')), ui.error.text));
  }
  return out;
}

/** Anthropic Messages · OpenAI Responses 응답의 부분(답 · 도구 호출)과 사용량 · 끝난 까닭 */
function agentParts(op, res) {
  const parts = [];
  const u = (res && res.usage) || {};
  if (op === 'run.messages') {
    for (const b of Array.isArray(res.content) ? res.content : []) {
      if (!b || typeof b !== 'object') continue;
      if (b.type === 'text') parts.push(['text', String(b.text || '')]);
      else if (b.type === 'tool_use') parts.push(['tool', `${b.name}(${JSON.stringify(b.input)})`]);
    }
    // Anthropic 의 input_tokens 는 캐시에서 읽지 않은 것만 — 캐시에서 읽은 것 · 캐시에 쓴 것과 더한 것이 입력 전체
    const cached = u.cache_read_input_tokens || 0;
    const inTok = typeof u.input_tokens === 'number' ? u.input_tokens + cached + (u.cache_creation_input_tokens || 0) : undefined;
    return { parts, stop: res.stop_reason, inTok, cached, outTok: u.output_tokens };
  }
  for (const it of Array.isArray(res.output) ? res.output : []) {
    if (!it || typeof it !== 'object') continue;
    if (it.type === 'message') {
      for (const c of Array.isArray(it.content) ? it.content : []) if (c && c.type === 'output_text') parts.push(['text', String(c.text || '')]);
    } else if (it.type === 'function_call') parts.push(['tool', `${it.name}(${it.arguments})`]);
  }
  return { parts, stop: res.status, inTok: u.input_tokens, cached: (u.input_tokens_details && u.input_tokens_details.cached_tokens) || 0, outTok: u.output_tokens };
}

function agentView(ctx, entry, d, text) {
  const a = agentParts(entry.operation, d.res || {});
  const label = { text: T('답'), tool: T('도구 호출') };
  return [
    el('p', { class: 'note-line' }, T('이 형식은 서버가 시드와 GeulPU 확장 필드를 받지 않아 토큰별 값(확률 · 마진 · 로짓 해시)이 없습니다. 온도 0 이면 같은 요청에 같은 답입니다.')),
    ...(a.parts.length ? a.parts.map(([k, t]) => el('div', { class: 'run-agent-part' }, el('div', { class: 'k' }, label[k]), el('pre', { class: 'xp-pre' }, t)))
      : [el('p', { class: 'muted' }, T('답에 텍스트가 없습니다.'))]),
    el('p', { class: 'muted small' }, T`입력 토큰 ${a.inTok ?? '-'}` + (a.cached ? ' ' + T`(캐시에서 ${a.cached})` : '') + ' · ' + T`출력 토큰 ${a.outTok ?? '-'}` + ' · ' + T`끝난 까닭 ${a.stop || '-'}`),
    el('div', { class: 'xp-actions' },
      el('button', { type: 'button', class: 'btn', disabled: ui.busy || !rawRequest(text) ? true : null, onclick: () => doRerun(ctx, d, text) }, T('재실행')),
      el('button', { type: 'button', class: 'btn ghost', onclick: () => newRun(ctx, d) }, T('이 설정으로 새 실행'))),
    el('h2', null, T('요청', '하나')), el('pre', { class: 'xp-pre' }, JSON.stringify(d.req, null, 2)),
    el('details', { class: 'run-raw' }, el('summary', null, T('응답(JSON)')), el('pre', { class: 'xp-pre' }, JSON.stringify(d.res, null, 2))),
  ];
}

/** 경로에 쓰는 실행 화면의 이름 */
export function runTitle(ctx, id) {
  const entry = entryOf(ctx, id);
  return entry ? `${T('실행', '경로')} ${id}` : id;
}

export function runView(ctx, id) {
  const entry = entryOf(ctx, id);
  if (!entry) return [el('div', { class: 'empty-box' }, el('div', { class: 'big' }, T`이 실험에 없는 실행입니다: ${id}`))];
  const c = recordOf(ctx, entry);
  const d = c.rec ? dataOf(entry, c.rec, c.text) : null;
  const head = [
    el('h1', null, `${id} `, el('span', { class: 'run-kind' }, kindLabel(entry))),
    el('p', { class: 'lead muted' }, describe(ctx, entry, d)),
  ];
  if (c.loading) return [...head, el('p', { class: 'muted' }, T('읽는 중…'))];
  if (c.error) return [...head, el('div', { class: 'empty-box' }, el('div', { class: 'big' }, T('기록 파일을 읽지 못했습니다')), el('div', null, c.error))];
  const out = [...head, runInfo(ctx, d), ...busyLine(ctx, id)];
  const mline = d && d.n && !AGENT_OPS.has(entry.operation) ? matchLine(ctx, d) : null;
  if (mline) out.push(mline);
  if (!d.res) {
    out.push(el('p', { class: 'muted' }, T`응답이 커서 기록 파일에 담지 않았습니다(${d.rec.response_bytes || '?'} 바이트).`));
    return out;
  }
  if (MEASURE_OPS.has(entry.operation)) {             // E4 — 점수 · 로짓 렌즈 실행은 그 표(steps.js)
    out.push(...measureView(ctx, entry, c));
    return out;
  }
  if (AGENT_OPS.has(entry.operation)) {
    out.push(...agentView(ctx, entry, d, c.text));
    return out;
  }
  if (!d.n) {
    out.push(el('p', { class: 'muted' }, T('생성한 토큰이 없습니다.')));
    out.push(el('h2', null, T('요청', '하나')), el('pre', { class: 'xp-pre' }, JSON.stringify(d.req, null, 2)));
    out.push(el('details', { class: 'run-raw' }, el('summary', null, T('응답(JSON)')), el('pre', { class: 'xp-pre' }, JSON.stringify(d.res, null, 2))));
    return out;
  }
  const key = ctx.path + '|' + id;
  const low = lowestMargin(d);
  let sel = ui.sel.has(key) ? ui.sel.get(key) : low >= 0 ? low : 0;
  sel = Math.max(0, Math.min(d.n - 1, sel));
  const onSel = (k, focus) => {
    ui.sel.set(key, k);
    ctx.rerender();                     // 바로 다시 그린다 — 초점도 바로 옮겨야 빠르게 누른 다음 키가 새 토큰에 간다
    if (focus) {
      const t = document.querySelector(`.run-strip .run-tok[data-k="${k}"]`);
      if (t) t.focus({ preventScroll: false });
    }
  };
  out.push(summaryView(ctx, d, sel, onSel));
  out.push(el('div', { class: 'run-split' },
    el('div', { class: 'run-main' },
      barSwitch(ctx),
      stripView(ctx, d, sel, low, onSel),
      el('div', { class: 'run-legend' },
        el('span', null, el('i', { 'aria-hidden': 'true' }), legend()),
        el('span', null, T('흐린 텍스트 = 프롬프트')),
        el('span', null, T('← → 이전 · 다음 스텝 · Enter 스텝 · 층 화면 · b 가장 높은 다른 후보로 분기')))),
    stepPanel(ctx, d, sel, c.text)));
  if (!d.texts) out.push(el('p', { class: 'muted' }, T('이 실행의 응답에는 토큰마다의 텍스트가 없어 토큰 번호로 보입니다(logprobs 를 넣으면 텍스트가 남습니다).')));
  out.push(...toolsView(ctx, d, c.text, !!ui.busy));
  return out;
}

// ---- 비교 ----

export function setCompare(ctx, a, b) {
  ui.cmp = { path: ctx.path, a, b };
}

function defaultPair(ctx) {
  // E4 — 점수 · 로짓 렌즈 실행(스텝 · 층 화면의 계산)은 빼고 생성한 실행끼리
  const all = runsOf(ctx);
  const runs = all.filter(r => !MEASURE_OPS.has(r.operation));
  if (runs.length < 2) return all.length >= 2 ? [all[all.length - 2].id, all[all.length - 1].id] : [null, null];
  // F3-1 — 가장 최근 생성 실행과 그 base, base 가 없으면 그 실험의 첫 생성 실행과(바로 앞 실행보다 연구의 기준 실행과 견줄 일이 많다)
  const last = runs[runs.length - 1];
  const base = last.base && runs.find(r => r.id === last.base) ? last.base : runs[0].id;
  return [base, last.id];
}

/** 위치 pos 의 토큰 글 — 생성 토큰 · echo 로 받은 프롬프트 토큰 */
function textAt(d, pos) {
  const p = d.ptoks ? d.ptoks.length : 0;
  if (pos >= p) return d.texts ? d.texts[pos - p] : '#' + d.ids[pos - p];
  if (d.ptexts && d.ptexts.length === p) return d.ptexts[pos];
  return null;
}

function cmpRow(d, c, s, from) {
  const p = d.ptoks.length;
  const end = s.length;
  const parts = [];
  let same = '';
  let diff = '';
  for (let pos = from; pos < end; pos++) {
    const t = textAt(d, pos);
    if (t === null) continue;
    if (pos < c.L) same += t;
    else diff += t;
  }
  if (from > 0 && from < p && typeof d.req.prompt === 'string') parts.push(el('span', { class: 'run-p' }, '…'));
  parts.push(el('span', { class: 'same-part' }, same), el('span', { class: 'fork' }, diff));
  return el('div', { class: 'run-cmp-row' }, el('span', { class: 'rid' }, d.entry.id), el('span', { class: 'txt' }, parts));
}

function flatSettings(req) {
  const out = {};
  for (const [k, v] of Object.entries(req || {})) {
    if (['prompt', 'messages', 'input', 'instructions', 'system', 'tools', 'tool_choice', 'stream', 'stream_options'].includes(k)) continue;
    if ((k === 'geulpu' || k === 'chat_template_kwargs') && v && typeof v === 'object' && !Array.isArray(v)) {
      for (const [k2, v2] of Object.entries(v)) out[k + '.' + k2] = v2;
    } else out[k] = v;
  }
  return out;
}

export function compareTab(ctx) {
  const runs = runsOf(ctx);
  if (runs.length < 2) return [el('div', { class: 'empty-box' }, el('div', { class: 'big' }, T('실행이 둘 이상 있어야 비교합니다')))];
  let [a, b] = ui.cmp && ui.cmp.path === ctx.path ? [ui.cmp.a, ui.cmp.b] : [null, null];
  if (!a || !b || !entryOf(ctx, a) || !entryOf(ctx, b)) [a, b] = defaultPair(ctx);
  const pick = (cur, which) => el('select', {
    class: 'input', 'aria-label': which === 'a' ? T('첫째 실행') : T('둘째 실행'),
    onchange: ev => { const x = ev.target.value; setCompare(ctx, which === 'a' ? x : a, which === 'a' ? b : x); ctx.rerender(); },
  }, runs.map(r => el('option', { value: r.id, selected: r.id === cur ? true : null }, `${r.id} · ${kindLabel(r)}`)));
  const out = [el('div', { class: 'xp-actions run-pick' }, pick(a, 'a'), el('span', { class: 'muted', 'aria-hidden': 'true' }, '↔'), pick(b, 'b'))];
  const ca = recordOf(ctx, entryOf(ctx, a));
  const cb = recordOf(ctx, entryOf(ctx, b));
  if (ca.loading || cb.loading) return [...out, el('p', { class: 'muted' }, T('읽는 중…'))];
  if (ca.error || cb.error) return [...out, el('p', { class: 'muted' }, ca.error || cb.error)];
  const da = dataOf(entryOf(ctx, a), ca.rec, ca.text);
  const db = dataOf(entryOf(ctx, b), cb.rec, cb.text);
  const pa = (da.rec.server || {}).precision;
  const pb = (db.rec.server || {}).precision;
  if (pa && pb && pa !== pb) {
    out.push(el('p', { class: 'xp-check run-warn' }, el('span', { class: 'tag err' }, '!'),
      T`계산 방식이 다릅니다 — ${a} ${modeLabel(pa)} · ${b} ${modeLabel(pb)}. 두 계산 방식은 서로 다른 비트를 냅니다.`));
  }
  const c = compareData(da, db);
  if (!c) {
    out.push(el('p', { class: 'muted' }, T('프롬프트 토큰이 기록되지 않은 실행이 있어 위치를 맞추지 못합니다(geulpu.prompt_tokens).')));
    return out;
  }
  const from = Math.min(da.ptoks.length - da.off, db.ptoks.length - db.off);   // 원래 프롬프트의 끝 — 답부터 보인다
  out.push(el('div', { class: 'run-cmp' }, cmpRow(da, c, c.sa, Math.max(0, from)), cmpRow(db, c, c.sb, Math.max(0, from))));
  // 요약 — 앱이 계산한 비교 값(experiment.compare, F3-3). 위의 두 줄과 아래의 해시 칩은 기록에서 바로 그린다
  const cv = compareOfRuns(ctx, da.entry, db.entry);
  const v = cv.val || null;
  if (!v) out.push(el('p', { class: 'muted' }, cv.error || T('읽는 중…')));
  else {
    const f = v.first;
    const tokLine = v.same ? T('토큰이 모두 같습니다')
      : f.step !== null ? T`토큰이 처음 다른 스텝: ${f.step}` : T`프롬프트가 다릅니다(위치 ${f.position})`;
    const h = v.logit_hashes;
    const hashLine = !h.compared ? T('두 실행 모두 로짓 해시가 있는 위치가 없습니다')
      : h.first_diff_step === null ? T`로짓 해시 ${h.equal} / ${h.compared} 같음` : T`로짓 해시가 처음 다른 스텝: ${h.first_diff_step} (${h.equal} / ${h.compared} 같음)`;
    out.push(el('dl', { class: 'run-kv run-sum' },
      el('dt', null, T('토큰')), el('dd', null, tokLine, v.compared_tokens ? T` · 같은 토큰 ${v.same_tokens} / ${v.compared_tokens}` : ''),
      el('dt', null, T('로짓 해시')), el('dd', null, hashLine)));
  }
  out.push(...compareExtras(ctx, da, db, v));
  // 해시 칩 — 처음 다른 스텝 앞뒤
  if (c.pos.length) {
    const at = c.firstDiff < 0 ? 0 : Math.max(0, c.pos.indexOf(c.firstDiff) - 4);
    const shown = c.pos.slice(at, at + 12);
    out.push(el('h2', null, T('스텝마다 로짓 해시')),
      el('div', { class: 'run-hashes' }, shown.map(pos => {
        const same = c.ha.get(pos) === c.hb.get(pos);
        return el('span', { class: 'run-hash' + (same ? ' same' : pos === c.firstDiff ? ' first' : ' diff') },
          el('span', { class: 'st' }, T`스텝 ${stepAt(da, pos)}`), el('code', null, String(c.ha.get(pos)).slice(0, 8)),
          same ? el('span', { class: 'ok' }, T('같음')) : el('code', null, String(c.hb.get(pos)).slice(0, 8)),
          pos === c.firstDiff ? el('span', { class: 'mark' }, T('처음 다른 스텝')) : null);
      })));
  }
  // 설정의 차이
  const fa = flatSettings(da.req);
  const fb = flatSettings(db.req);
  const keys = [...new Set([...Object.keys(fa), ...Object.keys(fb)])].filter(k => JSON.stringify(fa[k]) !== JSON.stringify(fb[k]));
  const promptSame = JSON.stringify(da.req.prompt === undefined ? da.req.messages : da.req.prompt) === JSON.stringify(db.req.prompt === undefined ? db.req.messages : db.req.prompt);
  out.push(el('h2', null, T('요청에서 다른 설정')));
  if (!keys.length && promptSame) out.push(el('p', { class: 'muted' }, T('요청이 같습니다')));
  else {
    const show = x => (x === undefined ? '–' : typeof x === 'string' ? x : JSON.stringify(x));
    out.push(el('table', { class: 'xp-files run-diff' }, el('tbody', null,
      el('tr', null, el('th', null, ''), el('th', null, a), el('th', null, b)),
      promptSame ? null : el('tr', null, el('td', null, T('프롬프트')), el('td', null, T('다름')), el('td', null, T('다름'))),
      keys.map(k => el('tr', null, el('td', null, el('code', null, k)), el('td', null, show(fa[k])), el('td', null, show(fb[k])))))));
  }
  return out;
}
