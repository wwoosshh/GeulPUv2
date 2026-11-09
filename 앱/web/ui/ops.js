// 앱리디자인 E1(문서/화면구현.md §1) — 기능 목록 클라이언트. 기능 목록(GET /api/operations)을 읽고, 기능 id 와 인자로 요청을 만들어 앱에
// 보낸다. 화면 코드는 앱을 이것으로만 부른다 — 앱 · 서버의 경로는 기능 목록에만 있다(E6 — 옛 app.js · eval.js 도 기능 id 로 부르고, CI 의
// 기능 목록 점검이 화면 코드에 경로가 없는지 본다).
import { index, requestFor } from './command.js';

let registryPromise = null;
let registryNow = null;        // 읽은 기능 목록(기다릴 수 없는 곳 — 링크의 주소)

/** 기능 목록(한 번 읽어 둔다 — 실패하면 다음에 다시 읽는다). */
export function registry() {
  if (!registryPromise) {
    registryPromise = fetch('/api/operations', { headers: { 'X-GeulPU-Source': 'gui' } })
      .then(r => {
        if (!r.ok) throw new Error('HTTP ' + r.status);
        return r.json();
      })
      .then(index)
      .then(reg => (registryNow = reg))
      .catch(e => {
        registryPromise = null;
        throw e;
      });
  }
  return registryPromise;
}

function headersOf(post, { source = 'gui', session = '', experiment = '' } = {}) {
  const headers = { 'X-GeulPU-Source': source };
  if (session) headers['X-GeulPU-Session'] = session;
  if (experiment) headers['X-GeulPU-Experiment'] = experiment;
  if (post) headers['Content-Type'] = 'application/json';
  return headers;
}

/**
 * 만든 요청({method, path, body})을 보낸다. source 는 출처(gui · cli), session 은 세션 이름, experiment 는 실험 폴더(명령 기록을 남길 때).
 * 답은 {status, ok, json, text} — 앱의 오류 답도 그대로 돌려준다(던지지 않는다). 앱에 닿지 못하면 던진다.
 */
export async function send(req, opt = {}) {
  const post = req.method === 'POST';
  const init = { method: req.method, headers: headersOf(post, opt), signal: opt.signal };
  if (post) init.body = req.body == null ? '{}' : req.body;
  const r = await fetch(req.path, init);
  const text = await r.text();
  let json = null;
  try { json = text ? JSON.parse(text) : null; } catch (_) { json = null; }
  return { status: r.status, ok: r.ok, json, text };
}

// 기능 id 와 인자의 요청. opt.body(글)를 주면 그 글을 그대로 본문으로 — 2⁵³ 위의 시드 · 큰 정수처럼 JSON.stringify 로 만들 수 없는 본문(POST).
// 내부용 전달 경로(method ANY — proxy.tokenize · proxy.detokenize)는 opt.method(기본 POST)로 — 화면을 그리려는 호출이라 이벤트가 나오지 않는다
async function requestOf(opId, params, opt = {}) {
  const reg = await registry();
  const o = reg.byId.get(opId);
  if (o && o.http && o.http.method === 'ANY' && !o.http.path.endsWith('*')) {
    const method = opt.method || 'POST';
    const body = method === 'POST' ? (opt.body !== undefined ? opt.body : JSON.stringify(params || {})) : null;
    return { op: opId, method, path: o.http.path, body };
  }
  const req = requestFor(reg, opId, opt.body === undefined ? params : {});
  if (opt.body !== undefined && req.method === 'POST') req.body = opt.body;
  return req;
}

/** 기능 하나를 부른다 — POST 는 params 를 본문으로, GET 은 쿼리 문자열로. opt.body 는 이미 만든 본문 글. */
export async function call(opId, params = {}, opt = {}) {
  return send(await requestOf(opId, params, opt), opt);
}

/** call 과 같되 답의 JSON 을 돌려주고, 앱이 오류로 답하면 그 메시지로 던진다(e.status · e.json) — 옛 app.js 의 api(path, body) 와 같은 형식. */
export async function callJ(opId, params = {}, opt = {}) {
  const r = await call(opId, params, opt);
  if (!r.ok) {
    const e = new Error((r.json && r.json.error && r.json.error.message) || ('HTTP ' + r.status));
    e.status = r.status;
    e.json = r.json;
    throw e;
  }
  return r.json;
}

/** 기능 하나를 부르고 fetch 의 Response 를 읽지 않고 돌려준다 — 스트리밍 응답(SSE)을 읽는 화면(채팅). */
export async function open(opId, params = {}, opt = {}) {
  const req = await requestOf(opId, params, opt);
  const post = req.method === 'POST';
  const init = { method: req.method, headers: headersOf(post, opt), signal: opt.signal };
  if (post) init.body = req.body == null ? '{}' : req.body;
  return fetch(req.path, init);
}

/** 기능 id 와 인자의 주소(GET — 링크 · 내려받기). 기능 목록을 아직 읽지 못했으면 null. */
export function urlFor(opId, params = {}) {
  return registryNow ? requestFor(registryNow, opId, params).path : null;
}

/** 경로의 기능 id(기록에 남은 경로 — 디버거의 실행 로그). 같은 경로의 기능이 여럿이면 목록의 첫 것. 기능 목록을 읽기 전이거나 없으면 null. */
export function opOfPath(method, path) {
  if (!registryNow) return null;
  const o = registryNow.operations.find(x => x.http && x.http.method === method && x.http.path === path && !x.internal);
  return o ? o.id : null;
}

/** 기능의 경로(기록의 경로와 비교할 때 — 디버거). 기능 목록을 읽기 전이면 null. */
export function pathOf(opId) {
  const o = registryNow ? registryNow.byId.get(opId) : null;
  return o && o.http ? o.http.path : null;
}
