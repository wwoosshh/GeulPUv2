// 앱리디자인 E(문서/화면구현.md) — 화면 모듈이 함께 쓰는 것: 요소 만들기, 아이콘, 화면 문구(app.js 의 T · TM — 한국어 키 → 영어 사전),
// 출처 이름, 시간 · 크기 쓰기, 복사 단추.

export const T = (s, ...v) => (typeof globalThis.T === 'function' ? globalThis.T(s, ...v) : typeof s === 'string' ? s : String.raw({ raw: s }, ...v));
export const TM = s => (typeof globalThis.TM === 'function' ? globalThis.TM(s) : s);
export const EN = () => document.documentElement.lang === 'en';

/** 요소 하나 — attrs 의 on… 은 이벤트, null · undefined · false 는 빼고, 자식은 DOM 노드 또는 문자열. */
export function el(tag, attrs, ...kids) {
  const e = document.createElement(tag);
  if (attrs) {
    for (const [k, v] of Object.entries(attrs)) {
      if (v === null || v === undefined || v === false) continue;
      if (k === 'class') e.className = v;
      else if (k.startsWith('on')) e.addEventListener(k.slice(2), v);
      else e.setAttribute(k, v === true ? '' : v);
    }
  }
  for (const k of kids.flat(Infinity)) if (k !== null && k !== undefined && k !== false) e.append(k instanceof Node ? k : String(k));
  return e;
}

/** 선 아이콘 하나(viewBox 24 × 24 의 경로). */
export function icon(d, size = 16) {
  const t = document.createElement('template');
  t.innerHTML = `<svg class="i" viewBox="0 0 24 24" width="${size}" height="${size}" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">${d}</svg>`;
  return t.content.firstChild;
}
export const CHEVRON = '<path d="m9.5 6 6 6-6 6"/>';

/** 기능 호출의 출처 — 화면설계.md §5: GUI · 사용자 CLI(세션 없음 · 명령 창의 입력 줄) · AI · 세션 이름 · 파이썬. */
export function sourceLabel(source, session) {
  if (source === 'cli' && !session) return [T('사용자 CLI'), ''];
  if (source === 'gui') return ['GUI', ''];
  if (source === 'cli') return ['AI · ' + session, 'ai'];
  if (source === 'python') return [T('파이썬'), ''];
  return [T('알 수 없음'), ''];
}

export const fmtMs = ms => (ms < 1000 ? `${ms} ms` : T`${(ms / 1000).toFixed(ms < 10000 ? 2 : 1)} 초`);
export const fmtBytes = n => (n >= 1048576 ? (n / 1048576).toFixed(1) + ' MB' : n >= 1024 ? Math.round(n / 1024) + ' KB' : n + ' B');
export const clock = t => {
  const d = new Date(t * 1000);
  return [d.getHours(), d.getMinutes(), d.getSeconds()].map(x => String(x).padStart(2, '0')).join(':');
};
export const baseName = p => String(p).replace(/[\\/]+$/, '').split(/[\\/]/).pop();

/** 누르면 텍스트를 클립보드에 넣는 단추(잠깐 '복사함'). */
export function copyButton(getText) {
  return el('button', {
    type: 'button', class: 'linkish',
    onclick: async ev => {
      const b = ev.currentTarget;
      try {
        await navigator.clipboard.writeText(getText());
        b.textContent = T('복사함');
      } catch (_) {
        b.textContent = T('복사하지 못함');
      }
      setTimeout(() => { b.textContent = T('명령 복사', '줄'); }, 1500);
    },
  }, T('명령 복사', '줄'));
}
