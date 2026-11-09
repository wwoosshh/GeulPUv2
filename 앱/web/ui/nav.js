// 앱리디자인 E5(문서/화면구현.md §5 · 화면설계.md §2 · §3) — 그룹의 탭과 경로. 사이드바는 그룹 다섯(실험 · 모델 · 재현 · 작업 · 설정)이고 화면을
// 고르는 일은 app.js 의 show 가 한다(화면 → 그룹의 표 · 그룹마다 마지막으로 연 화면). 여기서는 왼쪽 맨 위의 막대(#grp)에 지금 화면이 속한 그룹의
// 탭을 보이고, 사이드바에 없는 화면(채팅 · 디버거 · 에이전트)은 그 화면이 놓인 곳의 경로를 보인다.
import { T, el } from './dom.js';

const GROUPS = {
  model: [['models', () => T('파일')], ['eval', () => T('평가', '탭')], ['perf', () => T('성능')]],
  repro: [['llama', () => 'llama.cpp'], ['checks', () => T('검사')], ['results', () => T('검사 결과')]],
  settings: [['settings', () => T('앱', '탭')], ['device', () => T('장치', '화면')], ['cache', () => T('프롬프트 캐시')]],
};
const GROUP_NAME = { model: () => T('모델'), repro: () => T('재현'), settings: () => T('설정') };
const go = v => { if (typeof globalThis.show === 'function') globalThis.show(v); };
// F3-1 — 경로의 '실험'은 실험 목록으로(마지막에 연 실험이 아니라)
const goList = () => { document.dispatchEvent(new CustomEvent('geulpu:exp-list')); go('exp'); };
// 사이드바에 없는 화면 — 경로의 앞 항목은 누르면 그 화면으로
const PATHS = {
  chat: () => [[T('실험'), goList], [T('대화'), () => document.dispatchEvent(new CustomEvent('geulpu:open-chat-experiment'))]],
  debug: () => [[T('실험'), goList], [T('디버거')]],
  agents: () => [[T('명령 창')], [T('AI 세션')]],
};
const bar = document.getElementById('grp');

function render(view) {
  const grp = typeof globalThis.viewGroup === 'function' ? globalThis.viewGroup(view) : '';
  const tabs = GROUPS[grp];
  if (tabs) {
    bar.replaceChildren(
      el('span', { class: 'grp-name' }, GROUP_NAME[grp]()),
      el('div', { class: 'grp-tabs', role: 'tablist', 'aria-label': GROUP_NAME[grp]() },
        tabs.map(([v, label]) => el('button', {
          type: 'button', role: 'tab', class: 'grp-tab' + (v === view ? ' on' : ''), 'aria-selected': String(v === view), onclick: () => go(v),
        }, label()))));
    bar.hidden = false;
  } else if (PATHS[view]) {
    const items = PATHS[view]();
    bar.replaceChildren(el('div', { class: 'grp-path', role: 'navigation', 'aria-label': T('경로') },
      items.map(([t, f], i) => [
        i ? el('span', { class: 'sep', 'aria-hidden': 'true' }, '›') : null,
        f ? el('button', { type: 'button', onclick: f }, t) : el('span', { class: 'here' }, t),
      ])));
    bar.hidden = false;
  } else {
    bar.replaceChildren();
    bar.hidden = true;
  }
}

document.addEventListener('geulpu:view', ev => render(ev.detail && ev.detail.view));
// 모듈이 늦게 읽혔으면(앱이 이미 화면을 열었으면) 지금 화면으로
const active = document.querySelector('.view.active');
if (active) render(active.id.replace(/^view-/, ''));
