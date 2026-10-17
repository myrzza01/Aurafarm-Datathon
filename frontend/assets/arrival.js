(() => {
  'use strict';
  let stamp = 0;
  try {
    stamp = Number(sessionStorage.getItem('forte-portal-arrival'));
    sessionStorage.removeItem('forte-portal-arrival');
  } catch { return; }
  if (!stamp || Date.now() - stamp > 15000 || Date.now() < stamp || matchMedia('(prefers-reduced-motion: reduce)').matches) return;
  const root = document.documentElement;
  root.classList.add('portal-arrival');
  const clear = () => root.classList.remove('portal-arrival', 'portal-arrival-ready');
  const fallback = setTimeout(clear, 2500);
  document.addEventListener('DOMContentLoaded', () => {
    requestAnimationFrame(() => requestAnimationFrame(() => {
      root.classList.add('portal-arrival-ready');
      setTimeout(() => { clearTimeout(fallback); clear(); }, 1100);
    }));
  }, { once: true });
  window.addEventListener('pageshow', (event) => { if (event.persisted) clear(); });
})();
