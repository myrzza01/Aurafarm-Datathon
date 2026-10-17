(() => {
  'use strict';
  const $ = (id) => document.getElementById(id);
  const register = location.pathname === '/register';
  const reducedMotion = matchMedia('(prefers-reduced-motion: reduce)');
  const form = $('auth-form');
  let busy = false;
  let entering = false;
  let pageHasLeft = false;
  document.body.dataset.mode = register ? 'register' : 'login';

  document.title = `${register ? 'Регистрация' : 'Вход'} — Forte Vision`;
  $('auth-title').textContent = register ? 'Создать аккаунт' : 'С возвращением';
  $('auth-description').textContent = register ? 'Ваш доступ к проверке видеозаписей в одном месте.' : 'Войдите, чтобы перейти к проверке видеозаписей.';
  $('auth-submit-text').textContent = register ? 'Создать аккаунт' : 'Войти в сервис';
  $('login-tab').removeAttribute('aria-current');
  $('register-tab').removeAttribute('aria-current');
  $(register ? 'register-tab' : 'login-tab').setAttribute('aria-current', 'page');
  for (const id of ['name', 'confirm']) {
    $(`${id}-field`).hidden = !register;
    $(`auth-${id}`).disabled = !register;
    $(`auth-${id}`).required = register;
  }
  $('auth-password').autocomplete = register ? 'new-password' : 'current-password';
  $('auth-password').placeholder = register ? 'Придумайте пароль' : 'Введите пароль';
  if (register) {
    const text = document.createTextNode('Уже есть аккаунт? ');
    const link = document.createElement('a');
    link.href = '/login';
    link.textContent = 'Войти в сервис ↗';
    $('auth-switch').replaceChildren(text, link);
  }

  document.querySelectorAll('.password-toggle').forEach((button) => {
    button.addEventListener('click', () => {
      const input = $(button.dataset.target);
      const visible = input.type === 'password';
      input.type = visible ? 'text' : 'password';
      button.setAttribute('aria-pressed', String(visible));
      button.setAttribute('aria-label', `${visible ? 'Скрыть' : 'Показать'} ${input.id === 'auth-confirm' ? 'повтор пароля' : 'пароль'}`);
    });
  });
  for (const id of ['auth-password', 'auth-confirm']) {
    $(id).addEventListener('input', () => {
      $('auth-confirm').setCustomValidity('');
      $('auth-confirm').removeAttribute('aria-invalid');
    });
  }
  function error(message) {
    $('auth-error').textContent = message;
    $('auth-error').hidden = !message;
  }

  async function enterApp() {
    entering = true;
    if (frame !== null) cancelAnimationFrame(frame);
    frame = null;
    if (!reducedMotion.matches) {
      document.body.classList.add('entering');
      try {
        // Animation failures must never prevent an already authenticated user from entering.
        const flight = window.FortePortal?.play();
        if (flight) await flight;
        else await new Promise((resolve) => setTimeout(resolve, 500));
      } catch { /* Continue to the application if WebGL is unavailable. */ }
      if (pageHasLeft) return;
      try { sessionStorage.setItem('forte-portal-arrival', String(Date.now())); } catch { /* Storage is optional. */ }
    }
    if (!pageHasLeft) location.replace('/');
  }

  form.addEventListener('submit', async (event) => {
    event.preventDefault();
    if (busy) return;
    error('');
    if (register && $('auth-password').value !== $('auth-confirm').value) {
      $('auth-confirm').setCustomValidity('Пароли не совпадают. Повторите введённый пароль.');
      $('auth-confirm').setAttribute('aria-invalid', 'true');
      $('auth-confirm').reportValidity();
      return;
    }
    if (!form.reportValidity()) return;
    const data = { email: $('auth-email').value.trim(), password: $('auth-password').value };
    if (register) data.name = $('auth-name').value.trim();
    busy = true;
    form.setAttribute('aria-busy', 'true');
    $('auth-submit').disabled = true;
    $('auth-submit-text').textContent = register ? 'Создаём аккаунт…' : 'Входим…';
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 15000);
    let success = false;
    try {
      const response = await fetch(`/auth/api/${register ? 'register' : 'login'}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', 'X-UI-Request': '1' },
        body: JSON.stringify(data), signal: controller.signal,
      });
      const body = await response.json();
      if (!response.ok) {
        const message = typeof body.detail === 'string' ? body.detail : 'Проверьте email, имя и пароль. Пароль должен содержать от 12 до 128 символов.';
        throw new Error(message);
      }
      success = true;
      $('auth-status').textContent = register ? 'Аккаунт создан. Открываем приложение…' : 'Вы вошли. Открываем приложение…';
      $('auth-status').hidden = false;
      $('auth-submit-text').textContent = 'Готово';
      $('auth-password').value = '';
      $('auth-confirm').value = '';
      clearTimeout(timeout);
      await enterApp();
    } catch (reason) {
      error(reason.name === 'AbortError' ? 'Сервер не ответил вовремя. Попробуйте ещё раз.' : reason instanceof TypeError ? 'Нет соединения с сервером. Проверьте, что приложение запущено.' : reason.message);
    } finally {
      clearTimeout(timeout);
      if (!success) {
        busy = false;
        form.setAttribute('aria-busy', 'false');
        $('auth-submit').disabled = false;
        $('auth-submit-text').textContent = register ? 'Создать аккаунт' : 'Войти в сервис';
      }
    }
  });

  // A lightweight local canvas recreates the portal ambience without CDN scripts.
  const canvas = $('portal-background');
  const ctx = canvas.getContext('2d');
  let width = 0;
  let height = 0;
  let frame = null;
  let stars = [];
  const started = performance.now();
  function resize() {
    width = innerWidth;
    height = innerHeight;
    const scale = Math.min(devicePixelRatio || 1, 2);
    canvas.width = Math.round(width * scale);
    canvas.height = Math.round(height * scale);
    if (!ctx) return;
    ctx.setTransform(scale, 0, 0, scale, 0, 0);
    stars = Array.from({ length: width < 600 ? 45 : 100 }, () => ({ x: Math.random(), y: Math.random(), size: .5 + Math.random() * 1.1, phase: Math.random() * Math.PI * 2 }));
    if (reducedMotion.matches) draw(performance.now());
  }
  function draw(now) {
    if (!ctx) return;
    ctx.clearRect(0, 0, width, height);
    const time = (now - started) / 1000;
    for (const star of stars) {
      const alpha = .18 + Math.sin(time * .6 + star.phase) * .12;
      ctx.fillStyle = `rgba(252,147,133,${alpha})`;
      ctx.beginPath();
      ctx.arc(star.x * width, star.y * height, star.size, 0, Math.PI * 2);
      ctx.fill();
    }
    for (let i = 0; i < 15; i++) {
      const depth = ((i / 15) + time * (entering ? .8 : .008)) % 1;
      const scale = .12 + depth * depth * 1.7;
      const w = width * .72 * scale;
      const h = height * .85 * scale;
      ctx.strokeStyle = `rgba(${i % 2 ? '163,21,81' : '252,147,133'},${entering ? .25 : .045})`;
      ctx.lineWidth = entering ? 1.4 : .8;
      ctx.beginPath();
      ctx.rect((width - w) / 2, (height - h) / 2, w, h);
      ctx.stroke();
    }
  }
  function animate(now) {
    draw(now);
    if (!document.hidden && !reducedMotion.matches) frame = requestAnimationFrame(animate);
    else frame = null;
  }
  function startAnimation() {
    if (!ctx || frame !== null || document.hidden || reducedMotion.matches || entering) return;
    frame = requestAnimationFrame(animate);
  }
  window.addEventListener('resize', resize);
  document.addEventListener('visibilitychange', () => {
    if (document.hidden && frame !== null) { cancelAnimationFrame(frame); frame = null; }
    else startAnimation();
  });
  reducedMotion.addEventListener('change', () => {
    if (frame !== null) cancelAnimationFrame(frame);
    frame = null;
    draw(performance.now());
    startAnimation();
  });
  window.addEventListener('pagehide', () => { pageHasLeft = true; if (frame !== null) cancelAnimationFrame(frame); });
  window.addEventListener('pageshow', (event) => { if (event.persisted) location.reload(); });
  resize();
  startAnimation();
})();
