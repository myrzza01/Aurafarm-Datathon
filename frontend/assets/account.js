(() => {
  'use strict';
  const name = document.getElementById('account-name');
  const logout = document.getElementById('logout-button');
  const error = document.getElementById('account-error');
  const refreshAccount = () => fetch('/auth/api/me', { cache: 'no-store' }).then(async (response) => {
    if (response.status === 401) { location.replace('/login'); return; }
    if (!response.ok) throw new Error('Не удалось получить данные аккаунта');
    const body = await response.json();
    name.textContent = body.user.name;
    name.title = body.user.email;
  }).catch(() => { name.textContent = 'Личный кабинет'; });
  refreshAccount();
  window.addEventListener('pageshow', (event) => { if (event.persisted) refreshAccount(); });
  logout.addEventListener('click', async () => {
    logout.disabled = true;
    error.hidden = true;
    try {
      const response = await fetch('/auth/api/logout', { method: 'POST', headers: { 'X-UI-Request': '1' } });
      if (!response.ok) throw new Error('Не удалось выйти');
      location.replace('/login');
    } catch {
      error.textContent = 'Не удалось выйти из аккаунта. Проверьте соединение и повторите попытку.';
      error.hidden = false;
      logout.disabled = false;
    }
  });
})();
