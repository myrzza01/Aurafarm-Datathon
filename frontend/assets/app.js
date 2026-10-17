(() => {
  'use strict';
  const $ = (id) => document.getElementById(id);
  const input = $('video-file');
  const dropzone = $('dropzone');
  const startButton = $('analyze-button');
  const panel = $('analysis-panel');
  const progress = $('upload-progress');
  const MAX_BYTES = 100 * 1024 * 1024;
  let selectedFile = null;
  let busy = false;
  let runId = 0;
  let timer = null;
  let previewUrl = null;
  let preview = null;
  let activeGallery = null;
  const frameDialog = $('frame-dialog');

  function timecode(seconds) {
    const value = Math.max(0, Number(seconds) || 0);
    const tenths = Math.round(value * 10);
    const minutes = Math.floor(tenths / 600);
    const whole = Math.floor(tenths / 10) % 60;
    const tenth = tenths % 10;
    return `${String(minutes).padStart(2, '0')}:${String(whole).padStart(2, '0')}.${tenth}`;
  }

  function fileError(message) {
    $('file-error').textContent = message || '';
    $('file-error').hidden = !message;
  }

  function clearPreview() {
    if (preview) {
      preview.pause();
      preview.removeAttribute('src');
      preview.load();
      preview = null;
    }
    if (previewUrl) URL.revokeObjectURL(previewUrl);
    previewUrl = null;
  }

  function resetResult() {
    clearPreview();
    if (frameDialog.open) frameDialog.close();
    activeGallery = null;
    $('results').replaceChildren();
    $('demo-notice').hidden = true;
    $('demo-controls').hidden = true;
    $('new-review').hidden = true;
    $('retry-button').hidden = true;
    $('job-reference').hidden = true;
    $('elapsed-time').hidden = false;
    $('analysis-kicker').textContent = 'ПРОВЕРКА ВИДЕО';
    panel.hidden = true;
  }

  function selectFile(file) {
    if (busy) return;
    if (!file) return;
    if (!file.name.toLowerCase().endsWith('.mov')) {
      fileError('Выберите видео в формате .mov. Другие форматы пока не поддерживаются.');
      return;
    }
    if (!file.size || file.size > MAX_BYTES) {
      fileError(file.size ? 'Размер видео превышает 100 МиБ. Выберите файл поменьше.' : 'Файл пуст. Выберите другую видеозапись.');
      return;
    }
    runId += 1;
    resetResult();
    selectedFile = file;
    fileError('');
    $('file-name').textContent = file.name;
    $('file-size').textContent = `${(file.size / 1024 / 1024).toLocaleString('ru-RU', { maximumFractionDigits: 1 })} МиБ · MOV`;
    $('selected-file').hidden = false;
    startButton.disabled = false;
    panel.hidden = true;
  }

  input.addEventListener('change', () => {
    selectFile(input.files[0]);
    input.value = '';
  });
  $('remove-file').addEventListener('click', () => {
    if (busy) return;
    selectedFile = null;
    runId += 1;
    input.value = '';
    resetResult();
    $('selected-file').hidden = true;
    panel.hidden = true;
    startButton.disabled = true;
    fileError('');
  });

  for (const event of ['dragenter', 'dragover']) {
    dropzone.addEventListener(event, (e) => {
      e.preventDefault();
      if (!busy) dropzone.classList.add('dragging');
    });
  }
  for (const event of ['dragleave', 'drop']) {
    dropzone.addEventListener(event, (e) => {
      e.preventDefault();
      dropzone.classList.remove('dragging');
    });
  }
  dropzone.addEventListener('drop', (e) => {
    if (e.dataTransfer.files.length > 1) {
      fileError('Для одной проверки выберите один видеофайл.');
      return;
    }
    selectFile(e.dataTransfer.files[0]);
  });
  // Prevent accidental navigation if a file is dropped outside the upload area.
  window.addEventListener('dragover', (e) => e.preventDefault());
  window.addEventListener('drop', (e) => e.preventDefault());

  $('how-button').addEventListener('click', () => {
    const open = $('how-panel').hidden;
    $('how-panel').hidden = !open;
    $('how-button').setAttribute('aria-expanded', String(open));
    if (open) $('how-panel').scrollIntoView({ behavior: 'smooth', block: 'center' });
  });

  function setBusy(value) {
    busy = value;
    input.disabled = value;
    startButton.disabled = value || !selectedFile;
    startButton.firstChild.textContent = value ? 'Проверяем… ' : 'Проверить видео ';
    $('remove-file').disabled = value;
    $('demo-button').disabled = value;
    $('demo-events').disabled = value;
    $('demo-empty').disabled = value;
    $('new-review').disabled = value;
    dropzone.classList.toggle('busy', value);
    panel.setAttribute('aria-busy', String(value));
  }

  function showState(title, message, type = '') {
    panel.hidden = false;
    panel.className = `analysis-panel ${type}`;
    $('analysis-title').textContent = title;
    $('analysis-message').textContent = message;
    $('analysis-icon').querySelector('use').setAttribute('href', type === 'done' ? '#i-check' : type === 'error' ? '#i-close' : '#i-clock');
  }

  function upload(file) {
    return new Promise((resolve, reject) => {
      const xhr = new XMLHttpRequest();
      xhr.open('POST', '/app/api/v1/analyze');
      xhr.setRequestHeader('X-UI-Request', '1');
      xhr.timeout = 120000;
      xhr.upload.onprogress = (event) => {
        if (event.lengthComputable) progress.value = Math.round(event.loaded / event.total * 100);
      };
      xhr.onload = () => {
        let body;
        try { body = JSON.parse(xhr.responseText); }
        catch { reject(new Error('Сервис вернул некорректный ответ. Попробуйте ещё раз.')); return; }
        if (xhr.status >= 200 && xhr.status < 300 && body.job_id) resolve(body.job_id);
        else reject(new Error(xhr.status === 401 ? 'Сессия истекла. Обновите страницу и повторите загрузку.' : xhr.status === 413 ? 'Размер видео превышает допустимый лимит.' : [415, 429, 503].includes(xhr.status) && typeof body.detail === 'string' ? body.detail : 'Не удалось загрузить видео. Проверьте файл и повторите попытку.'));
      };
      xhr.onerror = () => reject(new Error('Нет соединения с сервером. Проверьте, что приложение запущено.'));
      xhr.ontimeout = () => reject(new Error('Загрузка заняла слишком много времени. Попробуйте ещё раз.'));
      const data = new FormData();
      data.append('file', file);
      xhr.send(data);
    });
  }

  const wait = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

  async function poll(jobId, currentRun) {
    const deadline = Date.now() + 295000;
    let networkFailures = 0;
    while (currentRun === runId && Date.now() < deadline) {
      let response;
      try {
        const controller = new AbortController();
        const timeout = setTimeout(() => controller.abort(), 10000);
        try { response = await fetch(`/app/api/v1/jobs/${encodeURIComponent(jobId)}`, { cache: 'no-store', signal: controller.signal }); }
        finally { clearTimeout(timeout); }
      } catch {
        networkFailures += 1;
        if (networkFailures >= 3) throw new Error('Соединение с сервером потеряно. Результат пока недоступен.');
        await wait(2000);
        continue;
      }
      networkFailures = 0;
      if (!response.ok) throw new Error(response.status === 401 ? 'Сессия истекла. Обновите страницу.' : 'Не удалось получить статус проверки. Возможно, сервер был перезапущен.');
      const job = await response.json();
      if (job.status === 'done') return job;
      if (job.status === 'failed') throw new Error(job.error || 'Не удалось обработать видео. Попробуйте другую запись.');
      if (job.status !== 'processing') throw new Error('Сервис вернул неизвестный статус обработки.');
      await wait(1500);
    }
    throw new Error('Результат не получен за 5 минут. Обработка на сервере может продолжаться.');
  }

  function node(tag, className, text) {
    const element = document.createElement(tag);
    if (className) element.className = className;
    if (text !== undefined) element.textContent = text;
    return element;
  }

  function seek(seconds) {
    if (!preview) return;
    const jump = () => {
      preview.currentTime = seconds;
      preview.scrollIntoView({ behavior: 'smooth', block: 'center' });
    };
    if (preview.readyState >= 1) jump();
    else preview.addEventListener('loadedmetadata', jump, { once: true });
  }

  function frameSource(frame, demo) {
    if (demo && /^\/assets\/demo\/[\w-]+\.svg$/.test(frame.demo_image_url || '')) return frame.demo_image_url;
    return `data:image/jpeg;base64,${frame.image_base64}`;
  }

  function updateFrameDialog() {
    if (!activeGallery) return;
    const { frames, index, demo, title } = activeGallery;
    const frame = frames[index];
    $('frame-dialog-title').textContent = title;
    $('frame-dialog-kicker').textContent = demo ? 'УЧЕБНАЯ СХЕМА / НЕ ВИДЕОКАДР' : 'КАДР-ДОКАЗАТЕЛЬСТВО';
    $('dialog-frame-image').src = frameSource(frame, demo);
    $('dialog-frame-image').alt = `${demo ? 'Учебная схема' : 'Кадр'}: ${title}, ${timecode(frame.timestamp_sec)}`;
    $('dialog-frame-position').textContent = `${index + 1} из ${frames.length} · ${timecode(frame.timestamp_sec)}`;
    $('previous-frame').disabled = index === 0;
    $('next-frame').disabled = index === frames.length - 1;
  }

  function moveFrame(step) {
    if (!activeGallery) return;
    const next = activeGallery.index + step;
    if (next < 0 || next >= activeGallery.frames.length) return;
    activeGallery.select(next);
    updateFrameDialog();
  }

  $('close-frame-dialog').addEventListener('click', () => frameDialog.close());
  $('previous-frame').addEventListener('click', () => moveFrame(-1));
  $('next-frame').addEventListener('click', () => moveFrame(1));
  frameDialog.addEventListener('click', (event) => {
    const bounds = frameDialog.getBoundingClientRect();
    if (event.target === frameDialog && (event.clientX < bounds.left || event.clientX > bounds.right || event.clientY < bounds.top || event.clientY > bounds.bottom)) frameDialog.close();
  });
  frameDialog.addEventListener('keydown', (event) => {
    if (event.key === 'ArrowLeft' || event.key === 'ArrowRight') {
      event.preventDefault();
      moveFrame(event.key === 'ArrowRight' ? 1 : -1);
    }
  });

  function renderGallery(event, demo) {
    const gallery = node('div', 'frame-gallery');
    const frames = event.evidence_frames || [];
    if (!frames.length) {
      gallery.append(node('p', 'review-note', 'Кадры не переданы. Для проверки события нужно визуальное подтверждение.'));
      return gallery;
    }
    const heading = node('div', 'gallery-heading');
    const position = node('span', 'gallery-position');
    heading.append(node('span', 'section-index', demo ? 'УЧЕБНЫЕ СХЕМЫ' : 'КАДРЫ-ДОКАЗАТЕЛЬСТВА'), position);
    const stage = node('button', 'frame-stage');
    stage.type = 'button';
    stage.setAttribute('aria-label', 'Увеличить выбранный кадр');
    const image = node('img');
    const expand = node('span', 'frame-expand', 'Открыть крупнее ↗');
    stage.append(image, expand);
    const thumbnails = node('div', 'frame-thumbnails');
    thumbnails.setAttribute('role', 'group');
    thumbnails.setAttribute('aria-label', 'Выбор кадра события');
    const buttons = frames.map((frame, index) => {
      const button = node('button', 'frame-thumbnail');
      button.type = 'button';
      button.setAttribute('aria-label', `${demo ? 'Учебная схема' : 'Кадр'} ${index + 1}, ${timecode(frame.timestamp_sec)}`);
      const thumb = node('img');
      thumb.src = frameSource(frame, demo);
      thumb.alt = '';
      button.append(thumb, node('span', '', timecode(frame.timestamp_sec)));
      button.addEventListener('click', () => select(index));
      button.addEventListener('keydown', (e) => {
        if (!['ArrowLeft', 'ArrowRight'].includes(e.key)) return;
        e.preventDefault();
        const next = (index + (e.key === 'ArrowRight' ? 1 : -1) + frames.length) % frames.length;
        select(next);
        buttons[next].focus();
      });
      return button;
    });
    function select(index) {
      const frame = frames[index];
      image.src = frameSource(frame, demo);
      image.alt = `${demo ? 'Учебная схема' : 'Кадр'}: ${event.type}, ${timecode(frame.timestamp_sec)}`;
      position.textContent = `${index + 1} из ${frames.length} · ${timecode(frame.timestamp_sec)}`;
      buttons.forEach((button, i) => button.setAttribute('aria-pressed', String(i === index)));
      activeGallery = { frames, index, demo, title: event.type, select };
      if (frameDialog.open) updateFrameDialog();
    }
    stage.addEventListener('click', () => {
      updateFrameDialog();
      frameDialog.showModal();
    });
    thumbnails.append(...buttons);
    select(0);
    gallery.append(heading, stage, thumbnails);
    return gallery;
  }

  function renderSource(event, demo) {
    const card = node('div', 'source-card');
    card.append(node('span', 'section-index', 'НОРМАТИВНОЕ ОСНОВАНИЕ'));
    const source = node('p', 'rule-source');
    const rule = event.rule_source;
    if (demo) {
      source.textContent = 'В учебном примере нормативный источник не указан. Здесь будет документ и пункт, связанный с событием.';
    } else if (rule?.status === 'unmapped') {
      source.textContent = rule.reason;
      card.append(node('strong', 'review-note', 'Требует классификации и проверки человеком'));
    } else if (rule) {
      source.append(node('strong', '', `${rule.document}${rule.section ? `, ${rule.section}` : ''}`));
      if (rule.url) {
        try {
          const url = new URL(rule.url);
          if (['https:', 'http:'].includes(url.protocol)) {
            const link = node('a', 'source-link', 'Открыть источник ↗');
            link.href = url.href;
            link.target = '_blank';
            link.rel = 'noopener noreferrer';
            source.append(link);
          }
        } catch { /* Keep the document title if its URL is invalid. */ }
      }
    } else {
      source.textContent = 'Источник правила не передан моделью. Требуется уточнение перед выводом о нарушении.';
    }
    card.append(source);
    return card;
  }

  function renderResult(job, { demo = false } = {}) {
    const events = [...(job.violations || [])].sort((a, b) => a.timecode_sec - b.timecode_sec);
    if (!['violation', 'no_violation'].includes(job.verdict) || (job.verdict === 'violation') !== (events.length > 0)) {
      throw new Error('Результат анализа содержит несогласованные данные. Требуется повторная проверка.');
    }
    clearPreview();
    const container = $('results');
    container.replaceChildren();
    activeGallery = null;
    const title = demo ? 'Пример результата проверки' : events.length ? 'Обнаружены признаки возможных нарушений' : 'Нарушение не обнаружено';
    const message = demo
      ? events.length ? 'Переключайте события, сравнивайте изображения и откройте любое из них крупнее.' : 'В этом учебном сценарии список событий пуст. Это не результат анализа видео.'
      : events.length ? `Событий для проверки: ${events.length}. Изучите кадры, описание и источник правила.` : 'В этой видеозаписи модель не выявила признаков возможных нарушений.';
    showState(title, message, 'done');
    panel.classList.toggle('demo', demo);
    $('analysis-kicker').textContent = demo ? 'ПРЕДПРОСМОТР / УЧЕБНЫЕ ДАННЫЕ' : 'РЕЗУЛЬТАТ ПРОВЕРКИ';
    $('demo-notice').hidden = !demo;
    $('demo-controls').hidden = !demo;
    $('elapsed-time').hidden = demo;
    $('new-review').hidden = false;
    const summary = node('dl', 'review-summary');
    for (const [label, value] of [
      [demo ? 'Примеры событий' : 'События', String(events.length).padStart(2, '0')],
      [demo ? 'Учебные схемы' : 'Кадры', String(events.reduce((total, event) => total + event.evidence_frames.length, 0)).padStart(2, '0')],
      ['Источник', demo ? 'Учебный пример' : selectedFile?.name || 'Видеозапись'],
    ]) {
      const metric = node('div', 'review-metric');
      metric.append(node('dt', '', label), node('dd', '', value));
      summary.append(metric);
    }
    container.append(summary);
    if (!events.length) {
      const empty = node('div', 'result-empty');
      empty.append(node('span', 'empty-check', '✓'), node('h3', '', demo ? 'Пример без событий' : 'Нет событий для просмотра'), node('p', '', demo ? 'Так выглядит завершённая проверка, в которой список событий пуст. Переключитесь на сценарий «С событиями», чтобы посмотреть кадры и таймкоды.' : 'Модель не выявила признаков возможных нарушений. Список событий и кадров пуст.'));
      container.append(empty);
    } else {
      const layout = node('div', 'review-layout');
      const sidebar = node('nav', 'event-sidebar');
      sidebar.setAttribute('aria-label', 'Список событий');
      sidebar.append(node('span', 'section-index', demo ? 'УЧЕБНЫЕ СОБЫТИЯ' : 'СОБЫТИЯ ДЛЯ ПРОВЕРКИ'));
      const content = node('div', 'review-content');
      if (!demo && selectedFile) {
        preview = node('video', 'video-preview');
        preview.controls = true;
        preview.preload = 'metadata';
        preview.playsInline = true;
        previewUrl = URL.createObjectURL(selectedFile);
        preview.src = previewUrl;
        preview.addEventListener('error', () => {
          content.prepend(node('p', 'review-note', 'Браузер не может воспроизвести кодек этой записи. Кадры-доказательства доступны ниже.'));
        }, { once: true });
        content.append(preview);
      }
      const timeline = node('nav', 'event-timeline');
      timeline.setAttribute('aria-label', 'Метки событий');
      timeline.append(node('span', 'section-index', 'ТАЙМКОДЫ'));
      const detail = node('article', 'event-detail');
      const announcement = node('p', 'sr-only');
      announcement.setAttribute('role', 'status');
      const eventButtons = [];
      const timeButtons = [];
      const selectEvent = (index, jump = false) => {
        const event = events[index];
        eventButtons.forEach((button, i) => button.setAttribute('aria-current', String(i === index)));
        timeButtons.forEach((button, i) => button.setAttribute('aria-pressed', String(i === index)));
        detail.replaceChildren();
        detail.append(node('span', 'section-index', `СОБЫТИЕ ${String(index + 1).padStart(2, '0')} / ${String(events.length).padStart(2, '0')}`), node('h3', 'event-detail-title', event.type), node('p', 'event-description', event.description), renderGallery(event, demo), renderSource(event, demo));
        announcement.textContent = `Выбрано событие ${index + 1}: ${event.type}, ${timecode(event.timecode_sec)}`;
        if (jump && !demo) seek(event.timecode_sec);
      };
      events.forEach((event, index) => {
        const button = node('button', 'event-select');
        button.type = 'button';
        const top = node('span', 'event-select-top');
        top.append(node('span', 'event-number', String(index + 1).padStart(2, '0')), node('span', 'event-select-time', timecode(event.timecode_sec)));
        button.append(top, node('strong', '', event.type), node('span', 'event-select-bottom', `${demo ? 'Учебных схем' : 'Кадров'}: ${event.evidence_frames.length} · Посмотреть →`));
        button.addEventListener('click', () => selectEvent(index, true));
        sidebar.append(button);
        eventButtons.push(button);
        const timeButton = node('button', 'timeline-marker', timecode(event.timecode_sec));
        timeButton.type = 'button';
        timeButton.setAttribute('aria-label', `Перейти к событию ${timecode(event.timecode_sec)}`);
        timeButton.addEventListener('click', () => selectEvent(index, true));
        timeline.append(timeButton);
        timeButtons.push(timeButton);
      });
      content.append(timeline, detail, announcement);
      layout.append(sidebar, content);
      container.append(layout);
      selectEvent(0);
    }
    container.append(node('p', 'review-note', demo ? 'Это только пример интерфейса. Для реального анализа загрузите собственное видео.' : 'Результат — подсказка для проверки человеком. Он не является окончательным юридическим или дисциплинарным заключением.'));
  }

  async function showDemo(scenario = 'events', scroll = true) {
    if (busy) return;
    const currentRun = ++runId;
    $('demo-button').disabled = true;
    try {
      const { demoScenarios } = await import('./demo.mjs');
      if (currentRun !== runId || busy) return;
      const focusedControl = document.activeElement?.id;
      resetResult();
      renderResult(demoScenarios[scenario], { demo: true });
      progress.hidden = true;
      $('demo-events').setAttribute('aria-pressed', String(scenario === 'events'));
      $('demo-empty').setAttribute('aria-pressed', String(scenario === 'empty'));
      if (['demo-events', 'demo-empty'].includes(focusedControl)) $(focusedControl).focus({ preventScroll: true });
      if (scroll) panel.scrollIntoView({ behavior: 'smooth', block: 'start' });
    } catch {
      showState('Пример не загрузился', 'Обновите страницу и попробуйте открыть пример ещё раз.', 'error');
    } finally {
      $('demo-button').disabled = busy;
    }
  }

  $('demo-button').addEventListener('click', () => showDemo());
  $('demo-events').addEventListener('click', () => showDemo('events', false));
  $('demo-empty').addEventListener('click', () => showDemo('empty', false));
  $('new-review').addEventListener('click', () => {
    if (busy) return;
    runId += 1;
    selectedFile = null;
    input.value = '';
    $('selected-file').hidden = true;
    startButton.disabled = true;
    fileError('');
    resetResult();
    $('workspace').scrollIntoView({ behavior: 'smooth', block: 'start' });
    input.focus({ preventScroll: true });
  });

  async function analyze() {
    if (!selectedFile || busy) return;
    const currentRun = ++runId;
    resetResult();
    fileError('');
    progress.hidden = false;
    progress.value = 0;
    setBusy(true);
    showState('Загружаем видео', 'Передаём запись на локальный сервер. Не закрывайте страницу.');
    panel.scrollIntoView({ behavior: 'smooth', block: 'center' });
    const started = Date.now();
    $('elapsed-time').textContent = '00:00';
    timer = setInterval(() => {
      const seconds = Math.floor((Date.now() - started) / 1000);
      $('elapsed-time').textContent = `${String(Math.floor(seconds / 60)).padStart(2, '0')}:${String(seconds % 60).padStart(2, '0')}`;
    }, 1000);
    try {
      const jobId = await upload(selectedFile);
      $('job-reference').textContent = `ID ПРОВЕРКИ / ${jobId}`;
      $('job-reference').hidden = false;
      progress.removeAttribute('value');
      showState('Анализируем запись', 'Проверяем визуальные события. Ожидаем ответ локальной модели.');
      const result = await poll(jobId, currentRun);
      if (currentRun === runId) renderResult(result);
    } catch (error) {
      if (currentRun === runId) {
        showState('Проверка не завершена', error.message || 'Произошла ошибка. Повторите попытку.', 'error');
        $('retry-button').hidden = false;
      }
    } finally {
      clearInterval(timer);
      timer = null;
      progress.hidden = true;
      setBusy(false);
    }
  }

  startButton.addEventListener('click', analyze);
  $('retry-button').addEventListener('click', analyze);
  window.addEventListener('pagehide', () => {
    clearInterval(timer);
    clearPreview();
  });
})();
