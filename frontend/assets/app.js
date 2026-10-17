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

  function timecode(seconds) {
    const value = Math.max(0, Number(seconds) || 0);
    const minutes = Math.floor(value / 60);
    const whole = Math.floor(value % 60);
    const tenth = Math.floor((value % 1) * 10);
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
    clearPreview();
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
    clearPreview();
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
        else reject(new Error(xhr.status === 401 ? 'Сессия истекла. Обновите страницу и повторите загрузку.' : xhr.status === 413 ? 'Размер видео превышает допустимый лимит.' : xhr.status === 415 ? 'Сервис принимает только файлы .mov.' : 'Не удалось загрузить видео. Проверьте файл и повторите попытку.'));
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
    const deadline = Date.now() + 290000;
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

  function renderResult(job) {
    const events = job.violations || [];
    if (!['violation', 'no_violation'].includes(job.verdict) || (job.verdict === 'violation') !== (events.length > 0)) {
      throw new Error('Результат анализа содержит несогласованные данные. Требуется повторная проверка.');
    }
    showState(job.verdict === 'violation' ? 'Обнаружены признаки возможных нарушений' : 'Нарушение не обнаружено', job.verdict === 'violation' ? `Событий для проверки: ${events.length}. Изучите кадры, описание и источник правила.` : 'В этой видеозаписи модель не выявила признаков возможных нарушений.', 'done');
    const container = $('results');
    if (events.length) {
      preview = node('video', 'video-preview');
      preview.controls = true;
      preview.preload = 'metadata';
      preview.playsInline = true;
      previewUrl = URL.createObjectURL(selectedFile);
      preview.src = previewUrl;
      preview.addEventListener('error', () => {
        container.prepend(node('p', 'review-note', 'Браузер не может воспроизвести кодек этой записи. Кадры-доказательства доступны ниже.'));
      }, { once: true });
      container.append(preview);
    }
    for (const event of events) {
      const article = node('article', 'result-event');
      const heading = node('div', 'event-header');
      const jump = node('button', 'timecode', timecode(event.timecode_sec));
      jump.type = 'button';
      jump.setAttribute('aria-label', `Перейти к событию ${timecode(event.timecode_sec)}`);
      jump.addEventListener('click', () => seek(event.timecode_sec));
      heading.append(jump, node('h3', '', event.type));
      article.append(heading, node('p', 'event-description', event.description));
      const frames = node('div', 'evidence-grid');
      for (const frame of event.evidence_frames || []) {
        const figure = node('figure');
        const image = node('img');
        image.src = `data:image/jpeg;base64,${frame.image_base64}`;
        image.alt = `Кадр события «${event.type}» в ${timecode(frame.timestamp_sec)}`;
        figure.append(image, node('figcaption', '', `КАДР / ${timecode(frame.timestamp_sec)}`));
        frames.append(figure);
      }
      article.append(frames);
      const rule = event.rule_source;
      const source = node('p', 'rule-source');
      source.append(node('strong', '', 'Основание: '));
      if (rule) {
        source.append(document.createTextNode(`${rule.document}${rule.section ? `, ${rule.section}` : ''}`));
        if (rule.url) {
          try {
            const url = new URL(rule.url);
            if (['https:', 'http:'].includes(url.protocol)) {
              const link = node('a', '', ' Открыть источник ↗');
              link.href = url.href;
              link.target = '_blank';
              link.rel = 'noopener noreferrer';
              source.append(link);
            }
          } catch { /* Invalid source URLs are displayed without a link. */ }
        }
      } else source.append(document.createTextNode('Источник правила не передан моделью. Требуется уточнение.'));
      article.append(source);
      container.append(article);
    }
    container.append(node('p', 'review-note', 'Результат — подсказка для проверки человеком. Он не является окончательным юридическим или дисциплинарным заключением.'));
  }

  async function analyze() {
    if (!selectedFile || busy) return;
    const currentRun = ++runId;
    clearPreview();
    $('results').replaceChildren();
    $('retry-button').hidden = true;
    $('job-reference').hidden = true;
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
