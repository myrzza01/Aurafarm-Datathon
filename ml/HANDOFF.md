# Передача ML-модуля участнику 4

Поставка содержит runtime, локальные веса, конфигурацию и зависимости. Excel,
исходные видео, manifest и обучение для инференса не нужны. Видео, кадры и описания
сцен внешним сервисам не передаются. Звук не декодируется; идентификации лиц нет.

## Установка и перенос

В Git checkout выполняйте команды из `ml/`. Здесь уже находятся `cashwatch/`,
`weights/`, зависимости и эта инструкция. Для ZIP-поставки рабочим каталогом
служит корень распакованного пакета.
Excel и исходный датасет для подключения backend не нужны. Для тестового запроса
используйте разрешённое командой локальное видео. Все пути к весам ниже относительны
к рабочему каталогу; при другом `cwd` backend задайте абсолютный путь к checkpoint.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu118
.\.venv\Scripts\python.exe -m pip install -r requirements-lock.txt --extra-index-url https://download.pytorch.org/whl/cu118
.\.venv\Scripts\python.exe main.py predict --video "C:\local-data\new.mov" --output artifacts/result.json
```

Для отдельного runtime-ZIP вместо `main.py predict` используется созданный при
упаковке `predict.py`: `python predict.py "C:\local-data\new.mov" --output result.json`.

Для полностью изолированной установки заранее скачайте wheels:

```powershell
python -m pip download -r requirements-lock.txt --extra-index-url https://download.pytorch.org/whl/cu118 --dest wheelhouse
```

Перенесите пакет и wheelhouse. В изолированном окружении:

```powershell
python -m pip install --no-index --find-links wheelhouse -r requirements-lock.txt
```

Lock соответствует Windows/Python 3.12/CUDA 11.8. Для Linux нужен wheelhouse для
Linux; Windows wheels несовместимы. Веса переносимы, поддержан также CPU. Отдельный
wheelhouse в текущем цикле не собирался; установленный runtime проверен офлайн.
`requirements-inference.txt` отдельно перечисляет минимальные прямые зависимости.
`weights/detector.pt` содержит baseline — среднее признаков 8 кадров.
`weights/temporal_candidate.pt` содержит экспериментальный temporal checkpoint: после
расширения данных он уменьшил число лишних событий на двух dev-видео, но ошибка
времени составляет 5 с у поставленного seed 42. Независимое test-качество ещё
не измерено. Конфигурации: `weights/config.json` и `weights/temporal_candidate.json`.
Для смены модели backend должен явно передать путь к checkpoint.

## Python и обязательный API

```python
from cashwatch.inference import analyze_video, VideoAnalyzer

result = analyze_video(video_path)
# Для сервера модель загружается один раз в отдельном worker:
analyzer = VideoAnalyzer()  # Находит ml/weights/detector.pt независимо от cwd.
result, diagnostics = analyzer.analyze(video_path, timeout_sec=240)
```

Ответ ML содержит только `verdict` и `violations`. У события: `type`, `timecode_sec`,
`description`, `evidence_frames` (1–3 JPEG/base64 с `timestamp_sec`, `image_base64`).
Для no_violation список пуст. Интервалы начала/конца — в `diagnostics.intervals`,
они не добавлены в обязательную схему жюри. Не логируйте кадры и описания сцен.

Backend добавляет `job_id`, `status=done`. Исключения файлов, декодирования,
таймаута и модели переводятся в `status=failed`, никогда в `no_violation`.

Участник 4 реализует `POST /api/v1/analyze` с multipart-полем `file`, ответом job_id;
`GET /api/v1/jobs/{job_id}` со статусами processing/done/failed. Оба метода проверяют
`X-API-Key`. HTTP-методы не являются частью этого ML-пакета.

## Ограничения и расширение

Модель бинарная. Тип обозначает неклассифицированный подозрительный фрагмент;
описание не выдумывает наблюдаемое действие и юридическое основание. Результат —
подсказка человеку, не окончательное заключение. Данных АБИС, проводок, журналов
нет; модель не делает выводы о них.

Для расширения `diagnostics.intervals[*].type_id` зарезервирован как null, а
`config.event_taxonomy` имеет статус unlabelled. После разметки типов потребуется
дополнительная обученная и проверенная голова классификации. Только проверенный
type_id можно связывать с локальным каталогом участника 1. Сейчас нормативных
ссылок нет: это требование ТЗ ещё открыто. Имя файла не используется как признак.

Порог и параметры объединения сохранены в checkpoint и подбираются только на dev.

## Время и память

Диагностика разделяет decode_sec, model_sec, evidence_sec, elapsed_sec и пиковые
torch CUDA allocated/reserved MB. Память драйвера и других процессов не включена.
Worker выполняет запросы последовательно. Внутренний бюджет 240 с не прерывает
зависший нативный декодер: backend должен контролировать worker-процесс с общим
лимитом 300 с, включая очередь, загрузку и сериализацию.

Полное время HTTP-запроса пока null: интеграции сервера участника 4 нет. После
интеграции его нужно измерить отдельно от времени локального ML-вызова.

## Проверка HTTP после получения backend участника 4

В командном репозитории есть каркас backend; ML-обработчик ещё не подключён.
Реальное время первого и последующих HTTP-анализов пока не измерено.
`tools/benchmark_http.py` — клиент обязательного API,
а не реализация сервера. Он отправляет multipart `file`, проверяет `job_id`, опрашивает
статус до `done` и сохраняет только технические агрегаты. Ответы с кадрами и описаниями
не выводятся и не записываются. Ключ берётся из `CASHWATCH_API_KEY` и должен совпадать
с `API_KEY` настоящего backend; ключ и командная
строка запуска сервера в отчёт не попадают. Допустимы только localhost, loopback и
числовые частные LAN-адреса; прокси и перенаправления не используются.

```powershell
# Сначала задайте CASHWATCH_API_KEY в окружении и запустите настоящий backend.
.\.venv\Scripts\python.exe tools/benchmark_http.py --url http://127.0.0.1:8000 --video "C:\путь\самый-длинный.mov" --warm-runs 3 --output artifacts/integration_checks/http_real.json
```

Отсчёт `full_http_request_sec` начинается перед загрузкой файла и заканчивается после
получения и проверки конечного JSON. Он включает загрузку, очередь, работу backend,
получение кадров и опрос статуса. Интервал опроса по умолчанию 0,2 с; он вносит задержку
наблюдения завершения. Отдельно записаны `upload_and_accept_sec`, `poll_until_done_sec`.
Срок 300 с относится ко всему циклу. Время модели берётся отдельно из диагностики
реального backend; клиент не выдаёт полное HTTP-время за время модели.

У уже работающего сервера первый запрос называется `first_observed_request`, и
`cold_start_measured=false`: это не доказательство холодного старта. Для измерения
запуска процесса создайте локальный JSON `server-start.json` с `argv` (точный массив
аргументов команды участника 4) и `cwd` (его рабочий каталог). Команда должна запускать
сервер в переднем плане и завершать worker-процессы при остановке основного процесса;
не используйте daemon/reload-режим. Затем:

```powershell
.\.venv\Scripts\python.exe tools/benchmark_http.py --url http://127.0.0.1:8000 --video "C:\путь\самый-длинный.mov" --start-command-file server-start.json --warm-runs 3 --output artifacts/integration_checks/http_cold.json
```

Клиент проверяет, что порт изначально свободен, запускает заданную команду без shell,
измеряет `startup_to_listening_sec`, первый запрос после запуска и
`startup_plus_first_request_sec`. В `finally` останавливается только запущенный им
основной процесс. Режим проверен на синтетическом loopback-сервере, который не
обрабатывает видео и не загружает модель. Эти проверки не являются измерением
готового сервиса. Проверка структуры JSON не подтверждает содержание обнаружений.

## Проверка автономности через Windows Firewall

`tools/verify_os_offline.ps1` проверяет права и состояние firewall, затем создаёт
временные исходящие правила Block только для Python-интерпретаторов эксперимента
(учитывается реальный exe виртуального окружения). Общие настройки firewall не
меняются. Правила проверяются в ActiveStore; в `finally` удаляются только собственные
правила с уникальными именами. Нужен уже открытый PowerShell с правами администратора;
скрипт не пытается самостоятельно повышать права. Текущая сессия этих прав не имеет,
поэтому реальная OS-проверка ещё не выполнена. Исходный локальный отчёт фиксирует
`status=blocked`; локальные отчёты экспериментов в Git не включены.

```powershell
# Из ml/ создайте пакет без training-кода и данных:
.\.venv\Scripts\python.exe tools/package_inference.py --output artifacts/runtime_baseline --checkpoint weights/detector.pt --config weights/config.json
# Затем из ml/ в PowerShell администратора:
powershell -NoProfile -ExecutionPolicy Bypass -File tools/verify_os_offline.ps1 -Bundle artifacts/runtime_baseline -Video "C:\local-data\longest.mov" -Output artifacts/integration_checks/os_offline.json
```

Если рабочая папка уже является распакованным runtime-ZIP, используйте `-Bundle .`
без повторной упаковки. Полный Git-каталог содержит training-код и специально
не принимается проверкой в качестве изолированного runtime.

Контроль до блокировки делает только TCP handshake с `1.1.1.1:443`, без HTTP,
видео, кадров и других прикладных данных. Адрес/порт можно задать через `ControlIp`
и `ControlPort`. После создания правил тот же интерпретатор должен получить нативный
Windows Access Denied (10013). Затем отдельный процесс выполняет локальный инференс
из runtime-пакета; Python socket guard не применяется. После удаления правил тот же
контроль должен снова подключиться. Недоступность контрольного узла до блокировки,
обычный timeout вместо доказанного отказа Windows и невозможность восстановления
связности дают явный неуспех/неопределённость, а не ложное подтверждение автономности.
Отчёт содержит время запуска и инференса, память и число событий; содержимое кадров
и описания сцен в него не входят. Этот тест проверяет ML runtime: автономность всего
HTTP-сервиса нужно дополнительно проверить в окружении участника 4.
