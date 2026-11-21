# Астрономический справочник и RAG

Самостоятельный FastAPI-сервис по [спецификации](SPEC.md): 501 термин Astro4Edu на английском и русском, исходные медиа, локализованные изображения, поиск FTS5 + FAISS и ответы LLM с проверяемыми ссылками на источники. Канонические записи лежат в `data/processed/`, исходные снимки — в `data/raw/`. Статусы редакторского ревью хранятся в записях.

## Запуск

```sh
docker compose up --build -d
curl http://127.0.0.1:8001/health
curl http://127.0.0.1:8001/docs
```

По умолчанию поиск лексический. Для гибридного поиска включите профиль `embeddings`: он скачивает и проверяет SHA-256 `Qwen3-Embedding-8B-Q4_K_M.gguf` (4,68 ГБ) в игнорируемый `data/models/` и запускает сервер эмбеддингов на CPU:

```sh
docker compose --profile embeddings up --build -d
```

На macOS можно вместо контейнера запустить `llama-server` с Metal и задать `EMBEDDING_URL=http://host.docker.internal:8082/v1/embeddings`:

```sh
llama-server -m data/models/Qwen3-Embedding-8B-Q4_K_M.gguf --embedding --pooling last -ngl 99 -c 4096 --host 0.0.0.0 --port 8082
```

Чтобы настроить ассистента, скопируйте `.env.example` в `.env` и заполните `DEEPSEEK_API_KEY`. Там же описаны дополнительные настройки Compose. `ASSISTANT_DEADLINE` задаётся в секундах и должен быть меньше 150 секунд, срока аренды слота Redis. В локальных командах `DATA_DIR` может указывать на отдельный каталог данных.

API хранит SQLite в Docker volume `catalog-db`. При запуске без Compose база по умолчанию создаётся в `data/glossary.db`. Путь можно переопределить через `CATALOG_DB_PATH`.

Для RunPod вместо DeepSeek задайте `ASSISTANT_PROVIDER=runpod`, `RUNPOD_API_KEY` и `RUNPOD_ENDPOINT_ID`. Без ключа LLM чтение и поиск работают, ассистент возвращает `503 assistant_unavailable`.

## Структура проекта

- `app/main.py` собирает FastAPI. Маршруты и схемы находятся в `app/api/`, HTTP middleware — в `app/middleware/`.
- `app/assistant/` содержит обработку вопросов, подбор источников, промпты и клиент LLM.
- `app/catalog/` содержит SQLite, медиа, FTS5 и векторный поиск.
- `app/pipeline/` содержит импорт Astro4Edu, перевод, аудит и чанкинг. Команды `python -m app.<команда>` запускают реализацию из пакетов.
- `app/infra/` содержит настройки окружения, файловые операции и Redis-лимиты.
- `scripts/index/` собирает FAISS и обслуживает кэш эмбеддингов. `scripts/evaluation/` проверяет поиск и область вопросов ассистента.
- `scripts/covers/` отвечает за подбор, импорт, манифест и отчёты по обложкам. `scripts/media/` содержит OCR и локализацию исходных медиа.
- `scripts/translation_review/` содержит этапы ревью переводов, а `legacy/` — прежние команды ревью кэша. Команды запуска приведены ниже.

## Данные и индекс

[Карта каталога `data/`](data/README.md) объясняет, где находятся источники, канонические записи, медиа, отчёты и локальные артефакты.

```sh
python3 -m app.importer --download-media
python3 -m app.translate
python3 -m app.audit
python3 -m app.storage
python3 -m app.chunks
```

Импорт и перевод возобновляемы. Проверка `app.audit` сверяет структуру, ссылки, тексты и файлы. `scripts/register_image_edits.py` обновляет `data/image_edits.json` и пути русских изображений в канонических записях; после него перестройте SQLite.

Для OAE Reviews импортируются 20 страниц тем и 23 PDF от OAE (19 обзоров, два методических комплекта, памятка и ответ на отзывы о воркшопе). Импорт PDF использует Poppler и Tesseract с английским языковым пакетом. Документы и темы доступны через `/api/v1/articles`, оригиналы лежат в `data/raw/reviews/`.

Для пересборки OAE Reviews из чистого клона установите зависимости из `requirements.txt` и выполните команды:

```sh
python -m pip install -r requirements-pdf.txt
python -m app.audit --reviews
python -m app.pipeline.review_exports
python -m app.chunks
python -m scripts.seed_embedding_cache data/index
```

Команды используют сохранённые тексты и изображения. Для проверки сборки в отдельной папке передайте `review_exports --output-dir /path/to/pdf`. Для пересборки одного выпуска передайте также `--glob 'oae-document-….json'`. Импорт источников запускается отдельно через `python -m app.pipeline.reviews --media`, извлечение иллюстраций для нового PDF — через `python -m app.pipeline.review_figures --glob 'oae-document-….json'`.

Оригиналы PDF выдаются через `GET /api/v1/articles/{article_id}/source.pdf`, русские издания — через `GET /api/v1/articles/{article_id}/translation.pdf`. Индекс строится по тексту страниц PDF, разделам тем и описаниям иллюстраций. Чанк хранит ID статьи и раздела или ID медиа, связанного с записью. Qwen3-Embedding индексирует текст, а сами файлы доступны по ссылкам API.

`python3 -m app.importer --refresh` сохраняет новые версии API и HTML рядом со старыми снимками и повторно проверяет исходные медиа. При изменении изображения прежняя русская версия снимается с выдачи до повторной локализации. `--refresh --ids 1 2` обновляет только указанные термины, но проверяет медиа всех записей.

`data/index/` создан на временном GPU Pod Runpod. Скрипт `scripts/build_index.py` получает `chunks.jsonl` и `chunk-manifest.json`, один и тот же GGUF Qwen3-Embedding-8B и URL `llama-server --embedding --pooling last`. Команда:

```sh
python -m scripts.build_index data/index --model-file data/models/Qwen3-Embedding-8B-Q4_K_M.gguf --embedding-url http://127.0.0.1:8080/v1/embeddings
```

Скрипт сохраняет FAISS `IndexFlatIP`, таблицу ID, кэш векторов по хэшу текста и манифест с хэшами модели/файлов. Публикуйте артефакты после проверки хэшей. Сервис сверяет их перед загрузкой. Для запросов подаётся инструкция Qwen `Instruct: ...\nQuery: ...`, для документов — исходный текст чанка. Pod для онлайн запросов не нужен.

Карта часовых поясов локализована из [векторного исходника Wikimedia Commons](https://commons.wikimedia.org/wiki/File:World_Time_Zones_Map.svg): 693 текстовые метки, словарь `data/media/localization/world_time_zones.json`. Исходный SVG лежит в `data/raw/world_time_zones_map_source.svg`. Для повторной сборки установите `rsvg-convert` и выполните команду.

```sh
python3 scripts/localize_timezone_svg.py data/raw/world_time_zones_map_source.svg data/media/localization/world_time_zones.json data/media/ru/media_jh312430Ry7.png
```

Оригинальный PNG Astro4Edu сохранён отдельно. Вектор обновлялся независимо от него и содержит дополнительные сведения о переходе на летнее время; различия зафиксированы в `data/image_edits.json`.

## Проверка

`requirements.txt` содержит зависимости сервиса и команд импорта и индексации. `requirements-media.txt` добавляет OCR. `requirements-dev.txt` включает оба набора и BasedPyright.

После правок в `requirements.in` обновите фиксированные версии для Python 3.12:

```sh
uv pip compile --universal --python-version 3.12 --no-annotate --output-file requirements.txt requirements.in
```

Для локальных команд используйте Python 3.12.

```sh
python3.12 -m venv .venv
.venv/bin/pip install -r requirements-dev.txt
```

```sh
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python -m app.audit
.venv/bin/python -m scripts.illustrations verify --complete
```

Проверка типов использует Python 3.12 и режим `standard` в `pyrightconfig.json`. Та же конфигурация читается Pyright и BasedPyright.

```sh
.venv/bin/basedpyright
```

Тесты сгруппированы в `tests/api/`, `tests/assistant/`, `tests/catalog/` и `tests/pipeline/`. Из корня проекта можно запустить отдельную группу или файл.

```sh
.venv/bin/python -m unittest discover -s tests/assistant -t . -v
.venv/bin/python -m unittest tests.api.test_limits -v
```

GitHub Actions запускает тесты, BasedPyright, аудит данных и проверку Docker. На чистом клоне `app.audit --skip-original-media` проверяет записи и локализованные медиа. Для полного аудита загрузите оригиналы командой `python3 -m app.importer --media-only --workers 4`. OpenAPI доступен в `/openapi.json`, Swagger UI — в `/docs`.

### Оценка помощника

`python3 -m scripts.evaluate --retrieval-only --output /tmp/rag-retrieval.json` проверяет, попали ли нужные разделы в реальный набор доказательств. Для гибридного прогона с хоста укажите `EMBEDDING_URL` локального сервера эмбеддингов. Основной eval — 50 полных запросов (25 RU, 25 EN), включая статьи, неполный ответ, отсутствие статьи и границы темы. Исторический отчёт лежит в `data/eval/assistant/report.json`.

Боевой прогон использует провайдера из окружения, а `--case` и `--limit` ограничивают число оплачиваемых вызовов. Записывайте отчёты кандидатов в разные файлы. При заданных тарифах eval считает оценку затрат всех завершённых кейсов на 1 000 прошедших рубрику. При отсутствии usage итоговая оценка затрат остаётся пустой. Проверки фактов по текстовым признакам требуют ручной сверки научной полноты на ответах из отчёта.

```sh
python3 -m scripts.evaluate --help
ASSISTANT_PROVIDER=runpod python3 -m scripts.evaluate --case partial-en --case explain-no-article-ru --output /tmp/rag-glm-pilot.json
ASSISTANT_PROVIDER=runpod RUNPOD_ENDPOINT_ID=qwen3-32b-awq \
  RUNPOD_MODEL=Qwen/Qwen3-32B-AWQ RUNPOD_THINKING=disabled \
  python3 -m scripts.evaluate --case partial-en --case explain-no-article-ru --output /tmp/rag-qwen-pilot.json
```

Для следующего этапа берите одинаковые диагностические кейсы `article-exoplanet-detection-en`, `explain-no-article-ru`, `general-1-ru`, `partial-en` и `scope-2-en`, затем расширьте до всех 50. Перед сравнением thinking в Qwen подтвердите поддержку параметра на Public Endpoint коротким вызовом. Снимите базу с thinking выключенным, затем включите его на сложных кейсах. Кандидата выбирайте по доле валидных ответов, критическим фактическим ошибкам, p95 и цене. При отсутствии URL/ключа для сторонней модели контракт адаптера проверяется HTTP-подменой, а тёплая/холодная задержка требует реального endpoint.


Если компактная модель проходит проверку фактов и ссылок, следующий кандидат — Qwen3‑4B‑Instruct‑2507 либо Qwen3.5‑4B на одном GPU RunPod. Сравнивайте thinking на причинных и составных вопросах при одинаковом пределе выходных токенов. Засчитывайте только завершённые и валидные ответы. Для включения reasoning ориентируйтесь на прирост полноты без критических ошибок и измеренный прирост p95.

`python3 -m scripts.smoke_http http://127.0.0.1:8001 --search-mode lexical` проверяет контейнерный запуск без LLM-ключей и с настоящим Redis. При запуске с эмбеддингами укажите `--search-mode hybrid`.

## Ревью переводов и иллюстрации

Для OCR в `scripts.ocr_audit` и `scripts.register_image_edits` установите `requirements-media.txt` и Tesseract с языками `eng` и `rus`. Импорт обложек из Wikimedia также использует ImageMagick (`magick`).

Команды `review_translations`, `resolve_translation_review` и `apply_translation_resolutions` относятся к прежнему ревью кэша переводов. Текущий проход по каноническим записям выполняет `scripts.translation_audit`.


```sh
python3 -m scripts.translation_audit all --workers 8        # полный проход: сбор, ревью, проверка, применение, отчёт
python3 -m scripts.translation_audit report                 # пересобрать отчёт по каждому из 501 ID
python3 -m scripts.illustrations manifest                   # манифест иллюстраций для всех терминов
python3 -m scripts.wikimedia_covers discover --workers 2   # поиск свободных файлов Wikimedia по одному запросу
python3 -m scripts.wikimedia_covers astro-fallback          # готовые медиа Astro4Edu как запасной источник
python3 -m scripts.wikimedia_covers import --workers 2     # загрузка обложек и OCR
python3 -m scripts.wikimedia_covers curated                # вручную подобранные файлы Commons для омонимов
python3 -m scripts.wikimedia_covers related                # явные привязки к близким темам
python3 -m scripts.illustrations verify --complete          # 501 файл, атрибуция, хэши и размеры
python3 -m scripts.wikimedia_covers report                 # CSV и страница визуального просмотра по всем ID
```

Кэш кандидатов Wikimedia, ручной отбор и связанные замены находятся в `data/illustrations/selection/`. Отчёты импорта, `coverage.csv` и `review.html` для визуального просмотра находятся в `data/illustrations/reports/`. Статус `approved` выставляется только после редакторской проверки.

## Проверка полноты медиа при развёртывании

Часть исходных изображений Astro4Edu не хранится в Git; на чистом checkout
приложение может выдавать `404` для этих оригиналов. CI сообщает о таких
отсутствиях, но не требует скачивать весь корпус медиа на GitHub runner.

```sh
python -m scripts.media.check_paths --report /tmp/astronomy-media.json
python -m app.importer --media-only  # восстановить отсутствующие исходники по source_url
python -m scripts.media.check_paths --strict --verify-hashes
```

`--strict` предназначен для развёртывания с полным датасетом, а не для
минимального тестового checkout. Если исходник более недоступен или отличается
по SHA-256, импорт сообщит об этом и не будет принимать подменённый файл.
Проверка согласованности опубликованных FAISS-артефактов (без GGUF и без
запуска embedding-сервера): `python -m scripts.index.verify data/index`.
Это **не** тест семантического качества: после изменения чанкинга или данных
перестройте индекс и проведите отдельную RU/EN оценку hybrid retrieval с Qwen.

Кэш перевода теперь контекстно-зависимый: старые текстовые ключи
`.translation-cache.json` не используются для новых полей, чтобы одинаковые
английские слова в разных материалах не получали ошибочно общий перевод.
