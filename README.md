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

`python3 -m scripts.evaluate` прогоняет 50 контрольных вопросов через провайдера из `.env` и записывает версию промпта, модель и метрики в `data/eval/assistant/report.json`. Для гибридного поиска при прогоне с хоста укажите `EMBEDDING_URL` локального сервера эмбеддингов.

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
