# Астрономический справочник и RAG

Самостоятельный FastAPI-сервис по [спецификации](SPEC.md): 501 термин Astro4Edu на английском и русском, исходные медиа, локализованные изображения, поиск FTS5 + FAISS и ответы LLM с проверяемыми ссылками на источники. Канонические записи лежат в `data/processed/`, исходные снимки — в `data/raw/`. Статусы редакторского ревью сохранены из исходного проекта.

Выделен из [airman66/astro](https://github.com/airman66/astro) на коммите `f5a1d1c`. История терминов, переводов, индекса и API перенесена в корень репозитория.

## Запуск

```sh
cp .env.example .env
# заполнить DEEPSEEK_API_KEY для помощника; без ключа справочник и поиск работают
docker compose up --build -d
curl http://localhost:8001/health
curl http://localhost:8001/docs
```

По умолчанию поиск лексический. Для гибридного поиска включите профиль `embeddings`: он скачивает и проверяет SHA-256 `Qwen3-Embedding-8B-Q4_K_M.gguf` (4,68 ГБ) в игнорируемый `data/models/` и запускает сервер эмбеддингов на CPU:

```sh
docker compose --profile embeddings up --build -d
```

На macOS можно вместо контейнера запустить `llama-server` с Metal и задать `EMBEDDING_URL=http://host.docker.internal:8082/v1/embeddings`:

```sh
llama-server -m data/models/Qwen3-Embedding-8B-Q4_K_M.gguf --embedding --pooling last -ngl 99 -c 4096 --host 0.0.0.0 --port 8082
```

Настройки Compose приведены в `.env.example`. `ASSISTANT_DEADLINE` задаётся в секундах и должен быть меньше 150 секунд, срока аренды слота Redis. В локальных командах `DATA_DIR` может указывать на отдельный каталог данных.

Для RunPod вместо DeepSeek задайте `ASSISTANT_PROVIDER=runpod`, `RUNPOD_API_KEY` и `RUNPOD_ENDPOINT_ID`. Без ключа LLM чтение и поиск работают, ассистент возвращает `503 assistant_unavailable`.

## Структура проекта

- `app/main.py` собирает FastAPI. Маршруты и схемы находятся в `app/api/`, HTTP middleware — в `app/middleware/`.
- `app/assistant/` содержит обработку вопросов, подбор источников, промпты и клиент LLM.
- `app/catalog/` содержит SQLite, медиа, FTS5 и векторный поиск.
- `app/pipeline/` содержит импорт Astro4Edu, перевод, аудит и чанкинг. Команды `python -m app.<команда>` запускают реализацию из пакетов.
- `app/infra/` содержит настройки окружения, файловые операции и Redis-лимиты.
- `scripts/translation_review/` содержит этапы ревью переводов. `scripts/covers/` содержит поиск, импорт и отчёт по обложкам. Для запуска используются команды ниже.

## Данные и индекс

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

Карта часовых поясов локализована из [векторного исходника Wikimedia Commons](https://commons.wikimedia.org/wiki/File:World_Time_Zones_Map.svg): 693 текстовые метки, словарь `data/processed/image_labels/world_time_zones.json`. Исходный SVG не входит в репозиторий; для повторной сборки сохраните его как `data/raw/world_time_zones_map_source.svg` и установите `rsvg-convert`:

```sh
python3 scripts/localize_timezone_svg.py data/raw/world_time_zones_map_source.svg data/processed/image_labels/world_time_zones.json data/media/ru/media_jh312430Ry7.png
```

Оригинальный PNG Astro4Edu сохранён отдельно. Вектор обновлялся независимо от него и содержит дополнительные сведения о переходе на летнее время; различия зафиксированы в `data/image_edits.json`.

## Проверка

Для локальных команд установите зависимости:

```sh
python3 -m venv .venv
.venv/bin/pip install -r requirements-dev.txt
```

```sh
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python -m app.audit
.venv/bin/python -m scripts.illustrations verify --complete
```

Оригиналы медиа и модель исключены из Git. После нового клонирования оригиналы восстанавливаются командой `python3 -m app.importer --media-only --workers 4`; без них полный `app.audit` сообщает об отсутствующих файлах, а API продолжает выдавать записи и локализованные медиа. Результат 50 контрольных вопросов в `data/eval/report.json` относится к прежней версии промпта; новый промпт нужно оценивать отдельно через `python3 -m scripts.eval_scope` с ключом LLM. OpenAPI доступен в `/openapi.json`, Swagger UI — в `/docs`.

## Ревью переводов и иллюстрации

Для OCR в `scripts.ocr_audit` и `scripts.register_image_edits` установите `requirements-media.txt` и Tesseract с языками `eng` и `rus`. Импорт обложек из Wikimedia также использует ImageMagick (`magick`).

Команды `review_translations`, `resolve_translation_review` и `apply_translation_resolutions` относятся к прежнему ревью кэша переводов. Текущий проход по каноническим записям выполняет `scripts.translation_audit`.


```sh
python3 -m scripts.translation_audit all --workers 8        # полный проход: сбор, ревью, проверка, применение, отчёт
python3 -m scripts.translation_audit report                 # пересобрать отчёт по каждому из 501 ID
python3 -m scripts.illustrations manifest                   # манифест иллюстраций для всех терминов
python3 -m scripts.wikimedia_covers discover --workers 2   # поиск свободных файлов Wikimedia по одному запросу
python3 -m scripts.wikimedia_covers astro-fallback          # готовые медиа Astro4Edu как запасной источник
python3 -m scripts.wikimedia_covers import --workers 2     # загрузка и OCR, без генерации
python3 -m scripts.wikimedia_covers curated                # вручную подобранные файлы Commons для омонимов
python3 -m scripts.wikimedia_covers related                # явные привязки к близким темам
python3 -m scripts.illustrations verify --complete          # 501 файл, атрибуция, хэши и размеры
python3 -m scripts.wikimedia_covers report                 # CSV и страница визуального просмотра по всем ID
```

Результат поиска кэшируется в `data/illustrations/wikimedia_candidates.json`; прямые исключения и связи с близкими темами записаны по ID в `curated_wikimedia.json` и `related_fallbacks.json`. Отчёты первичного поиска — `wikimedia_report.json` и `astro_fallback_report.json`. Итоговое покрытие и атрибуция — `coverage.csv`, визуальный просмотр — `review.html`. Статус `approved` выставляется только после редакторской проверки.
