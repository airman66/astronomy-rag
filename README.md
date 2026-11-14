# Astro4Edu glossary RAG

Самостоятельный двуязычный справочник и RAG API по утверждённой [спецификации](SPEC.md). Канонические записи лежат в `data/processed/`, исходные снимки — в `data/raw/`. Записи и изображения в API остаются черновиками до редакторского ревью.

## Запуск

```sh
cp .env.example .env
# заполнить DEEPSEEK_API_KEY в .env; для импорта и переиндексации — RUNPOD_API_KEY
docker compose up --build -d
curl http://localhost:8001/health
curl http://localhost:8001/docs
```

При первом запуске Compose скачивает и проверяет SHA-256 официального `Qwen3-Embedding-8B-Q4_K_M.gguf` (4,68 ГБ) в игнорируемый `data/models/`. Сервис эмбеддингов на CPU прогревается при первом запросе. На macOS можно запустить `llama-server` с Metal на хосте и передать `EMBEDDING_URL=http://host.docker.internal:8082/v1/embeddings` в Compose:

```sh
llama-server -m data/models/Qwen3-Embedding-8B-Q4_K_M.gguf --embedding --pooling last -ngl 99 -c 4096 --host 0.0.0.0 --port 8082
```

Сервер на хосте в таком режиме доступен в локальной сети; ограничьте интерфейс или используйте только доверенную сеть. Без ключа DeepSeek чтение и поиск работают, ассистент возвращает `503 assistant_unavailable`.

## Данные и индекс

```sh
python3 -m app.importer --download-media
python3 -m app.translate
python3 -m app.audit
python3 -m app.storage
python3 -m app.chunks
```

Импорт и перевод возобновляемы. Проверка `app.audit` сверяет структуру, ссылки, тексты и файлы. `scripts/register_image_edits.py` обновляет `data/image_edits.json` и пути русских изображений в канонических записях; после него перестройте SQLite.

`data/index/` создан на временном GPU Pod Runpod. Скрипт `scripts/build_index.py` получает `chunks.jsonl` и `chunk-manifest.json`, один и тот же GGUF Qwen3-Embedding-8B и URL `llama-server --embedding --pooling last`. Команда:

```sh
python scripts/build_index.py data/index --model-file data/models/Qwen3-Embedding-8B-Q4_K_M.gguf --embedding-url http://127.0.0.1:8080/v1/embeddings
```

Скрипт сохраняет FAISS `IndexFlatIP`, таблицу ID, кэш векторов по хэшу текста и манифест с хэшами модели/файлов. Публикуйте артефакты после проверки хэшей. Сервис сверяет их перед загрузкой. Для запросов подаётся инструкция Qwen `Instruct: ...\nQuery: ...`, для документов — исходный текст чанка. Pod для онлайн запросов не нужен.

Карта часовых поясов локализована из [векторного исходника Wikimedia Commons](https://commons.wikimedia.org/wiki/File:World_Time_Zones_Map.svg): 693 текстовые метки, словарь `data/processed/image_labels/world_time_zones.json`. Исходный SVG не входит в репозиторий; для повторной сборки сохраните его как `data/raw/world_time_zones_map_source.svg` и установите `rsvg-convert`:

```sh
python3 scripts/localize_timezone_svg.py data/raw/world_time_zones_map_source.svg data/processed/image_labels/world_time_zones.json data/media/ru/media_jh312430Ry7.png
```

Оригинальный PNG Astro4Edu сохранён отдельно. Вектор обновлялся независимо от него и содержит дополнительные сведения о переходе на летнее время; различия зафиксированы в `data/image_edits.json`.

## Проверка

```sh
python3 -m unittest discover -s tests -v
python3 -m app.audit
python3 -m scripts.evaluate --answers 10
```

Для последней команды нужны `DEEPSEEK_API_KEY` и запущенный сервер эмбеддингов. Результат 50 контрольных вопросов — `data/eval/report.json`; сравнение размеров чанков — `data/eval/chunk_config.json`. OpenAPI доступен в `/openapi.json`, Swagger UI — в `/docs`.
