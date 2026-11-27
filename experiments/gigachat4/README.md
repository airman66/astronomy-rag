# Astronomy RAG Benchmark v1

[Результаты первого прогона](REPORT.md)

100 самостоятельных вопросов, 50 RU и 50 EN. 16 определений, 16 объяснений по глоссарию, 24 задачи по статьям, 14 сопоставлений, 12 проверок политики источников и режимов, 10 общих вопросов по астрономии, 8 запросов за пределами области. Вопросы включают расчёты, ссылку на открытую карточку, разграничение статьи и PDF-обзора, недоступный источник и частичный ответ.

- [Вопросы](../../data/eval/assistant/astronomy_v1_questions.json) содержат только данные пользовательского запроса.
- [Рубрика](../../data/eval/assistant/astronomy_v1_rubric.json) хранит ожидаемый режим, полноту и проверяемые утверждения с ссылками на разделы. Где важна конкретная фраза, проверяется её наличие в реально переданном фрагменте.
- `freeze` использует боевые функции поиска и упаковки. `run` посылает обеим моделям одинаковые системный промпт, пользовательский JSON и лимит 8192 токена на рассуждение и ответ. Модель вызывается один раз на кейс. Проверка связи использует отдельную арифметическую задачу.
- `audit` разделяет промахи поиска, выбор режима моделью, итоговый режим, полноту, схему и ссылки. `judge` отдельно, вслепую сверяет язык, факты и связь каждого утверждения с цитируемым блоком. В отчёте также время LLM, токены и результаты по категориям.
- Checkpoint возобновляет незавершённые запросы и сохраняет HTTP-ошибки, обрывы и ответы вне схемы. SHA-256 привязывает результаты к одному корпусу, индексу, набору вопросов и промпту. Результаты с другим конфигом запускаются в другом каталоге.

## Запуск

Подними сервер эмбеддингов из корневого README. `CLOUD_RU_API_KEY` — статический сервисный ключ Foundation Models. `DEEPSEEK_API_KEY` — ключ прямого API DeepSeek. Оба читаются из окружения или локального `.env`.

```bash
out="$PWD/.benchmark-runs/astronomy-v1"
python -m experiments.gigachat4.grade validate
EMBEDDING_URL=http://127.0.0.1:8082/v1/embeddings \
  python -m experiments.gigachat4.run freeze --out "$out"
python -m experiments.gigachat4.run run --out "$out"
python -m experiments.gigachat4.grade audit --out "$out"

ollama pull qwen3.5:9b
python -m experiments.gigachat4.grade judge --out "$out" --concurrency 1
```

`run --provider deepseek` или `run --provider gigachat` продолжает одну сторону пары. Второй запуск `judge` дополняет checkpoint и строит `report.json`. С `--partial` оценка берёт дописанные строки текущего прогона. Для внешнего судьи есть `--judge runpod` с существующим конфигом проекта. Слепые оценки можно импортировать командой `grade judge --manual FILE`. Файл содержит `reviewer`, `rubric_sha256` и `grades` по ID из `blind-packet.json`, по одному `answer_sha256` и объекту `grade` на ответ. Оценщик проверяет хэши и полный охват.

## GigaChat 4 708B

708 млрд параметров, около 41 млрд активны на токен. Из 64 слоёв 48 используют Gated DeltaNet, 16 — MLA с DeepSeek Sparse Attention. Дообучение длинного контекста шло от 32K через 128K до 1 048 576 токенов. После SFT отдельных специалистов обучали RL, затем их навыки перенесли в общую модель через multi-teacher on-policy distillation. Один checkpoint поддерживает Reasoning и Instant. [Статья команды на Хабре](https://habr.com/ru/companies/sberbank/articles/1112118/) · [Технический отчёт](https://giga.chat/blog/en/posts/gigachat-4-tech-report)

Карточка self-hosted модели управляет двумя режимами через `chat_template_kwargs.reasoning`. Cloud.ru вернул поле `message.reasoning` и с `true`, и с `false`. Через Cloud.ru `reasoning_effort=high` принялся и вернул рассуждение, `max` и `xhigh` дали HTTP 422. API не подтвердил, насколько `high` меняет бюджет относительно `low`. В сравнении используются **GigaChat 4 с запрошенным `high`** и **DeepSeek V4.1 Flash с подтверждённым `max`**. [Карточка GigaChat 4](https://huggingface.co/ai-sage/GigaChat4-708B-A41B) · [Cloud.ru Foundation Models](https://cloud.ru/docs/foundation-models/ug/topics/api-ref) · [DeepSeek Thinking Mode](https://api-docs.deepseek.com/guides/thinking_mode/)

Из обучения на длинных документах следует гипотеза для отдельного эксперимента с увеличенным окном. В v1 окно и промпт едины для обеих моделей. Так метрики по ссылкам показывают пригодность модели в текущем RAG, а промахи поиска видны отдельно. Рубрика и контрольные цитаты доступны лишь оценщику. Для новых гипотез используй новые данные и фиксируй версию до запуска.
