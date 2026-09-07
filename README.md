# Документация проекта Neo4j Assistant

## 1. Обзор системы и назначение

Проект представляет собой текстовый ИИ-ассистент для интерактивного взаимодействия с графовой базой знаний Neo4j. Ассистент выступает в роли Q&A слоя (Graph-RAG) для данных БД. Интерфейс — SPA (`web/`), API — FastAPI (`server/`).

Предметная область базы данных: *food-science, микробиология, умная упаковка (smart packaging)*.

**Ключевой принцип работы:** Ассистент опирается строго на данные из базы, всегда сопровождая ответ провенансом (`evidence`, файлом-источником и `confidence`).

Голосовой контур (STT/TTS) в коде сохранён, но по умолчанию выключен (`audio_enabled: false` в `server/config.yaml`). Основной путь — чат по SSE, не WAV.

---

## 2. Архитектура и обработка запроса (Pipeline)

Система построена на асинхронном пайплайне FastAPI (`server/core/app.py`, `server/core/pipeline.py`). Состояние диалога (сообщения, ветки, checkpoint, UNIT, карточки) лежит в SQLite (`server/core/app_store.py`).

### Схема полного пайплайна

```
Текстовый запрос из SPA
    │
    ▼ POST /process_text_stream
    │   либо POST /api/conversations/{id}/branches/{branch_id}/turns,
    │   если открыта конкретная ветка.
    │
    ▼ LLM Orchestrator
    │   Модель читает промпт режима (auto / staged / card)
    │   и при необходимости вызывает ask_subgraph.
    │
    ▼ Tool Call: ask_subgraph
    │   Английские subquestions + retrieval S1–S5
    │   (embed → ANN → опционально CE → графы → carousel → UNIT).
    │
    ▼ Оценка полноты данных (LLM)
    │   Если информации хватает ──► переход к финальному ответу.
    │   Если нет ──► повторный вызов ask_subgraph (бюджет: max_turns, в yaml — 2).
    │   Режим «По этапам»: пауза approval_required, поиск после подтверждения.
    │
    ▼ LLM (финальный ответ)
    │   Формирует ответ по UNIT-блокам; факты помечает (source:N).
    │
    ▼ Checkpoint (SQLite)
    │   Сообщения, accepted chains, снимок retrieval (версия retrieval-carousel-v1),
    │   graph_run для панели графа.
    │
    ▼ Клиент (UI)
        Рисует markdown-ответ. После done с graph_run_id справа открывается граф.

Пользователь смотрит граф ответа
    │
    ▼ POST /graph_viz  или  GET /api/checkpoints/{id}/graph
    │   Сервер берёт accepted chains из SQLite и дочитывает
    │   свойства узлов/связей из Neo4j по elementId, без embedding-полей.
    │
    ▼ Клиент (UI)
        vis-network: отдельные цепи и объединенный граф.
```

В чате у модели один инструмент — `ask_subgraph`. Заполнение карточки идёт отдельным промптом и tool `submit_card`, без GraphRAG.

---

## 3. Инструменты и retrieval

Ядро извлечения данных: `server/tools/` + `server/algorithm/`.

### 3.1 `ask_subgraph` — поиск цепочек evidence
Оркестратор вызывает tool `ask_subgraph` с английскими subquestions. Пайплайн
(`server/algorithm/pipeline.py`) поднимает accepted chains (UNIT-туры в порядке
обхода, `@Hub` на лучах хаба, `source_file` и `confidence` на строке evidence).

Стадии одной строкой: **S1** embed sq → **S2** ANN → **S2b** cross-encoder (если `rerank_enabled`) → **S3** графы на sq (якоря + мосты, один раз) → **S4** carousel hop-DP → **S5** дедуп по spine-evidence. Детали стадий — в `server/algorithm/README.md`.

Инварианты вызова живут в коде, а не в промпте (`server/core/turn_state.py`,
`server/tools/subgraph_search.py`):
*   **Глубина поиска** (`low|medium|high`) приходит из интерфейса
    (`search_depth` в теле запроса), модель её не выбирает и не видит.
    В UI: Компактно / Обычно / Расширенно. Это бюджет числа UNIT (`max_paths_*`), не «ширина ANN».
*   **Бюджет ответа** — не больше `max_turns` вызовов `ask_subgraph` на один ход
    пользователя; лишний вызов не запускает пайплайн, а возвращает `TOOL_ERROR`.
*   **Контракт subquestions** проверяется до поиска: только английские
    утверждения, без `?`, без повторов внутри вызова и между вызовами хода,
    не больше шести.
*   Ответы инструмента начинаются с однозначных маркеров `NO_RESULTS` /
    `TOOL_ERROR`, поведение на каждый маркер задано в промпте одной строкой.

Корпус ANN и мостов S3 фильтруется по `run_id` текущего аккаунта. Значение
`server/config.yaml` используется только как bootstrap для существующих строк
при миграции и как default для новых аккаунтов. Пустой `run_id` запрещён.
В ANN условие `r.run_id = $run_id` находится внутри `SEARCH` до `LIMIT`, поэтому
лимит набирается только из рёбер нужного корпуса. Эмбеддинг retrieval — локальный
Ollama, задаётся `EMBED__*` в `.env`, не yaml.

### 3.2 Граф ответа — визуализация без LLM
В UI рисуется ровно тот подграф, который вошёл в accepted chains checkpoint (или последнего `graph_run_id`).

```
ask_subgraph accepted chains
    │
    ▼ SQLite (graph_runs + checkpoint)
    │   Цепи ответа и снимок retrieval на checkpoint.
    │   SSE done получает graph_run_id и graph_chain_count.
    │
    ▼ POST /graph_viz
    │   По elementId рёбер дочитывает из Neo4j тип связи, направление,
    │   имена/labels узлов, evidence, source_file, chunk_id,
    │   confidence и run_id. Embedding-поля не выбираются.
    │
    ▼ GET /api/checkpoints/{id}/graph?scope=…
    │   Тот же payload: context (все данные ветки),
    │   new_in_answer (новые в ответе), all_branches (другие версии).
    │
    ▼ Web UI
        Правая панель vis-network: стрелки листают цепи,
        инспектор показывает evidence, source и confidence.
```

---

## 4. Схема Neo4j графа

Текстовые ссылки на вершины используют только свойство `name`. Labels могут
отсутствовать, быть единственными или множественными; приложение хранит их как
необязательные метаданные explorer-фильтров, но не выбирает «главный» label и
не ветвит алгоритм по классам. Типы связей также полностью задаёт корпус.

*   **Свойства связей (обязательный провенанс):**
    *   `evidence`: подготовленный агентом фрагмент данных по файлу; он может быть структурированным обобщением содержимого PDF.
    *   `source_file`: имя файла-источника.
    *   `chunk_id`: идентификатор текстового фрагмента.
    *   `confidence`: уверенность при экстракции (0.0–1.0; отсутствие свойства не подменяется на 1.0).
    *   `run_id`: идентификатор загрузки корпуса.

---

## 5. LLM-подсистема (`server/llm/`)

*   **`BaseLLMProvider` / `OpenAIProvider`:** Работа с LLM через OpenAI SDK. Каталог в `llm.ui_profiles`:
    *   `auto` — первая живая модель из `llm.auto_order` (QwenCloud), с дефолтным thinking; при исчерпании бесплатной квоты модели переход к следующей. Баны хранятся в SQLite по sha256 ключа.
    *   Десять облачных профилей DashScope (`qwen38_flash` … `qwen37_flash`). Ключ наследуют у `qwen_cloud` (`LLM__PROFILES__QWEN_CLOUD__*`).
    *   Запасной `ollama` (Gemma 4 31B) не в UI: серверный ключ `LLM__PROFILES__OLLAMA__API_KEY`, если облако недоступно.
*   **`current_profile`:** по умолчанию `auto`. Неизвестный id в запросе — HTTP 400. `qwen_cloud` остаётся credential-родителем, в селекторе его нет.
*   **Tool Call Loop:** `generate_response_stream` крутит вызовы инструментов до `max_turns`. Ротация Auto только до первого thinking/content/tool_call. SSE `model` отдаёт фактический id/label.
*   **Промпты** (`prompts/`, загрузчик `server/llm/prompt_loader.py`):
    *   `assistant_logic.md` — режим автоматически;
    *   `assistant_staged.md` — режим «По этапам»;
    *   `card_generation.md` — заполнение карточки.
    Файлы `output_quality.md` / `output_speed.md` подключаются только при `audio_enabled`.

---

## 6. Речь: STT и TTS

В `server/config.yaml` стоит `audio_enabled: false`. CPU-образ Compose (`server/Dockerfile.cpu`) не ставит Faster Whisper и TTS-модели. `/process` и `/stt` отвечают **503**. Микрофон в SPA скрыт.

Код STT/TTS (`server/stt/`, `server/tts/`) остаётся для отдельного GPU-образа (`server/Dockerfile`, Compose его не собирает). Пока флаг выключен, этот контур не является способом пользоваться системой.

---

## 7. Серверная архитектура (`server/`)

*   **FastAPI:** Ручки в `server/core/app.py`. Почти все требуют сессию: cookie `ui_session` (форма `/login`) или HTTP Basic. Публичные: `GET/POST /login`, `GET /healthz`, статика логина и `/ui/assets/*`.

| Метод | Эндпоинт | Назначение и формат данных |
| :--- | :--- | :--- |
| `POST` | `/process_text_stream` | **Основной чат.** JSON `{"text":"…"}` плюс `profile`, `search_depth`, `reasoning_effort`, `mode`. SSE: `model`, `thinking`, `tool_call`, `tool_result`, `content`, `approval_required`, `done`, `error`. |
| `POST` | `/api/conversations/{id}/branches/{branch_id}/turns` | Тот же SSE на явной ветке. SPA ходит сюда, когда выбрана версия диалога. |
| `POST` | `/process_text` | Текстовый ход без SSE. При выключенном аудио — JSON `{"answer":"…"}`; при включённом — PCM. |
| `POST` | `/process_text_test` | Как `/process_text` без TTS: JSON `{"answer":"…"}` для скриптов. |
| `POST` | `/process` | WAV → PCM. При `audio_enabled=false` — 503. |
| `POST` | `/stt` | WAV → `{"text":"…"}`. При `audio_enabled=false` — 503. |
| `GET` | `/login` | HTML-форма входа. После успеха — cookie и редирект на `/ui/`. |
| `POST` | `/logout` | Отзыв сессии. |
| `GET` | `/api/me` | Текущий аккаунт. |
| `GET/POST` | `/api/conversations` | Список чатов и создание. |
| `GET/PATCH/DELETE` | `/api/conversations/{id}` | Открытие, переименование, удаление. |
| `POST` | `/api/conversations/{id}/forks` | Ветка («версия») от `checkpoint_id`. |
| `PATCH` | `/api/branches/{id}` | Переименование ветки. |
| `POST` | `/api/branches/{id}/agenda-events` | Исследовательские вопросы: добавить / закрыть / переставить. |
| `POST` | `/api/tool-approvals/{id}/resolve` | Staged: `approve` / `revise` → SSE; `cancel` → JSON. |
| `GET/POST` | `/api/card-templates` | Шаблоны карточек (JSON Schema). |
| `GET` | `/api/cards` | Сохранённые карточки. |
| `POST` | `/api/card-drafts` | Черновик. |
| `POST` | `/api/card-drafts/generate` | Заполнение карточки моделью по checkpoint (без `ask_subgraph`). |
| `POST` | `/graph_viz` | Граф accepted chains по `graph_run_id`. |
| `GET` | `/api/checkpoints/{id}/graph` | Граф checkpoint; `scope=context\|new_in_answer\|all_branches`. |
| `POST` | `/graph_explore` | Поиск по корпусу (имя / label / цитата). Cypher с клиента не принимается. Пустой `q` не сэмплит граф. |
| `POST` | `/api/graph/expand` | Соседи узла (клик в explorer). |
| `GET` | `/health` | `{status, checks}`. Без cookie или Basic — 401. |
| `GET` | `/healthz` | Публичный probe Compose: `{status: ready\|degraded}`. |
| `GET` | `/ui_config` | Флаги UI: модели, глубина, `audio` / `staged` / `cards`. Без ключей. |
| `GET` | `/ui/` | SPA (`web/dist`). Без сессии браузер уходит на `/login`. |

*   **Lifespan:** при старте проверяются `config.yaml`, промпты, пароль Neo4j (placeholder `password123` не принимается) и текущий LLM-профиль.

---

## 8. Клиентская часть (`web/`)

Продакшен-шелл на Vite + React + TypeScript: сайдбар, чат по центру, граф ответа справа. Бренд в UI — «Neo4j Assistant».

*   Обычно Node на машине не нужен: `docker compose up --build` собирает SPA внутри образа и кладёт её в `/ui/`. Локально без Docker: `cd web && npm install && npm run build`. `npm run dev` (порт 5173) проксирует на `:8000` только часть путей из `web/vite.config.ts` и **не** проксирует `/api/*` — ветки, карточки и approvals в этом режиме не работают. Для полного UI нужен собранный `/ui/` на `:8000`.
*   Логин — серверная форма `/login` (без JS). После cookie открывается SPA.
*   Чаты, ветки, checkpoint и карточки хранятся в SQLite на сервере и изолированы по аккаунтам. В браузере остаются настройки UI (глубина, режим, ключи LLM).
*   Композер: **Автоматически** vs **По этапам** (`staged_enabled`); глубина поиска; модель из `ui_profiles` (Авто + QwenCloud); reasoning effort скрыт для Авто. Микрофон только при `audio_enabled`.
*   **По этапам:** стрим останавливается на `approval_required`; пользователь правит английские SQ и подтверждает. Один поиск на ответ, по одному новому UNIT на открытый SQ.
*   **Версии диалога** — fork от checkpoint, переключение ветки, список исследовательских вопросов SQ («Вопросы»).
*   **Карточки** — шаблоны (JSON Schema), генерация из checkpoint, черновики, сохранение, вставка в ветку. Пункт сайдбара «Библиотека» / «Библиотека статей» — заглушка («В разработке»), это не корпус PDF и не библиотека карточек.
*   Граф ответа: `vis-network`, физика `forceAtlas2Based`, инспектор evidence / source / confidence, фильтры «Все данные / Новые / Другие версии».
*   Сайдбар «Граф знаний» — полноэкранный explorer: текстовый поиск, лимит 1–5000 (`POST /graph_explore`), раскрытие соседей (`POST /api/graph/expand`).

---

## 9. Установка и запуск

Канонический сценарий — CPU Docker. Neo4j в Compose нет: база должна быть доступна с хоста (из контейнера — через `host.docker.internal`). GPU-образ `server/Dockerfile` (CUDA) Compose не использует.

### Требования к системе
*   Docker и Docker Compose
*   Запущенная база Neo4j
*   Локальный Ollama с моделью эмбеддингов `embeddinggemma:300m-qat-q8_0` (retrieval)

### 9.1 Конфигурация и переменные окружения

Настройки имеют двухуровневую иерархию (Pydantic Settings):
1. **Базовые настройки:** `server/config.yaml` (флаги `audio_enabled` / `rerank_enabled` / `staged_enabled` / `cards_enabled`, `run_id`, профили LLM, порт).
2. **Секреты:** `.env` в корне репозитория, переопределяют yaml. Вложенные ключи — через двойное подчёркивание (`__`).

Скопируйте пример и заполните реальные значения:
```bash
cp .env.example .env
```

**Переменные, без которых сервис не работает как задумано:**

| Переменная | Назначение |
| :--- | :--- |
| `NEO4J__URI` | Bolt-адрес графа. Из контейнера — `bolt://host.docker.internal:7687` (в `.env.example` стоит `localhost`, это URI для процесса на хосте). |
| `NEO4J__USER` / `NEO4J__PASSWORD` | Учётная запись Neo4j. Пароль `password123` и пустой сервер отвергает при старте. |
| `APP_DB_PATH` | SQLite аккаунтов и истории; в Compose `/app/data/assistant.db`. |
| `AUTH_COOKIE_SECURE` | `true` за HTTPS, локально `false`. |
| `AUTH_SESSION_DAYS` | Срок cookie, по умолчанию 30. |
| `AUTH_TRUSTED_ORIGINS` | Дополнительные origin через запятую для reverse proxy. |
| `EMBED__BASE_URL` | Ollama embeddings. Из Docker обычно `http://host.docker.internal:11434/v1`. |
| `EMBED__MODEL` | `embeddinggemma:300m-qat-q8_0` |
| `EMBED__API_KEY` | Для локальной Ollama достаточно `ollama`. |
| `LLM__PROFILES__OLLAMA__API_KEY` | Серверный ключ Ollama Cloud (запасной Gemma). Не задаётся в UI. |
| `LLM__PROFILES__OLLAMA__BASE_URL` | `https://ollama.com/v1` |
| `LLM__PROFILES__QWEN_CLOUD__API_KEY` | Ключ DashScope / QwenCloud. Пустое поле в UI использует этот env. |
| `LLM__PROFILES__QWEN_CLOUD__BASE_URL` | `https://dashscope-intl.aliyuncs.com/compatible-mode/v1` |

> **Совет:** Активную модель чата переключают `llm.current_profile` в `server/config.yaml` или селектором в UI (`ui_profiles`). Глубина поиска и reasoning effort задаются в композере, не моделью.

`scripts/vectorize_edges.py --run-id RUN_ID` добавляет вектора всем рёбрам этого
корпуса, у которых есть непустой `evidence`, но нет `evidence_embedding`, а затем
добавляет недостающие индексы. Индекс один на каждый тип связи и общий для всех
`run_id`; имя новых индексов имеет вид
`rel_ev_v1_<slug-типа>_<sha256-первые-10>`, а `run_id` включён как filter property
через `WITH [r.run_id]`. Обычный запуск существующие индексы не удаляет.
Фильтрованный `SEARCH` и дополнительные свойства vector index требуют Neo4j
2026.01 или новее.
`--recreate-indexes --yes` — отдельная явная операция DROP/CREATE для всех
relationship-индексов `evidence_embedding`. После смены размерности сначала
нужно перевекторизовать все корпуса одной моделью и только затем пересоздать
индексы. Это хостовый скрипт, не способ стартовать приложение; на маленькой VM
остановите контейнер `app`.

### 9.2 Запуск контейнеров
Один шаг: сборка фронта (Node в образе) + бэкенд (`restart: unless-stopped`). Node.js на хосте не ставится.
```bash
docker compose up --build -d
```
`--build` нужен при первом запуске и после правок UI или сервера. Сервис `app` слушает порт `8000`. Реранкер в этот запуск не входит (`rerank_enabled: false`).

Чтобы поднять сервис `reranker` (CPU, порт 7997 внутри сети compose):
```bash
docker compose --profile rerank up -d
```
App ходит на `http://reranker:7997`. Поиск начнёт звать CE только если в yaml включить `rerank_enabled: true`.

### 9.3 Аккаунты и доступ

После первого запуска создайте аккаунт (пароль 12–128 символов):
```bash
docker compose exec app python -m server.manage_users create technologist
```
Он получит bootstrap `run_id` из `server/config.yaml`; явно задать корпус можно
через `create LOGIN --run-id RUN_ID`. Переключение существующего аккаунта:
```bash
docker compose exec app python -m server.manage_users set-run-id technologist RUN_ID
```
Новый чат фиксирует текущий `run_id`. Старые чаты другого корпуса остаются
доступны для просмотра, переименования и удаления, но не принимают новые
сообщения/ветки/изменения контента; возврат аккаунта на прежний `run_id` снова
их разблокирует. `list` показывает назначенный корпус. Доступны также
`reset-password LOGIN`, `disable LOGIN`, `enable LOGIN` и
`revoke-sessions LOGIN`. Сброс пароля и блокировка отзывают открытые сессии.

*   **Web-интерфейс:** [http://localhost:8000/ui/](http://localhost:8000/ui/) перенаправляет на `/login`. Сессия — отзывная HttpOnly cookie, пароль — Argon2id в SQLite.
*   Скрипты: `/health` без авторизации — **401**. `curl -I -u technologist:ПАРОЛЬ http://127.0.0.1:8000/health`. Compose-probe — публичный `/healthz`.
*   **Neo4j Browser (если установлен локально):** [http://localhost:7474](http://localhost:7474)

SQLite лежит в volume `assistant_data` и переживает пересоздание контейнера. Backup:
```bash
docker compose exec app python -m server.manage_db backup /tmp/assistant-backup.db
docker compose cp app:/tmp/assistant-backup.db ./assistant-backup.db
```

Restore (сначала остановить app; страховая копия создаётся автоматически):
```bash
docker compose stop app
docker compose run --rm -v "$PWD:/restore:ro" app \
  python -m server.manage_db restore /restore/assistant-backup.db --force
docker compose up -d app
```

---

## 10. Тестирование

### 10.1 Unit-тесты
```bash
python -m pytest tests -q
```

### 10.2 Регрессия промпта
Прогоняет кейсы `tests/prompt_regression/cases.json` через `/process_text_stream`
и проверяет автоматом: бюджет вызовов, язык и уникальность subquestions,
служебные утечки в ответе, наличие `### GAPS`. Пункты рубрики выводятся для проверки
глазами. Нужен запущенный сервер. Задайте аккаунт для HTTP Basic:
```bash
export ASSISTANT_USER=technologist ASSISTANT_PASSWORD=ПАРОЛЬ
python -m tests.prompt_regression.run
python -m tests.prompt_regression.run --case catalog_freshness_indicators
# отчёт → tests/reports/prompt_regression/report.md
```

### 10.3 Оценка retrieval
Нужны доступный Neo4j и JSON-датасет (`tests/qa_open_20.json` или `tests/qa_evidence_50.json`). Имя скрипта историческое:
```bash
python tests/evaluate_v6.py --effort auto --limit 1 --sq-cache
```
Отчёты пишутся в `tests/reports/v6/` (каталог каждый прогон пересоздаётся); кэш subquestions и графов — `tests/reports/v6_cache/`. Подробности стадий пайплайна — в `server/algorithm/README.md`.
