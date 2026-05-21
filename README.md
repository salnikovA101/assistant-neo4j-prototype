# 🧬 Neo4j Voice Assistant — Prototype

Голосовой ИИ-ассистент для работы с графовой базой знаний Neo4j.
Задавай вопросы голосом или текстом — ассистент обращается к knowledge graph,
формирует ответ с цитатами и провенансом, и озвучивает его.

## Архитектура

```
┌─────────────┐    HTTP/PCM    ┌──────────────────────────────────────┐
│   Client    │ ◄────────────► │             Server (FastAPI)          │
│             │                │                                      │
│ • Recorder  │                │  ┌─────┐   ┌─────┐   ┌─────────┐   │
│ • Player    │                │  │ STT │──►│ LLM │──►│   TTS   │   │
│ • PTT keys  │                │  └─────┘   └──┬──┘   └─────────┘   │
│             │                │               │                     │
│             │                │          ┌────┴────┐                │
│             │                │          │ GraphQA │                │
│             │                │          │ (Neo4j) │                │
│             │                │          └─────────┘                │
└─────────────┘                └──────────────────────────────────────┘
```

**Pipeline:** Аудио → STT (Whisper) → LLM (Gemini/OpenRouter) → Tool Calls (Neo4j Cypher) → TTS → Аудио

## Структура проекта

```
neo4j-prototype/
├── server/                    # Серверная часть (FastAPI)
│   ├── core/                  # Приложение и pipeline
│   │   ├── app.py             # FastAPI endpoints (с /stt, /process_text и др.)
│   │   └── pipeline.py        # STT → LLM → TTS pipeline
│   ├── static/                # Веб-интерфейс (HTML, CSS, JS)
│   │   ├── index.html         # Структура веб-интерфейса
│   │   ├── style.css          # Стилизация (микро-анимации, премиальная темная тема)
│   │   └── app.js             # Логика клиента (запись, воспроизведение, Barge-in, оптимизированный STT)
│   ├── llm/                   # Работа с языковыми моделями
│   │   ├── base.py            # Базовый провайдер (OpenAI-совместимый)
│   │   ├── manager.py         # Менеджер LLM + история + инструменты
│   │   ├── history_manager.py # Управление контекстом диалога
│   │   ├── prompt_loader.py   # Загрузка промптов из .md файлов
│   │   └── providers/         # Конкретные провайдеры LLM
│   ├── stt/                   # Speech-to-Text (Faster Whisper)
│   ├── tts/                   # Text-to-Speech (Silero / Qwen / Cloud)
│   │   ├── base.py            # Абстрактный интерфейс TTS
│   │   ├── manager.py         # Менеджер с горячим переключением
│   │   └── providers/         # speed (Silero), quality (Qwen), cloud (OpenRouter)
│   ├── tools/                 # Инструменты для LLM
│   │   ├── graph_qa.py        # Text-to-Cypher pipeline с retry
│   │   └── registry.py        # Реестр инструментов (OpenAI function calling)
│   └── utils/                 # Конфигурация, константы, трейсинг
│
├── client/                    # Консольный клиент (Push-to-Talk)
│   ├── main.py                # Основной цикл: PTT + текстовый ввод
│   ├── recorder.py            # Захват аудио с микрофона
│   ├── player.py              # Стриминговое воспроизведение PCM
│   └── audio_cues.py          # Звуковые сигналы PTT
│
├── prompts/                   # Системные промпты (редактируемые)
│   ├── logic.md               # Логика поведения ассистента
│   ├── output_quality.md      # Формат вывода для качественного TTS
│   ├── output_speed.md        # Формат вывода для Silero TTS
│   └── graph_qa/              # Промпты для генерации Cypher
│
├── docker-compose.yml         # Оркестрация: app + Phoenix (трейсинг)
├── .env.example               # Шаблон переменных окружения
└── voices/                    # Референсные аудио для voice cloning
```

## Быстрый старт

### Требования

- Docker + Docker Compose
- NVIDIA GPU + NVIDIA Container Toolkit (для STT/TTS моделей)
- Neo4j база данных (доступ по Bolt)

> [!IMPORTANT]
> **Конфигурация CUDA и видеокарты:**
> По умолчанию в [Dockerfile](file:///c:/neo4j-prototype/server/Dockerfile) прописана сборка под **CUDA 12.8** (образ `nvidia/cuda:12.8.1` и PyTorch с индексом `cu128`).
> Если ваша видеокарта или установленная версия драйверов требуют другую версию CUDA (например, CUDA 12.1, 11.8) или вы хотите запустить проект **только на CPU**, обязательно скорректируйте следующие строки в [Dockerfile](file:///c:/neo4j-prototype/server/Dockerfile):
> 1. Базовый образ: `FROM nvidia/cuda:<версия>-cudnn-runtime-ubuntu22.04` (строка 1).
> 2. Индекс PyTorch: `--index-url https://download.pytorch.org/whl/...` (строка 11) в соответствии с официальным сайтом [PyTorch](https://pytorch.org/).
> 3. В случае CPU-only сборки смените базовый образ на обычный `python:3.11-slim` и ставьте стандартный `torch` без CUDA-индексов.

### 1. Настройка окружения

```bash
cp .env.example .env
# Заполни .env реальными API-ключами и адресом Neo4j
```

### 2. Запуск сервера

```bash
docker compose up --build
```

Сервер будет доступен на `http://localhost:8000`.
Phoenix UI (трейсинг) — `http://localhost:6006`.

### 3. Проверка готовности

```bash
curl http://localhost:8000/health
# {"status": "ready"}
```

### 4. Веб-интерфейс (Web UI)

Интерактивный веб-клиент доступен по адресу:
👉 **[http://localhost:8000/ui/](http://localhost:8000/ui/)**

**Функциональность сайта:**
* **Голосовой ввод:** Нажми на иконку микрофона для начала/остановки записи.
* **Текстовый ввод:** Ввод вопроса через классическое текстовое поле ввода.
* **Мгновенный STT:** Текст вашего голосового запроса распознается и выводится на экран практически мгновенно благодаря разделению фазы транскрибации и фазы генерации/озвучивания.
* **Потоковый звук:** Ответ ассистента озвучивается в реальном времени порциями (AudioContext API).
* **Barge-in (Прерывание):** Нажатие на кнопку записи или паузы во время проигрывания ответа мгновенно останавливает воспроизведение и отменяет текущие запросы на сервере.

### 5. Консольный клиент (Push-to-Talk)

При желании можно запустить локальный python-клиент для терминала:

```bash
pip install -r client/requirements.txt
python -m client.main
```

Удерживай `Right Ctrl` для голосового ввода, или печатай текст в консоли.

### 6. Запуск тестов

```bash
python -m tests.test_runner
```

## API Endpoints

| Метод | Путь | Описание |
|---|---|---|
| `POST` | `/stt` | Аудио WAV → JSON с распознанным текстом. Быстрая транскрибация. |
| `POST` | `/process_text` | Текст JSON → стрим PCM с ответом (заголовок `LLM-Response` содержит текст ответа). |
| `POST` | `/process` | Аудио WAV → стрим PCM (полный монолитный pipeline: STT + LLM + TTS). Оставлен для обратной совместимости. |
| `POST` | `/process_text_test` | Текст JSON → JSON ответ (без TTS, для тестов). |
| `GET` | `/health` | Проверка готовности сервера. |
| `GET` | `/ui` | Статический эндпоинт, раздающий Web UI. |

## Конфигурация

- `server/config.yaml` — параметры сервера, STT, TTS, LLM
- `client/config.yaml` — параметры клиента
- `.env` — секреты (API-ключи, пароли)
