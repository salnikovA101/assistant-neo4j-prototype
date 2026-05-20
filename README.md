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
│   │   ├── app.py             # FastAPI endpoints
│   │   └── pipeline.py        # STT → LLM → TTS pipeline
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
├── client/                    # Клиентская часть (Push-to-Talk)
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

### 4. Запуск клиента

```bash
pip install -r client/requirements.txt
python -m client.main
```

Удерживай `Right Ctrl` для голосового ввода, или печатай текст в консоли.

## API Endpoints

| Метод | Путь | Описание |
|---|---|---|
| `POST` | `/process` | Аудио WAV → стрим PCM (полный pipeline) |
| `POST` | `/process_text` | Текст JSON → стрим PCM (без STT) |
| `POST` | `/process_text_test` | Текст JSON → JSON ответ (без TTS, для тестов) |
| `GET` | `/health` | Проверка готовности сервера |

## Конфигурация

- `server/config.yaml` — параметры сервера, STT, TTS, LLM
- `client/config.yaml` — параметры клиента
- `.env` — секреты (API-ключи, пароли)
