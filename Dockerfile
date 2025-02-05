# Используем многоэтапную сборку для уменьшения размера финального образа
FROM python:3.12-slim AS builder

# Устанавливаем системные зависимости
RUN apt-get update && apt-get install -y \
    build-essential \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Устанавливаем переменные окружения для кэширования
ENV TRANSFORMERS_CACHE=/app/cache/transformers
ENV TORCH_HOME=/app/cache/torch

# Копируем и устанавливаем Python-зависимости
WORKDIR /app

# Копируем только requirements.txt для кэширования зависимостей
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Предзагрузка моделей
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/paraphrase-multilingual-mpnet-base-v2')"
RUN python -c "import torch; model, example_text = torch.hub.load(repo_or_dir='snakers4/silero-models', model='silero_tts', language='ru', speaker='v4_ru', trust_repo=True)"

# Финальный этап
FROM python:3.12-slim

# Устанавливаем рабочую директорию
WORKDIR /app

# Копируем кэш моделей из builder
COPY --from=builder /app/cache /root/.cache

# Устанавливаем переменные окружения для кэширования
ENV TRANSFORMERS_CACHE=/root/.cache/transformers
ENV TORCH_HOME=/root/.cache/torch

# Копируем только requirements.txt для установки зависимостей
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Копируем остальные файлы проекта
COPY . .

# Команда для запуска бота
CMD ["python", "models/telegram_bot.py"]