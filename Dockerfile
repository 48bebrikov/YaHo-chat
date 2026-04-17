# Используем легковесный образ Python
FROM python:3.12-slim

# Устанавливаем рабочую директорию в контейнере
WORKDIR /app

# Устанавливаем системные зависимости, необходимые для сборки некоторых Python пакетов (например, SQLite, crypto)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Копируем файл с зависимостями (сначала только его, чтобы использовать кэш слоев Docker)
COPY requirements.txt .

# Устанавливаем Python зависимости
RUN pip install --no-cache-dir -r requirements.txt

# Chromium for browse_url_visual (Playwright PDF)
RUN python -m playwright install-deps && python -m playwright install chromium

# Предзагружаем модель SentenceTransformer (deepvk), чтобы она не скачивалась каждый раз при запуске контейнера
# Это делает образ тяжелее, но запуск быстрее и стабильнее
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('deepvk/USER2-base')"

# Копируем остальной код проекта в контейнер
COPY . .

# Команда для запуска приложения
CMD ["python", "main.py"]
