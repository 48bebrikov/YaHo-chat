# Lightweight Python image
FROM python:3.12-slim

WORKDIR /app

# System packages needed to build some Python wheels (SQLite, crypto) and ffmpeg for audio
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first so Docker can cache the dependency layer
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Chromium for browse_url_visual (Playwright PDF)
RUN python -m playwright install-deps && python -m playwright install chromium

# Preload the SentenceTransformer (deepvk) so it is not downloaded on every container start.
# Makes the image larger, but startup is faster and more reliable.
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('deepvk/USER2-base')"

COPY . .

CMD ["python", "main.py"]
