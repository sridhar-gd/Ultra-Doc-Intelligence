FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update && apt-get install -y \
    libxcb1 libx11-6 libxext6 libxrender1 libxkbcommon0 libgl1 \
    tesseract-ocr \
 && rm -rf /var/lib/apt/lists/*

COPY backend/requirements.txt /app/backend/requirements.txt
RUN pip install --no-cache-dir -r /app/backend/requirements.txt

COPY backend /app/backend
WORKDIR /app/backend

CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"]
