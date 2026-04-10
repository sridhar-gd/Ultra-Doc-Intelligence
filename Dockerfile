FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
# Avoid GUI backends trying to open a display (Docling / OCR stacks).
ENV QT_QPA_PLATFORM=offscreen
ENV OPENCV_IO_MAX_IMAGE_PIXELS=2147483647

WORKDIR /app

# Native runtime dependencies for Docling / RapidOCR / OpenCV-style stacks on Linux.
# libxcb.so.1 comes from libxcb1; extra XCB/X11 libs avoid missing .so at runtime.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libxcb1 \
    libxcb-xfixes0 \
    libxcb-render0 \
    libxcb-shm0 \
    libxcb-shape0 \
    libx11-6 \
    libx11-xcb1 \
    libxext6 \
    libxrender1 \
    libxkbcommon0 \
    libsm6 \
    libglib2.0-0 \
    libgomp1 \
    libgl1 \
    tesseract-ocr \
    tesseract-ocr-eng \
 && rm -rf /var/lib/apt/lists/* \
 && ldconfig \
 && (test -f /usr/lib/x86_64-linux-gnu/libxcb.so.1 || test -f /usr/lib/aarch64-linux-gnu/libxcb.so.1) \
 && echo "libxcb.so.1 present"

COPY backend/requirements.txt /app/backend/requirements.txt
RUN pip install --no-cache-dir -r /app/backend/requirements.txt

COPY backend /app/backend

WORKDIR /app/backend

# Use sh so "cd" is never the container argv[0] (matches railway.toml deploy.startCommand).
CMD ["sh", "-c", "exec uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000} --proxy-headers --forwarded-allow-ips='*'"]
