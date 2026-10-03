FROM python:3.11-slim-bookworm

WORKDIR /app

# Install system graphics dependencies for OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Expose common web ports
EXPOSE 7860 8080 5000

# Bind dynamically to Railway's $PORT or fallback to 7860
CMD exec gunicorn --bind 0.0.0.0:${PORT:-7860} --workers 1 --threads 4 --timeout 120 app:app
