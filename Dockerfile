# Public deployment image: the API serves the website and keeps uploads in memory only.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PORT=8000
WORKDIR /app

COPY backend/requirements.txt backend/requirements.txt
RUN pip install --no-cache-dir -r backend/requirements.txt

COPY backend/main.py backend/main.py
COPY backend/core backend/core
COPY frontend frontend

# Run as an unprivileged user
RUN useradd --create-home sdap
USER sdap

WORKDIR /app/backend
EXPOSE 8000
# One worker: datasets live in this process's memory, so every request must reach the same process
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT} --workers 1 --proxy-headers --forwarded-allow-ips='*'"]
