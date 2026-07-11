FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Copy dependency manifest first so Docker can cache the pip install layer
# and skip it on rebuilds that only change application code.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY final/ ./final/

RUN useradd --create-home appuser && chown -R appuser:appuser /app
USER appuser

WORKDIR /app/final

EXPOSE 8000

CMD gunicorn --bind 0.0.0.0:${PORT:-8000} --workers 2 --timeout 60 app:app
