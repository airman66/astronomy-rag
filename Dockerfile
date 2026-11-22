FROM python:3.12-slim@sha256:5e2bdb1b0d542d5039cffb8181c65930093949194a65c365dfb1b0fe13a4fa5b

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt \
    && useradd --create-home --uid 10001 appuser \
    && install -d -o appuser -g appuser /app/data /app/state

COPY app ./app
COPY scripts ./scripts

USER appuser
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/ready', timeout=2)"]

# Reverse-proxy identity is handled only by client_identity.py and TRUSTED_PROXIES.
CMD ["sh", "-ec", "if [ -z \"$DATA_RELEASE\" ]; then python -m scripts.catalog_release prepare; fi; export CATALOG_READ_ONLY=1; exec uvicorn app.main:app --host 0.0.0.0 --port 8000 --no-proxy-headers"]
