FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    AGENTV4_HOST=0.0.0.0 \
    AGENTV4_PORT=8094

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src
COPY keys ./keys

RUN pip install --no-cache-dir .

RUN adduser --disabled-password --gecos "" --uid 10001 agentv4
USER agentv4

EXPOSE 8094

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8094/health', timeout=3).read()"

CMD ["sh", "-c", "uvicorn agentv4.main:app --host ${AGENTV4_HOST} --port ${AGENTV4_PORT}"]
