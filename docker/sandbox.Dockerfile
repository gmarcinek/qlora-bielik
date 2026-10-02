FROM python:3.11-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
    poppler-utils tesseract-ocr tesseract-ocr-pol tesseract-ocr-eng \
    libreoffice-writer-nogui libreoffice-impress-nogui libreoffice-calc-nogui \
    ripgrep file jq less procps \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONPATH=/app
COPY sandbox/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt
COPY sandbox /app/sandbox

RUN useradd --create-home --uid 1000 sandbox && mkdir -p /sessions /shared && chown sandbox /sessions
USER sandbox

EXPOSE 8080
CMD ["uvicorn", "sandbox.server:app", "--host", "0.0.0.0", "--port", "8080"]
