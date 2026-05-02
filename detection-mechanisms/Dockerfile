# Detector service: flow-anomaly-detector CLI for daemon mode
FROM python:3.11-slim

WORKDIR /app
ENV PYTHONUNBUFFERED=1

COPY . .
RUN pip install --no-cache-dir -e ".[viz]"

ENTRYPOINT ["detect"]
