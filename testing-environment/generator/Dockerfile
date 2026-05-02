FROM python:3.9-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1

# Minimal script: sends periodic HTTP requests to receiver (simulates traffic)
COPY ping.py .

CMD ["python", "ping.py"]
