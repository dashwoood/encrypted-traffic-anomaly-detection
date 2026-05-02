FROM python:3.9-slim

RUN apt-get update && apt-get install -y --no-install-recommends tcpdump \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

ENV PYTHONUNBUFFERED=1

EXPOSE 8000

COPY server.py entrypoint.sh .
RUN mkdir -p /app/logs && chmod +x entrypoint.sh

CMD ["./entrypoint.sh"]
