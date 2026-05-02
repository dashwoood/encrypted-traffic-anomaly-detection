#!/usr/bin/env python3
"""
HTTP server with comprehensive flow-level metadata for anomaly detection.
Anomaly detection for encrypted traffic operates at FLOW level (NetFlow/IPFIX):
metadata only - payload is not inspected.

Output format: flows.csv uses the canonical flow schema (reference dataset structure)
so it matches the format used for benchmark datasets (CICIDS2017, UNSW-NB15) after
normalization. See docs/canonical-flow-format.md for the full column list and types.
"""
import csv
import logging
import os
import threading
import time
from collections import defaultdict, deque
from datetime import datetime
from http.server import HTTPServer, SimpleHTTPRequestHandler
from urllib.parse import urlparse

LOG_PATH = os.environ.get("LOG_PATH", "/app/logs/requests.log")
FLOWS_PATH = os.environ.get("FLOWS_PATH", "/app/logs/flows.csv")
ROLLING_WINDOW_SEC = int(os.environ.get("ROLLING_WINDOW_SEC", "60"))

LOG_DIR = os.path.dirname(LOG_PATH)
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(os.path.dirname(FLOWS_PATH), exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[
        logging.FileHandler(LOG_PATH),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger()

# Per-client state (thread-safe)
_state_lock = threading.Lock()
_client_last_time = {}
_client_request_count = {}
_client_paths = defaultdict(set)
_client_timestamps = defaultdict(lambda: deque(maxlen=1000))

# Canonical flow schema (reference dataset structure). Order must match row in log_flow().
# See docs/canonical-flow-format.md. Same structure as normalized CICIDS2017/UNSW-NB15.
FLOW_HEADER = [
    "timestamp", "client_ip", "method", "path", "path_length", "query_length",
    "duration_ms", "response_code", "response_size", "request_content_length",
    "header_count", "header_size_bytes", "user_agent", "user_agent_length",
    "referer", "referer_present", "accept", "accept_language",
    "inter_arrival_ms", "request_sequence", "requests_last_60s",
    "unique_paths_count", "hour_utc", "minute", "day_of_week",
    "ground_truth",
]


def ensure_flow_header():
    if not os.path.exists(FLOWS_PATH) or os.path.getsize(FLOWS_PATH) == 0:
        with open(FLOWS_PATH, "w", newline="") as f:
            csv.writer(f).writerow(FLOW_HEADER)


def _compute_client_metrics(client_ip: str, path: str) -> dict:
    now = time.time()
    with _state_lock:
        last = _client_last_time.get(client_ip)
        inter_arrival_ms = (now - last) * 1000 if last is not None else -1
        _client_last_time[client_ip] = now

        _client_request_count[client_ip] = _client_request_count.get(client_ip, 0) + 1
        seq = _client_request_count[client_ip]

        _client_paths[client_ip].add(path.split("?")[0])
        unique_paths = len(_client_paths[client_ip])

        timestamps = _client_timestamps[client_ip]
        timestamps.append(now)
        cutoff = now - ROLLING_WINDOW_SEC
        while timestamps and timestamps[0] < cutoff:
            timestamps.popleft()
        requests_last_60s = len(timestamps)

    return {
        "inter_arrival_ms": inter_arrival_ms,
        "request_sequence": seq,
        "requests_last_60s": requests_last_60s,
        "unique_paths_count": unique_paths,
    }


def log_flow(
    client_ip: str,
    method: str,
    path: str,
    duration_ms: float,
    code: int,
    response_size: int,
    headers: "http.client.HTTPMessage",
    content_length: int,
    client_metrics: dict,
):
    parsed = urlparse(path)
    path_part = parsed.path or "/"
    query_part = parsed.query
    path_length = len(path_part)
    query_length = len(query_part)

    header_count = len(headers)
    header_size = sum(len(k) + len(v) + 4 for k, v in headers.items())

    user_agent = (headers.get("User-Agent") or "").strip()[:200]
    user_agent_length = len(user_agent)
    referer = (headers.get("Referer") or "").strip()[:200]
    referer_present = 1 if referer else 0
    accept = (headers.get("Accept") or "").strip()[:100]
    accept_language = (headers.get("Accept-Language") or "").strip()[:50]
    ground_truth = 1 if (headers.get("X-Test-Label") or "").strip().lower() == "anomaly" else 0

    ts = datetime.utcnow()
    hour_utc = ts.hour
    minute = ts.minute
    day_of_week = ts.weekday()
    timestamp = ts.isoformat() + "Z"

    # Row order must match FLOW_HEADER (canonical flow schema)
    inter_arrival = client_metrics["inter_arrival_ms"]
    row = [
        timestamp,
        client_ip,
        method,
        path_part,
        path_length,
        query_length,
        f"{duration_ms:.2f}",
        code,
        response_size,
        content_length,
        header_count,
        header_size,
        user_agent,
        user_agent_length,
        referer,
        referer_present,
        accept,
        accept_language,
        f"{inter_arrival:.2f}" if inter_arrival >= 0 else "",
        client_metrics["request_sequence"],
        client_metrics["requests_last_60s"],
        client_metrics["unique_paths_count"],
        hour_utc,
        minute,
        day_of_week,
        ground_truth,
    ]
    with open(FLOWS_PATH, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
        w.writerow(row)

    logger.info(
        f"{timestamp} | {method} {path_part} | {client_ip} | {code} | "
        f"{duration_ms:.1f}ms | seq={client_metrics['request_sequence']}"
    )


class LoggingHandler(SimpleHTTPRequestHandler):
    def log_message(self, format, *args):
        pass

    def _wrap_handler(self, handler):
        client_ip = self.client_address[0]
        parsed = urlparse(self.path)
        path_part = parsed.path or "/"
        content_length = int(self.headers.get("Content-Length", 0) or 0)

        client_metrics = _compute_client_metrics(client_ip, self.path)

        start = time.perf_counter()
        code = 200
        try:
            handler()
        except Exception:
            code = 500
            raise
        finally:
            duration_ms = (time.perf_counter() - start) * 1000
            response_size = -1  # SimpleHTTPRequestHandler does not expose this easily
            log_flow(
                client_ip,
                self.command,
                self.path,
                duration_ms,
                code,
                response_size,
                self.headers,
                content_length,
                client_metrics,
            )

    def do_GET(self):
        self._wrap_handler(lambda: SimpleHTTPRequestHandler.do_GET(self))

    def do_POST(self):
        self._wrap_handler(lambda: SimpleHTTPRequestHandler.do_POST(self))


if __name__ == "__main__":
    ensure_flow_header()
    port = int(os.environ.get("RECEIVER_PORT", "8000"))
    logger.info(
        f"Receiver on 0.0.0.0:{port} | flows→{FLOWS_PATH} | window={ROLLING_WINDOW_SEC}s"
    )
    with HTTPServer(("0.0.0.0", port), LoggingHandler) as httpd:
        httpd.serve_forever()
