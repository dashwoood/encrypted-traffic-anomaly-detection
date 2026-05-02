#!/usr/bin/env python3
"""
Continuously generate HTTP traffic to receiver with varied metadata.

The receiver captures each request as a flow row in flows.csv using the canonical
flow schema (same structure as reference datasets: CICIDS2017/UNSW-NB15). See
docs/canonical-flow-format.md.

Anomaly handling:
- Every request marked as anomaly carries header X-Test-Label: anomaly (ground truth).
- If SIMULATE_ATTACKS=1, anomaly requests also use attack-like behaviour so that
  flow-level metadata (paths, timing, headers) differs from normal traffic. That
  allows detectors to learn from behaviour, not just from a magic header.
"""
import os
import random
import time
import urllib.request

RECEIVER_HOST = os.environ.get("RECEIVER_HOST", "receiver")
RECEIVER_PORT = os.environ.get("RECEIVER_PORT", "8000")
INTERVAL_SEC = float(os.environ.get("INTERVAL_SEC", "2"))
ANOMALY_RATE = float(os.environ.get("ANOMALY_RATE", "0.1"))  # Fraction of requests marked as anomaly (ground truth)
SIMULATE_ATTACKS = os.environ.get("SIMULATE_ATTACKS", "1").strip().lower() in ("1", "true", "yes")
BASE_URL = f"http://{RECEIVER_HOST}:{RECEIVER_PORT}"

# Normal traffic
PATHS = ["/", "/api/status", "/api/health", "/metrics", "/favicon.ico"]
QUERIES = ["", "?v=1", "?id=123", "?debug=false", "?limit=10&offset=0"]

USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/120.0.0.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) Safari/605.1",
    "python-requests/2.31.0",
    "curl/7.88.1",
    "Monitoring-Bot/1.0",
]

# Attack-like: scanners, exploit tools, bot signatures (flow metadata differs)
USER_AGENTS_ATTACK = [
    "sqlmap/1.0",
    "Nmap Scripting Engine",
    "masscan/1.0",
    "ZmEu",
    "Python-urllib/3.10",
    "Nikto/2.1.6",
    "Go-http-client/1.1",
]

# Paths that mimic probing / path traversal (long path, many segments)
PATHS_ATTACK = [
    "/admin",
    "/.env",
    "/wp-admin/wp-login.php",
    "/../../../etc/passwd",
    "/api/v1/users/1/admin",
    "/.git/config",
    "/console",
    "/actuator/health",
    "/?id=1' OR '1'='1",
]

ACCEPT = ["*/*", "application/json", "text/html,application/xhtml+xml"]
ACCEPT_LANG = ["en-US,en;q=0.9", "cs,en;q=0.9", ""]

print(
    f"Generator: {BASE_URL} every {INTERVAL_SEC}s | anomaly_rate={ANOMALY_RATE} | simulate_attacks={SIMULATE_ATTACKS}"
)


def send_request(url: str, headers: dict) -> None:
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=5) as r:
            print(f"OK {r.status} {url}")
    except Exception as e:
        print(f"ERR {url}: {e}")


def next_interval(is_anomaly: bool) -> float:
    """Return sleep time in seconds. Attack bursts use shorter delay."""
    if is_anomaly and SIMULATE_ATTACKS and random.random() < 0.5:
        return random.uniform(0.05, 0.3)  # burst: many requests in short time
    return INTERVAL_SEC


def build_headers_and_path(is_anomaly: bool):
    """Return (headers, path, query) for one request. Anomaly requests may use attack-like path/headers."""
    path = random.choice(PATHS)
    query = random.choice(QUERIES) if random.random() < 0.3 else ""
    headers = {
        "User-Agent": random.choice(USER_AGENTS),
        "Accept": random.choice(ACCEPT),
        "Accept-Language": random.choice(ACCEPT_LANG),
        "Referer": BASE_URL + "/" if random.random() < 0.6 else "",
    }
    if is_anomaly:
        headers["X-Test-Label"] = "anomaly"
        if SIMULATE_ATTACKS:
            headers["User-Agent"] = random.choice(USER_AGENTS_ATTACK)
            if random.random() < 0.7:
                path = random.choice(PATHS_ATTACK)
                query = "?id=1" if random.random() < 0.5 else ""
    url = BASE_URL + path + query
    return headers, path, query


while True:
    is_anomaly = random.random() < ANOMALY_RATE
    headers, path, query = build_headers_and_path(is_anomaly)
    url = BASE_URL + path + query
    send_request(url, headers)
    time.sleep(next_interval(is_anomaly))
