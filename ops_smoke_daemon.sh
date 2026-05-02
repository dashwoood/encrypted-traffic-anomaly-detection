#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${PROJECT_ROOT}"

LOG_DIR="data/logs"
FLOWS_PATH="${LOG_DIR}/flows.csv"
DAEMON_LOG="${LOG_DIR}/daemon_structured.log"

echo "[ops-smoke] Ensuring log directory exists at ${LOG_DIR} ..."
mkdir -p "${LOG_DIR}"

if [ ! -s "${FLOWS_PATH}" ]; then
  echo "[ops-smoke] flows.csv is empty or missing. Run 'make env-smoke-http' or a demo first." >&2
  exit 1
fi

echo "[ops-smoke] Starting daemon with structured logging to ${DAEMON_LOG} ..."
detect daemon \
  --flows "${FLOWS_PATH}" \
  --model isolation_forest \
  --interval 2 \
  --min-fit-flows 10 \
  --output "${DAEMON_LOG}" \
  --log-level INFO &
DAEMON_PID=$!

sleep 10

echo "[ops-smoke] Stopping daemon (pid=${DAEMON_PID}) ..."
kill "${DAEMON_PID}" || true
sleep 2

if [ ! -s "${DAEMON_LOG}" ]; then
  echo "[ops-smoke] Daemon log ${DAEMON_LOG} is empty – something went wrong." >&2
  exit 1
fi

echo "[ops-smoke] Daemon log populated. Ops smoke test passed."

