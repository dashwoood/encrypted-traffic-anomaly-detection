#!/usr/bin/env bash
#
# End-to-end demo:
#   1) Start Docker testing environment (receiver + generator)
#   2) Wait for synthetic traffic and flows.csv
#   3) Run the detection daemon on live flows and stream anomalies
#
# Usage:
#   chmod +x demo.sh
#   ./demo.sh
#
# Requirements:
#   - Docker + Docker Compose V2
#   - detection-mechanisms package installed (so the `detect` CLI is available),
#     e.g. `make install` from the project root.

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${PROJECT_ROOT}"

LOG_DIR="data/logs"
FLOWS_PATH="${LOG_DIR}/flows.csv"

echo "[demo] Ensuring log directory exists at ${LOG_DIR} ..."
mkdir -p "${LOG_DIR}"

echo "[demo] Starting testing environment (receiver + generator) via docker compose ..."
docker compose up -d

echo "[demo] Waiting for flows to appear in ${FLOWS_PATH} (this may take ~60–90s) ..."
SECONDS_WAITED=0
MAX_WAIT=120
until [ -s "${FLOWS_PATH}" ]; do
  if [ "${SECONDS_WAITED}" -ge "${MAX_WAIT}" ]; then
    echo "[demo] Timed out waiting for ${FLOWS_PATH}. Check 'docker compose logs' for issues." >&2
    exit 1
  fi
  sleep 5
  SECONDS_WAITED=$((SECONDS_WAITED + 5))
done

echo "[demo] Flows detected in ${FLOWS_PATH}."

# Resolve CLI command: prefer installed `detect`, fall back to module execution.
if command -v detect >/dev/null 2>&1; then
  DETECT_CMD=(detect)
else
  echo "[demo] 'detect' CLI not found on PATH. Falling back to 'python3 -m cli' from detection-mechanisms/." >&2
  DETECT_CMD=(python3 detection-mechanisms/cli.py)
fi

echo
echo "[demo] Starting detection daemon on live synthetic traffic."
echo "       Press Ctrl+C to stop the daemon. Containers will remain running; use 'make stop-environment' to stop them."
echo

"${DETECT_CMD[@]}" daemon \
  --flows "${FLOWS_PATH}" \
  --model isolation_forest \
  --interval 5 \
  --min-fit-flows 50 \
  --log-level INFO

