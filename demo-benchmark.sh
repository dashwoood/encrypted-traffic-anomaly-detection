#!/usr/bin/env bash
#
# Benchmark-mode demo: replay canonical flows (UNSW-NB15 style) into flows.csv
# and run the detection daemon. Data is in-distribution with trained models.
#
# Usage:
#   chmod +x demo-benchmark.sh
#   ./demo-benchmark.sh
#
# Requirements:
#   - Docker + Docker Compose V2
#   - data/datasets/canonical_val.csv (run 'make prepare-data' first)
#   - detection-mechanisms package installed for local CLI fallback

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${PROJECT_ROOT}"

LOG_DIR="data/logs"
FLOWS_PATH="${LOG_DIR}/flows.csv"
DATASET="${1:-data/datasets/canonical_val.csv}"

if [ ! -f "${DATASET}" ]; then
  echo "[demo-benchmark] Dataset not found: ${DATASET}" >&2
  echo "  Run 'make prepare-data' first, or pass a path to a canonical CSV as first argument." >&2
  exit 1
fi

echo "[demo-benchmark] Ensuring log directory exists at ${LOG_DIR} ..."
mkdir -p "${LOG_DIR}"

echo "[demo-benchmark] Clearing previous flows.csv ..."
: > "${FLOWS_PATH}"

echo "[demo-benchmark] Starting benchmark environment (flow-replayer + detector) ..."
docker compose -f docker-compose.benchmark.yml up -d

echo "[demo-benchmark] Flow replayer is streaming from ${DATASET} to ${FLOWS_PATH}"
echo "[demo-benchmark] Detector is polling flows.csv. Wait ~30s for enough flows, then check:"
echo "  docker compose -f docker-compose.benchmark.yml logs -f detector"
echo ""
echo "To run the daemon locally instead (see live output):"
echo "  detect daemon --flows ${FLOWS_PATH} --model cnn --model-dir data/models --interval 3"
echo ""
echo "To stop: docker compose -f docker-compose.benchmark.yml down"
