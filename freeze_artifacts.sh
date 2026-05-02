#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${PROJECT_ROOT}"

OUT_DIR="${1:-artifacts/final}"

echo "[freeze] Cleaning previous artifacts and results..."
make clean || true

echo "[freeze] Running full evaluation pipeline..."
make evaluate-all

echo "[freeze] Collecting artifacts into ${OUT_DIR} ..."
mkdir -p "${OUT_DIR}"

cp -v experiments/results/experiment_*.json "${OUT_DIR}" || true
cp -v experiments/results/statistical_comparison.json "${OUT_DIR}" || true
cp -v experiments/results/experiment_summary.json "${OUT_DIR}" || true
cp -v experiments/results/ai_benefit_summary.json "${OUT_DIR}" || true
cp -v experiments/results/cv_*.json "${OUT_DIR}" 2>/dev/null || true
cp -v experiments/results/cv_summary.json "${OUT_DIR}" 2>/dev/null || true
cp -v experiments/results/run_manifest.json "${OUT_DIR}" || true

if [ -d "experiments/results/figures" ]; then
  mkdir -p "${OUT_DIR}/figures"
  cp -v experiments/results/figures/* "${OUT_DIR}/figures" || true
fi

if ls data/datasets/dataset_report_*.json >/dev/null 2>&1; then
  mkdir -p "${OUT_DIR}/dataset-reports"
  cp -v data/datasets/dataset_report_*.json "${OUT_DIR}/dataset-reports" || true
fi

echo "[freeze] Writing artifact checklist..."
CHECKLIST="${OUT_DIR}/CHECKLIST.txt"
{
  echo "Thesis artifact bundle checklist"
  echo "Generated at: $(date -u +"%Y-%m-%dT%H:%M:%SZ")"
  echo
  echo "Commands to reproduce from clean repo:"
  echo "  make prepare-data"
  echo "  make clean && make evaluate-all"
  echo
  echo "Artifacts expected in this directory:"
  echo "  - experiment_*.json"
  echo "  - experiment_summary.json"
  echo "  - statistical_comparison.json"
  echo "  - ai_benefit_summary.json"
  echo "  - cv_*.json, cv_summary.json (if cross-validation enabled)"
  echo "  - run_manifest.json"
  echo "  - figures/ (if matplotlib available)"
  echo "  - dataset-reports/dataset_report_*.json (if prepare-data wrote them)"
} > "${CHECKLIST}"

echo "[freeze] Artifact bundle ready in ${OUT_DIR}."

