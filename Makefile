.PHONY: install install-all start-environment stop-environment prepare-data \
       train-all-models evaluate-all generate-thesis-results clean \
       env-benchmark-up env-benchmark-down demo-benchmark \
       env-smoke-http env-smoke-benchmark ops-smoke-daemon freeze-artifacts

PYTHON       ?= python3
PIP          ?= pip
DETECT_DIR   := detection-mechanisms
EXP_DIR      := experiments
DATA_DIR     := data
RESULTS_DIR  := $(EXP_DIR)/results
FIGURES_DIR  := $(RESULTS_DIR)/figures

# ------------------------------------------------------------------
# Installation
# ------------------------------------------------------------------

install:
	$(PIP) install -e $(DETECT_DIR)

install-all:
	$(PIP) install -e "$(DETECT_DIR)[all]"

# ------------------------------------------------------------------
# Testing environment (Docker)
# ------------------------------------------------------------------

start-environment:
	docker compose up -d

stop-environment:
	docker compose down

logs:
	docker compose logs -f

# Convenience aliases for environment management
env-up: start-environment

env-down: stop-environment

env-logs: logs

demo:
	chmod +x demo.sh
	./demo.sh

# ------------------------------------------------------------------
# Benchmark testing environment (Docker)
# ------------------------------------------------------------------

env-benchmark-up:
	docker compose -f docker-compose.benchmark.yml up -d

env-benchmark-down:
	docker compose -f docker-compose.benchmark.yml down

demo-benchmark:
	chmod +x demo-benchmark.sh
	./demo-benchmark.sh

# ------------------------------------------------------------------
# Environment smoke checks
# ------------------------------------------------------------------

env-smoke-http:
	@echo "[make] Running HTTP synthetic environment smoke test..."
	chmod +x demo.sh
	./demo.sh &
	DEMO_PID=$$!
	sleep 30
	test -s data/logs/flows.csv
	echo "[make] flows.csv exists and is non-empty for HTTP synthetic mode."
	kill $$DEMO_PID || true

env-smoke-benchmark:
	@echo "[make] Running benchmark replay environment smoke test..."
	chmod +x demo-benchmark.sh
	./demo-benchmark.sh &
	DEMO_BENCH_PID=$$!
	sleep 30
	test -s data/logs/flows.csv
	echo "[make] flows.csv exists and is non-empty for benchmark mode."
	kill $$DEMO_BENCH_PID || true

ops-smoke-daemon:
	@echo "[make] Running operational daemon smoke test..."
	chmod +x ops_smoke_daemon.sh
	./ops_smoke_daemon.sh

# ------------------------------------------------------------------
# Data preparation
# ------------------------------------------------------------------

prepare-data:
	cd $(DETECT_DIR) && $(PYTHON) -m datasets.prepare \
		--data-dir ../$(DATA_DIR) --skip-download 2>&1 || \
	$(PYTHON) -c "from datasets.prepare import prepare_datasets; \
		prepare_datasets('$(DATA_DIR)', download_benchmarks=False)"

# ------------------------------------------------------------------
# Training
# ------------------------------------------------------------------

train-all-models:
	cd $(DETECT_DIR) && $(PYTHON) cli.py train \
		--data-dir ../$(DATA_DIR) --model isolation_forest --skip-download
	cd $(DETECT_DIR) && $(PYTHON) cli.py train \
		--data-dir ../$(DATA_DIR) --model pca --skip-download
	cd $(DETECT_DIR) && $(PYTHON) cli.py train \
		--data-dir ../$(DATA_DIR) --model kmeans --skip-download

# ------------------------------------------------------------------
# Evaluation
# ------------------------------------------------------------------

evaluate-all:
	$(PYTHON) $(EXP_DIR)/run_experiment.py \
		--train $(DATA_DIR)/datasets/canonical_train.csv \
		--test  $(DATA_DIR)/datasets/canonical_val.csv \
		--train-ordered $(DATA_DIR)/datasets/canonical_train_ordered.csv \
		--test-ordered  $(DATA_DIR)/datasets/canonical_val_ordered.csv \
		--all \
		--output $(RESULTS_DIR) \
		--window-size 16 \
		--seed 42 \
		--cv-folds 10 \
		--run-manifest $(RESULTS_DIR)/run_manifest.json

evaluate-traditional:
	$(PYTHON) $(EXP_DIR)/run_experiment.py \
		--train $(DATA_DIR)/datasets/canonical_train.csv \
		--test  $(DATA_DIR)/datasets/canonical_val.csv \
		--traditional \
		--output $(RESULTS_DIR)

evaluate-ai:
	$(PYTHON) $(EXP_DIR)/run_experiment.py \
		--train $(DATA_DIR)/datasets/canonical_train.csv \
		--test  $(DATA_DIR)/datasets/canonical_val.csv \
		--train-ordered $(DATA_DIR)/datasets/canonical_train_ordered.csv \
		--test-ordered  $(DATA_DIR)/datasets/canonical_val_ordered.csv \
		--ai \
		--output $(RESULTS_DIR) \
		--window-size 16

# ------------------------------------------------------------------
# Full pipeline for thesis
# ------------------------------------------------------------------

generate-thesis-results: evaluate-all
	@echo "Results in $(RESULTS_DIR)/"
	@echo "Figures in $(FIGURES_DIR)/"
	@test -f $(RESULTS_DIR)/statistical_comparison.json && \
		echo "Statistical comparison available." || true

freeze-artifacts:
	@echo "[make] Freezing final thesis artifacts..."
	chmod +x freeze_artifacts.sh
	./freeze_artifacts.sh

# ------------------------------------------------------------------
# Cleanup
# ------------------------------------------------------------------

clean:
	rm -rf $(RESULTS_DIR)/experiment_*.json
	rm -rf $(RESULTS_DIR)/cv_*.json
	rm -rf $(RESULTS_DIR)/cv_summary.json
	rm -rf $(RESULTS_DIR)/experiment_summary.json
	rm -rf $(RESULTS_DIR)/figures/
	rm -rf $(RESULTS_DIR)/statistical_comparison.json
	rm -rf $(RESULTS_DIR)/run_manifest.json
	find . -name __pycache__ -type d -exec rm -rf {} + 2>/dev/null || true
