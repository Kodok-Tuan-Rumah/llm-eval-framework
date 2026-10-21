.PHONY: help install test test-cov lint eval eval-clean eval-distractor eval-paraphrase report render-analysis verify clean clean-artifacts

MODEL ?= mistral
PYTHON ?= python3
REPORTS_DIR := reports
RESULTS_DIR := reports/results

help:  ## Tampilkan help
	@echo "LLM Eval Framework - Make targets"
	@echo "------------------------------------"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort

install:  ## Install dependencies
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r requirements.txt

test:  ## Run tests
	$(PYTHON) -m pytest tests/ -v

test-cov:  ## Run tests with coverage
	$(PYTHON) -m pytest tests/ --cov=. --cov-report=term-missing

lint:  ## Static check & syntax compilation
	$(PYTHON) -m py_compile *.py evaluators/*.py pipeline/*.py reports/*.py scripts/*.py tests/*.py
	@if command -v ruff >/dev/null 2>&1; then ruff check .; fi
	@if command -v black >/dev/null 2>&1; then black --check .; fi

eval-clean:  ## Evaluasi faithfulness - clean dataset
	$(PYTHON) -m pipeline.runner --model $(MODEL) --dataset clean

eval-distractor:  ## Evaluasi faithfulness - distractor dataset
	$(PYTHON) -m pipeline.runner --model $(MODEL) --dataset distractor

eval-paraphrase:  ## Evaluasi consistency - paraphrase dataset
	$(PYTHON) -m pipeline.runner --model $(MODEL) --dataset paraphrase

eval: eval-clean eval-distractor  ## Jalankan clean + distractor (paraphrase terpisah)

report:  ## Generate charts + HTML report
	@if [ -z "$$(find $(RESULTS_DIR) -name "*_faithfulness.json" 2>/dev/null | head -1)" ]; then \
		echo "[ERROR] Tidak ada artefak faithfulness di $(RESULTS_DIR)."; \
		echo "Jalankan: make eval"; \
		exit 1; \
	fi
	$(PYTHON) reports/visualize.py --chart all
	$(PYTHON) reports/generate_report.py --embed-charts

render-analysis:  ## Regenerate numeric tables di analysis.md dari artefak
	$(PYTHON) reports/render_analysis.py

verify:  ## Verifikasi reproducibility (provenance + angka README vs artefak)
	$(PYTHON) scripts/check_reproducibility.py

clean-artifacts:  ## Hapus checkpoints & log (hindari hapus results report.html charts)
	rm -rf logs/checkpoints/* logs/*.log 2>/dev/null || true
