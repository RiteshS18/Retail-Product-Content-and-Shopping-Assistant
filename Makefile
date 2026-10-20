# Makefile for Retail Shopping Assistant
# Usage: make <target>
# Compatible with GNU Make on Linux/macOS. On Windows use 'python -m ...' directly.

.PHONY: help setup data train-index train-classifier train-generator eval serve test lint clean

PYTHON ?= python
PIP    ?= pip

help:
	@echo "Retail Shopping Assistant – available targets:"
	@echo "  setup            Install dependencies (pip install -r requirements.txt -e .)"
	@echo "  data             Download dataset and generate splits"
	@echo "  train-index      Build FAISS semantic search index"
	@echo "  train-classifier Fine-tune ResNet18 image classifier"
	@echo "  train-efficientnet Fine-tune EfficientNet-B0 image classifier"
	@echo "  train-generator  Fine-tune Flan-T5 content generator"
	@echo "  train            Run all training steps"
	@echo "  eval             Run evaluation notebook (papermill)"
	@echo "  serve            Start API + Streamlit (two background processes)"
	@echo "  test             Run pytest"
	@echo "  lint             Run ruff linter"
	@echo "  clean            Remove generated artefacts"

setup:
	$(PIP) install -r requirements.txt
	$(PIP) install -e .

data:
	$(PYTHON) scripts/prepare_dataset.py

train-index:
	$(PYTHON) -m train.build_index

train-classifier:
	$(PYTHON) -m train.train_classifier --backbone resnet18

train-efficientnet:
	$(PYTHON) -m train.train_classifier --backbone efficientnet_b0

train-generator:
	$(PYTHON) -m train.finetune_content_gen

train: train-index train-classifier train-generator

eval:
	$(PYTHON) -m jupyter nbconvert --to notebook --execute notebooks/evaluation.ipynb \
		--output notebooks/evaluation_executed.ipynb

serve:
	@echo "Starting API on http://127.0.0.1:8000 ..."
	start uvicorn retail_assistant.api:app --reload --app-dir src
	@echo "Starting Streamlit on http://localhost:8501 ..."
	streamlit run app/streamlit_app.py

test:
	$(PYTHON) -m pytest tests/ -v --tb=short

lint:
	$(PYTHON) -m ruff check src/ train/ scripts/ tests/

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
	rm -rf models_out/index models_out/*.pt models_out/content_gen
	rm -rf reports/figures reports/*.json reports/*.md
