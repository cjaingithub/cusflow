# CusFlow Makefile
# Production-Ready Learning-to-Rank Recommendation System

.PHONY: help install dev test lint format clean docker-build docker-up docker-down \
        generate-data train evaluate ab-sim serve load-features

# Default target
help:
	@echo "CusFlow - Learning-to-Rank Recommendation System"
	@echo ""
	@echo "Setup Commands:"
	@echo "  make install        Install production dependencies"
	@echo "  make dev            Install development dependencies"
	@echo ""
	@echo "Development Commands:"
	@echo "  make test           Run tests"
	@echo "  make lint           Run linter"
	@echo "  make format         Format code"
	@echo "  make clean          Clean generated files"
	@echo ""
	@echo "Data & Training Commands:"
	@echo "  make generate-data  Generate synthetic data"
	@echo "  make train          Train the ranking model"
	@echo "  make evaluate       Evaluate the model"
	@echo "  make ab-sim         Run A/B simulation"
	@echo ""
	@echo "Service Commands:"
	@echo "  make serve          Start API server"
	@echo "  make load-features  Load features into Redis"
	@echo ""
	@echo "Docker Commands:"
	@echo "  make docker-build   Build Docker images"
	@echo "  make docker-up      Start all services"
	@echo "  make docker-down    Stop all services"
	@echo "  make docker-dev     Start development environment"

# ============================================================================
# Setup
# ============================================================================

install:
	pip install -e .

dev:
	pip install -e ".[dev]"
	pre-commit install

# ============================================================================
# Development
# ============================================================================

test:
	pytest tests/ -v --cov=src --cov-report=term-missing

lint:
	ruff check src/ tests/ scripts/
	mypy src/ --ignore-missing-imports

format:
	ruff format src/ tests/ scripts/
	ruff check --fix src/ tests/ scripts/

clean:
	rm -rf __pycache__ .pytest_cache .mypy_cache .ruff_cache
	rm -rf *.egg-info build dist
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true

# ============================================================================
# Data & Training
# ============================================================================

generate-data:
	python -m src.cli generate-data --domain hotel --items 1000 --users 500 --events 10000

generate-data-wealth:
	python -m src.cli generate-data --domain wealth_report --items 500 --users 200 --events 5000

generate-data-ecommerce:
	python -m src.cli generate-data --domain ecommerce --items 2000 --users 1000 --events 20000

train:
	python scripts/train_model.py --domain hotel --estimators 500

train-with-bias:
	python scripts/train_model.py --domain hotel --estimators 500 --bias-correction

evaluate:
	python scripts/evaluate.py --bootstrap

ablation:
	python scripts/evaluate.py --ablation

ab-sim:
	python scripts/run_ab_sim.py --control popularity --treatment models/lambdamart_v1.joblib

# ============================================================================
# Services
# ============================================================================

serve:
	python -m src.cli serve --host 0.0.0.0 --port 8000

serve-dev:
	python -m src.cli serve --host 0.0.0.0 --port 8000 --reload

load-features:
	python scripts/load_features.py --generate --embeddings

demo:
	python scripts/demo_api.py

# ============================================================================
# Docker
# ============================================================================

docker-build:
	docker compose build

docker-up:
	docker compose up -d api redis

docker-down:
	docker compose down

docker-dev:
	docker compose --profile dev up -d

docker-logs:
	docker compose logs -f api

docker-redis:
	docker run -d --name cusflow-redis -p 6379:6379 redis:7-alpine

# ============================================================================
# Full Pipeline
# ============================================================================

pipeline: generate-data train evaluate
	@echo "✅ Full pipeline complete!"

quick-start: docker-redis generate-data load-features train serve
	@echo "✅ Quick start complete!"

# ============================================================================
# Notebooks
# ============================================================================

jupyter:
	jupyter notebook --notebook-dir=notebooks/

# ============================================================================
# Documentation
# ============================================================================

docs:
	@echo "Documentation available in docs/ folder"
	@echo "  - architecture.md: System architecture"
	@echo "  - technical_guide.md: Technical details"
	@echo "  - quick_reference.md: Quick reference"
