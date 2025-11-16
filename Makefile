# Makefile for FakeScope project automation

.PHONY: help install test coverage lint format clean docker-build docker-run

help:  ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install:  ## Install dependencies
	pip install -r requirements.txt
	pip install -e .
	python -m spacy download en_core_web_sm
	python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt')"

install-dev:  ## Install development dependencies
	pip install -r requirements.txt
	pip install pytest pytest-cov pytest-mock black flake8 isort pre-commit
	pre-commit install

test:  ## Run unit tests
	pytest tests/ -v

coverage:  ## Run tests with coverage report
	pytest tests/ -v --cov=src --cov-report=html --cov-report=term-missing
	@echo "Coverage report: htmlcov/index.html"

lint:  ## Run linting checks
	flake8 src/ tests/
	black --check src/ tests/
	isort --check-only src/ tests/

format:  ## Format code with black and isort
	black src/ tests/
	isort src/ tests/

clean:  ## Clean cache and build files
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache htmlcov .coverage coverage.xml
	rm -rf build dist *.egg-info

docker-build:  ## Build Docker image
	docker build -f Dockerfile.prod -t fakescope-api:latest .

docker-run:  ## Run Docker container
	docker run -d -p 8000:8000 --name fakescope-api fakescope-api:latest

mlflow-ui:  ## Start MLFlow UI
	mlflow ui --port 5000

train:  ## Run training pipeline
	python src/data_pipeline.py
	python scripts/train_all_models.py

deploy:  ## Deploy model (placeholder)
	@echo "Deployment would happen here"
