.PHONY: dev fmt test help
PY ?= python

help:
	@echo "make dev   - install project deps from requirements.txt"
	@echo "make fmt   - autoformat code with black + isort"
	@echo "make test  - run pytest"

dev:
	$(PY) -m pip install --upgrade pip
	$(PY) -m pip install -r requirements.txt

fmt:
	black encoders tests scripts
	isort encoders tests scripts

test:
	pytest -q
