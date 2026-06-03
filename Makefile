PYTHON = venv/bin/python
PIP    = venv/bin/python -m pip

.PHONY: help setup install run clean freeze lint venv

help:
	@echo "Usage: make <target>"
	@echo ""
	@echo "  setup    Create virtual environment and install dependencies"
	@echo "  install  Install dependencies into existing venv"
	@echo "  run      Run the app"
	@echo "  freeze   Update requirements.txt from current venv"
	@echo "  lint     Run basic syntax check on all Python files"
	@echo "  clean    Remove venv and __pycache__"

setup: venv install

venv:
	python3.13 -m venv venv

install: venv
	venv/bin/python -m pip install --upgrade pip
	venv/bin/python -m pip install -r requirements.txt

run:
	$(PYTHON) app.py

freeze:
	$(PIP) freeze > requirements.txt

lint:
	$(PYTHON) -m py_compile app.py src/loader.py src/split.py src/embedding.py src/qa.py src/qa_openai.py
	@echo "Syntax OK"

clean:
	rm -rf venv
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
