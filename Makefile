# libervia-ia-Core v1 - Makefile
.PHONY: help setup install dev test lint format clean build train infer
.DEFAULT_GOAL := help

PYTHON := python3
VENV_DIR := venv
PROFILE ?= dev
MODEL ?= 2b
PORT ?= 8000

# Colors
GREEN := \033[0;32m
YELLOW := \033[0;33m
BLUE := \033[0;34m
NC := \033[0m

help: ## 📖 Mostrar este help
	@echo "$(BLUE)libervia-ia-Core v1 - Comandos Disponíveis$(NC)"
	@echo ""
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  $(GREEN)%-15s$(NC) %s\n", $$1, $$2}' $(MAKEFILE_LIST)

setup: ## 🚀 Setup completo inicial
	@echo "$(BLUE)Setup do libervia-ia...$(NC)"
	@$(PYTHON) -m venv $(VENV_DIR)
	@$(VENV_DIR)/bin/pip install --upgrade pip setuptools wheel
	@$(MAKE) install
	@$(MAKE) dev
	@echo "$(GREEN)✅ Setup completo! Execute: source $(VENV_DIR)/bin/activate$(NC)"

install: ## 📦 Instalar dependências
	@echo "$(BLUE)Instalando dependências...$(NC)"
	@$(VENV_DIR)/bin/pip install -e ".[dev]"

dev: ## 🛠️ Setup ambiente de desenvolvimento
	@echo "$(BLUE)Configurando desenvolvimento...$(NC)"
	@$(VENV_DIR)/bin/pre-commit install
	@echo "$(GREEN)✅ Ambiente dev configurado$(NC)"

test: ## 🧪 Executar testes
	@echo "$(BLUE)Executando testes...$(NC)"
	@$(VENV_DIR)/bin/pytest -v

lint: ## 🔍 Linting
	@echo "$(BLUE)Executando linting...$(NC)"
	@$(VENV_DIR)/bin/ruff check packages/
	@$(VENV_DIR)/bin/mypy packages/

format: ## 🎨 Formatar código
	@echo "$(BLUE)Formatando código...$(NC)"
	@$(VENV_DIR)/bin/black packages/ apps/ scripts/
	@$(VENV_DIR)/bin/isort packages/ apps/ scripts/

train: ## 🎯 Treinar modelo
	@echo "$(BLUE)Treinando modelo $(MODEL)...$(NC)"
	@$(VENV_DIR)/bin/python -m libervia_training --config-path=configs train=$(PROFILE) model=core_$(MODEL)

infer: ## 🤖 Servidor de inferência
	@echo "$(BLUE)Iniciando servidor $(PORT)...$(NC)"
	@$(VENV_DIR)/bin/python -m libervia_inference --config-path=configs infer=$(PROFILE) server.port=$(PORT)

clean: ## 🧹 Limpar temporários
	@echo "$(BLUE)Limpando temporários...$(NC)"
	@find . -type f -name "*.pyc" -delete
	@find . -type d -name "__pycache__" -exec rm -rf {} +
	@rm -rf .pytest_cache/ .mypy_cache/ .coverage htmlcov/

status: ## ℹ️ Status do projeto
	@echo "$(BLUE)Status libervia-ia:$(NC)"
	@echo "  Python: $(shell $(PYTHON) --version 2>/dev/null || echo 'Não encontrado')"
	@echo "  VEnv: $(shell test -d $(VENV_DIR) && echo 'OK' || echo 'Não criado')"
	@echo "  Git: $(shell git --version 2>/dev/null | cut -d' ' -f3 || echo 'Não encontrado')"
