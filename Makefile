# Start server in background (daemon mode)
start:
	uv run anthropic-proxy start

# Stop the running server
stop:
	uv run anthropic-proxy stop

# Restart the server
restart:
	uv run anthropic-proxy restart

# Show server status
status:
	uv run anthropic-proxy status

# Start server on custom port
start-port:
	uv run anthropic-proxy start --port 8082

# Start server on custom host and port
start-custom:
	uv run anthropic-proxy start --host 0.0.0.0 --port 8082

# Initialize config files
init:
	uv run anthropic-proxy --init

# Force reinitialize config files
init-force:
	uv run anthropic-proxy --init-force

# Print current configuration
print-config:
	uv run anthropic-proxy --print-config

# Development: Run server directly in foreground (not as daemon)
dev:
	uv run -m anthropic_proxy.server --host 0.0.0.0 --port 8082

# Testing
test:
	uv run pytest tests/ -v

test-cov:
	uv run pytest --cov=anthropic_proxy tests/ -v

test-cov-html:
	uv run pytest --cov=anthropic_proxy tests/ --cov-report html

test-routing:
	uv run pytest tests/test_routing.py -v

test-hooks:
	uv run pytest tests/test_hooks.py -v

test-conversion:
	uv run pytest tests/test_conversions.py -v

# Code quality
lint:
	uv run ruff check . --fix

format:
	uv run ruff format anthropic_proxy/**/*.py tests/*.py

install: lint format
	uv tool install . --reinstall

# Help command
help:
	@echo "Available commands:"
	@echo ""
	@echo "Server Control (daemon mode):"
	@echo "  make start            - Start server in background"
	@echo "  make stop             - Stop server"
	@echo "  make restart          - Restart server"
	@echo "  make status           - Show server status"
	@echo "  make start-port       - Start on port 8082"
	@echo "  make start-custom     - Start on 0.0.0.0:8082"
	@echo ""
	@echo "Configuration:"
	@echo "  make init             - Initialize config files"
	@echo "  make init-force       - Force reinitialize config files"
	@echo "  make print-config     - Print current configuration"
	@echo ""
	@echo "Development:"
	@echo "  make dev              - Run server directly in foreground"
	@echo ""
	@echo "Testing:"
	@echo "  make test             - Run all tests"
	@echo "  make test-cov         - Generate coverage report"
	@echo "  make test-cov-html    - Generate HTML coverage report"
	@echo "  make test-routing     - Run routing tests"
	@echo "  make test-hooks       - Run hook tests"
	@echo "  make test-conversion  - Run conversion tests"
	@echo ""
	@echo "Code Quality:"
	@echo "  make lint             - Check and fix code with ruff"
	@echo "  make format           - Format code with ruff"

.PHONY: install start stop restart status start-port start-custom init init-force print-config dev test test-cov test-cov-html test-routing test-hooks test-conversion lint format help
