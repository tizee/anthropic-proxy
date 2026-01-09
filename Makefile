dev:
	@if [ ! -f .env ]; then echo "🔴 ERROR: .env file not found! Please create .env file from .env.example"; exit 1; fi
	uv run anthropic-proxy --reload --host 0.0.0.0 --port 8082

dev-stable:
	@if [ ! -f .env ]; then echo "🔴 ERROR: .env file not found! Please create .env file from .env.example"; exit 1; fi
	uv run anthropic-proxy --host 0.0.0.0 --port 8082

run:
	@if [ ! -f .env ]; then echo "🔴 ERROR: .env file not found! Please create .env file from .env.example"; exit 1; fi
	@echo "Starting server with auto-reload..."
	@uv run anthropic-proxy --reload > uvicorn.log 2>&1 & (sleep 2 && pgrep -f "anthropic-proxy" | head -n 1 > uvicorn.pid && echo "Server started with PID $$(cat uvicorn.pid).")

run-stable:
	@if [ ! -f .env ]; then echo "🔴 ERROR: .env file not found! Please create .env file from .env.example"; exit 1; fi
	@echo "Starting server..."
	@uv run anthropic-proxy > uvicorn.log 2>&1 & (sleep 2 && pgrep -f "anthropic-proxy" | head -n 1 > uvicorn.pid && echo "Server started with PID $$(cat uvicorn.pid).")

stop:
	@./scripts/stop.sh

restart: stop
	@sleep 1
	@echo "Restarting server..."
	@$(MAKE) -s run-stable

.PHONY:stop restart run run-stable dev-stable test lint format

# Modern pytest framework (recommended)
test-pytest:
	-uv run pytest tests/ -v

# Test Coverage
test-cov:
	uv run pytest --cov=anthropic_proxy tests/ -v

test-cov-html:
	uv run pytest --cov=anthropic_proxy tests/ --cov-report html

test-routing:
	-uv run pytest tests/test_routing.py -v

# Default test command (pytest)
test: test-pytest

test-conversion:
	-uv run pytest tests/test_conversions.py -v

test-hooks:
	-uv run pytest tests/test_hooks.py -v

lint: format
	uv run ruff check . --fix

format:
	uv run ruff format anthropic_proxy/**/*.py tests/*.py performance_test.py

# Help command
help:
	@echo "Available commands:"
	@echo "IMPORTANT: Make sure you have a .env file (copy from .env.example) before running server commands!"
	@echo ""
	@echo "  make run              - Start development server with auto-reload"
	@echo "  make run-stable       - Start server without auto-reload (for editing server code)"
	@echo "  make dev-stable       - Start foreground server without auto-reload"
	@echo "  make test             - Run pytest suite (recommended)"
	@echo "  make test-pytest      - Run pytest suite"
	@echo "  make test-cov         - Generate terminal test coverage report"
	@echo "  make test-cov-html    - Generate HTML test coverage report"
	@echo "  make test-routing     - Run routing-specific tests"
	@echo "  make test-hooks       - Run hook-specific tests"
	@echo "  make test-conversion  - Run conversion-specific tests"
	@echo ""
	@echo "  make lint             - Check code quality with ruff"
	@echo "  make format           - Format code with ruff"
	@echo "  make stop             - Stop the running server"
	@echo "  make restart          - Restart the server"
	@echo "  make help             - Show this help message"
