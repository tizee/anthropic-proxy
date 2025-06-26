dev:
	uv run uvicorn server:app --reload --host 0.0.0.0 --port 8082

dev-stable:
	uv run uvicorn server:app --host 0.0.0.0 --port 8082

run:
	uv run python server.py --reload > uvicorn.log 2>&1 & echo $$! > uvicorn.pid

run-stable:
	uv run python server.py > uvicorn.log 2>&1 & echo $$! > uvicorn.pid

stop:
	@stopped=false; \
	if [ -f uvicorn.pid ]; then \
	  PID=$$(cat uvicorn.pid); \
	  if ps -p $$PID > /dev/null; then \
	    CMDLINE=$$(ps -p $$PID -o args=); \
	    if echo "$$CMDLINE" | grep -q "python server.py"; then \
	      echo "Stopping server (from 'run') with PID $$PID..."; \
	      kill $$PID && rm uvicorn.pid; \
	      echo "Server stopped."; \
	      stopped=true; \
	    fi; \
	  else \
	    echo "Stale PID file found. Removing it."; \
	    rm uvicorn.pid; \
	  fi; \
	fi; \
	if [ "$$stopped" = "false" ]; then \
	  PIDS=$$(pgrep -f "uvicorn server:app"); \
	  if [ -n "$$PIDS" ]; then \
	    echo "Stopping server (from 'dev') with PID(s) $$PIDS..."; \
	    kill $$PIDS; \
	    echo "Server stopped."; \
	    stopped=true; \
	  fi; \
	fi; \
	if [ "$$stopped" = "false" ]; then \
	  echo "No running server found."; \
	fi

restart: stop
	sleep 1
	uv run python server.py --reload > uvicorn.log 2>&1 & echo $$! > uvicorn.pid

.PHONY:stop restart run run-stable dev-stable test test-unittest lint format

# Modern pytest framework (recommended)
test-pytest:
	-uv run pytest tests/ -v

# Modern unittest framework
test-unittest:
	-uv run python tests/test_unittest.py

# Test Coverage
test-cov:
	uv run pytest --cov=server --cov=models tests/ -v

test-cov-html:
	uv run pytest --cov=server --cov=models tests/ --cov-report html

test-routing:
	-uv run pytest tests/test_routing.py -v

# Default test command (pytest)
test: test-pytest

# Quick test commands for development
test-basic:
	-uv run python -m unittest tests.test_unittest.TestBasicRequests -v

test-tools:
	-uv run python -m unittest tests.test_unittest.TestToolRequests -v

test-custom:
	-uv run python -m unittest tests.test_unittest.TestCustomModels -v

test-comparison:
	-uv run python -m unittest tests.test_unittest.TestAnthropicComparison -v

# Specific tool tests
test-gemini:
	-uv run python -m unittest tests.test_unittest.TestCustomModels.test_gemini_tool_conversion -v

test-calculator:
	-uv run python -m unittest tests.test_unittest.TestToolRequests.test_calculator_tool -v

test-complex:
	-uv run python -m unittest tests.test_unittest.TestComplexScenarios -v

test-conversion:
	-uv run python tests/test_conversions.py

test-hooks:
	-uv run pytest tests/test_hooks.py -v

lint: format
	uv run ruff check . --fix

format:
	uv run ruff format .

# Help command
help:
	@echo "Available commands:"
	@echo "  make run              - Start development server with auto-reload"
	@echo "  make run-stable       - Start server without auto-reload (for editing server code)"
	@echo "  make dev-stable       - Start foreground server without auto-reload"
	@echo "  make test             - Run pytest suite (recommended)"
	@echo "  make test-pytest      - Run pytest suite"
	@echo "  make test-unittest    - Run unittest suite"
	@echo ""
	@echo "  make test-basic       - Quick basic request tests"
	@echo "  make test-tools       - Tool usage tests"
	@echo "  make test-custom      - Custom model tests"
	@echo "  make test-complex     - Complex tests"
	@echo "  make test-comparison  - Proxy vs Anthropic API comparison"
	@echo "  make test-gemini      - Gemini model conversion test"
	@echo "  make test-calculator  - Calculator tool test"
	@echo "  make test-cov         - Generate terminal test coverage report"
	@echo "  make test-cov-html  - Generate HTML test coverage report""
	@echo "  make lint             - Check code quality with ruff"
	@echo "  make format           - Format code with ruff"
	@echo "  make help             - Show this help message"
