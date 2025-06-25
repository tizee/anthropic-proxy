run:
	uv run uvicorn server:app --reload --host 0.0.0.0 --port 8082 > /dev/null 2>&1 & echo $$! > uvicorn.pid

stop:
		@if [ -f uvicorn.pid ]; then \
	  PID=$$(cat uvicorn.pid); \
	  if ps -p $$PID > /dev/null; then \
	    CMDLINE=$$(ps -p $$PID -o args=); \
	    echo "$$CMDLINE" | grep -q "server:app" && echo "$$CMDLINE" | grep -q "8082" && kill $$PID && rm uvicorn.pid || echo 'pid exists but is not the fastapi API'; \
	  else \
	    echo "No running process with PID"; \
	    rm uvicorn.pid; \
	  fi \
	else \
	  echo "No PID file found"; \
	fi

restart: stop
	sleep 1
	uv run uvicorn server:app --reload --host 0.0.0.0 --port 8082 > /dev/null 2>&1 & echo $$! > uvicorn.pid

.PHONY:stop restart run test test-unittest lint format

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

lint:
	uv run ruff check . --fix

format:
	uv run ruff format .

# Help command
help:
	@echo "Available commands:"
	@echo "  make run              - Start development server"
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
