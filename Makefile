run:
	-uv run uvicorn server:app --host 0.0.0.0 --port 8082 --reload

.PHONY: run test test-unittest lint format

# Modern pytest framework (recommended)
test-pytest:
	-uv run pytest tests/ -v

# Modern unittest framework
test-unittest:
	-uv run python tests/test_unittest.py

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
	@echo "  make test-conversion  - Format conversion tests"
	@echo "  make lint             - Check code quality with ruff"
	@echo "  make format           - Format code with ruff"
	@echo "  make help             - Show this help message"
