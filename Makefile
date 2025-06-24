run:
	-uv run uvicorn server:app --host 0.0.0.0 --port 8082 --reload

.PHONY: run test test-unittest lint format

# Modern unittest framework (recommended)
test-unittest:
	-uv run tests_unittest.py

# Default test command (modern unittest)
test: test-unittest test-conversion

# Quick test commands for development
test-basic:
	-python -m unittest tests_unittest.TestBasicRequests -v

test-tools:
	-python -m unittest tests_unittest.TestToolRequests -v

test-custom:
	-python -m unittest tests_unittest.TestCustomModels -v

test-comparison:
	-python -m unittest tests_unittest.TestAnthropicComparison -v

# Specific tool tests
test-gemini:
	-python -m unittest tests_unittest.TestCustomModels.test_gemini_tool_conversion -v

test-calculator:
	-python -m unittest tests_unittest.TestToolRequests.test_calculator_tool -v

test-complex:
	-python -m unittest tests_unittest.TestComplexScenarios -v

test-conversion:
	-python test_conversions.py

lint:
	ruff check server.py --select F,E,W --ignore E501 --fix

format:
	ruff format server.py tests_unittest.py

# Help command
help:
	@echo "Available commands:"
	@echo "  make run              - Start development server"
	@echo "  make test             - Run modern unittest suite (recommended)"
	@echo "  make test-unittest    - Run modern unittest suite"
	@echo ""
	@echo "  make test-basic       - Quick basic request tests"
	@echo "  make test-tools       - Tool usage tests"
	@echo "  make test-custom      - Custom model tests"
	@echo "  make test-complex     - Complex tests"
	@echo "  make test-comparison  - Proxy vs Anthropic API comparison"
	@echo "  make test-gemini      - Gemini model conversion test"
	@echo "  make test-calculator  - Calculator tool test"
	@echo "  make lint             - Check code quality"
	@echo "  make format           - Format code"
	@echo "  make help             - Show this help message"
