run:
	-uv run uvicorn server:app --host 0.0.0.0 --port 8082 --reload
.PHONY: run lint format

test:
	-uv run tests.py

lint:
	ruff check server.py --select F,E,W --ignore E501 --fix

format:
	ruff format server.py tests.py
