# Run `make help` for the list. Everything installs into .venv (Python 3.12 if available).
PY   ?= $(shell command -v python3.12 || command -v python3)
VENV := .venv
BIN  := $(VENV)/bin
PORT ?= 8000

.PHONY: help install ml demo dev test clean

help:
	@echo "make demo    run the public demo page locally; no API keys needed  -> http://localhost:$(PORT)"
	@echo "make test    run the test suite (no network, no keys)"
	@echo "make dev     run the full session API; needs GROQ_API_KEY (env or .env)"
	@echo "make ml      optional: install sentence-transformers (PyTorch, ~5.6 GB) for the real embedder"
	@echo "make clean   remove .venv"

$(BIN)/uvicorn: requirements.txt requirements-dev.txt
	$(PY) -m venv $(VENV)
	$(BIN)/pip install -q --upgrade pip
	$(BIN)/pip install -q -r requirements.txt -r requirements-dev.txt
	@touch $@

install: $(BIN)/uvicorn

ml: install
	$(BIN)/pip install -q -r requirements-ml.txt

# Demo mode: the journal-style page at /, plus /api/research and /health. Search and
# ranking are real; without GROQ_API_KEY the planner, critic and writer are simulated.
demo: install
	DEMO_MODE=true EMBEDDER_BACKEND=hashing $(BIN)/uvicorn main:app --reload --port $(PORT)

# Full mode: POST /research, SSE stream, sessions, SQLite. Uses the real embedder if
# `make ml` was run, otherwise falls back to the lexical one with a warning.
dev: install
	@if [ -z "$$GROQ_API_KEY" ] && ! grep -qs '^GROQ_API_KEY=..*' .env; then \
		echo "GROQ_API_KEY is not set. Get a free key at https://console.groq.com, then:"; \
		echo "  GROQ_API_KEY=your_key make dev    (or: cp .env.example .env and edit it)"; \
		exit 1; fi
	@if ! $(BIN)/python -c "import sentence_transformers" 2>/dev/null; then \
		echo "Note: sentence-transformers is not installed (make ml); the app falls back to the lexical embedder."; fi
	$(BIN)/uvicorn main:app --reload --port $(PORT)

test: install
	$(BIN)/python -m pytest tests -q

clean:
	rm -rf $(VENV)
