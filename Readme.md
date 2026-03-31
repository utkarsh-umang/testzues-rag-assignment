# AcmeCloud KB Agent

A RAG-based agent that answers customer support questions using a keyword-search retrieval system over a local knowledge base.

## Knowledge Base

All KB files live in the [kb/](kb/) folder. The files are small and intentionally use keyword search for deterministic, reproducible retrieval.

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure your OpenAI API key

Copy `.env` (already present) and set your key before running anything:

```
OPENAI_API_KEY=your-openai-api-key-here
```

> **Note:** `.env` is listed in `.gitignore` — it will never be committed.

## Running conversations

The `out/` folder (where outputs are written) is listed in `.gitignore` and is not tracked by git.

### Conversation 1

```bash
python -m agent.run --fixture fixtures/conversation_1.json --kb kb --out out
```

### Conversation 2

```bash
python -m agent.run --fixture fixtures/conversation_2.json --kb kb --out out
```

Outputs (`answer.json`, `answer.md`, `memory.json`) are written to the `out/` directory after each run.

## Running tests

```bash
pytest
```

To run a specific test file:

```bash
pytest tests/test_retriever.py
pytest tests/test_memory.py
pytest tests/test_output.py
pytest tests/test_e2e.py
```
