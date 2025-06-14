# Heuristics-based Guardrail Service

A fast, rule-based LLM input guardrail service that uses configurable heuristics to detect potentially unsafe queries.

## Features

- Keyword blacklist matching
- Regex pattern detection for common attack patterns
- Length-based validation
- Weighted scoring system for combining multiple rules
- Configurable via Hydra

## Quick Start

```bash
# Install dependencies
uv sync

# Run tests
uv run pytest

# Run the service
uv run uvicorn src.main:app --reload --port 8001
```

## Configuration

Edit `configs/config.yaml` to customize:
- Blacklisted keywords
- Regex patterns
- Score thresholds
- Rule weights

## API

- `POST /predict` - Classify a query as safe/unsafe
- `GET /health` - Health check endpoint 