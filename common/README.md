# Common Module for LLM Guardrails

Shared models, utilities, and types used across all guardrail services.

## Contents

- **models/**: Pydantic models for API requests, responses, and exceptions
- **utils/**: Utility functions for timing and other common operations

## Installation

This module is installed as an editable package in each service:

```bash
cd algorithms/heuristics
uv pip install -e ../../common
```

## Usage

```python
from common.models import GuardrailRequest, GuardrailResponse, ResultType
from common.utils import Timer

# Use the models and utilities in your service
``` 