# Product Requirements Document: LLM Input Guardrails

## Introduction/Overview

This feature implements a comprehensive LLM Input Guardrails system that detects and blocks potentially unsafe or dangerous user queries before they reach the main LLM application. The system showcases three different algorithmic approaches (Heuristics, Machine Learning, and LLM-as-a-Guardrail) to demonstrate the trade-offs between speed, accuracy, cost, and implementation complexity in building safety systems for AI applications.

The project serves as an educational resource for a blog post highlighting the intersection of Machine Learning and Software Engineering best practices, demonstrating how to build production-ready ML systems with proper testing, containerization, and evaluation frameworks.

## Goals

1. **Implement three distinct guardrail algorithms** that can classify user queries as "safe" or "unsafe"
2. **Demonstrate best practices** in ML/SWE including containerization, API design, testing, and configuration management
3. **Create a comprehensive evaluation framework** to compare algorithm performance across latency, accuracy, and cost dimensions
4. **Provide production-ready code** that can serve as a reference implementation for building safety systems
5. **Enable easy comparison** of different approaches through standardized APIs and evaluation metrics

## User Stories

1. **As a developer**, I want to integrate a guardrails API into my application so that I can prevent unsafe queries from reaching my LLM
2. **As a developer**, I want to compare different guardrail approaches so that I can choose the best trade-off for my use case
3. **As an ML engineer**, I want to understand how to productionize ML models with proper API design and containerization
4. **As a researcher**, I want to evaluate guardrail performance on out-of-distribution samples to understand generalization capabilities
5. **As a blog reader**, I want to see clear examples of ML/SWE best practices in action

## Functional Requirements

### Core Algorithm Implementations

1. **Heuristics-based Guardrail**
   - 1.1 Must implement configurable rule-based checks including keyword blacklists, regex patterns, and length checks
   - 1.2 Must use weighted scoring system to combine multiple rule outputs
   - 1.3 Must load rules from configuration file using Hydra
   - 1.4 Must return classification result without confidence score

2. **Machine Learning Model Guardrail**
   - 2.1 Must fine-tune BERT model on the provided dataset from HuggingFace
   - 2.2 Must implement model inference with confidence score output
   - 2.3 Must handle model loading and initialization efficiently
   - 2.4 Must return both classification and confidence score

3. **LLM-as-a-Guardrail**
   - 3.1 Must use Claude-4-Sonnet via OpenRouter API
   - 3.2 Must implement ReAct prompting approach for classification
   - 3.3 Must include retry logic for API failures
   - 3.4 Must track API costs per request
   - 3.5 Must return classification with confidence score

### API Requirements

4. **All algorithms must expose identical REST API**
   - 4.1 Endpoint: `POST /predict`
   - 4.2 Request format: `{"query": "user input text"}`
   - 4.3 Response format: `{"result": "safe"/"unsafe", "confidence": 0.0-1.0, "processing_time_ms": 123, "algorithm": "name", "version": "1.0.0"}`
   - 4.4 Must handle JSON content-type
   - 4.5 Must include health check endpoint at `GET /health`

### Infrastructure Requirements

5. **Containerization and Deployment**
   - 5.1 Each algorithm must have its own Dockerfile
   - 5.2 Must use Python 3.12 with UV package manager
   - 5.3 Must expose services on ports: 8001 (heuristics), 8002 (ML), 8003 (LLM)
   - 5.4 Must include docker-compose.yml for orchestrating all services
   - 5.5 Services must be addressable by name (e.g., `http://heuristics-guardrail:8001`)

6. **Configuration Management**
   - 6.1 All services must use Hydra for configuration
   - 6.2 Configurations must be environment-specific and overridable
   - 6.3 Must support configuration for model paths, API keys, thresholds, etc.

### Testing Requirements

7. **Unit and Integration Tests**
   - 7.1 Each algorithm must have comprehensive pytest unit tests
   - 7.2 Must include integration tests for API endpoints
   - 7.3 Must achieve at least 80% code coverage
   - 7.4 Must include tests for edge cases and error conditions

### Evaluation Framework

8. **Evaluation Service Requirements**
   - 8.1 Must be a separate service under `/evaluation` directory
   - 8.2 Must query all three algorithms via their service names
   - 8.3 Must measure and report latency for each request
   - 8.4 Must calculate accuracy metrics on test dataset
   - 8.5 Must generate comparison report with visualizations
   - 8.6 Must include sample out-of-distribution test cases

## Non-Goals (Out of Scope)

1. **Will NOT implement** multiple model versions or A/B testing capabilities
2. **Will NOT include** authentication or authorization for API endpoints
3. **Will NOT implement** rate limiting or request throttling
4. **Will NOT support** batch prediction endpoints
5. **Will NOT include** model retraining or online learning capabilities
6. **Will NOT implement** infrastructure cost estimation (only API costs for LLM)
7. **Will NOT handle** multi-language support (English only)
8. **Will NOT implement** detailed logging or monitoring systems

## Design Considerations

### Directory Structure
```
cerberus/
├── algorithms/
│   ├── heuristics/
│   │   ├── Dockerfile
│   │   ├── pyproject.toml
│   │   ├── src/
│   │   ├── tests/
│   │   └── configs/
│   ├── ml_model/
│   │   ├── Dockerfile
│   │   ├── pyproject.toml
│   │   ├── src/
│   │   ├── tests/
│   │   └── configs/
│   └── llm_guardrail/
│       ├── Dockerfile
│       ├── pyproject.toml
│       ├── src/
│       ├── tests/
│       └── configs/
├── evaluation/
│   ├── Dockerfile
│   ├── pyproject.toml
│   ├── src/
│   └── tests/
├── docker-compose.yml
└── README.md
```

### API Response Examples
```json
// Heuristics Response
{
  "result": "unsafe",
  "confidence": null,
  "processing_time_ms": 5,
  "algorithm": "heuristics",
  "version": "1.0.0"
}

// ML Model Response
{
  "result": "safe",
  "confidence": 0.92,
  "processing_time_ms": 45,
  "algorithm": "bert-classifier",
  "version": "1.0.0"
}

// LLM Guardrail Response
{
  "result": "unsafe",
  "confidence": 0.98,
  "processing_time_ms": 850,
  "algorithm": "claude-guardrail",
  "version": "1.0.0",
  "cost_usd": 0.0012
}
```

## Technical Considerations

1. **FastAPI** for all API implementations for async support and automatic OpenAPI documentation
2. **UV** for fast, reliable Python package management
3. **Hydra** for configuration management across all services
4. **PyTorch** for BERT model implementation
5. **httpx** for async HTTP requests to LLM API
6. **Pytest** with pytest-asyncio for testing async endpoints
7. **Docker** multi-stage builds for optimized container sizes
8. **OpenRouter API** integration with proper error handling and retries

## Success Metrics

1. **All three algorithms deployed** and accessible via standardized API
2. **Evaluation framework successfully compares** all algorithms on test dataset
3. **Latency targets achieved**:
   - Heuristics: < 10ms
   - ML Model: < 100ms  
   - LLM Guardrail: < 2000ms
4. **Test coverage** > 80% for all components
5. **Documentation** clear enough for junior developers to understand and extend
6. **Blog post demonstrates** clear trade-offs between approaches

## Open Questions

1. Should we implement caching for the LLM guardrail to reduce costs on repeated queries?
2. What specific out-of-distribution test cases would be most valuable for the evaluation?
3. Should the ML model support batch inference for better throughput?
4. Do we need to implement graceful degradation if one service is down during evaluation?
5. Should we add Prometheus metrics endpoints for production monitoring scenarios? 