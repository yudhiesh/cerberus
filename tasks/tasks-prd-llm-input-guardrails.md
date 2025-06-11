## Relevant Files

- `algorithms/heuristics/src/main.py` - FastAPI application for heuristics-based guardrail
- `algorithms/heuristics/src/rules_engine.py` - Rule processing and weighted scoring logic
- `algorithms/heuristics/configs/config.yaml` - Hydra configuration for rules and thresholds
- `algorithms/heuristics/tests/test_api.py` - API endpoint tests for heuristics service
- `algorithms/heuristics/tests/test_rules_engine.py` - Unit tests for rules engine
- `algorithms/heuristics/Dockerfile` - Container definition for heuristics service
- `algorithms/heuristics/pyproject.toml` - UV package configuration for heuristics service
- `algorithms/ml_model/src/main.py` - FastAPI application for ML-based guardrail
- `algorithms/ml_model/src/model.py` - BERT model training and inference logic
- `algorithms/ml_model/src/data_loader.py` - Dataset loading from HuggingFace
- `algorithms/ml_model/configs/config.yaml` - Hydra configuration for model parameters
- `algorithms/ml_model/tests/test_api.py` - API endpoint tests for ML service
- `algorithms/ml_model/tests/test_model.py` - Unit tests for model inference
- `algorithms/ml_model/Dockerfile` - Container definition for ML service
- `algorithms/ml_model/pyproject.toml` - UV package configuration for ML service with PyTorch
- `algorithms/llm_guardrail/src/main.py` - FastAPI application for LLM-based guardrail
- `algorithms/llm_guardrail/src/llm_client.py` - OpenRouter API client with retry logic
- `algorithms/llm_guardrail/src/prompt_manager.py` - ReAct prompting implementation
- `algorithms/llm_guardrail/configs/config.yaml` - Hydra configuration including API keys
- `algorithms/llm_guardrail/tests/test_api.py` - API endpoint tests for LLM service
- `algorithms/llm_guardrail/tests/test_llm_client.py` - Unit tests for LLM client
- `algorithms/llm_guardrail/Dockerfile` - Container definition for LLM service
- `algorithms/llm_guardrail/pyproject.toml` - UV package configuration for LLM service
- `evaluation/src/main.py` - Evaluation framework main script
- `evaluation/src/metrics.py` - Performance metrics calculation
- `evaluation/src/test_data.py` - OOD test data generation
- `evaluation/tests/test_evaluation.py` - Tests for evaluation framework
- `evaluation/pyproject.toml` - UV package configuration for evaluation framework
- `shared/pyproject.toml.template` - Base template for pyproject.toml files
- `shared/pyproject.toml` - UV package configuration for shared module
- `shared/models/__init__.py` - Shared models package initialization
- `shared/models/api_models.py` - Pydantic models for API requests and responses
- `shared/models/exceptions.py` - Custom exception classes for error handling
- `shared/utils/__init__.py` - Shared utilities package initialization
- `shared/utils/timing.py` - Timing utilities for measuring execution time
- `docker-compose.yml` - Multi-service orchestration configuration
- `README.md` - Project documentation and setup instructions
- `.pre-commit-config.yaml` - Pre-commit hooks configuration for code quality
- `.bandit` - Bandit security linter configuration
- `.gitignore` - Git ignore patterns for Python and project files
- `Justfile` - Development commands and automation tasks
- `.github/workflows/ci.yml` - Main CI/CD pipeline workflow
- `.github/dependabot.yml` - Automated dependency updates configuration

### Notes

- Each algorithm service follows the same directory structure for consistency
- All services use UV for dependency management with `pyproject.toml`
- Hydra configurations allow environment-specific overrides
- Tests should achieve 80%+ coverage using pytest
- Docker images should use multi-stage builds for optimization

## Tasks

- [x] 1.0 Set up project structure and common infrastructure
  - [x] 1.1 Create directory structure for algorithms/, evaluation/, and shared components
  - [x] 1.2 Set up base pyproject.toml template with common dependencies (FastAPI, Hydra, pytest, httpx)
  - [x] 1.3 Create shared response models and types for API consistency
  - [x] 1.4 Implement base Dockerfile template with UV and Python 3.12
  - [x] 1.5 Set up pre-commit hooks for code quality (black, ruff, mypy)
  - [x] 1.6 Create GitHub Actions workflow for CI/CD pipeline

- [ ] 2.0 Implement heuristics-based guardrail algorithm
  - [ ] 2.1 Initialize heuristics service directory with UV and pyproject.toml
  - [ ] 2.2 Create FastAPI application with /predict and /health endpoints
  - [ ] 2.3 Implement rules engine with keyword blacklist functionality
  - [ ] 2.4 Add regex pattern matching for common attack patterns
  - [ ] 2.5 Implement weighted scoring system to combine rule outputs
  - [ ] 2.6 Create Hydra configuration for rules and thresholds
  - [ ] 2.7 Add input validation and error handling
  - [ ] 2.8 Write unit tests for rules engine (test_rules_engine.py)
  - [ ] 2.9 Write integration tests for API endpoints (test_api.py)
  - [ ] 2.10 Create Dockerfile with multi-stage build

- [ ] 3.0 Implement ML model-based guardrail algorithm
  - [ ] 3.1 Initialize ml_model service directory with UV and pyproject.toml
  - [ ] 3.2 Create FastAPI application with /predict and /health endpoints
  - [ ] 3.3 Implement data loader for HuggingFace dataset
  - [ ] 3.4 Create BERT model training script with fine-tuning logic
  - [ ] 3.5 Implement model inference with confidence score calculation
  - [ ] 3.6 Add model serialization and loading functionality
  - [ ] 3.7 Create Hydra configuration for model parameters
  - [ ] 3.8 Implement model warm-up on service startup
  - [ ] 3.9 Write unit tests for model inference (test_model.py)
  - [ ] 3.10 Write integration tests for API endpoints (test_api.py)
  - [ ] 3.11 Create Dockerfile with model artifacts and dependencies

- [ ] 4.0 Implement LLM-as-a-guardrail algorithm
  - [ ] 4.1 Initialize llm_guardrail service directory with UV and pyproject.toml
  - [ ] 4.2 Create FastAPI application with /predict and /health endpoints
  - [ ] 4.3 Implement OpenRouter API client with proper authentication
  - [ ] 4.4 Add retry logic with exponential backoff for API failures
  - [ ] 4.5 Implement ReAct prompting strategy for classification
  - [ ] 4.6 Add cost tracking functionality per request
  - [ ] 4.7 Create Hydra configuration with API keys and model settings
  - [ ] 4.8 Implement response parsing and confidence extraction
  - [ ] 4.9 Write unit tests for LLM client with mocked responses (test_llm_client.py)
  - [ ] 4.10 Write integration tests for API endpoints (test_api.py)
  - [ ] 4.11 Create Dockerfile with environment variable handling

- [ ] 5.0 Create evaluation framework and benchmarking tools
  - [ ] 5.1 Initialize evaluation service directory with UV and pyproject.toml
  - [ ] 5.2 Create main evaluation script with service discovery
  - [ ] 5.3 Implement HTTP client to query all three algorithms
  - [ ] 5.4 Create metrics calculation module (accuracy, precision, recall, F1)
  - [ ] 5.5 Implement latency measurement and percentile calculations
  - [ ] 5.6 Generate out-of-distribution (OOD) test samples
  - [ ] 5.7 Create visualization module for comparison charts
  - [ ] 5.8 Implement CSV/JSON export for benchmark results
  - [ ] 5.9 Add support for loading custom test datasets
  - [ ] 5.10 Write tests for evaluation framework
  - [ ] 5.11 Create evaluation report template

- [ ] 6.0 Set up Docker containerization and orchestration
  - [ ] 6.1 Create docker-compose.yml with all services defined
  - [ ] 6.2 Configure service names and networking for inter-service communication
  - [ ] 6.3 Set up environment variable management for configurations
  - [ ] 6.4 Add health check configurations for each service
  - [ ] 6.5 Create docker-compose.override.yml for local development
  - [ ] 6.6 Write comprehensive README.md with setup instructions
  - [ ] 6.7 Add Makefile for common Docker operations
  - [ ] 6.8 Create .env.example file with required environment variables
  - [ ] 6.9 Test full stack deployment and service communication
  - [ ] 6.10 Document troubleshooting steps and common issues 