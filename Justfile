# Justfile for LLM Guardrails Project
# https://just.systems/

# Default recipe to display help
default:
    @just --list

# Install development dependencies and set up the project
setup:
    @echo "🔧 Setting up LLM Guardrails project..."
    @echo "📦 Installing UV if not already installed..."
    @command -v uv >/dev/null 2>&1 || curl -LsSf https://astral.sh/uv/install.sh | sh
    @echo "🎣 Setting up pre-commit hooks..."
    @just setup-precommit
    @echo "✅ Setup complete!"

# Set up pre-commit hooks
setup-precommit:
    @echo "📦 Installing pre-commit..."
    uv pip install pre-commit
    @echo "🎣 Installing git hooks..."
    pre-commit install
    pre-commit install --hook-type commit-msg
    @echo "✅ Pre-commit hooks installed!"

# Run pre-commit on all files
lint:
    pre-commit run --all-files

# Run pre-commit on specific files
lint-file FILE:
    pre-commit run --files {{FILE}}

# Update pre-commit hooks to latest versions
update-hooks:
    pre-commit autoupdate

# Format code with black
format:
    black algorithms/ evaluation/ shared/ --line-length 100

# Run type checking with mypy
typecheck:
    mypy algorithms/ evaluation/ shared/ --python-version 3.12 --strict

# Run security checks with bandit
security:
    bandit -r algorithms/ evaluation/ shared/ -ll

# Build all Docker images
build-all:
    @echo "🐳 Building all Docker images..."
    @just build-heuristics
    @just build-ml
    @just build-llm
    @echo "✅ All images built!"

# Build heuristics service Docker image
build-heuristics:
    docker build -f algorithms/heuristics/Dockerfile -t guardrails-heuristics:latest .

# Build ML model service Docker image
build-ml:
    docker build -f algorithms/ml_model/Dockerfile -t guardrails-ml:latest .

# Build LLM service Docker image
build-llm:
    docker build -f algorithms/llm_guardrail/Dockerfile -t guardrails-llm:latest .

# Run all services with docker-compose
up:
    docker-compose up -d

# Stop all services
down:
    docker-compose down

# View logs for all services
logs:
    docker-compose logs -f

# View logs for specific service
logs-service SERVICE:
    docker-compose logs -f {{SERVICE}}

# Run tests for a specific service
test SERVICE:
    cd algorithms/{{SERVICE}} && uv run pytest -v

# Run all tests
test-all:
    @echo "🧪 Running all tests..."
    @just test heuristics
    @just test ml_model
    @just test llm_guardrail
    cd evaluation && uv run pytest -v

# Run tests with coverage
test-coverage SERVICE:
    cd algorithms/{{SERVICE}} && uv run pytest --cov=src --cov-report=html --cov-report=term

# Clean up Python cache files
clean:
    find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
    find . -type f -name "*.pyc" -delete
    find . -type f -name "*.pyo" -delete
    find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
    find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
    find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
    find . -type f -name ".coverage" -delete
    find . -type d -name "htmlcov" -exec rm -rf {} + 2>/dev/null || true

# Clean Docker resources
clean-docker:
    docker-compose down -v
    docker system prune -f

# Install dependencies for a specific service
install SERVICE:
    cd algorithms/{{SERVICE}} && uv sync

# Install all service dependencies
install-all:
    @echo "📦 Installing all service dependencies..."
    cd algorithms/heuristics && uv sync
    cd algorithms/ml_model && uv sync
    cd algorithms/llm_guardrail && uv sync
    cd evaluation && uv sync
    cd shared && uv sync

# Run the evaluation framework
evaluate:
    cd evaluation && uv run python src/main.py

# Create a new feature branch
feature NAME:
    git checkout -b feature/{{NAME}}

# Run GitHub Actions locally (existing command)
ga-local-data-generation:
    act -W .github/workflows/data_generation_ci.yml --container-architecture linux/arm64

# Check if all services are healthy
health-check:
    @echo "🏥 Checking service health..."
    @curl -s http://localhost:8001/health | jq '.' || echo "❌ Heuristics service not responding"
    @curl -s http://localhost:8002/health | jq '.' || echo "❌ ML service not responding"
    @curl -s http://localhost:8003/health | jq '.' || echo "❌ LLM service not responding"

# Generate requirements.txt from pyproject.toml for a service
requirements SERVICE:
    cd algorithms/{{SERVICE}} && uv pip compile pyproject.toml -o requirements.txt

# Watch for file changes and run tests
watch SERVICE:
    cd algorithms/{{SERVICE}} && watchexec -e py -c "uv run pytest -v"

# Run a specific service locally (without Docker)
run SERVICE PORT:
    cd algorithms/{{SERVICE}} && uv run uvicorn src.main:app --reload --port {{PORT}}

# Profile a service for performance
profile SERVICE:
    cd algorithms/{{SERVICE}} && uv run python -m cProfile -o profile.stats src/main.py

# Create .env file from example
env:
    cp .env.example .env
    @echo "📝 Created .env file. Please update it with your configuration."
