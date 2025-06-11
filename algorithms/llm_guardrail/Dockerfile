# Dockerfile for LLM-based Guardrail Service

# Stage 1: Build stage using official UV image
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS builder

# Configure UV for optimal builds
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=0

# Set working directory
WORKDIR /app

# Install shared module first (for better caching)
COPY shared/ /shared/
RUN --mount=type=cache,target=/root/.cache/uv \
    cd /shared && uv venv && uv pip install -e .

# Copy and install service dependencies
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=algorithms/llm_guardrail/pyproject.toml,target=pyproject.toml \
    uv venv && uv sync --no-install-project --no-dev

# Copy service code
COPY algorithms/llm_guardrail/ /app/

# Stage 2: Runtime stage
FROM python:3.12-slim-bookworm

# Install only runtime dependencies
RUN apt-get update && apt-get install -y \
    --no-install-recommends \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN groupadd -r guardrail && useradd -r -g guardrail -u 1000 -m -s /bin/bash guardrail

# Set working directory
WORKDIR /app

# Copy the virtual environment from builder
COPY --from=builder --chown=guardrail:guardrail /app/.venv /app/.venv

# Copy shared module
COPY --from=builder --chown=guardrail:guardrail /shared /app/shared

# Copy application code
COPY --from=builder --chown=guardrail:guardrail /app/src /app/src
COPY --from=builder --chown=guardrail:guardrail /app/configs /app/configs

# Install shared module in the runtime environment
RUN /app/.venv/bin/pip install -e /app/shared

# Switch to non-root user
USER guardrail

# Set environment variables
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app:${PYTHONPATH} \
    SERVICE_NAME=claude-guardrail \
    SERVICE_VERSION=1.0.0

# Expose port
EXPOSE 8003

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import httpx; r = httpx.get('http://localhost:8003/health'); r.raise_for_status()"

# Run the service
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8003"] 