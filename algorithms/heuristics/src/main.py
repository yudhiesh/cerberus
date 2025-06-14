"""Main FastAPI application for heuristics-based guardrail service."""

from contextlib import asynccontextmanager

from common.models import (
    GuardrailRequest,
    GuardrailResponse,
    HealthResponse,
)
from common.utils import Timer
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from config import Config, load_config
from rules_engine import DEFAULT_RULES, RulesEngine

# Global instances
config: Config | None = None
rules_engine: RulesEngine | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handle startup and shutdown events."""
    global config, rules_engine

    try:
        config = load_config()
        print(f"Starting {config.service.name} v{config.service.version}")

        rules_engine = RulesEngine(
            rules=config.rules_engine.rules,
            unsafe_threshold=config.rules_engine.unsafe_threshold
        )
        print(f"Loaded {len(rules_engine.rules)} rules from configuration")

    except Exception as e:
        import traceback
        print(f"Failed to load configuration: {e}")
        print(f"Error type: {type(e).__name__}")
        print("Traceback:")
        traceback.print_exc()
        print("Falling back to default rules")
        config = None
        rules_engine = RulesEngine(rules=DEFAULT_RULES, unsafe_threshold=0.5)
        print(f"Loaded {len(rules_engine.rules)} default rules")

    yield

    # Shutdown
    service_name = config.service.name if config else "heuristics-guardrail"
    print(f"Shutting down {service_name}")


app = FastAPI(
    title="Heuristics-based LLM Guardrail",
    description="Fast rule-based guardrail for detecting unsafe LLM inputs",
    version="1.0.0",
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root() -> dict[str, str]:
    """Root endpoint."""
    return {
        "service": config.service.name if config else "heuristics-guardrail",
        "version": config.service.version if config else "1.0.0",
        "status": "running",
    }


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        algorithm=config.service.name if config else "heuristics-guardrail",
        version=config.service.version if config else "1.0.0",
    )


@app.post("/predict", response_model=GuardrailResponse)
async def predict(request: GuardrailRequest) -> GuardrailResponse:
    """
    Predict if the input query is safe or unsafe using heuristics.
    
    Args:
        request: The guardrail request containing the query to check
        
    Returns:
        GuardrailResponse with classification result
    """
    if not rules_engine:
        raise HTTPException(
            status_code=503,
            detail="Rules engine not initialized"
        )

    # Input validation
    if not request.query or not request.query.strip():
        raise HTTPException(
            status_code=400,
            detail="Query cannot be empty"
        )

    # Check query length if config is available
    if config and len(request.query) > config.performance.max_query_length:
        raise HTTPException(
            status_code=400,
            detail=f"Query exceeds maximum length of {config.performance.max_query_length} characters"
        )

    # Start timing
    with Timer() as timer:
        try:
            # Check query against all rules
            result, score, matches = rules_engine.check_query(request.query)

            # Log matches for debugging if enabled
            if config and config.features.log_matches and matches:
                match_summary = ", ".join(
                    f"{m.rule_name}:{m.matched_pattern}"
                    for m in matches[:5]  # First 5 matches
                )
                if len(matches) > 5:
                    match_summary += f" ... and {len(matches) - 5} more"
                print(f"Query matched rules: {match_summary}")

        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Error processing request: {str(e)}"

            ) from e
    # Return response with timing
    return GuardrailResponse(
        result=result,
        confidence=score,  # Use the normalized score as confidence
        processing_time_ms=timer.elapsed_ms,
        algorithm=config.service.name if config else "heuristics-guardrail",
        version=config.service.version if config else "1.0.0",
    )


if __name__ == "__main__":
    import uvicorn

    standalone_config = load_config()
    port = standalone_config.service.port
    host = standalone_config.service.host

    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        reload=True,
    )

