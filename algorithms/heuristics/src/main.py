"""Main FastAPI application for heuristics-based guardrail service."""

import time
from contextlib import asynccontextmanager
from typing import Any, Dict

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from common.models import (
    GuardrailRequest,
    GuardrailResponse,
    HealthResponse,
    ResultType,
)
from common.utils import Timer


# Service metadata
SERVICE_NAME = "heuristics-guardrail"
SERVICE_VERSION = "1.0.0"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handle startup and shutdown events."""
    # Startup
    print(f"Starting {SERVICE_NAME} v{SERVICE_VERSION}")
    # TODO: Load rules configuration on startup
    yield
    # Shutdown
    print(f"Shutting down {SERVICE_NAME}")


# Create FastAPI app
app = FastAPI(
    title="Heuristics-based LLM Guardrail",
    description="Fast rule-based guardrail for detecting unsafe LLM inputs",
    version=SERVICE_VERSION,
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
async def root() -> Dict[str, str]:
    """Root endpoint."""
    return {
        "service": SERVICE_NAME,
        "version": SERVICE_VERSION,
        "status": "running",
    }


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        algorithm=SERVICE_NAME,
        version=SERVICE_VERSION,
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
    # Start timing
    with Timer() as timer:
        try:
            # TODO: Implement actual rules engine logic
            # For now, just return a placeholder response
            
            # Placeholder logic - mark queries containing certain keywords as unsafe
            unsafe_keywords = ["hack", "exploit", "injection", "malicious"]
            query_lower = request.query.lower()
            
            is_unsafe = any(keyword in query_lower for keyword in unsafe_keywords)
            
            result = ResultType.UNSAFE if is_unsafe else ResultType.SAFE
            
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Error processing request: {str(e)}"
            )
    
    # Return response with timing
    return GuardrailResponse(
        result=result,
        confidence=None,  # Heuristics don't have confidence scores
        processing_time_ms=timer.elapsed_ms,
        algorithm=SERVICE_NAME,
        version=SERVICE_VERSION,
    )


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8001,
        reload=True,
    ) 