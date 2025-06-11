"""Setup configuration for heuristics-guardrail service."""

from pathlib import Path
from setuptools import setup

# Resolve path to the local common module
common_path: str = (Path(__file__).parent.parent / "common").resolve().as_uri()

setup(
    install_requires=[
        f"common @ {common_path}",
        "fastapi>=0.104.0",
        "uvicorn[standard]>=0.24.0",
        "hydra-core>=1.3.2",
        "httpx>=0.25.0",
        "pydantic>=2.5.0",
        "pydantic-settings>=2.1.0",
        "python-multipart>=0.0.6",
        "pyyaml>=6.0.1",
    ]
) 