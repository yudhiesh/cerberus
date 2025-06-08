"""Common fixtures for testing."""

import os
from pathlib import Path
import pytest
from typer.testing import CliRunner


@pytest.fixture
def runner():
    """Create a CliRunner instance for testing Typer apps."""
    return CliRunner()


@pytest.fixture
def test_data_dir():
    """Return the path to the test data directory."""
    return Path(__file__).parent / "data"


@pytest.fixture
def mock_env_vars(monkeypatch):
    """Set up mock environment variables for testing."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "test_openrouter_key")
    monkeypatch.setenv("HF_TOKEN", "test_hf_token")
    monkeypatch.setenv("ARGILLA_API_KEY", "test_argilla_key")
