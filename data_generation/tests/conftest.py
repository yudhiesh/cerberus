"""Common fixtures for testing."""

from pathlib import Path
import pytest
from typer.testing import CliRunner
import json
from unittest.mock import patch, MagicMock
import pandas as pd


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


@pytest.fixture
def sample_argilla_export_record():
    # This matches the real Argilla record structure provided by the user
    return {
        "id": ["e1368429-9d7d-4460-a135-f31162fa490f"],
        "fields": [
            {
                "prompt": "I'm building a real-time analytics dashboard for a SaaS platform that processes 500K events per second. Using Apache Kafka 3.6 with 100 partitions, ClickHouse as the OLAP database, and Redis for caching frequently accessed aggregations. The dashboard needs to show metrics with <2 second latency for the last 5 minutes and support drill-down queries on historical data up to 12 months. Currently experiencing memory pressure on ClickHouse nodes during peak traffic and some Kafka consumer lag on complex aggregation topics. How should I optimize the data pipeline architecture to handle traffic spikes, implement efficient pre-aggregation strategies, and design the materialized views for both real-time and historical queries?"
            }
        ],
        "responses": [
            {
                "label": [
                    {
                        "value": "unsafe",
                        "user_id": "fd2222ba-dd2b-4936-b961-424324f0ed16",
                    }
                ]
            }
        ],
    }


@pytest.fixture
def sample_judged_jsonl(tmp_path):
    # The input_prompt matches the Argilla record's prompt
    record = {
        "input_prompt": "I'm building a real-time analytics dashboard for a SaaS platform that processes 500K events per second. Using Apache Kafka 3.6 with 100 partitions, ClickHouse as the OLAP database, and Redis for caching frequently accessed aggregations. The dashboard needs to show metrics with <2 second latency for the last 5 minutes and support drill-down queries on historical data up to 12 months. Currently experiencing memory pressure on ClickHouse nodes during peak traffic and some Kafka consumer lag on complex aggregation topics. How should I optimize the data pipeline architecture to handle traffic spikes, implement efficient pre-aggregation strategies, and design the materialized views for both real-time and historical queries?",
        "original_label": "unsafe",
    }
    jsonl_file = tmp_path / "judged.jsonl"
    with open(jsonl_file, "w") as f:
        f.write(json.dumps(record) + "\n")
    return jsonl_file


@pytest.fixture
def mock_argilla(sample_argilla_export_record):
    with (
        patch("src.data_generation.dataset_preprocess.rg.Argilla") as mock_client,
        patch(
            "src.data_generation.dataset_preprocess.rg.Dataset"
        ) as mock_dataset_class,
    ):
        mock_dataset = MagicMock()
        mock_dataset.records = MagicMock()
        mock_dataset.records.to_dict.return_value = sample_argilla_export_record
        mock_dataset_class.from_disk.return_value = mock_dataset
        yield mock_client


@pytest.fixture
def safe_contexts_file(tmp_path):
    """Create a sample safe contexts file."""
    file_path = tmp_path / "safe_contexts.txt"
    with open(file_path, "w") as f:
        f.write("Generate a safe prompt about {topic}")
    return file_path


@pytest.fixture
def unsafe_contexts_file(tmp_path):
    """Create a sample unsafe contexts file."""
    file_path = tmp_path / "unsafe_contexts.txt"
    with open(file_path, "w") as f:
        f.write("Generate an unsafe prompt about {topic}")
    return file_path


@pytest.fixture
def mock_pipeline_run():
    """Patch only the Pipeline.run method to return the desired mock output for generations."""
    with patch(
        "distilabel.pipeline.Pipeline.run",
        return_value={
            "default": {
                "train": {
                    "generation": [
                        json.dumps(
                            {
                                "query": "How to write a Python function?",
                                "label": "safe",
                                "explanation": "This is a safe prompt about programming.",
                            }
                        )
                    ]
                }
            }
        },
    ):
        yield


@pytest.fixture
def sample_data():
    """Load sample data for testing."""
    data_path = Path("tests/data/sample_deduplicate.jsonl")
    return pd.read_json(data_path, lines=True)
