"""Tests for the deduplicate.py module."""
import json
from pathlib import Path
import pytest
from typer.testing import CliRunner
from unittest.mock import patch, mock_open
import pandas as pd

from deduplicate import app

@pytest.fixture
def sample_data():
    """Load sample data for testing."""
    data_path = Path("tests/data/sample_deduplicate.jsonl")
    return pd.read_json(data_path, lines=True)

def test_deduplicate_command(tmp_path, runner: CliRunner, sample_data):
    """Test the deduplicate command with valid input."""
    output_dir = tmp_path / "output"
    result = runner.invoke(
        app,
        [
            "--input-file",
            "tests/data/sample_deduplicate.jsonl",
            "--output-dir",
            str(output_dir),
        ],
    )
    assert result.exit_code == 0
    assert "Deduplication Summary:" in result.stdout
    assert "Original size:" in result.stdout
    assert "Deduplicated size:" in result.stdout
    # Check that output files are created
    dedup_file = output_dir / "sample_deduplicate_deduplicated.jsonl"
    metrics_file = output_dir / "sample_deduplicate_deduplication_metrics.json"
    assert dedup_file.exists()
    assert metrics_file.exists()

def test_deduplicate_invalid_file(tmp_path, runner: CliRunner):
    """Test deduplicate command with invalid input file."""
    output_dir = tmp_path / "output"
    result = runner.invoke(
        app,
        [
            "--input-file",
            "nonexistent.jsonl",
            "--output-dir",
            str(output_dir),
        ],
    )
    assert result.exit_code != 0
    assert "Error" in result.stdout

def test_deduplicate_missing_columns(tmp_path, runner: CliRunner):
    """Test deduplicate command with missing required columns."""
    # Create a file with missing columns
    bad_file = tmp_path / "bad.jsonl"
    with open(bad_file, "w") as f:
        f.write(json.dumps({"invalid_column": "data"}) + "\n")
    output_dir = tmp_path / "output"
    result = runner.invoke(
        app,
        [
            "--input-file",
            str(bad_file),
            "--output-dir",
            str(output_dir),
        ],
    )
    assert result.exit_code != 0
    assert "Error" in result.stdout
    assert "must contain 'query' and 'label' columns" in result.stdout

def test_deduplicate_metrics(tmp_path, runner: CliRunner):
    """Test that deduplication metrics are calculated and outputted."""
    output_dir = tmp_path / "output"
    result = runner.invoke(
        app,
        [
            "--input-file",
            "tests/data/sample_deduplicate.jsonl",
            "--output-dir",
            str(output_dir),
        ],
    )
    assert result.exit_code == 0
    assert "Duplicate ratio:" in result.stdout
    assert "Exact duplicate ratio:" in result.stdout
    # Check that metrics file exists and is valid JSON
    metrics_file = output_dir / "sample_deduplicate_deduplication_metrics.json"
    assert metrics_file.exists()
    with open(metrics_file) as f:
        metrics = json.load(f)
    assert "duplicate_ratio" in metrics
    assert "exact_duplicate_ratio" in metrics

def test_deduplicate_output_files(tmp_path, runner: CliRunner):
    """Test that output files are created correctly."""
    output_dir = tmp_path / "output"
    result = runner.invoke(
        app,
        [
            "--input-file",
            "tests/data/sample_deduplicate.jsonl",
            "--output-dir",
            str(output_dir),
        ],
    )
    assert result.exit_code == 0
    dedup_file = output_dir / "sample_deduplicate_deduplicated.jsonl"
    metrics_file = output_dir / "sample_deduplicate_deduplication_metrics.json"
    assert dedup_file.exists()
    assert metrics_file.exists() 