"""Tests for the generate.py module."""

from typer.testing import CliRunner
import pandas as pd

from src.data_generation.generate import app


def test_generate_dataset_basic(
    tmp_path,
    safe_contexts_file,
    unsafe_contexts_file,
    mock_pipeline_run,
):
    """Test basic dataset generation with valid inputs."""
    runner = CliRunner()
    output_dir = tmp_path / "output"
    result = runner.invoke(
        app,
        [
            "--num-golden",
            "20",
            "--model",
            "mistralai/mistral-small:free",
            "--output-dir",
            str(output_dir),
            "--safe-ratio",
            "0.9",
            "--safe-contexts-file",
            str(safe_contexts_file),
            "--unsafe-contexts-file",
            str(unsafe_contexts_file),
        ],
    )
    assert result.exit_code == 0
    assert output_dir.exists()
    output_file = output_dir / "synthetic_20.jsonl"
    assert output_file.exists()

    # Verify output format using pandas
    df = pd.read_json(output_file, lines=True)
    assert not df.empty
    assert "query" in df.columns
    assert "label" in df.columns


def test_generate_dataset_invalid_num_golden(
    tmp_path,
    safe_contexts_file,
    unsafe_contexts_file,
):
    """Test dataset generation with invalid num_golden."""
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "--num-golden",
            "5",  # Less than minimum of 10
            "--model",
            "mistralai/mistral-small:free",
            "--output-dir",
            str(tmp_path / "output"),
            "--safe-ratio",
            "0.9",
            "--safe-contexts-file",
            str(safe_contexts_file),
            "--unsafe-contexts-file",
            str(unsafe_contexts_file),
        ],
    )
    assert result.exit_code != 0
    assert "Error: --num-golden must be at least 10" in result.stdout


def test_generate_dataset_invalid_safe_ratio(
    tmp_path,
    safe_contexts_file,
    unsafe_contexts_file,
):
    """Test dataset generation with invalid safe_ratio."""
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "--num-golden",
            "20",
            "--model",
            "mistralai/mistral-small:free",
            "--output-dir",
            str(tmp_path / "output"),
            "--safe-ratio",
            "1.5",  # Invalid ratio
            "--safe-contexts-file",
            str(safe_contexts_file),
            "--unsafe-contexts-file",
            str(unsafe_contexts_file),
        ],
    )
    assert result.exit_code != 0
    assert "Error: --safe-ratio must be between 0 and 1" in result.stdout
