"""Happy path test for dataset_preprocess.py using real Argilla record structure."""

import pandas as pd
from typer.testing import CliRunner
from src.data_generation.dataset_preprocess import app


def test_generate_dataset_happy_path(tmp_path, sample_judged_jsonl, mock_argilla):
    runner = CliRunner()
    output_file = tmp_path / "output.jsonl"
    result = runner.invoke(
        app,
        [
            "generate-dataset",
            "--argilla-export-dir",
            str(tmp_path),  # Directory is not used due to mocking
            "--judged-jsonl",
            str(sample_judged_jsonl),
            "--output-file",
            str(output_file),
        ],
    )
    assert result.exit_code == 0
    assert output_file.exists()
    df = pd.read_json(output_file, lines=True)
    assert "input_prompt" in df.columns
    assert "label" in df.columns
    assert len(df) == 1
