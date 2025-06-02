import json

import pytest
from typer.testing import CliRunner
from pathlib import Path

from main import app

runner = CliRunner()

@pytest.fixture
def temp_dir(tmp_path) -> Path:
    """
    Provides a fresh temporary directory for each test.
    """
    return tmp_path

def test_generation_creates_jsonl(temp_dir: Path):
    """
    Test that 'generation' command writes a JSONL with the correct number of records.
    """
    # Arrange: define num_golden and model, route output to temp_dir
    num_golden = 5
    model_name = "test-model"
    result = runner.invoke(
        app,
        [
            "generation",
            "--num-golden",
            str(num_golden),
            "--model",
            model_name,
            "--output-dir",
            str(temp_dir),
        ],
    )

    breakpoint()
    # Assert: no errors
    assert result.exit_code == 0
    # The CLI output should mention the saved file
    expected_file = temp_dir / f"synthetic_{num_golden}.jsonl"
    assert expected_file.exists()

    # Read the JSONL and confirm line count and content
    lines = expected_file.read_text(encoding="utf-8").strip().split("\n")
    assert len(lines) == num_golden

    for raw in lines:
        entry = json.loads(raw)
        assert "prompt" in entry and "label" in entry
        # The model field was only used in the echo; deep check is textual
        assert entry["label"] in {"safe", "unsafe"}

def test_evaluate_produces_judged_jsonl(temp_dir: Path):
    """
    Test that 'evaluate' command reads a JSONL and writes a judged JSONL.
    """
    # Arrange: create a dummy input JSONL in temp_dir
    sample_data = [
        {"prompt": "Test prompt 1", "label": "safe"},
        {"prompt": "Test prompt 2", "label": "unsafe"},
    ]
    input_file = temp_dir / "dummy.jsonl"
    with open(input_file, "w", encoding="utf-8") as f:
        for rec in sample_data:
            f.write(json.dumps(rec) + "\n")

    # Act: invoke the 'evaluate' command
    result = runner.invoke(
        app,
        [
            "evaluate",
            str(input_file),
            "--model",
            "test-model",  # This fake model won't actually be called if Outlines isn't reachable
            "--output-dir",
            str(temp_dir),
        ],
    )

    # Assert: command exits successfully
    assert result.exit_code == 0

    # The judged file should be named 'dummy_judged.jsonl' (input_file.stem + '_judged.jsonl')
    judged_file = temp_dir / f"{input_file.stem}_judged.jsonl"
    assert judged_file.exists()

    # Read the judged file; since we supplied a non-functional model, the code may write 'unsafe' for each
    lines = judged_file.read_text(encoding="utf-8").strip().split("\n")
    assert len(lines) == len(sample_data)

    for raw in lines:
        rec = json.loads(raw)
        # Ensure keys exist
        assert {"prompt", "original_label", "judged_label", "explanation", "label_changed"}.issubset(rec.keys())
