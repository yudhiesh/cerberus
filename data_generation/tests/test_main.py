"""Tests for the main CLI application."""

import pytest
from typer.testing import CliRunner

from main import app


def test_app_initialization(runner: CliRunner):
    """Test that the app initializes correctly."""
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "CLI for generating and evaluating LLM safety datasets" in result.stdout


def test_command_registration(runner: CliRunner):
    """Test that all commands are registered correctly."""
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0

    # Check that all expected commands are present
    expected_commands = [
        "generate",
        "evaluate",
        "annotate",
        "preprocess",
        "deduplicate",
    ]
    for cmd in expected_commands:
        assert cmd in result.stdout


def test_generate_command_help(runner: CliRunner):
    """Test the generate command help message."""
    result = runner.invoke(app, ["generate", "--help"])
    assert result.exit_code == 0
    assert "Generate synthetic LLM prompts" in result.stdout


def test_evaluate_command_help(runner: CliRunner):
    """Test the evaluate command help message."""
    result = runner.invoke(app, ["evaluate", "--help"])
    assert result.exit_code == 0
    assert "Evaluate synthetic prompts" in result.stdout


def test_annotate_command_help(runner: CliRunner):
    """Test the annotate command help message."""
    result = runner.invoke(app, ["annotate", "--help"])
    assert result.exit_code == 0
    assert "Manage data annotation workflow" in result.stdout


def test_preprocess_command_help(runner: CliRunner):
    """Test the preprocess command help message."""
    result = runner.invoke(app, ["preprocess", "--help"])
    assert result.exit_code == 0
    assert "Dataset preprocessing and merging" in result.stdout


def test_deduplicate_command_help(runner: CliRunner):
    """Test the deduplicate command help message."""
    result = runner.invoke(app, ["deduplicate", "--help"])
    assert result.exit_code == 0
    assert "Deduplicate the dataset" in result.stdout


def test_invalid_command(runner: CliRunner):
    """Test handling of invalid commands."""
    result = runner.invoke(app, ["invalid-command"])
    assert result.exit_code != 0
    assert "Error" in result.stdout


def test_command_without_args(runner: CliRunner):
    """Test commands without required arguments."""
    # Test generate command without args
    result = runner.invoke(app, ["generate"])
    assert result.exit_code != 0
    assert "Error" in result.stdout

    # Test evaluate command without args
    result = runner.invoke(app, ["evaluate"])
    assert result.exit_code != 0
    assert "Error" in result.stdout
