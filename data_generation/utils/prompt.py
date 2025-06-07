import typer
from pathlib import Path


def format_context_as_instruction(context: str, label: str) -> str:
    """Format the context from file into an instruction for the LLM."""
    return (
        f"{context}\n\n"
        f"Based on the above context, generate a single example prompt. "
        f'Output only a JSON object with the following fields: "query" (the prompt) and "label" (always "{label}").'
    )


def load_prompt_template(file_path: Path) -> str:
    """Load prompt template from a text file."""
    try:
        with open(file_path, encoding="utf-8") as f:
            return f.read().strip()
    except FileNotFoundError:
        typer.echo(f"Error: File not found: {file_path}", err=True)
        raise typer.Exit(code=1)
    except Exception as e:
        typer.echo(f"Error reading file {file_path}: {e}", err=True)
        raise typer.Exit(code=1)
