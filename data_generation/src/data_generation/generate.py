#!/usr/bin/env python3

import json
import random
from pathlib import Path
import typer
from src.data_generation.openrouter_distilabel_llm import OpenRouterLLM
from distilabel.pipeline import Pipeline
from distilabel.steps import LoadDataFromDicts
from distilabel.steps.tasks import TextGeneration
from utils.prompt import load_prompt_template, format_context_as_instruction
from utils.models import OutPrompt

app = typer.Typer(
    help="Generate synthetic LLM prompts labeled safe/unsafe, enforcing a configurable split using distilabel and OpenRouterLLM."
)


@app.command("run")
def generate_dataset(
    num_golden: int = typer.Option(
        100,
        help="Total number of examples to generate (must be >= 10).",
    ),
    model: str = typer.Option(
        "mistralai/devstral-small:free",
        help="The OpenRouter model to use (e.g., 'google/gemma-3-27b-it:free').",
    ),
    output_dir: Path = typer.Option(
        Path("data"),
        help="Directory where the output JSONL will be saved.",
    ),
    safe_ratio: float = typer.Option(
        0.9,
        help="Proportion of safe examples (e.g., 0.9 for 90:10 split). Must be in (0,1).",
    ),
    temperature: float = typer.Option(
        1.0,
        help="Sampling temperature for the LLM (higher values = more diverse output, e.g., 1.0 or 1.2).",
    ),
    safe_contexts_file: Path = typer.Option(
        Path("v2_safe_contexts.txt"),
        help="Path to the file containing safe prompt contexts.",
    ),
    unsafe_contexts_file: Path = typer.Option(
        Path("v2_unsafe_contexts.txt"),
        help="Path to the file containing unsafe prompt contexts.",
    ),
    max_new_tokens: int = typer.Option(
        2048,
        help="Maximum number of tokens to generate.",
    ),
    input_batch_size: int = typer.Option(
        32,
        help="Batch size for the LLM.",
    ),
    use_cache: bool = typer.Option(
        False,
        help="Whether to use cache.",
    ),
):
    """
    Generate a synthetic dataset of LLM prompts with a configurable safe/unsafe split using distilabel and OpenRouterLLM.
    - num_golden: Total number of examples (must be >= 10).
    - model: OpenRouter model identifier.
    - output_dir: Directory to save `synthetic_{num_golden}.jsonl`.
    - safe_ratio: Proportion of safe examples (default: 0.9 for 90:10 split).
    - safe_contexts_file: Path to file with safe prompt contexts.
    - unsafe_contexts_file: Path to file with unsafe prompt contexts.
    """
    if num_golden < 10:
        typer.echo(
            "Error: --num-golden must be at least 10.",
            err=True,
        )
        raise typer.Exit(code=1)
    if not (0 < safe_ratio < 1):
        typer.echo(
            "Error: --safe-ratio must be between 0 and 1 (exclusive).",
            err=True,
        )
        raise typer.Exit(code=1)

    typer.echo(f"Loading safe contexts from: {safe_contexts_file}")
    safe_context = load_prompt_template(safe_contexts_file)

    typer.echo(f"Loading unsafe contexts from: {unsafe_contexts_file}")
    unsafe_context = load_prompt_template(unsafe_contexts_file)

    safe_count = int(num_golden * safe_ratio)
    unsafe_count = num_golden - safe_count
    typer.echo(
        f"Desired distribution: {safe_count} safe, {unsafe_count} unsafe (total={num_golden}, safe_ratio={safe_ratio:.2f})."
    )

    # Format contexts into instructions
    safe_instruction = format_context_as_instruction(safe_context, "safe")
    unsafe_instruction = format_context_as_instruction(unsafe_context, "unsafe")

    # Prepare instructions for each type
    safe_instructions = [safe_instruction for _ in range(safe_count)]
    unsafe_instructions = [unsafe_instruction for _ in range(unsafe_count)]
    all_instructions = [(instr, "safe") for instr in safe_instructions] + [
        (instr, "unsafe") for instr in unsafe_instructions
    ]
    random.shuffle(all_instructions)

    # Prepare data for distilabel pipeline
    data = [
        {
            "system_prompt": "You are a prompt generator for LLM safety datasets.",
            "instruction": instr,
        }
        for instr, _ in all_instructions
    ]

    # Set up distilabel pipeline
    with Pipeline("llm-guardrail-prompt-generation") as pipeline:
        load_dataset = LoadDataFromDicts(
            name="load_instructions",
            data=data,
        )
        llm = OpenRouterLLM(
            model=model,
            structured_output={"format": "json", "schema": OutPrompt},
        )
        text_generation = TextGeneration(
            name="text_generation_guardrail",
            llm=llm,
            input_batch_size=input_batch_size,
            output_mappings={"model_name": "generation_model"},
        )
        load_dataset >> text_generation

    typer.echo(f"Using model: {model}")
    typer.echo(f"Generating {num_golden} prompts...")

    distiset = pipeline.run(
        parameters={
            text_generation.name: {
                "llm": {
                    "generation_kwargs": {
                        "max_new_tokens": max_new_tokens,
                        "temperature": temperature,
                    }
                }
            }
        },
        use_cache=use_cache,
    )

    generations = distiset["default"]["train"]["generation"]
    typer.echo(f"Obtained {len(generations)} generations.")

    output_records: list[OutPrompt] = []
    for g in generations:
        try:
            record = OutPrompt.model_validate_json(g)
            output_records.append(record)
        except Exception as e:
            typer.echo(f"Skipping malformed output: {g} (error: {e})", err=True)

    typer.echo(f"Valid records: {len(output_records)}")

    random.shuffle(output_records)

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"synthetic_{num_golden}.jsonl"
    typer.echo(f"Saving examples to: {output_path}")
    with open(output_path, "w", encoding="utf-8") as sf:
        for rec in output_records:
            sf.write(json.dumps(rec.model_dump(), ensure_ascii=False) + "\n")

    typer.echo("\nGeneration complete.")
