#!/usr/bin/env python3

import json
import uuid
from collections import Counter
from pathlib import Path
import typer

from deepeval.synthesizer import Synthesizer
from deepeval.synthesizer.config import StylingConfig
from open_router_llm import OpenRouterOpenAI
from typing_extensions import Annotated
from deepeval.synthesizer.config import ContextConstructionConfig

app = typer.Typer(
    help="Generate synthetic LLM prompts labeled safe/unsafe, enforcing a 90:10 split."
)


@app.command("run")
def generate_dataset(
    num_golden: Annotated[
        int,
        typer.Option(
            help="Total number of goldens to generate (must be ≥ 10 for 90:10 split).",
        ),
    ] = 10,
    model: Annotated[
        str,
        typer.Option(
            help="The OpenRouter model to use (e.g., 'google/gemma-3-27b-it:free').",
        ),
    ] = "google/gemma-3-27b-it:free",
    output_dir: Annotated[
        Path,
        typer.Option(
            help="Directory where all outputs (two JSONL files + metadata) will be saved.",
        ),
    ] = Path("data"),
):
    """
    Generate a synthetic dataset of LLM prompts with a 90% safe / 10% unsafe split.

    - num_golden: Total number of examples (must be ≥ 10).
    - model: OpenRouter/OpenAI-compatible model identifier.
    - output_dir: Directory to save `synthetic_{run_id}_safe.jsonl`, `synthetic_{run_id}_unsafe.jsonl`, and `metadata_{run_id}.json`.
    """
    if num_golden < 10:
        typer.echo(
            "Error: --num-golden must be at least 10 to enforce a 90:10 split.",
            err=True,
        )
        raise typer.Exit(code=1)

    safe_count = int(num_golden * 0.9)
    unsafe_count = num_golden - safe_count
    typer.echo(
        f"Desired distribution: {safe_count} safe, {unsafe_count} unsafe (total={num_golden})."
    )

    run_id = uuid.uuid4().hex
    typer.echo(f"Assigned run_id: {run_id}")

    styling_config = StylingConfig(
        input_format=(
            "A single user-supplied prompt in English. "
            "Prompts may be simple factual questions, moderately complex explanations, "
            "or code-related tasks that could be either benign or malicious."
        ),
        expected_output_format="Either 'safe' or 'unsafe'.",
        task=("Classify whether the given user prompt is 'safe' or 'unsafe'."),
        scenario=("User prompts range from trivial factual questions to code tasks."),
    )
    deepeval_model = OpenRouterOpenAI(model_name=model)
    synthesizer = Synthesizer(
        styling_config=styling_config, async_mode=False, model=deepeval_model
    )
    safe_context_construction_config = ContextConstructionConfig(
        min_contexts_per_document=safe_count, critic_model=deepeval_model
    )
    unsafe_context_construction_config = ContextConstructionConfig(
        min_contexts_per_document=unsafe_count, critic_model=deepeval_model
    )
    typer.echo(f"Using model: {model}")

    typer.echo(f"Generating up to {safe_count} safe examples...")
    synthesizer.generate_goldens_from_docs(
        document_paths=["safe_contexts.txt"],
        context_construction_config=safe_context_construction_config,
    )
    safe_goldens = synthesizer.synthetic_goldens
    typer.echo(f"Obtained {len(safe_goldens)} safe examples.")

    typer.echo(f"Generating up to {unsafe_count} unsafe examples...")
    synthesizer.generate_goldens_from_docs(
        document_paths=["unsafe_contexts.txt"],
        context_construction_config=unsafe_context_construction_config,
    )
    synthetic_goldens = synthesizer.synthetic_goldens

    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / f"synthetic_{run_id}.jsonl"
    typer.echo(f"Saving examples to: {output_path}")
    with open(output_path, "w", encoding="utf-8") as sf:
        for g in synthetic_goldens:
            record = {
                "prompt": g.input.strip(),
                "label": g.expected_output.strip().lower(),
            }
            sf.write(json.dumps(record, ensure_ascii=False) + "\n")

    metadata = {
        "run_id": run_id,
        "model": model,
        "requested_num_golden": num_golden,
    }
    metadata_path = output_dir / f"metadata_{run_id}.json"
    typer.echo(f"Writing metadata to: {metadata_path}")
    with open(metadata_path, "w", encoding="utf-8") as mf:
        json.dump(metadata, mf, indent=2)

    safe_labels = Counter(g.expected_output.strip().lower() for g in synthetic_goldens)
    typer.echo(f"Safe label distribution: {safe_labels}")

    typer.echo("\nGeneration complete. All files share run_id: " + run_id)
