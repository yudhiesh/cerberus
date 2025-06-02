#!/usr/bin/env python3

import json
from collections import Counter
from pathlib import Path
import typer

from deepeval.synthesizer import Synthesizer
from deepeval.synthesizer.config import StylingConfig
from open_router_llm import OpenRouterOpenAI
from typing_extensions import Annotated

app = typer.Typer(help="Generate synthetic LLM prompts labeled safe/unsafe.")

@app.command("run")
def generate_dataset(
    num_golden: Annotated[int, typer.Option(
        help="Number of golden examples to generate (default: 100).",

    )] = 100,
    model: Annotated[str, typer.Option(
        help="The model to use leveraging OpenRouter https://openrouter.ai/models/?q=free",

    )] = "google/gemma-3-27b-it:free",
    output_dir: Annotated[Path, typer.Option(
        help="Output directory to save all datasets"
    )] = Path("data")
):
    """
    Generate synthetic examples (prompts + labels) using DeepEval’s Synthesizer.
    The output is a JSONL file of format: {\"prompt\": ..., \"label\": ...}
    """
    styling_config = StylingConfig(
        input_format=(
            "A single user-supplied prompt in English. "
            "Prompts may be simple factual questions, moderately complex explanations, "
            "or code-related tasks that could be either benign or malicious (e.g., hacking instructions)."
        ),
        expected_output_format="Either 'safe' or 'unsafe'.",
        task=(
            "Classify whether the given user prompt is 'safe' or 'unsafe' "
            "(requests disallowed instructions, hacking advice, or malicious code)."
        ),
        scenario=(
            "User prompts range from trivial factual questions to code tasks. "
            "99% of prompts are benign (safe); 1% are malicious (unsafe)."
            "Unsafe code can take many forms, such as NSFW, security issues, "
            "national risks, religious extremism, hate speech, etc."
        ),
    )
    deepeval_model = OpenRouterOpenAI(model_name=model)
    synthesizer = Synthesizer(styling_config=styling_config, async_mode=False, model=deepeval_model)

    typer.echo(f"Generating {num_golden} synthetic examples with model '{model}'...")
    contexts = ["Unsafe code can take many forms, such as NSFW, security issues, national risks, religious extremism, hate speech, etc."]
    synthesizer.generate_goldens_from_contexts(contexts = [contexts])
    golden_list = synthesizer.synthetic_goldens
    typer.echo(f"Completed generation: obtained {len(golden_list)} examples.")

    label_counts = Counter(g.input.lower() for g in golden_list)
    typer.echo(f"Label distribution: {label_counts}")

    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"synthetic_{num_golden}.jsonl"
    typer.echo(f"Saving synthetic dataset to: {output_file}")

    with open(output_file, "w", encoding="utf-8") as out_f:
        for golden in golden_list:
            record = {
                "prompt": golden.input.strip(),
                "label": golden.expected_output.strip().lower(),
            }
            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")

    typer.echo("Done writing synthetic dataset.")
    typer.echo(
        "\nResults saved. "
        "You can now load the JSONL file for downstream training or evaluation. "
        "If distribution is skewed, adjust --num-golden or sample separately."
    )

