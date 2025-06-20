#!/usr/bin/env python3

import typer
import sys
from src.data_generation.generate import app as generate_app
from src.data_generation.evaluate import app as evaluate_app
from dotenv import load_dotenv
from src.data_generation.annotate import app as annotate_app
from src.data_generation.dataset_preprocess import app as preprocess_app
from src.data_generation.deduplicate import app as deduplicate_app

load_dotenv()

app = typer.Typer(help="CLI for generating and evaluating LLM safety datasets.")

app.add_typer(generate_app, name="generate")
app.add_typer(evaluate_app, name="evaluate")
app.add_typer(annotate_app, name="annotate")
app.add_typer(preprocess_app, name="preprocess")
app.add_typer(deduplicate_app, name="deduplicate")

if __name__ == "__main__":
    sys.exit(app())
