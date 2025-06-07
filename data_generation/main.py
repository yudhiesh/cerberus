#!/usr/bin/env python3

import typer
import sys
from generate import app as generate_app
from evaluate import app as evaluate_app
from dotenv import load_dotenv
from annotate import app as annotate_app
from dataset_preprocess import app as preprocess_app

load_dotenv()

app = typer.Typer(help="CLI for generating and evaluating LLM safety datasets.")

app.add_typer(generate_app, name="generate")
app.add_typer(evaluate_app, name="evaluate")
app.add_typer(annotate_app, name="annotate")
app.add_typer(preprocess_app, name="preprocess")

if __name__ == "__main__":
    sys.exit(app())
