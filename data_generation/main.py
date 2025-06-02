#!/usr/bin/env python3

import typer
import sys
from generate import app as generate_app
from evaluate import app as evaluate_app
from dotenv import load_dotenv

load_dotenv()

app = typer.Typer(help="CLI for generating and evaluating LLM safety datasets.")

app.add_typer(generate_app, name="generate")
app.add_typer(evaluate_app, name="evaluate")

if __name__ == "__main__":
    sys.exit(app())
