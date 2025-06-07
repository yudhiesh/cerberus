import typer
import json
from pathlib import Path
from typing import Optional
import pandas as pd
import argilla as rg
from utils.models import LabelEnum

app = typer.Typer(help="Manage data annotation workflow with Argilla.")

def init_argilla() -> rg.Argilla:
    """Initialize Argilla client with default settings."""
    return rg.Argilla(
        api_url="http://localhost:6900",
        api_key="argilla.apikey"
    )

@app.command("upload")
def upload_to_argilla(
    eval_file: Path = typer.Option(..., help="Path to evaluation results JSONL file."),
    dataset_name: str = typer.Option("misclassified-prompts", help="Name of the dataset in Argilla."),
):
    """Upload evaluation results to Argilla for annotation using the FeedbackDataset API."""
    try:
        # Initialize Argilla client
        client = init_argilla()

        # Create dataset with fields and questions if it doesn't exist
        settings = rg.Settings(
            guidelines="Classify the prompt as safe or unsafe.",
            fields=[rg.TextField(name="prompt")],
            questions=[rg.LabelQuestion(name="label", labels=["safe", "unsafe"])]
        )
        dataset = client.datasets(dataset_name)
        if dataset is None:
            dataset = rg.Dataset(name=dataset_name, settings=settings)
            dataset.create()
            dataset = client.datasets(dataset_name)

        # Load and process data
        df = pd.read_json(eval_file, lines=True)
        df = df[df["success"] == False].copy()
        if "pred_label" in df.columns:
            df = df.rename(columns={"pred_label": "label"})
        elif "predicted_label" in df.columns:
            df = df.rename(columns={"predicted_label": "label"})
        else:
            def get_pred_label(row):
                if row["success"]:
                    return row["original_label"]
                else:
                    return "unsafe" if row["original_label"] == "safe" else "safe"
            df["label"] = df.apply(get_pred_label, axis=1)
        records = df[["input_prompt", "label"]].to_dict(orient="records")
        dataset.records.log(records=records, mapping={"input_prompt": "prompt", "label": "label"})

        typer.echo(f"Successfully uploaded {len(df)} records to Argilla dataset '{dataset_name}'")
        typer.echo("Please annotate the data in the Argilla UI at http://localhost:6900")

    except Exception as e:
        typer.echo(f"Error uploading to Argilla: {str(e)}", err=True)
        raise typer.Exit(1)

@app.command("download")
def download_from_argilla(
    dataset_name: str = typer.Option("misclassified-prompts", help="Name of the dataset in Argilla."),
    output_dir: Path = typer.Option("misclassified-prompts_annotated", help="Directory to save the annotated dataset."),
    api_url: str = typer.Option("http://localhost:6900", help="Argilla API URL."),
    api_key: str = typer.Option("argilla.apikey", help="Argilla API key."),
):
    """Download annotated data from Argilla FeedbackDataset and save locally using to_disk."""
    try:
        # Ensure output directory exists
        output_dir.mkdir(parents=True, exist_ok=True)
        # Initialize Argilla client
        client = rg.Argilla(api_url=api_url, api_key=api_key)
        # Fetch the dataset
        dataset = client.datasets(name=dataset_name)
        # Save to disk (directory)
        dataset.to_disk(str(output_dir))
        typer.echo(f"Successfully saved annotated dataset to {output_dir}")
    except Exception as e:
        typer.echo(f"Error downloading from Argilla: {str(e)}", err=True)
        raise typer.Exit(1)

if __name__ == "__main__":
    app() 