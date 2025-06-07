import typer
import json
from pathlib import Path
import pandas as pd
import argilla as rg
from datasets import Dataset

app = typer.Typer(help="Dataset preprocessing and merging for guardrails.")

@app.command()
def generate_dataset(
    argilla_export_dir: Path = typer.Option(..., help="Path to Argilla export directory (with records.json)."),
    judged_jsonl: Path = typer.Option(..., help="Path to judged dataset JSONL file."),
    output_file: Path = typer.Option("full_dataset.jsonl", help="Path to save the merged and deduplicated dataset."),
    api_url: str = typer.Option("http://localhost:6900", help="Argilla API URL."),
    api_key: str = typer.Option("argilla.apikey", help="Argilla API key."),
):
    """Merge Argilla-annotated and judged datasets, deduplicate, and output a clean dataset."""
    rg.Argilla(api_url=api_url, api_key=api_key)
    dataset = rg.Dataset.from_disk(argilla_export_dir.as_posix())
    
    # Convert records directly to DataFrame
    df = pd.DataFrame.from_dict(dataset.records.to_dict())
    
    # Extract required fields and create argilla_df
    argilla_df = pd.DataFrame({
        "input_prompt": df["fields"].apply(lambda x: x.get("prompt")),
        "label": df["responses"].apply(lambda x: next((resp["value"] for resp in x.get("label", []) if "value" in resp), None))
    })
    
    # Filter out rows with missing values
    argilla_df = argilla_df.dropna()

    # Load judged JSONL file directly with pandas
    judged_df = pd.read_json(judged_jsonl, lines=True)
    judged_df["input_prompt"] = judged_df["prompt"]
    # Robustly handle missing columns, now including 'original_label'
    if "predicted_label" in judged_df.columns and "label" in judged_df.columns:
        judged_df["label"] = judged_df["predicted_label"].combine_first(judged_df["label"])
    elif "predicted_label" in judged_df.columns:
        judged_df["label"] = judged_df["predicted_label"]
    elif "label" in judged_df.columns:
        judged_df["label"] = judged_df["label"]
    elif "original_label" in judged_df.columns:
        judged_df["label"] = judged_df["original_label"]
    else:
        raise ValueError("No label column found in judged JSONL file (expected one of 'predicted_label', 'label', or 'original_label').")
    judged_df = judged_df[["input_prompt", "label"]].dropna()

    merged_df = pd.concat([argilla_df, judged_df]).drop_duplicates(subset=["input_prompt"], keep="first")

    # Save merged DataFrame as JSONL using pandas
    merged_df.to_json(output_file, orient="records", lines=True, force_ascii=False)
    typer.echo(f"Saved merged dataset to {output_file}")

@app.command()
def upload_to_huggingface(
    processed_jsonl: Path = typer.Option(..., help="Path to the processed JSONL file (output of generate-dataset)."),
    repo_id: str = typer.Option(..., help="HuggingFace repo name, e.g. 'yudhiesh/cerberus-guardrails'."),
    split: str = typer.Option("train", help="Dataset split name (default: 'train')."),
    api_url: str = typer.Option("http://localhost:6900", help="Argilla API URL."),
    api_key: str = typer.Option("argilla.apikey", help="Argilla API key."),
):
    """Upload the processed dataset to HuggingFace Datasets Hub."""
    rg.Argilla(api_url=api_url, api_key=api_key) 
    typer.echo(f"Loading dataset from {processed_jsonl}")
    dataset = Dataset.from_json(str(processed_jsonl))
    typer.echo(f"Uploading to HuggingFace Hub repo: {repo_id} (split: {split})")
    dataset.push_to_hub(repo_id, split=split)
    typer.echo(f"Successfully uploaded to {repo_id}")

if __name__ == "__main__":
    app() 