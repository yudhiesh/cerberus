import typer
from pathlib import Path
import pandas as pd
import argilla as rg
from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split

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

    # Extract required fields and create argilla_df
    df = pd.DataFrame.from_dict(dataset.records.to_dict())
    argilla_df = pd.DataFrame({
        "input_prompt": df["fields"].apply(lambda x: x.get("prompt")),
        "label": df["responses"].apply(lambda x: next((resp["value"] for resp in x.get("label", []) if "value" in resp), None))
    })
    # Filter out rows with missing values
    argilla_df = argilla_df.dropna()

    # Load judged JSONL file directly with pandas
    judged_df = pd.read_json(judged_jsonl, lines=True)
    judged_df = judged_df[["input_prompt", "original_label"]].dropna()

    typer.echo(f"Argilla annotations: {len(argilla_df)} rows")
    typer.echo(f"Original judged data: {len(judged_df)} rows")

    # Method 1: Proper merge to preserve all original data
    # Left join judged_df with argilla_df to preserve all original entries
    merged_df = judged_df.merge(argilla_df, on='input_prompt', how='left')

    # Combine labels: use argilla label where available, otherwise use original_label
    merged_df['final_label'] = merged_df['label'].fillna(merged_df['original_label'])

    # Keep only the columns we need
    result_df = merged_df[['input_prompt', 'final_label']].rename(columns={'final_label': 'label'})

    typer.echo(f"After merge: {len(result_df)} rows")

    # Check for and handle duplicates
    duplicates_count = result_df.duplicated(subset=['input_prompt']).sum()
    typer.echo(f"Duplicate input_prompts found: {duplicates_count}")

    if duplicates_count > 0:
        # Check for conflicting labels
        conflicts = result_df.groupby('input_prompt')['label'].nunique()
        conflicting_prompts = conflicts[conflicts > 1]
        typer.echo(f"Input_prompts with conflicting labels: {len(conflicting_prompts)}")
        
        if len(conflicting_prompts) > 0:
            typer.echo("Examples of conflicts:")
            for prompt in conflicting_prompts.head(3).index:
                conflict_labels = result_df[result_df['input_prompt'] == prompt]['label'].unique()
                typer.echo(f"  - '{prompt[:60]}...': {conflict_labels}")
            
            # Prioritize argilla labels over original labels for conflicts
            # Go back to merged_df to handle priorities properly
            priority_df = merged_df.copy()
            priority_df['has_argilla'] = priority_df['label'].notna()
            # Sort by priority (argilla labels first) and keep first occurrence
            priority_df = priority_df.sort_values(['input_prompt', 'has_argilla'], ascending=[True, False])
            priority_df = priority_df.drop_duplicates(subset=['input_prompt'], keep='first')
            priority_df['final_label'] = priority_df['label'].fillna(priority_df['original_label'])
            result_df = priority_df[['input_prompt', 'final_label']].rename(columns={'final_label': 'label'})
            typer.echo(f"After prioritizing argilla labels: {len(result_df)} rows")
        else:
            # No conflicts, just remove exact duplicates
            result_df = result_df.drop_duplicates(subset=['input_prompt'], keep='first')
            typer.echo(f"After removing duplicates: {len(result_df)} rows")

    # Final statistics
    argilla_used = len(merged_df[merged_df['label'].notna()])
    original_used = len(result_df) - argilla_used
    typer.echo(f"\nFinal dataset statistics:")
    typer.echo(f"- Total rows: {len(result_df)}")
    typer.echo(f"- Rows using argilla labels: {argilla_used}")
    typer.echo(f"- Rows using original labels: {original_used}")
    typer.echo(f"- Coverage: {argilla_used/len(result_df)*100:.1f}% argilla annotations")

    # Verify no missing labels
    missing_labels = result_df['label'].isna().sum()
    if missing_labels > 0:
        typer.echo(f"WARNING: {missing_labels} rows have missing labels!")
    else:
        typer.echo("✓ All rows have labels")

    merged_df = result_df  
    merged_df.to_json(output_file, orient="records", lines=True, force_ascii=False)
    typer.echo(f"Saved merged dataset to {output_file}")

@app.command()
def upload_to_huggingface(
    input_file: Path = typer.Option(..., help="Path to the processed JSONL file (output of generate-dataset)."),
    repo_id: str = typer.Option("yudhiesh/cerberus-guardrails-small", help="HuggingFace repo name, e.g. 'yudhiesh/cerberus-guardrails-small'."),
    test_size: float = typer.Option(0.2, help="Proportion of dataset to include in the test split."),
    api_url: str = typer.Option("http://localhost:6900", help="Argilla API URL."),
    api_key: str = typer.Option("argilla.apikey", help="Argilla API key."),
):
    """Upload the processed dataset to HuggingFace Datasets Hub with train/test splits."""
    rg.Argilla(api_url=api_url, api_key=api_key) 
    typer.echo(f"Loading dataset from {input_file}")
    
    # Load the dataset and ensure we only have the required columns
    df = pd.read_json(input_file, lines=True)
    df = df[["input_prompt", "label"]]
    
    # Split the dataset with stratification
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        stratify=df["label"],
        random_state=42
    )
    
    # Reset index to prevent __index_level_0__ column
    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)
    
    # Convert to HuggingFace datasets
    train_dataset = Dataset.from_pandas(train_df)
    test_dataset = Dataset.from_pandas(test_df)
    
    # Create a DatasetDict with both splits
    dataset_dict = DatasetDict({
        "train": train_dataset,
        "test": test_dataset
    })
    
    typer.echo(f"Uploading to HuggingFace Hub repo: {repo_id}")
    try:
        # Upload the dataset with both splits
        dataset_dict.push_to_hub(
            repo_id,
            commit_message="Upload dataset with train/test splits",
            commit_description="Dataset uploaded with stratified train/test splits"
        )
        typer.echo(f"Successfully uploaded to {repo_id} with train/test splits")
    except Exception as e:
        typer.echo(f"Error uploading to HuggingFace Hub: {str(e)}", err=True)
        raise typer.Exit(1)

if __name__ == "__main__":
    app() 