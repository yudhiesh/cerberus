import typer
from pathlib import Path
from semhash import SemHash
import pandas as pd
import json

app = typer.Typer(help="Deduplicate the dataset.")


@app.command("run")
def deduplicate(
    input_file: Path = typer.Option(
        ...,
        "--input-file",
        "-i",
        help="Path to the input JSONL file to be deduplicated.",
    ),
    output_dir: Path = typer.Option(
        Path("data"),
        "--output-dir",
        "-o",
        help="Directory to save deduplicated data and metrics.",
    ),
):
    """
    Deduplicate the dataset using SemHash, preserving labels and collecting metrics.
    """
    try:
        # Load the data
        df = pd.read_json(input_file, lines=True)
        if "query" not in df.columns or "label" not in df.columns:
            raise ValueError("Input file must contain 'query' and 'label' columns")

        # Create SemHash instance and perform deduplication
        queries = df["query"]
        semhash = SemHash.from_records(records=queries)
        deduplication_result = semhash.self_deduplicate()

        # Get deduplicated texts and find their indices in the original DataFrame
        deduplicated_texts = deduplication_result.selected
        deduplicated_df = df[df["query"].isin(deduplicated_texts)].copy()

        # Collect metrics
        deduplication_metadata = {
            "duplicate_ratio": deduplication_result.duplicate_ratio,
            "exact_duplicate_ratio": deduplication_result.exact_duplicate_ratio,
            "least_similar_text_similarity": float(
                deduplication_result.get_least_similar_from_duplicates()[0][-1]
            ),
            "original_size": len(df),
            "deduplicated_size": len(deduplicated_df),
            "removed_count": len(df) - len(deduplicated_df),
        }

        # Create output directory if it doesn't exist
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save deduplicated data
        output_data_path = output_dir / f"{input_file.stem}_deduplicated.jsonl"
        deduplicated_df.to_json(output_data_path, orient="records", lines=True)

        # Save metrics
        output_metrics_path = (
            output_dir / f"{input_file.stem}_deduplication_metrics.json"
        )
        with open(output_metrics_path, "w") as f:
            json.dump(deduplication_metadata, f, indent=2)

        # Print summary
        typer.echo("\nDeduplication Summary:")
        typer.echo(f"Original size: {deduplication_metadata['original_size']}")
        typer.echo(f"Deduplicated size: {deduplication_metadata['deduplicated_size']}")
        typer.echo(f"Removed {deduplication_metadata['removed_count']} duplicates")
        typer.echo(f"Duplicate ratio: {deduplication_metadata['duplicate_ratio']:.2f}")
        typer.echo(
            f"Exact duplicate ratio: {deduplication_metadata['exact_duplicate_ratio']:.2f}"
        )
        typer.echo(f"\nSaved deduplicated data to: {output_data_path}")
        typer.echo(f"Saved metrics to: {output_metrics_path}")

    except Exception as e:
        typer.echo(f"Error during deduplication: {str(e)}", err=True)
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
