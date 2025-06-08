# LLM Safety Dataset Generation Pipeline

This repository contains a comprehensive pipeline for generating, curating, and validating LLM safety datasets. The pipeline uses multiple tools to ensure high-quality, deduplicated, and properly labeled data.

## Overview

The pipeline consists of the following steps:
1. Generate synthetic data using Distilabel and OpenRouter
2. Deduplicate the data using SemHash
3. Evaluate the data using DeepEval's G-Eval (LLM-as-a-judge)
4. Annotate disputed labels using Argilla
5. Merge and preprocess the final dataset
6. Upload to HuggingFace with stratified train/test splits

## Prerequisites

- Python 3.12+
- Docker and Docker Compose (for Argilla)
- Just command runner (`brew install just` on macOS)
- HuggingFace account and API token
- OpenRouter API key

## Environment Variables

Before starting, you need to set up your environment variables. The repository includes a `.env.example` file that you can use as a template:

```bash
# Copy the example environment file
cp .env.example .env
```

Then edit the `.env` file and set the following required variables:

- `OPENROUTER_API_KEY`: Your OpenRouter API key (get it from [OpenRouter](https://openrouter.ai/))
- `HF_TOKEN`: Your HuggingFace API token (get it from [HuggingFace Settings](https://huggingface.co/settings/tokens))
- `ARGILLA_API_KEY`: Your Argilla API key (default is "argilla.apikey" for local development)

⚠️ **Important**: Never commit your `.env` file to version control. It's already included in `.gitignore` to prevent accidental commits.

## Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd <repository-name>
```

2. Install dependencies:
```bash
just sync
```

3. Start Argilla (required for annotation):
```bash
just up
```

## Workflow

### 1. Generate Synthetic Data

Generate synthetic prompts with safe/unsafe labels using Distilabel and OpenRouter.

```bash
just generate --num-golden 1000 --model "mistralai/devstral-small:free" --safe-ratio 0.9
```

This will create a JSONL file in the `data` directory with the generated prompts.

### 2. Deduplicate the Data

Remove duplicate and similar prompts using SemHash.

```bash
just deduplicate --input-file data/synthetic_1000.jsonl
```

This will create a deduplicated version of the dataset with metrics about the deduplication process.

### 3. Evaluate the Data

Use DeepEval's G-Eval to validate the labels using an LLM-as-a-judge approach.

```bash
just evaluate --input-file data/synthetic_1000_deduplicated.jsonl --model "mistralai/devstral-small:free"
```

This will create a new JSONL file with evaluation results and a confusion matrix visualization.

### 4. Annotate Disputed Labels

Upload disputed labels to Argilla for human annotation.

```bash
just annotate upload --eval-file data/synthetic_1000_deduplicated_judged.jsonl
```

Then:
1. Open http://localhost:6900 in your browser
2. Log in with the default credentials
3. Annotate the disputed labels
4. Download the annotated data:
```bash
just annotate download --dataset-name "misclassified-prompts"
```

### 5. Preprocess and Merge

Merge the annotated data with the original dataset, prioritizing human annotations.

```bash
just preprocess generate-dataset --argilla-export-dir misclassified-prompts_annotated --judged-jsonl data/synthetic_1000_deduplicated_judged.jsonl
```

### 6. Upload to HuggingFace

Upload the final dataset to HuggingFace with stratified train/test splits.

```bash
just preprocess upload-to-huggingface --input-file data/full_dataset.jsonl --repo-id "your-username/dataset-name"
```

## Output Files

- `data/synthetic_*.jsonl`: Generated synthetic data
- `data/synthetic_*_deduplicated.jsonl`: Deduplicated data
- `data/synthetic_*_deduplicated_judged.jsonl`: Data with LLM judgments
- `data/synthetic_*_evaluation_confusion_matrix.png`: Evaluation visualization
- `data/full_dataset.jsonl`: Final merged dataset

## Troubleshooting

### Common Issues

1. **Argilla Connection Issues**
   - Ensure Docker is running
   - Check if Argilla is accessible at http://localhost:6900
   - Verify the API key in your environment

2. **OpenRouter API Issues**
   - Verify your API key
   - Check rate limits
   - Ensure the model is available

3. **Dataset Upload Issues**
   - Verify your HuggingFace token
   - Check repository permissions
   - Ensure the dataset format is correct

### Getting Help

For issues with:
- Distilabel: [Distilabel Documentation](https://distilabel.argilla.io/)
- DeepEval: [DeepEval Documentation](https://deepeval.com/docs)
- Argilla: [Argilla Documentation](https://docs.argilla.io/)
- SemHash: [SemHash Documentation](https://github.com/argilla-io/semhash)
