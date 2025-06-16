import marimo

__generated_with = "0.13.15"
app = marimo.App(width="full")


@app.cell
def _():
    from dotenv import load_dotenv
    import marimo as mo

    load_dotenv()
    return (mo,)


@app.cell
def _(mo):
    mo.md(
        """
    # BERTopic Analysis of Guardrails Dataset

    This notebook performs topic modeling on the Cerberus Guardrails dataset using BERTopic.
    We'll discover the main themes and patterns in the input prompts, visualize topic clusters,
    and analyze how topics are distributed across different labels.

    ## 1. Loading the Dataset

    First, we'll load the guardrails dataset from Hugging Face and look at the train set. This dataset contains prompts
    that are used to test AI safety guardrails. 
    """
    )
    return


@app.cell
def _():
    from datasets import load_dataset

    ds = load_dataset("yudhiesh/cerberus-guardrails-small", split="train")
    return (ds,)


@app.cell
def _(ds, mo):
    mo.md(
        f"""
    ## 2. Data Preparation

    The dataset contains **{len(ds)}** samples. Let's extract the input prompts and their 
    corresponding labels for analysis.

    Dataset columns: {list(ds.column_names)}
    """
    )
    return


@app.cell
def _(ds):
    input_prompt = ds["input_prompt"]
    label = ds["label"]
    return input_prompt, label


@app.cell
def _(mo):
    mo.md(
        """
    ## 3. BERTopic Configuration

    Now we'll set up BERTopic with custom components for optimal topic discovery:

    - **Embedding Model**: `thenlper/gte-large` - A high-quality sentence transformer for semantic embeddings
    - **UMAP**: Reduces embedding dimensions while preserving local and global structure
    - **HDBSCAN**: Density-based clustering that finds topics of varying sizes
    - **CountVectorizer**: Extracts meaningful keywords using n-grams and English stop words

    This configuration balances quality with computational efficiency.
    """
    )
    return


@app.cell
def _():
    import pandas as pd
    import numpy as np
    from bertopic import BERTopic
    from sentence_transformers import SentenceTransformer
    from umap import UMAP
    from hdbscan import HDBSCAN
    from sklearn.feature_extraction.text import CountVectorizer
    import plotly.graph_objects as go
    import matplotlib.pyplot as plt

    # UMAP for dimensionality reduction
    umap_model = UMAP(
        n_neighbors=15, n_components=5, min_dist=0.0, metric="cosine", random_state=42
    )

    # HDBSCAN for clustering
    hdbscan_model = HDBSCAN(
        min_cluster_size=15,
        metric="euclidean",
        cluster_selection_method="eom",
        prediction_data=True,
    )

    # Sentence transformer for embeddings
    embedding_model = SentenceTransformer("thenlper/gte-large")

    # Vectorizer for keyword extraction
    vectorizer_model = CountVectorizer(
        stop_words="english", min_df=2, ngram_range=(1, 2)
    )

    # Initialize BERTopic
    topic_model = BERTopic(
        embedding_model=embedding_model,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer_model,
        top_n_words=10,
        verbose=True,
    )
    return UMAP, embedding_model, topic_model


@app.cell
def _(mo):
    mo.md(
        """
    ## 4. Fitting the Topic Model

    Now we'll fit BERTopic on our input prompts. 
    This process:

    1. Embeds all prompts into high-dimensional vectors
    2. Reduces dimensions with UMAP
    3. Clusters similar prompts with HDBSCAN
    4. Extracts representative keywords for each topic using c-TF-IDF

    This may take a few minutes depending on dataset size...
    """
    )
    return


@app.cell
def _(input_prompt, topic_model):
    topics, probs = topic_model.fit_transform(input_prompt)

    # Get topic info
    topic_info = topic_model.get_topic_info()
    num_topics = len(set(topics)) - 1  # -1 for outliers
    num_outliers = sum(1 for t in topics if t == -1)

    print(f"Topics discovered: {num_topics}")
    print(f"Outliers: {num_outliers} ({num_outliers / len(topics) * 100:.1f}%)")
    return num_outliers, num_topics, topic_info, topics


@app.cell
def _(mo, num_outliers, num_topics):
    mo.md(
        f"""
    ## 5. Topic Overview

    The model discovered **{num_topics}** distinct topics in the dataset, with **{num_outliers}** 
    prompts classified as outliers (not fitting well into any topic).

    ### Interactive Topic Visualization

    This visualization shows the relationships between topics:

    - **Size** indicates topic frequency
    - **Distance** shows semantic similarity
    - **Hover** to see top keywords for each topic
    """
    )
    return


@app.cell
def _(topic_model):
    fig = topic_model.visualize_topics()
    return (fig,)


@app.cell
def _(fig, mo):
    mo.ui.plotly(fig)
    return


@app.cell
def _(mo):
    mo.md(
        """
    ## 6. Topic Similarity Heatmap

    This heatmap shows how similar topics are to each other based on their keyword distributions.
    Darker colors indicate higher similarity. 
    This helps identify:

    - Topic clusters that might be merged
    - Truly distinct topic areas
    - The overall thematic structure of your data
    """
    )
    return


@app.cell
def _(mo, topic_model):
    mo.ui.plotly(topic_model.visualize_heatmap())
    return


@app.cell
def _(mo):
    mo.md(
        """
    ## 7. Topics by Label Analysis

    Since our dataset includes labels, we can analyze how topics are distributed across 
    different categories. 
    This reveals:

    - Which topics are specific to certain labels
    - Which topics appear across multiple labels
    - The characteristic themes of each label category
    """
    )
    return


@app.cell
def _(input_prompt, label, topic_model):
    topics_per_class = topic_model.topics_per_class(input_prompt, classes=label)
    return (topics_per_class,)


@app.cell
def _(mo, topic_model, topics_per_class):
    mo.ui.plotly(topic_model.visualize_topics_per_class(topics_per_class))
    return


@app.cell
def _(mo):
    mo.md(
        """
    ## 8. Document Clustering Visualization

    This scatter plot shows all prompts in 2D space, colored by their assigned topic.
    Points that are close together have similar semantic content. 
    This visualization helps:

    - Identify outliers and edge cases
    - See topic boundaries and overlaps
    - Understand the overall structure of your prompt space

    **Note**: If this cell errors, we'll need to compute the reduced embeddings first.
    """
    )
    return


@app.cell
def _(UMAP, embedding_model, input_prompt, mo, topic_model, topics):
    # First, we need to get the embeddings and reduce them to 2D
    embeddings = embedding_model.encode(input_prompt)

    # Use UMAP to reduce to 2D for visualization
    umap_2d = UMAP(n_components=2, n_neighbors=15, min_dist=0.1, random_state=42)
    reduced_embeddings_2d = umap_2d.fit_transform(embeddings)

    # Create the visualization
    fig_docs = topic_model.visualize_documents(
        input_prompt,
        topics=topics,
        reduced_embeddings=reduced_embeddings_2d,
        sample=0.5 if len(input_prompt) > 2000 else 1.0,  # Sample if dataset is large
    )

    mo.ui.plotly(fig_docs)
    return


@app.cell
def _(mo):
    mo.md(
        """
    ## 9. Topic Details

    Let's examine the top topics in more detail, including their keywords and representative examples.
    """
    )
    return


@app.cell
def _(mo, topic_info, topic_model):
    # Create a formatted view of top topics
    top_topics_md = "### Top 10 Topics by Frequency\n\n"

    for _, row in topic_info.head(11).iterrows():
        if row["Topic"] != -1:  # Skip outlier topic
            topic_num = row["Topic"]
            count = row["Count"]
            name = row["Name"]

            # Get representative docs
            rep_docs = topic_model.get_representative_docs(topic_num)

            top_topics_md += f"**Topic {topic_num}** ({count} documents): *{name}*\n"
            top_topics_md += "Representative examples:\n"
            for i, doc in enumerate(rep_docs[:2], 1):
                top_topics_md += f"- {doc[:150]}...\n"
            top_topics_md += "\n"

    mo.md(top_topics_md)
    return


if __name__ == "__main__":
    app.run()
