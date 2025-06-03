from typing import List, Optional
from langchain_openai import AzureOpenAIEmbeddings
from deepeval.models import DeepEvalBaseEmbeddingModel


class CustomEmbeddingModel(DeepEvalBaseEmbeddingModel):
    def __init__(self):
        pass

    def load_model(self):
        return AzureOpenAIEmbeddings(
            openai_api_version="...",
            azure_deployment="...",
            azure_endpoint="...",
            openai_api_key="...",
        )

    def embed_text(self, text: str) -> List[float]:
        embedding_model = self.load_model()
        return embedding_model.embed_query(text)

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        embedding_model = self.load_model()
        return embedding_model.embed_documents(texts)

    async def a_embed_text(self, text: str) -> List[float]:
        embedding_model = self.load_model()
        return await embedding_model.aembed_query(text)

    async def a_embed_texts(self, texts: List[str]) -> List[List[float]]:
        embedding_model = self.load_model()
        return await embedding_model.aembed_documents(texts)

    def get_model_name(self):
        "Custom Azure Embedding Model"
