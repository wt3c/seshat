from langchain_core.embeddings import Embeddings
from langchain_voyageai import VoyageAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from seshat.config import settings


def get_embeddings() -> Embeddings:
    """
    Retorna o modelo de embeddings configurado.

    Modelos disponíveis (sentence-transformers):
    - all-MiniLM-L6-v2: Rápido, leve (384 dims) - bom para começar
    - all-mpnet-base-v2: Melhor qualidade (768 dims)
    - multilingual-e5-large: Melhor para português (1024 dims)
    """
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},  # ou "cuda" se tiver GPU
        encode_kwargs={"normalize_embeddings": True},
    )