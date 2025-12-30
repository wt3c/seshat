from pathlib import Path
from langchain_chroma import Chroma
from langchain_core.documents import Document

from seshat.config import settings
from seshat.retrieval.embeddings import get_embeddings


class VectorStoreManager:
    """Gerencia o ChromaDB para armazenamento e busca de vetores."""

    COLLECTION_NAME = "seshat_documents"

    def __init__(self):
        self.embeddings = get_embeddings()
        self._vector_store: Chroma | None = None

    @property
    def vector_store(self) -> Chroma:
        """Lazy loading do vector store."""
        if self._vector_store is None:
            self._vector_store = Chroma(
                collection_name=self.COLLECTION_NAME,
                embedding_function=self.embeddings,
                persist_directory=str(settings.chroma_persist_dir),
            )
        return self._vector_store

    def add_documents(self, documents: list[Document]) -> None:
        """Adiciona documentos ao vector store."""
        print(f"🔄 Gerando embeddings para {len(documents)} chunks...")

        # Adiciona em batches para evitar timeout
        batch_size = 100
        for i in range(0, len(documents), batch_size):
            batch = documents[i : i + batch_size]
            self.vector_store.add_documents(batch)
            print(f"   ✅ Batch {i // batch_size + 1} concluído")

        print("💾 Vector store persistido com sucesso!")

    def similarity_search(self, query: str, k: int = 4, filter_dict: dict | None = None) -> list[Document]:
        """Busca documentos similares à query."""
        return self.vector_store.similarity_search(query=query, k=k, filter=filter_dict)

    def as_retriever(self, search_kwargs: dict | None = None):
        """Retorna um retriever para uso em chains."""
        return self.vector_store.as_retriever(search_type="similarity", search_kwargs=search_kwargs or {"k": 4})

    def get_collection_stats(self) -> dict:
        """Retorna estatísticas da coleção."""
        collection = self.vector_store._collection
        return {"name": collection.name, "count": collection.count()}

    def clear_collection(self) -> None:
        """Limpa toda a coleção (use com cuidado!)."""
        self.vector_store.delete_collection()
        self._vector_store = None
        print("🗑️ Coleção removida.")
