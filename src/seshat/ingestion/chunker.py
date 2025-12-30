from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)
from langchain_core.documents import Document

from seshat.config import settings


class SemanticChunker:
    """
    Chunking em 2 estágios para Markdown:
    1. Divide por headers (mantém contexto semântico)
    2. Subdivide chunks grandes (respeita limite de tokens)
    """

    def __init__(self, chunk_size: int = settings.chunk_size, chunk_overlap: int = settings.chunk_overlap):
        # Estágio 1: Divisão por estrutura Markdown
        self.header_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=[
                ("#", "header_1"),
                ("##", "header_2"),
                ("###", "header_3"),
            ],
            strip_headers=False,  # Mantém headers no conteúdo
        )

        # Estágio 2: Divisão por tamanho
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
            length_function=len,
        )

    def chunk_document(self, content: str, source: str) -> list[Document]:
        """Processa um documento em chunks otimizados."""
        # Estágio 1: Divide por headers
        header_splits = self.header_splitter.split_text(content)

        # Estágio 2: Subdivide se necessário
        final_chunks = []
        for doc in header_splits:
            if len(doc.page_content) > settings.chunk_size:
                # Subdivide mantendo metadados
                sub_chunks = self.text_splitter.split_text(doc.page_content)
                for i, chunk in enumerate(sub_chunks):
                    final_chunks.append(
                        Document(
                            page_content=chunk,
                            metadata={**doc.metadata, "source": source, "chunk_index": i},
                        )
                    )
            else:
                doc.metadata["source"] = source
                final_chunks.append(doc)

        return final_chunks

    def process_all(self, documents: list[dict]) -> list[Document]:
        """Processa múltiplos documentos."""
        all_chunks = []
        for doc in documents:
            chunks = self.chunk_document(doc["content"], doc["source"])
            all_chunks.extend(chunks)
            print(f"📑 {doc['source']}: {len(chunks)} chunks criados")

        print(f"\n📊 Total: {len(all_chunks)} chunks")
        return all_chunks