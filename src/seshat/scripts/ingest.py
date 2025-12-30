#!/usr/bin/env python
"""Script CLI para ingestão de documentos."""

import click
from seshat.ingestion.pdf_processor import PDFProcessor
from seshat.ingestion.chunker import SemanticChunker
from seshat.retrieval.vector_store import VectorStoreManager


@click.command()
@click.option("--clear", is_flag=True, help="Limpa o vector store antes de indexar")
def ingest(clear: bool):
    """Processa PDFs e indexa no vector store."""
    vector_store = VectorStoreManager()

    if clear:
        click.echo("🗑️ Limpando vector store...")
        vector_store.clear_collection()

    click.echo("📄 Processando PDFs...")
    processor = PDFProcessor()
    docs = processor.process_directory()

    if not docs:
        click.echo("❌ Nenhum documento encontrado!")
        return

    click.echo("✂️ Criando chunks...")
    chunker = SemanticChunker()
    chunks = chunker.process_all(docs)

    click.echo("🔄 Indexando...")
    vector_store.add_documents(chunks)

    click.echo(f"✅ Concluído! {len(chunks)} chunks indexados.")


if __name__ == "__main__":
    ingest()
