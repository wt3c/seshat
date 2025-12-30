from pathlib import Path
from docling.document_converter import DocumentConverter
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import PdfFormatOption

from seshat.config import settings


class PDFProcessor:
    """Processa PDFs e converte para Markdown usando Docling."""

    def __init__(self):
        # Configuração otimizada para PDFs complexos
        pipeline_options = PdfPipelineOptions()
        pipeline_options.do_ocr = True  # Habilita OCR para imagens/scans
        pipeline_options.do_table_structure = True  # Extrai estrutura de tabelas

        self.converter = DocumentConverter(
            format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)}
        )

    def process_single(self, pdf_path: Path) -> str:
        """Converte um único PDF para Markdown."""
        result = self.converter.convert(pdf_path)
        return result.document.export_to_markdown()

    def process_directory(self) -> list[dict]:
        """Processa todos os PDFs do diretório de entrada."""
        processed_docs = []

        for pdf_file in settings.pdf_input_dir.glob("*.pdf"):
            print(f"📄 Processando: {pdf_file.name}")

            try:
                markdown_content = self.process_single(pdf_file)

                # Salva o markdown processado
                output_file = settings.processed_output_dir / f"{pdf_file.stem}.md"
                output_file.write_text(markdown_content, encoding="utf-8")

                processed_docs.append(
                    {
                        "source": pdf_file.name,
                        "content": markdown_content,
                        "output_path": str(),
                    }
                )
                print(f"   ✅ Salvo em: {output_file}")

            except Exception as e:
                print(f"   ❌ Erro: {e}")

        return processed_docs

    def load_processed(self) -> list[dict]:
        """Carrega documentos Markdown já processados."""
        docs = []
        for md_file in settings.processed_output_dir.glob("*.md"):
            docs.append(
                {
                    "source": md_file.stem,
                    "content": md_file.read_text(encoding="utf-8"),
                }
            )
        return docs
