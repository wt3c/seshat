from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # API Keys
    ANTHROPODS_API_KEY: str
    # voyageai_api_key: str | None = None
    openai_api_key: str | None = None

    # Diretórios
    chroma_persist_dir: Path = Path("./vectorstore")
    pdf_input_dir: Path = Path("./data/pdfs")
    processed_output_dir: Path = Path("./data/processed")

    # Modelos
    claude_model: str = "claude-sonnet-4-20250514"
    embedding_model: str = "voyage-3-lite"

    # Chunking
    chunk_size: int = 1000
    chunk_overlap: int = 200

    def ensure_directories(self) -> None:
        """Cria os diretórios necessários se não existirem."""
        for dir_path in [
            self.chroma_persist_dir,
            self.pdf_input_dir,
            self.processed_output_dir,
        ]:
            dir_path.mkdir(parents=True, exist_ok=True)


settings = Settings()
settings.ensure_directories()
