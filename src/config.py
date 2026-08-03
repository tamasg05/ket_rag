"""Configuration loaded from .env, with deliberately conservative defaults."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")


@dataclass(frozen=True)
class Settings:
    """Environment-backed paths, model choices, and indexing parameters."""
    data_file: Path = ROOT / "data" / "TheProjectGutenbergeBookofTheAdventuresofSherlockHolmes.txt"
    cache_dir: Path = ROOT / ".rag_cache"
    url_corpus_dir: Path = Path(
        os.getenv("URL_CORPUS_DIR", str(ROOT / "data" / "url_corpora"))
    )
    google_api_key: str | None = os.getenv("GOOGLE_API_KEY")

    # Bound URL-corpus downloads so one accidental page cannot dominate a build.
    max_url_pages: int = int(os.getenv("MAX_URL_PAGES", "20"))
    url_timeout_seconds: float = float(os.getenv("URL_TIMEOUT_SECONDS", "20"))
    max_url_page_bytes: int = int(os.getenv("MAX_URL_PAGE_BYTES", "5000000"))

    # Stable, inexpensive Google models. Change them in .env without editing code.
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "gemini-embedding-001")
    generation_model: str = os.getenv("GENERATION_MODEL", "gemini-3.1-flash-lite")
    extraction_model: str = os.getenv("EXTRACTION_MODEL", "gemini-3.1-flash-lite")
    embedding_dimensions: int = int(os.getenv("EMBEDDING_DIMENSIONS", "768"))
    embed_batch_size: int = int(os.getenv("EMBED_BATCH_SIZE", "64"))
    # 64 inputs every 1.4s stays below a 3,000-input/minute project quota.
    embed_batch_delay: float = float(os.getenv("EMBED_BATCH_DELAY", "1.4"))

    # Word-based chunks keep the demo easy to understand.
    chunk_words: int = int(os.getenv("CHUNK_WORDS", "450"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "60"))
    extraction_batch_size: int = int(os.getenv("EXTRACTION_BATCH_SIZE", "5"))

    def require_api_key(self) -> str:
        """
        Return the configured Google API key or fail with a clear message.

        Inputs:
            None; the key is read from this settings instance.

        Returns:
            The non-empty Google API key.
        """
        if not self.google_api_key:
            raise RuntimeError("GOOGLE_API_KEY is missing from .env")
        return self.google_api_key
