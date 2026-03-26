# rag_store.py — owned by Person D
#
# Responsibilities:
#   1. Chunk text from PDFs or raw strings
#   2. Embed with sentence-transformers (all-MiniLM-L6-v2)
#   3. Store and search with FAISS
#   4. Convert regulatory DataFrames into searchable text chunks
#
# Interface everyone relies on (import rag_store, then call the singleton):
#   rag_store.ingest_pdf(path)          -> int   (chunk count)
#   rag_store.ingest_text(text)         -> int
#   rag_store.ingest_dataframe(df, desc)-> int
#   rag_store.retrieve(query, k)        -> List[str]
#   rag_store.size                      -> int

from typing import List, Optional
import numpy as np
import faiss
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer


class RAGStore:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.embedder = SentenceTransformer(model_name)
        self.index: Optional[faiss.IndexFlatL2] = None
        self.chunks: List[str] = []

    # ── Chunking ──────────────────────────────────────────────────────────────

    def _chunk(self, text: str, size: int = 400, overlap: int = 50) -> List[str]:
        """
        Word-level sliding window.
        Analogy: overlapping Post-it notes so no sentence gets cut at the edge.
        """
        words = text.split()
        return [
            " ".join(words[i : i + size])
            for i in range(0, len(words), size - overlap)
            if words[i : i + size]
        ]

    # ── Embedding + indexing ──────────────────────────────────────────────────

    def _add(self, chunks: List[str]) -> None:
        if not chunks:
            return
        vecs = self.embedder.encode(chunks, show_progress_bar=False).astype("float32")
        if self.index is None:
            self.index = faiss.IndexFlatL2(vecs.shape[1])
        self.index.add(vecs)
        self.chunks.extend(chunks)

    # ── Public ingest API ─────────────────────────────────────────────────────

    def ingest_pdf(self, path: str) -> int:
        """Extract text from a PDF and add to the index."""
        reader = PdfReader(path)
        text   = "\n".join(page.extract_text() or "" for page in reader.pages)
        chunks = self._chunk(text)
        self._add(chunks)
        return len(chunks)

    def ingest_text(self, text: str) -> int:
        """Add raw text (10-K excerpt, press release, etc.) to the index."""
        chunks = self._chunk(text)
        self._add(chunks)
        return len(chunks)

    def ingest_dataframe(self, df, description: str = "") -> int:
        """
        Convert a market_risk_df into a single dense text chunk.
        Analogy: transcribing a spreadsheet into a memo the LLM can search.
        Expected df: index=metric name, columns include 'Amount $millions', 'Source'.
        """
        lines = [f"Market Risk Data — {description}"] if description else ["Market Risk Data"]
        for idx, row in df.iterrows():
            val = row.get("Amount $millions", "")
            src = row.get("Source", "")
            lines.append(f"  {idx}: {val}  (source: {src})")
        self._add(["\n".join(lines)])
        return 1

    # ── Public retrieval API ──────────────────────────────────────────────────

    def retrieve(self, query: str, k: int = 5) -> List[str]:
        """Return the k most relevant chunks for a query."""
        if self.index is None or not self.chunks:
            return []
        q   = self.embedder.encode([query]).astype("float32")
        k   = min(k, len(self.chunks))
        _, idxs = self.index.search(q, k)
        return [self.chunks[i] for i in idxs[0] if 0 <= i < len(self.chunks)]

    @property
    def size(self) -> int:
        return len(self.chunks)


# Module-level singleton — import this everywhere
rag_store = RAGStore()
