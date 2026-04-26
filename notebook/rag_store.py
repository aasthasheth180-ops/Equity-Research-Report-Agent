# rag_store.py — owned by Person D
#
# Interface everyone relies on (import rag_store, then call the singleton):
#   rag_store.ingest_pdf(path)           -> int   (chunk count)
#   rag_store.ingest_text(text)          -> int
#   rag_store.ingest_dataframe(df, desc) -> int
#   rag_store.retrieve(query, k)         -> list[str]
#   rag_store.size                       -> int

import numpy as np
import faiss
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer


class RAGStore:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.embedder = SentenceTransformer(model_name)
        self.index: faiss.IndexFlatL2 | None = None
        self.chunks: list[str] = []

    @staticmethod
    def _chunk(text: str, size: int = 400, overlap: int = 50) -> list[str]:
        words = text.split()
        return [
            " ".join(words[i : i + size])
            for i in range(0, len(words), size - overlap)
            if words[i : i + size]
        ]

    def _add(self, chunks: list[str]) -> None:
        if not chunks:
            return
        vecs = self.embedder.encode(chunks, show_progress_bar=False).astype("float32")
        if self.index is None:
            self.index = faiss.IndexFlatL2(vecs.shape[1])
        self.index.add(vecs)
        self.chunks.extend(chunks)

    def ingest_pdf(self, path: str) -> int:
        reader = PdfReader(path)
        text = "\n".join(page.extract_text() or "" for page in reader.pages)
        chunks = self._chunk(text)
        self._add(chunks)
        return len(chunks)

    def ingest_text(self, text: str) -> int:
        chunks = self._chunk(text)
        self._add(chunks)
        return len(chunks)

    def ingest_dataframe(self, df, description: str = "") -> int:
        lines = [f"Market Risk Data — {description}" if description else "Market Risk Data"]
        for idx, row in df.iterrows():
            lines.append(f"  {idx}: {row.get('Amount $millions', '')}  (source: {row.get('Source', '')})")
        self._add(["\n".join(lines)])
        return 1

    def retrieve(self, query: str, k: int = 5) -> list[str]:
        if self.index is None or not self.chunks:
            return []
        q = self.embedder.encode([query]).astype("float32")
        k = min(k, len(self.chunks))
        _, idxs = self.index.search(q, k)
        return [self.chunks[i] for i in idxs[0] if 0 <= i < len(self.chunks)]

    @property
    def size(self) -> int:
        return len(self.chunks)


rag_store = RAGStore()
