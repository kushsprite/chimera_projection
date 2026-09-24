"""Word-based overlapping chunks (same approach as 07_llm_explainer)."""
from __future__ import annotations

from dataclasses import dataclass

from .sources import Document


@dataclass
class Chunk:
    text: str
    doc: Document
    index: int

    @property
    def chunk_id(self) -> str:
        return f"{self.doc.doc_id}:{self.index}"


def chunk_text(text: str, chunk_size: int = 300, overlap: int = 50, min_chunk_words: int = 50) -> list[str]:
    """Split into overlapping chunks.

    Stops as soon as a chunk reaches the end of the text, so there is never a
    final chunk made only of overlap (the one-line stub from notebook 07). A
    small trailing piece is merged into the previous chunk.
    """
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")
    words = text.split()
    chunks: list[str] = []
    start = 0
    while start < len(words):
        piece = words[start:start + chunk_size]
        if len(piece) < min_chunk_words and chunks:
            chunks[-1] = chunks[-1] + " " + " ".join(piece[overlap:])  # skip words already in the previous chunk
            break
        chunks.append(" ".join(piece))
        if start + chunk_size >= len(words):
            break
        start += chunk_size - overlap
    return chunks


def chunk_documents(docs: list[Document], chunk_size: int = 300, overlap: int = 50,
                    min_chunk_words: int = 50) -> list[Chunk]:
    out = []
    for d in docs:
        for i, t in enumerate(chunk_text(d.text, chunk_size, overlap, min_chunk_words)):
            out.append(Chunk(text=t, doc=d, index=i))
    return out
