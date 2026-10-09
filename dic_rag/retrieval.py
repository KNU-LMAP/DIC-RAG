"""Dense retrieval: sentence-transformers embeddings + exact FAISS index (IndexFlatL2)."""

from typing import List, Tuple

import numpy as np

from .data import ContentChunk

DEFAULT_EMBEDDING_MODEL = "paraphrase-multilingual-mpnet-base-v2"


class VectorStore:
    def __init__(self, embedding_model: str = DEFAULT_EMBEDDING_MODEL, batch_size: int = 50, encoder=None):
        """`encoder` may be any object with `.encode(list_of_str, ...)` (used for testing)."""
        if encoder is None:
            from sentence_transformers import SentenceTransformer

            encoder = SentenceTransformer(embedding_model)
        self.embedding_model = encoder
        self.batch_size = batch_size
        self.index = None
        self.chunks: List[ContentChunk] = []
        self.dimension = None

    def build_index(self, chunks: List[ContentChunk]) -> None:
        import faiss

        if not chunks:
            raise ValueError("No chunks provided")
        n_batches = (len(chunks) + self.batch_size - 1) // self.batch_size
        for b in range(n_batches):
            batch = chunks[b * self.batch_size:(b + 1) * self.batch_size]
            texts = [c.content if c.content_type == "text" else c.description for c in batch]
            embeddings = self.embedding_model.encode(texts, show_progress_bar=False, batch_size=32)
            for chunk, emb in zip(batch, embeddings):
                chunk.embedding = emb
            matrix = np.array([c.embedding for c in batch]).astype("float32")
            if self.index is None:
                self.dimension = matrix.shape[1]
                self.index = faiss.IndexFlatL2(self.dimension)
            self.index.add(matrix)
            self.chunks.extend(batch)
        n_text = sum(c.content_type == "text" for c in self.chunks)
        print(f"index built: {len(self.chunks)} chunks ({n_text} text, {len(self.chunks) - n_text} table)")

    def search(self, query: str, k: int = 20) -> List[Tuple[ContentChunk, float]]:
        """Top-k chunks, best first. score = 1 / (1 + distance) in (0, 1]."""
        if self.index is None:
            raise ValueError("Index not built. Call build_index first.")
        q = np.array([self.embedding_model.encode([query])[0].astype("float32")])
        distances, indices = self.index.search(q, k)
        return [(self.chunks[i], float(1.0 / (1.0 + d))) for i, d in zip(indices[0], distances[0]) if i >= 0]
