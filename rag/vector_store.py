"""In-memory vector store with cosine similarity search."""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field

@dataclass
class Document:
    id: str
    text: str
    embedding: list[float]
    metadata: dict = field(default_factory=dict)

class VectorStore:
    def __init__(self) -> None:
        self._documents: dict[str, Document] = {}

    def add(self, doc_id: str, text: str, embedding: list[float], metadata: dict | None = None) -> None:
        self._documents[doc_id] = Document(id=doc_id, text=text, embedding=embedding, metadata=metadata or {})

    def add_many(self, documents: list[Document]) -> int:
        for doc in documents:
            self._documents[doc.id] = doc
        return len(documents)

    def search(self, query_embedding: list[float], top_k: int = 5, min_score: float = 0.0, filter_metadata: dict | None = None) -> list[tuple[Document, float]]:
        if not self._documents:
            return []
        q = np.array(query_embedding)
        q_norm = float(np.linalg.norm(q))
        if q_norm == 0:
            return []
        results: list[tuple[Document, float]] = []
        for doc in self._documents.values():
            if filter_metadata and not all(doc.metadata.get(k) == v for k, v in filter_metadata.items()):
                continue
            d = np.array(doc.embedding)
            d_norm = float(np.linalg.norm(d))
            if d_norm == 0:
                continue
            score = float(np.dot(q, d) / (q_norm * d_norm))
            if score >= min_score:
                results.append((doc, score))
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    def delete(self, doc_id: str) -> bool:
        return self._documents.pop(doc_id, None) is not None

    def count(self) -> int:
        return len(self._documents)

    def clear(self) -> None:
        self._documents.clear()
