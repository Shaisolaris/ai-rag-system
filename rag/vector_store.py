"""In-memory vector store with cosine similarity search."""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import Any

@dataclass
class Document:
    id: str
    text: str
    embedding: list[float]
    metadata: dict[str, Any] = field(default_factory=dict)

@dataclass
class SearchResult:
    document: Document
    score: float

class VectorStore:
    def __init__(self):
        self._documents: dict[str, Document] = {}

    def add(self, doc: Document) -> None:
        self._documents[doc.id] = doc

    def add_many(self, docs: list[Document]) -> int:
        for doc in docs:
            self._documents[doc.id] = doc
        return len(docs)

    def search(self, query_embedding: list[float], top_k: int = 5, min_score: float = 0.0) -> list[SearchResult]:
        if not self._documents: return []
        q = np.array(query_embedding)
        q_norm = np.linalg.norm(q)
        if q_norm == 0: return []
        results = []
        for doc in self._documents.values():
            d = np.array(doc.embedding)
            d_norm = np.linalg.norm(d)
            if d_norm == 0: continue
            score = float(np.dot(q, d) / (q_norm * d_norm))
            if score >= min_score:
                results.append(SearchResult(document=doc, score=score))
        results.sort(key=lambda r: r.score, reverse=True)
        return results[:top_k]

    def delete(self, doc_id: str) -> bool:
        return self._documents.pop(doc_id, None) is not None

    @property
    def count(self) -> int:
        return len(self._documents)
