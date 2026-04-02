"""RAG pipeline — ingest, embed, retrieve, generate."""
from __future__ import annotations
import logging
from openai import AsyncOpenAI
from rag.chunker import chunk_text
from rag.vector_store import VectorStore, Document

logger = logging.getLogger(__name__)
client = AsyncOpenAI()

class RAGPipeline:
    def __init__(self, embedding_model: str = "text-embedding-3-small", chat_model: str = "gpt-4o") -> None:
        self.store = VectorStore()
        self.embedding_model = embedding_model
        self.chat_model = chat_model

    async def ingest(self, text: str, source: str = "unknown", chunk_size: int = 500) -> int:
        chunks = chunk_text(text, chunk_size=chunk_size, metadata={"source": source})
        if not chunks:
            return 0
        texts = [c.text for c in chunks]
        response = await client.embeddings.create(model=self.embedding_model, input=texts)
        docs = [Document(id=f"{source}_{c.index}", text=c.text, embedding=e.embedding, metadata={**c.metadata, "chunk_index": c.index}) for c, e in zip(chunks, response.data)]
        return self.store.add_many(docs)

    async def retrieve(self, query: str, top_k: int = 5, min_score: float = 0.3) -> list[tuple[Document, float]]:
        response = await client.embeddings.create(model=self.embedding_model, input=query)
        return self.store.search(response.data[0].embedding, top_k=top_k, min_score=min_score)

    async def query(self, question: str, top_k: int = 5) -> dict:
        results = await self.retrieve(question, top_k=top_k)
        if not results:
            return {"answer": "No relevant documents found.", "sources": [], "chunks_used": 0}
        context = "\n\n---\n\n".join([f"[Source: {doc.metadata.get('source', '?')}]\n{doc.text}" for doc, _ in results])
        response = await client.chat.completions.create(
            model=self.chat_model,
            messages=[{"role": "system", "content": f"Answer based on the context. Cite sources.\n\nContext:\n{context}"}, {"role": "user", "content": question}],
            temperature=0,
        )
        return {"answer": response.choices[0].message.content or "", "sources": [{"text": doc.text[:200], "score": round(s, 4), "source": doc.metadata.get("source", "")} for doc, s in results], "chunks_used": len(results)}

    @property
    def document_count(self) -> int:
        return self.store.count()
