"""RAG pipeline — ingest, embed, retrieve, generate."""
from __future__ import annotations
import logging
from openai import AsyncOpenAI
from rag.chunker import Chunk, chunk_by_tokens, chunk_by_paragraphs
from rag.vector_store import VectorStore, Document, SearchResult

logger = logging.getLogger(__name__)
client = AsyncOpenAI()

class RAGPipeline:
    def __init__(self, embedding_model: str = "text-embedding-3-small", chat_model: str = "gpt-4o"):
        self.store = VectorStore()
        self.embedding_model = embedding_model
        self.chat_model = chat_model

    async def ingest(self, text: str, source: str = "document", chunk_strategy: str = "tokens", max_tokens: int = 500) -> int:
        chunks = chunk_by_tokens(text, max_tokens) if chunk_strategy == "tokens" else chunk_by_paragraphs(text)
        if not chunks: return 0
        texts = [c.text for c in chunks]
        resp = await client.embeddings.create(model=self.embedding_model, input=texts)
        docs = []
        for chunk, emb_data in zip(chunks, resp.data):
            doc = Document(id=f"{source}_{chunk.index}", text=chunk.text, embedding=emb_data.embedding, metadata={**chunk.metadata, "source": source})
            docs.append(doc)
        count = self.store.add_many(docs)
        logger.info(f"Ingested {count} chunks from '{source}'")
        return count

    async def retrieve(self, query: str, top_k: int = 5) -> list[SearchResult]:
        resp = await client.embeddings.create(model=self.embedding_model, input=query)
        return self.store.search(resp.data[0].embedding, top_k=top_k)

    async def generate(self, query: str, top_k: int = 5, system_prompt: str | None = None) -> dict:
        results = await self.retrieve(query, top_k=top_k)
        context = "\n\n---\n\n".join([f"[Source: {r.document.metadata.get('source', '?')}]\n{r.document.text}" for r in results])
        sys = system_prompt or "Answer based on the provided context. If the context doesn't contain relevant information, say so."
        messages = [
            {"role": "system", "content": f"{sys}\n\nContext:\n{context}"},
            {"role": "user", "content": query},
        ]
        resp = await client.chat.completions.create(model=self.chat_model, messages=messages, temperature=0.3)
        answer = resp.choices[0].message.content or ""
        return {"answer": answer, "sources": [{"text": r.document.text[:200], "source": r.document.metadata.get("source", ""), "score": round(r.score, 4)} for r in results], "chunks_used": len(results)}
