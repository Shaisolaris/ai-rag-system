"""FastAPI for the RAG system."""
from __future__ import annotations
from fastapi import FastAPI
from pydantic import BaseModel, Field
from rag.pipeline import RAGPipeline

app = FastAPI(title="RAG System API", version="1.0.0")
_pipeline = RAGPipeline()

class IngestRequest(BaseModel):
    text: str = Field(..., min_length=1)
    source: str = "upload"
    chunk_size: int = 500

class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1)
    top_k: int = 5

@app.get("/health")
def health():
    return {"status": "healthy", "documents": _pipeline.document_count}

@app.post("/ingest")
async def ingest(req: IngestRequest):
    count = await _pipeline.ingest(req.text, source=req.source, chunk_size=req.chunk_size)
    return {"chunks_ingested": count, "total_documents": _pipeline.document_count}

@app.post("/query")
async def query(req: QueryRequest):
    return await _pipeline.query(req.question, top_k=req.top_k)

@app.post("/retrieve")
async def retrieve(req: QueryRequest):
    results = await _pipeline.retrieve(req.question, top_k=req.top_k)
    return {"results": [{"text": doc.text[:300], "score": round(s, 4), "metadata": doc.metadata} for doc, s in results]}

@app.delete("/documents")
def clear():
    _pipeline.store.clear()
    return {"cleared": True}
