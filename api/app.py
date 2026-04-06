import os
# Demo mode: runs with sample data when no API keys configured
DEMO_MODE = os.getenv('DEMO_MODE', 'false').lower() == 'true' or not os.getenv('DATABASE_URL')
"""FastAPI for RAG system."""
from __future__ import annotations
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from rag.pipeline import RAGPipeline

app = FastAPI(title="RAG System API", version="1.0.0")
_pipeline = RAGPipeline()

class IngestRequest(BaseModel):
    text: str = Field(..., min_length=1)
    source: str = "document"
    chunk_strategy: str = "tokens"
    max_tokens: int = 500

class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1)
    top_k: int = 5
    system_prompt: str | None = None

@app.get("/health")
def health(): return {"status": "healthy", "documents": _pipeline.store.count}

@app.post("/ingest")
async def ingest(req: IngestRequest):
    count = await _pipeline.ingest(req.text, req.source, req.chunk_strategy, req.max_tokens)
    return {"chunks_ingested": count, "total_documents": _pipeline.store.count}

@app.post("/query")
async def query(req: QueryRequest):
    try:
        result = await _pipeline.generate(req.query, req.top_k, req.system_prompt)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/retrieve")
async def retrieve(req: QueryRequest):
    try:
        results = await _pipeline.retrieve(req.query, req.top_k)
        return {"results": [{"text": r.document.text[:300], "source": r.document.metadata.get("source"), "score": round(r.score, 4)} for r in results]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
