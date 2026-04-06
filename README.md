# ai-rag-system

![CI](https://github.com/Shaisolaris/ai-rag-system/actions/workflows/ci.yml/badge.svg)

Retrieval-Augmented Generation system with document chunking (token-based, paragraph-based, header-based), OpenAI embeddings, in-memory vector store with cosine similarity search, and GPT-4o answer generation with source attribution. FastAPI serving layer.

## Stack
- **AI:** OpenAI (text-embedding-3-small, GPT-4o)
- **Search:** In-memory vector store with numpy cosine similarity
- **API:** FastAPI

## Pipeline: Ingest → Embed → Retrieve → Generate

## API Endpoints
| Method | Endpoint | Description |
|---|---|---|
| POST | `/ingest` | Chunk and embed a document |
| POST | `/query` | RAG: retrieve + generate answer with sources |
| POST | `/retrieve` | Retrieve similar chunks without generation |
| GET | `/health` | Status + document count |

## Architecture
```
rag/chunker.py       — 3 chunking strategies (tokens, paragraphs, headers)
rag/vector_store.py  — In-memory store with cosine similarity search
rag/pipeline.py      — RAGPipeline: ingest, retrieve, generate
api/app.py           — FastAPI endpoints
```

## Setup
```bash
git clone https://github.com/Shaisolaris/ai-rag-system.git
cd ai-rag-system && pip install -r requirements.txt
export OPENAI_API_KEY=sk-...
python main.py
```

## License
MIT

## Architecture

```
.editorconfig
.env.example
.github/workflows/ci.yml
.gitignore
Dockerfile
LICENSE
README.md
api/__init__.py
api/app.py
main.py
rag/__init__.py
rag/chunker.py
rag/pipeline.py
rag/vector_store.py
requirements.txt
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Open a Pull Request

## License

MIT
