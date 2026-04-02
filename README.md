# ai-rag-system

Retrieval-Augmented Generation system with document chunking (character, paragraph, header-based), OpenAI embeddings, in-memory vector store with cosine similarity search, metadata filtering, and a query pipeline that retrieves context and generates answers via GPT-4o. FastAPI serving layer.

## Stack

- **AI:** OpenAI (text-embedding-3-small, GPT-4o)
- **API:** FastAPI
- **Vector Store:** In-memory with numpy cosine similarity

## Pipeline

```
Document → Chunker → Embeddings → Vector Store
                                       ↑
Query → Embedding → Similarity Search → Context → GPT-4o → Answer
```

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| POST | `/ingest` | Ingest text: chunk, embed, store |
| POST | `/query` | RAG query: retrieve + generate answer |
| POST | `/retrieve` | Retrieve similar chunks only |
| DELETE | `/documents` | Clear vector store |
| GET | `/health` | Status + document count |

## Chunking Strategies

- **Character-based:** Fixed size with overlap, sentence boundary detection
- **Paragraph-based:** Split on double newlines, merge small paragraphs
- **Header-based:** Split markdown by h1-h3 headers with level metadata

## Setup

```bash
git clone https://github.com/Shaisolaris/ai-rag-system.git
cd ai-rag-system
pip install -r requirements.txt
export OPENAI_API_KEY=sk-...
python main.py
```

## License

MIT
