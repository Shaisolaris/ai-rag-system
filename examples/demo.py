"""
RAG Demo: Query the sample knowledge base without any API keys.
Uses mock embeddings to demonstrate the retrieval pipeline.
Run: python examples/demo.py
"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

class MockRAG:
    """Demonstrates RAG pipeline with local document chunks."""
    
    def __init__(self, knowledge_base_path: str):
        self.chunks = self._load_and_chunk(knowledge_base_path)
        print(f"📚 Loaded {len(self.chunks)} chunks from knowledge base")
    
    def _load_and_chunk(self, path: str) -> list:
        text = open(path).read()
        sections = text.split("\n## ")
        chunks = []
        for section in sections:
            if section.strip():
                lines = section.strip().split("\n")
                title = lines[0].replace("# ", "")
                content = "\n".join(lines[1:]).strip()
                if content:
                    chunks.append({"title": title, "content": content})
        return chunks
    
    def query(self, question: str) -> dict:
        """Simple keyword-based retrieval (production uses vector similarity)."""
        question_lower = question.lower()
        scored = []
        for chunk in self.chunks:
            score = sum(1 for word in question_lower.split() if word in chunk["content"].lower())
            scored.append((score, chunk))
        
        scored.sort(key=lambda x: x[0], reverse=True)
        top_chunks = [c for _, c in scored[:2] if _ > 0]
        
        if not top_chunks:
            return {"answer": "I don't have information about that in my knowledge base.", "sources": []}
        
        context = "\n".join(c["content"] for c in top_chunks)
        answer = f"Based on the knowledge base: {context[:300]}..."
        
        return {
            "answer": answer,
            "sources": [c["title"] for c in top_chunks],
            "chunks_searched": len(self.chunks),
            "chunks_retrieved": len(top_chunks),
        }

def main():
    print("🔍 RAG System Demo")
    print("=" * 50)
    
    kb_path = os.path.join(os.path.dirname(__file__), "..", "data", "sample_knowledge_base.txt")
    rag = MockRAG(kb_path)
    
    queries = [
        "What pricing plans are available?",
        "What tech stack does the platform use?",
        "How do I fix login issues?",
        "Does it integrate with GitHub?",
        "What is the support email?",
    ]
    
    for q in queries:
        print(f"\n❓ {q}")
        result = rag.query(q)
        print(f"📎 Sources: {result['sources']}")
        print(f"💬 {result['answer'][:200]}")
        print(f"   ({result['chunks_retrieved']}/{result['chunks_searched']} chunks used)")

if __name__ == "__main__":
    main()
