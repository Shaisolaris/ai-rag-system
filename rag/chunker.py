"""Document chunking strategies for RAG."""
from __future__ import annotations
import re
from dataclasses import dataclass

@dataclass
class Chunk:
    text: str
    index: int
    metadata: dict

def chunk_by_tokens(text: str, max_tokens: int = 500, overlap: int = 50) -> list[Chunk]:
    words = text.split()
    chunks = []
    step = max(1, max_tokens - overlap)
    for i in range(0, len(words), step):
        chunk_words = words[i:i + max_tokens]
        if len(chunk_words) < 20 and i > 0: break
        chunks.append(Chunk(text=" ".join(chunk_words), index=len(chunks), metadata={"start_word": i, "word_count": len(chunk_words)}))
    return chunks

def chunk_by_paragraphs(text: str, max_chars: int = 2000) -> list[Chunk]:
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks, current, idx = [], "", 0
    for para in paragraphs:
        if len(current) + len(para) > max_chars and current:
            chunks.append(Chunk(text=current.strip(), index=idx, metadata={"type": "paragraph"}))
            current, idx = "", idx + 1
        current += para + "\n\n"
    if current.strip():
        chunks.append(Chunk(text=current.strip(), index=idx, metadata={"type": "paragraph"}))
    return chunks

def chunk_by_headers(text: str) -> list[Chunk]:
    sections = re.split(r'\n(#{1,3}\s+.+)', text)
    chunks, current_header, current_text, idx = [], "", "", 0
    for section in sections:
        if re.match(r'^#{1,3}\s+', section):
            if current_text.strip():
                chunks.append(Chunk(text=f"{current_header}\n{current_text}".strip(), index=idx, metadata={"header": current_header.strip()}))
                idx += 1
            current_header = section
            current_text = ""
        else:
            current_text += section
    if current_text.strip():
        chunks.append(Chunk(text=f"{current_header}\n{current_text}".strip(), index=idx, metadata={"header": current_header.strip()}))
    return chunks
