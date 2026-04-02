"""Document chunking strategies for RAG."""
from __future__ import annotations
import re
from dataclasses import dataclass, field

@dataclass
class Chunk:
    text: str
    index: int
    metadata: dict = field(default_factory=dict)
    start_char: int = 0
    end_char: int = 0

def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50, metadata: dict | None = None) -> list[Chunk]:
    chunks, start, idx = [], 0, 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        if end < len(text):
            last_period = text.rfind('. ', start, end)
            if last_period > start + chunk_size // 2:
                end = last_period + 1
        t = text[start:end].strip()
        if t:
            chunks.append(Chunk(text=t, index=idx, metadata=metadata or {}, start_char=start, end_char=end))
            idx += 1
        start = max(start + 1, end - overlap)
    return chunks

def chunk_by_paragraph(text: str, max_size: int = 1000, metadata: dict | None = None) -> list[Chunk]:
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
    chunks, current, idx = [], "", 0
    for para in paragraphs:
        if len(current) + len(para) + 2 > max_size and current:
            chunks.append(Chunk(text=current.strip(), index=idx, metadata=metadata or {}))
            idx += 1
            current = para
        else:
            current = f"{current}\n\n{para}" if current else para
    if current.strip():
        chunks.append(Chunk(text=current.strip(), index=idx, metadata=metadata or {}))
    return chunks

def chunk_by_headers(text: str, metadata: dict | None = None) -> list[Chunk]:
    sections = re.split(r'\n(?=#{1,3}\s)', text)
    chunks = []
    for i, section in enumerate(sections):
        section = section.strip()
        if not section:
            continue
        meta = {**(metadata or {})}
        m = re.match(r'^(#{1,3})\s+(.+)', section)
        if m:
            meta['header'] = m.group(2)
            meta['level'] = len(m.group(1))
        chunks.append(Chunk(text=section, index=i, metadata=meta))
    return chunks
