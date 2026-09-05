from __future__ import annotations

import re


def normalize_transcript(text: str) -> str:
    """Normalize user-provided meeting text without destroying speaker lines."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def chunk_text(text: str, chunk_size: int = 2600, overlap: int = 300) -> list[str]:
    """Lightweight paragraph-aware chunking for transcript retrieval."""
    text = normalize_transcript(text)
    if not text:
        return []
    if len(text) <= chunk_size:
        return [text]

    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks: list[str] = []
    current = ""

    for paragraph in paragraphs:
        candidate = f"{current}\n\n{paragraph}".strip() if current else paragraph
        if len(candidate) <= chunk_size:
            current = candidate
            continue

        if current:
            chunks.append(current)
            tail = current[-overlap:] if overlap else ""
            current = f"{tail}\n\n{paragraph}".strip()
        else:
            # One paragraph is larger than the target; split it directly.
            start = 0
            step = max(1, chunk_size - overlap)
            while start < len(paragraph):
                chunks.append(paragraph[start : start + chunk_size])
                start += step
            current = ""

    if current:
        chunks.append(current)

    return [c for c in chunks if c.strip()]
