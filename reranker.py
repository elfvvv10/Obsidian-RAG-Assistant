"""Lightweight local reranking helpers."""

from __future__ import annotations

import re

from utils import RetrievedChunk


def rerank_chunks(query: str, chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
    """Rerank retrieved chunks with a simple lexical overlap heuristic."""
    if not chunks:
        return []

    query_terms = set(_tokenize(query))

    def score(chunk: RetrievedChunk) -> float:
        chunk_terms = set(_tokenize(chunk.text))
        overlap = len(query_terms & chunk_terms)
        distance = chunk.distance_or_score if chunk.distance_or_score is not None else 1.0
        similarity = max(0.0, 1.0 - distance)
        return overlap * 2.0 + similarity

    return sorted(chunks, key=score, reverse=True)


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())
