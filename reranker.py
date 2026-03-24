"""Lightweight local reranking helpers."""

from __future__ import annotations

import re

from utils import RetrievedChunk


def rerank_chunks(
    query: str,
    chunks: list[RetrievedChunk],
    *,
    boost_tags: tuple[str, ...] = (),
    tag_boost_weight: float = 3.0,
) -> list[RetrievedChunk]:
    """Rerank retrieved chunks with a simple lexical overlap heuristic."""
    if not chunks:
        return []

    query_terms = set(_tokenize(query))
    normalized_boost_tags = {tag.lower() for tag in boost_tags if tag}

    def score(chunk: RetrievedChunk) -> float:
        chunk_terms = set(_tokenize(chunk.text))
        overlap = len(query_terms & chunk_terms)
        distance = chunk.distance_or_score if chunk.distance_or_score is not None else 1.0
        similarity = max(0.0, 1.0 - distance)
        metadata_tags = set(_metadata_tags(chunk))
        tag_bonus = len(normalized_boost_tags & metadata_tags) * tag_boost_weight
        source_penalty = 0.25 if _is_saved_answer(chunk) else 0.0
        return overlap * 2.0 + similarity + tag_bonus - source_penalty

    return sorted(chunks, key=score, reverse=True)


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def _metadata_tags(chunk: RetrievedChunk) -> tuple[str, ...]:
    serialized = chunk.metadata.get("tags_serialized", "")
    if not isinstance(serialized, str) or not serialized:
        return ()
    return tuple(part for part in serialized.split("|") if part)


def _is_saved_answer(chunk: RetrievedChunk) -> bool:
    return str(chunk.metadata.get("source_kind", "")).strip().lower() == "saved_answer"
