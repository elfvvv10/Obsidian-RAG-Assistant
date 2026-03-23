"""Tests for Phase 1 retrieval improvements."""

from __future__ import annotations

import unittest

from llm import build_prompt
from reranker import rerank_chunks
from utils import RetrievedChunk


class RerankerTests(unittest.TestCase):
    def test_reranker_promotes_lexically_relevant_chunk(self) -> None:
        chunks = [
            RetrievedChunk(
                text="General note about planning.",
                metadata={"note_title": "Planning", "source_path": "planning.md"},
                distance_or_score=0.05,
            ),
            RetrievedChunk(
                text="AI agents use tools and retrieval for grounded answers.",
                metadata={"note_title": "Agents", "source_path": "agents.md"},
                distance_or_score=0.25,
            ),
        ]

        reranked = rerank_chunks("How do AI agents use retrieval tools?", chunks)

        self.assertEqual(reranked[0].metadata["note_title"], "Agents")


class PromptFormattingTests(unittest.TestCase):
    def test_build_prompt_includes_structured_context(self) -> None:
        chunks = [
            RetrievedChunk(
                text="Agents use retrieval to ground answers.",
                metadata={
                    "note_title": "Agents",
                    "source_path": "notes/agents.md",
                    "heading_context": "Retrieval",
                },
                distance_or_score=0.12345,
            )
        ]

        prompt = build_prompt("What do my notes say about agents?", chunks)

        self.assertIn("[Source 1]", prompt)
        self.assertIn("Title: Agents | Section: Retrieval", prompt)
        self.assertIn("Path: notes/agents.md", prompt)
        self.assertIn("Relevance distance: 0.1235", prompt)
        self.assertIn("Content:", prompt)
