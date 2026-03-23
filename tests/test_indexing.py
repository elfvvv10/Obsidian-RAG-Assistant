"""Tests for incremental indexing and retrieval filters."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from config import AppConfig
from main import run_index
from retriever import Retriever
from utils import RetrievalFilters, RetrievalOptions, RetrievedChunk
from vector_store import VectorStore


def make_config(root: Path) -> AppConfig:
    return AppConfig(
        obsidian_vault_path=root / "vault",
        obsidian_output_path=root / "output",
        chroma_db_path=root / "chroma",
        ollama_base_url="http://localhost:11434",
        ollama_chat_model="hermes3",
        ollama_embedding_model="nomic-embed-text",
        top_k_results=3,
    )


class IncrementalIndexingTests(unittest.TestCase):
    def test_index_skips_unchanged_notes_and_updates_changed_ones(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            vault = root / "vault"
            vault.mkdir()
            (root / "output").mkdir()
            note_path = vault / "agents.md"
            note_path.write_text("# Agents\n\nInitial content", encoding="utf-8")
            config = make_config(root)

            embed_calls: list[list[str]] = []

            def fake_embed_texts(texts: list[str]) -> list[list[float]]:
                embed_calls.append(list(texts))
                return [[float(index + 1), 0.0, 0.0] for index, _ in enumerate(texts)]

            with patch("main.OllamaEmbeddingClient.embed_texts", side_effect=fake_embed_texts):
                run_index(config, reset_store=True)
                first_count = VectorStore(config).count()
                run_index(config, reset_store=False)
                second_count = VectorStore(config).count()

                note_path.write_text("# Agents\n\nUpdated content", encoding="utf-8")
                run_index(config, reset_store=False)
                third_count = VectorStore(config).count()

                note_path.unlink()
                run_index(config, reset_store=False)
                fourth_count = VectorStore(config).count()

            self.assertEqual(first_count, 1)
            self.assertEqual(second_count, 1)
            self.assertEqual(third_count, 1)
            self.assertEqual(fourth_count, 0)
            self.assertEqual(len(embed_calls), 2)

    def test_vector_store_applies_folder_and_path_filters(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            (root / "vault").mkdir()
            (root / "output").mkdir()
            config = make_config(root)
            store = VectorStore(config)
            store.reset()

            from chunker import chunk_notes
            from utils import Note

            notes = [
                Note(path="projects/agents.md", title="Agents", content="# Agents\n\nAgent systems use tools."),
                Note(path="ideas/notes.md", title="Ideas", content="# Ideas\n\nGeneral brainstorming."),
            ]
            chunks = chunk_notes(notes, chunk_size=1000, overlap=100)
            embeddings = [[1.0, 0.0], [0.0, 1.0]]
            store.upsert_chunks(chunks, embeddings)

            folder_results = store.query([1.0, 0.0], 3, filters=RetrievalFilters(folder="projects"))
            path_results = store.query([1.0, 0.0], 3, filters=RetrievalFilters(path_contains="agents"))

            self.assertEqual(len(folder_results), 1)
            self.assertEqual(folder_results[0].metadata["source_dir"], "projects")
            self.assertEqual(len(path_results), 1)
            self.assertEqual(path_results[0].metadata["source_path"], "projects/agents.md")

    def test_retriever_uses_candidate_count_and_final_top_k(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            (root / "vault").mkdir()
            (root / "output").mkdir()
            config = make_config(root)

            class StubEmbeddingClient:
                def embed_text(self, text: str) -> list[float]:
                    return [1.0, 0.0]

            class StubVectorStore:
                def __init__(self) -> None:
                    self.requested_top_k = None

                def count(self) -> int:
                    return 3

                def query(self, query_embedding: list[float], top_k: int, filters=None) -> list[RetrievedChunk]:
                    self.requested_top_k = top_k
                    return [
                        RetrievedChunk("one", {"note_title": "One", "source_path": "one.md"}, 0.1),
                        RetrievedChunk("two", {"note_title": "Two", "source_path": "two.md"}, 0.2),
                        RetrievedChunk("three", {"note_title": "Three", "source_path": "three.md"}, 0.3),
                    ]

            store = StubVectorStore()
            retriever = Retriever(config, StubEmbeddingClient(), store)

            results = retriever.retrieve(
                "question",
                options=RetrievalOptions(top_k=2, candidate_count=3, rerank=False),
            )

            self.assertEqual(store.requested_top_k, 3)
            self.assertEqual(len(results), 2)

    def test_retriever_reranks_when_enabled(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            (root / "vault").mkdir()
            (root / "output").mkdir()
            config = make_config(root)

            class StubEmbeddingClient:
                def embed_text(self, text: str) -> list[float]:
                    return [1.0, 0.0]

            class StubVectorStore:
                def count(self) -> int:
                    return 2

                def query(self, query_embedding: list[float], top_k: int, filters=None) -> list[RetrievedChunk]:
                    return [
                        RetrievedChunk("general planning note", {"note_title": "Planning", "source_path": "planning.md"}, 0.05),
                        RetrievedChunk("ai agents use retrieval tools", {"note_title": "Agents", "source_path": "agents.md"}, 0.25),
                    ]

            retriever = Retriever(config, StubEmbeddingClient(), StubVectorStore())

            results = retriever.retrieve(
                "How do AI agents use retrieval tools?",
                options=RetrievalOptions(top_k=1, candidate_count=2, rerank=True),
            )

            self.assertEqual(len(results), 1)
            self.assertEqual(results[0].metadata["note_title"], "Agents")
