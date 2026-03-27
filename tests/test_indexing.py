"""Tests for incremental indexing and retrieval filters."""

from __future__ import annotations

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from config import AppConfig
from services.index_service import IndexService
from services.models import CollaborationWorkflow, DomainProfile, SessionTask, TrackContext
from services.prompt_service import build_citation_sources
from main import run_ask, run_index
from retriever import Retriever
from utils import Note, RetrievalFilters, RetrievalOptions, RetrievedChunk
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

    def test_arrangement_chunks_preserve_metadata_in_vector_store(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            (root / "vault").mkdir()
            (root / "output").mkdir()
            config = make_config(root)
            store = VectorStore(config)
            store.reset()

            from chunker import chunk_notes
            from utils import Note

            note = Note(
                path="Track Context/My Track/arrangement.md",
                title="My Track Arrangement",
                content=(
                    "# Arrangement Overview\n\n"
                    "## Global Notes\n"
                    "- Goal: stronger second drop\n\n"
                    "# Sections\n\n"
                    "## S1 - Intro\n"
                    "Bars: 1-8\n"
                    "Energy: 2\n"
                    "Purpose: establish groove\n"
                ),
                frontmatter={
                    "type": "track_arrangement",
                    "track_name": "My Track",
                    "genre": "progressive_house",
                    "arrangement_version": 1,
                },
                source_type="track_arrangement",
            )
            chunks = chunk_notes([note], chunk_size=400, overlap=40)
            store.upsert_chunks(chunks, [[1.0, 0.0] for _ in chunks])

            results = store.get_all_chunks()
            arrangement_rows = [metadata for _, metadata, _ in results if metadata.get("source_type") == "track_arrangement"]

            self.assertTrue(arrangement_rows)
            self.assertTrue(any(row.get("arrangement_track_name") == "My Track" for row in arrangement_rows))
            self.assertTrue(any(row.get("arrangement_section_id") == "S1" for row in arrangement_rows))

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

    def test_retriever_weights_track_genre_and_problem_signals(self) -> None:
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

                def query(self, query_embedding: list[float], top_k: int, filters=None, **kwargs) -> list[RetrievedChunk]:
                    return [
                        RetrievedChunk(
                            "general club arrangement advice with weak overlap",
                            {
                                "note_title": "Generic Club Notes",
                                "source_path": "generic.md",
                                "chunk_index": 0,
                                "content_category": "curated_knowledge",
                            },
                            0.05,
                        ),
                        RetrievedChunk(
                            "progressive house drop contrast and euphoric tension ideas for a flat second half",
                            {
                                "note_title": "Progressive House Drop Notes",
                                "source_path": "ph-drop.md",
                                "chunk_index": 0,
                                "content_category": "curated_knowledge",
                                "import_genre": "progressive house",
                                "heading_context": "drop contrast",
                            },
                            0.22,
                        ),
                    ]

            retriever = Retriever(config, StubEmbeddingClient(), StubVectorStore())

            results = retriever.retrieve(
                "How do I improve this drop?",
                options=RetrievalOptions(top_k=1, candidate_count=2, rerank=False),
                track_context=TrackContext(
                    track_id="moonlit_driver",
                    genre="progressive house",
                    vibe=["euphoric"],
                    current_problem="flat second half of the drop",
                ),
            )

            self.assertEqual(results[0].metadata["note_title"], "Progressive House Drop Notes")

    def test_retriever_weights_workflow_relevance_for_arrangement_chunks(self) -> None:
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

                def query(self, query_embedding: list[float], top_k: int, filters=None, **kwargs) -> list[RetrievedChunk]:
                    return [
                        RetrievedChunk(
                            "drop arrangement and energy flow notes",
                            {
                                "note_title": "Track Arrangement",
                                "source_path": "arrangement.md",
                                "chunk_index": 0,
                                "source_type": "track_arrangement",
                                "arrangement_section_name": "Drop",
                                "content_category": "curated_knowledge",
                            },
                            0.22,
                        ),
                        RetrievedChunk(
                            "drop impact notes from a general article",
                            {
                                "note_title": "General Notes",
                                "source_path": "general.md",
                                "chunk_index": 0,
                                "content_category": "curated_knowledge",
                            },
                            0.05,
                        ),
                    ]

            retriever = Retriever(config, StubEmbeddingClient(), StubVectorStore())

            debug = retriever.retrieve_with_debug(
                "How can I improve the drop arrangement?",
                options=RetrievalOptions(top_k=1, candidate_count=2, rerank=False),
                collaboration_workflow=CollaborationWorkflow.TRACK_CONCEPT_CRITIQUE,
                section_focus="drop",
                domain_profile=DomainProfile.ELECTRONIC_MUSIC,
            )

            self.assertEqual(debug.primary_chunks[0].metadata["note_title"], "Track Arrangement")
            self.assertTrue(debug.reranking_details)
            top_detail = debug.reranking_details[0]
            self.assertIn("workflow_relevance", top_detail.component_scores)
            self.assertIn("track_context_relevance", top_detail.component_scores)
            self.assertIn("section_focus_match", top_detail.component_scores)

    def test_retriever_prefers_current_problem_and_open_task_aligned_drop_note(self) -> None:
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

                def query(self, query_embedding: list[float], top_k: int, filters=None, **kwargs) -> list[RetrievedChunk]:
                    return [
                        RetrievedChunk(
                            "breakdown re-entry urgency improves when the fill accelerates density across the last four bars",
                            {
                                "note_title": "Breakdown Re-entry Fixes",
                                "source_path": "breakdown.md",
                                "chunk_index": 0,
                                "source_type": "track_arrangement",
                                "arrangement_section_name": "Breakdown",
                                "arrangement_genre": "progressive house",
                                "heading_context": "re-entry urgency",
                                "content_category": "curated_knowledge",
                            },
                            0.12,
                        ),
                        RetrievedChunk(
                            "if the first drop loses contrast after 8 bars, use bar 49 as a pivot and vary the bass motif",
                            {
                                "note_title": "Drop Pivot Note",
                                "source_path": "drop.md",
                                "chunk_index": 0,
                                "source_type": "track_arrangement",
                                "arrangement_section_name": "Drop",
                                "arrangement_genre": "progressive house",
                                "heading_context": "bar 49 pivot",
                                "content_category": "curated_knowledge",
                            },
                            0.22,
                        ),
                    ]

            retriever = Retriever(config, StubEmbeddingClient(), StubVectorStore())

            debug = retriever.retrieve_with_debug(
                "What should I do next to get this track unstuck?",
                options=RetrievalOptions(top_k=1, candidate_count=2, rerank=False),
                collaboration_workflow=CollaborationWorkflow.ARRANGEMENT_PLANNER,
                track_context=TrackContext(
                    track_id="moonlit_driver",
                    genre="progressive house",
                    current_problem="first drop loses contrast after the initial 8 bars",
                    known_issues=["breakdown re-entry feels too polite"],
                ),
                current_tasks=[
                    SessionTask(
                        id="1",
                        text="Increase drop contrast with pre-drop subtraction",
                        status="open",
                        source="user",
                        created_at="2026-03-28 10:00:00",
                        linked_section="drop",
                    )
                ],
            )

            self.assertEqual(debug.primary_chunks[0].metadata["note_title"], "Drop Pivot Note")
            top_detail = debug.reranking_details[0]
            self.assertGreater(top_detail.component_scores["current_problem_match"], 0.0)
            self.assertGreater(top_detail.component_scores["task_relevance"], 0.0)
            second_detail = debug.reranking_details[1]
            self.assertEqual(second_detail.note_title, "Breakdown Re-entry Fixes")
            self.assertEqual(second_detail.component_scores["task_relevance"], 0.0)

    def test_retriever_prefers_exact_drop_arrangement_chunk_in_close_call(self) -> None:
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

                def query(self, query_embedding: list[float], top_k: int, filters=None, **kwargs) -> list[RetrievedChunk]:
                    return [
                        RetrievedChunk(
                            "progressive house drop contrast improves when the motif evolves in the second half",
                            {
                                "note_title": "Progressive House Drop Dynamics",
                                "source_path": "genre-note.md",
                                "chunk_index": 0,
                                "source_type": "youtube_video",
                                "import_genre": "progressive house",
                                "video_section_title": "Drop contrast",
                                "content_category": "curated_knowledge",
                            },
                            0.18,
                        ),
                        RetrievedChunk(
                            "moonlit driver drop blueprint: keep bars 33 to 40 stable, then create a bar 49 pivot",
                            {
                                "note_title": "Moonlit Driver Drop Blueprint",
                                "source_path": "arrangement.md",
                                "chunk_index": 0,
                                "source_type": "track_arrangement",
                                "arrangement_track_name": "Moonlit Driver",
                                "arrangement_section_name": "Drop",
                                "arrangement_genre": "progressive house",
                                "heading_context": "drop payoff and bar 49 pivot",
                                "content_category": "curated_knowledge",
                            },
                            0.22,
                        ),
                    ]

            retriever = Retriever(config, StubEmbeddingClient(), StubVectorStore())

            debug = retriever.retrieve_with_debug(
                "How can I improve the second half of the drop without making it busier?",
                options=RetrievalOptions(top_k=1, candidate_count=2, rerank=False),
                collaboration_workflow=CollaborationWorkflow.ARRANGEMENT_PLANNER,
                section_focus="drop",
                track_context=TrackContext(
                    track_id="moonlit_driver",
                    genre="progressive house",
                    current_problem="first drop loses contrast after the initial 8 bars",
                ),
            )

            self.assertEqual(debug.primary_chunks[0].metadata["note_title"], "Moonlit Driver Drop Blueprint")
            top_detail = debug.reranking_details[0]
            self.assertGreater(top_detail.component_scores["section_focus_match"], 0.0)
            self.assertGreater(top_detail.component_scores["current_problem_match"], 0.0)

    def test_index_version_mismatch_requires_rebuild(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            vault = root / "vault"
            vault.mkdir()
            (root / "output").mkdir()
            note_path = vault / "agents.md"
            note_path.write_text("# Agents\n\nInitial content", encoding="utf-8")
            config = make_config(root)

            with patch("main.OllamaEmbeddingClient.embed_texts", return_value=[[1.0, 0.0, 0.0]]):
                run_index(config, reset_store=True)

            store = VectorStore(config)
            store.write_index_version("old-schema")

            with self.assertRaisesRegex(RuntimeError, "Run `python main.py rebuild`"):
                run_index(config, reset_store=False)

    def test_saved_answers_remain_excluded_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            vault = root / "vault"
            output_dir = vault / "research_answers"
            vault.mkdir()
            output_dir.mkdir()
            (vault / "agents.md").write_text("# Agents\n\nPrimary note", encoding="utf-8")
            (output_dir / "saved.md").write_text("# Saved\n\nDerived note", encoding="utf-8")
            config = make_config(root)
            config = replace(config, obsidian_output_path=output_dir)

            index_service = IndexService(config)
            with patch(
                "services.index_service.OllamaEmbeddingClient.embed_texts",
                return_value=[[1.0, 0.0, 0.0]],
            ):
                response = index_service.index(reset_store=True)

            self.assertEqual(response.notes_loaded, 1)

    def test_saved_answers_are_indexed_when_enabled(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            vault = root / "vault"
            output_dir = vault / "research_answers"
            vault.mkdir()
            output_dir.mkdir()
            (vault / "agents.md").write_text("# Agents\n\nPrimary note", encoding="utf-8")
            (output_dir / "saved.md").write_text("# Saved\n\nDerived note", encoding="utf-8")
            config = make_config(root)
            config = replace(config, obsidian_output_path=output_dir, index_saved_answers=True)

            index_service = IndexService(config)
            with patch(
                "services.index_service.OllamaEmbeddingClient.embed_texts",
                return_value=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            ):
                response = index_service.index(reset_store=True)

            self.assertEqual(response.notes_loaded, 2)

    def test_import_and_saved_research_folders_remain_excluded_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            vault = root / "vault"
            vault.mkdir()
            (root / "output").mkdir()
            (vault / "agents.md").write_text("# Agents\n\nPrimary note", encoding="utf-8")
            (vault / "Saved Outputs" / "research").mkdir(parents=True)
            (vault / "Saved Outputs" / "research" / "session.md").write_text(
                "# Session\n\nGenerated research output",
                encoding="utf-8",
            )
            (vault / "ingested_webpages").mkdir()
            (vault / "ingested_webpages" / "page.md").write_text("# Page\n\nImported webpage", encoding="utf-8")
            (vault / "ingested_youtube").mkdir()
            (vault / "ingested_youtube" / "video.md").write_text("# Video\n\nImported transcript", encoding="utf-8")
            config = make_config(root)

            index_service = IndexService(config)
            with patch(
                "services.index_service.OllamaEmbeddingClient.embed_texts",
                return_value=[[1.0, 0.0, 0.0]],
            ):
                response = index_service.index(reset_store=True)

            self.assertEqual(response.notes_loaded, 1)

    def test_import_folders_can_be_indexed_when_enabled(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            vault = root / "vault"
            vault.mkdir()
            (root / "output").mkdir()
            (vault / "agents.md").write_text("# Agents\n\nPrimary note", encoding="utf-8")
            (vault / "ingested_webpages").mkdir()
            (vault / "ingested_webpages" / "page.md").write_text(
                '---\nsource_type: "webpage_import"\n---\n\n# Page\n\nImported webpage',
                encoding="utf-8",
            )
            (vault / "ingested_youtube").mkdir()
            (vault / "ingested_youtube" / "video.md").write_text(
                '---\nsource_type: "youtube_import"\n---\n\n# Video\n\nImported transcript',
                encoding="utf-8",
            )
            config = make_config(root)
            config = replace(config, index_webpage_imports=True, index_youtube_imports=True)

            index_service = IndexService(config)
            with patch(
                "services.index_service.OllamaEmbeddingClient.embed_texts",
                return_value=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            ):
                response = index_service.index(reset_store=True)

            self.assertEqual(response.notes_loaded, 3)

    def test_saved_answer_chunks_are_downranked_relative_to_primary_notes(self) -> None:
        chunks = [
            RetrievedChunk(
                text="Agents use retrieval.",
                metadata={
                    "note_title": "Saved Answer",
                    "source_path": "research_answers/saved.md",
                    "source_kind": "saved_answer",
                },
                distance_or_score=0.1,
            ),
            RetrievedChunk(
                text="Agents use retrieval.",
                metadata={
                    "note_title": "Primary Note",
                    "source_path": "agents.md",
                    "source_kind": "primary_note",
                },
                distance_or_score=0.1,
            ),
        ]

        class StubEmbeddingClient:
            def embed_text(self, text: str) -> list[float]:
                return [1.0, 0.0]

        class StubVectorStore:
            def count(self) -> int:
                return 2

            def query(self, query_embedding: list[float], top_k: int, filters=None) -> list[RetrievedChunk]:
                return list(chunks)

        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            (root / "vault").mkdir()
            (root / "output").mkdir()
            config = make_config(root)
            retriever = Retriever(config, StubEmbeddingClient(), StubVectorStore())
            results = retriever.retrieve("agents retrieval", options=RetrievalOptions(top_k=1, candidate_count=2, rerank=True))

        self.assertEqual(results[0].metadata["source_kind"], "primary_note")

    def test_retriever_can_exclude_saved_answers_per_question(self) -> None:
        chunks = [
            RetrievedChunk(
                text="Saved synthesis about agents.",
                metadata={
                    "note_title": "Saved Answer",
                    "source_path": "research_answers/saved.md",
                    "source_kind": "saved_answer",
                },
                distance_or_score=0.05,
            ),
            RetrievedChunk(
                text="Primary note about agents.",
                metadata={
                    "note_title": "Agents",
                    "source_path": "agents.md",
                    "source_kind": "primary_note",
                },
                distance_or_score=0.2,
            ),
        ]

        class StubEmbeddingClient:
            def embed_text(self, text: str) -> list[float]:
                return [1.0, 0.0]

        class StubVectorStore:
            def count(self) -> int:
                return 2

            def query(
                self,
                query_embedding: list[float],
                top_k: int,
                filters=None,
                include_saved_answers=None,
            ) -> list[RetrievedChunk]:
                if include_saved_answers is False:
                    return [chunk for chunk in chunks if chunk.metadata["source_kind"] != "saved_answer"]
                return list(chunks)

        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            (root / "vault").mkdir()
            (root / "output").mkdir()
            config = make_config(root)
            retriever = Retriever(config, StubEmbeddingClient(), StubVectorStore())
            results = retriever.retrieve(
                "agents",
                options=RetrievalOptions(top_k=2, candidate_count=2, include_saved_answers=False),
            )

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].metadata["source_kind"], "primary_note")

    def test_saved_answer_sources_use_saved_label(self) -> None:
        sources, _ = build_citation_sources(
            [
                RetrievedChunk(
                    text="Saved synthesis",
                    metadata={
                        "note_title": "Collaborator Output",
                        "source_path": "research_answers/answer.md",
                        "source_kind": "saved_answer",
                    },
                    distance_or_score=0.1,
                ),
                RetrievedChunk(
                    text="Primary note",
                    metadata={
                        "note_title": "Agents",
                        "source_path": "agents.md",
                        "source_kind": "primary_note",
                    },
                    distance_or_score=0.1,
                ),
            ],
            [],
        )

        self.assertIn("[Saved 1] Collaborator Output (research_answers/answer.md)", sources)
        self.assertIn("[Local 1] Agents (agents.md)", sources)
