"""Tests for genre-aware imported content retrieval."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from config import AppConfig
from retriever import RetrievalDebugResult
from services.models import QueryRequest, RetrievalMode, TrackContext
from services.query_service import QueryService
from utils import RetrievedChunk, RetrievalOptions


def make_config(root: Path) -> AppConfig:
    return AppConfig(
        obsidian_vault_path=root / "vault",
        obsidian_output_path=root / "output",
        chroma_db_path=root / "chroma",
        ollama_base_url="http://localhost:11434",
        ollama_chat_model="deepseek",
        ollama_embedding_model="nomic-embed-text",
        top_k_results=2,
    )


class ImportGenreQueryTests(unittest.TestCase):
    def test_query_service_filters_imported_chunks_to_generic_plus_track_genre(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            config = make_config(Path(tmp_dir))
            response = QueryService(
                config,
                embedding_client_cls=StubEmbeddingClient,
                chat_client_cls=StubChatClient,
                retriever_cls=StubRetriever,
                vector_store_cls=StubVectorStore,
                web_search_service_cls=StubWebSearchService,
            ).ask(
                QueryRequest(
                    question="bassline ideas",
                    retrieval_mode=RetrievalMode.LOCAL_ONLY,
                    options=RetrievalOptions(top_k=3),
                    track_context=TrackContext(track_id="moonlit_driver", genre="Techno"),
                    track_id="moonlit_driver",
                    use_track_context=True,
                )
            )

        source_paths = [chunk.metadata["source_path"] for chunk in response.retrieved_chunks]
        self.assertIn("Imports/Web Imports/Generic/generic.md", source_paths)
        self.assertIn("Imports/Web Imports/Techno/techno.md", source_paths)
        self.assertNotIn("Imports/Web Imports/Progressive House/progressive.md", source_paths)
        self.assertEqual(response.debug.imported_genres_eligible, ("Generic", "Techno"))

    def test_query_service_uses_generic_only_when_track_genre_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            config = make_config(Path(tmp_dir))
            response = QueryService(
                config,
                embedding_client_cls=StubEmbeddingClient,
                chat_client_cls=StubChatClient,
                retriever_cls=StubRetriever,
                vector_store_cls=StubVectorStore,
                web_search_service_cls=StubWebSearchService,
            ).ask(
                QueryRequest(
                    question="bassline ideas",
                    retrieval_mode=RetrievalMode.LOCAL_ONLY,
                    options=RetrievalOptions(top_k=3),
                    track_context=TrackContext(track_id="moonlit_driver"),
                    track_id="moonlit_driver",
                    use_track_context=True,
                )
            )

        source_paths = [chunk.metadata["source_path"] for chunk in response.retrieved_chunks]
        self.assertIn("Imports/Web Imports/Generic/generic.md", source_paths)
        self.assertNotIn("Imports/Web Imports/Techno/techno.md", source_paths)
        self.assertNotIn("Imports/Web Imports/Progressive House/progressive.md", source_paths)
        self.assertEqual(response.debug.imported_genres_eligible, ("Generic",))


class StubEmbeddingClient:
    def __init__(self, config: AppConfig) -> None:
        del config

    def embed_text(self, text: str) -> list[float]:
        del text
        return [1.0, 0.0]


class StubChatClient:
    def __init__(self, config: AppConfig, model_override: str | None = None) -> None:
        del config
        self.model = model_override or "deepseek"

    def answer_with_prompt(self, payload) -> str:
        del payload
        return "[Local 1] Filtered answer."


class StubRetriever:
    def __init__(self, config: AppConfig, embedding_client, vector_store) -> None:
        del config, embedding_client, vector_store

    def retrieve_with_debug(self, query: str, filters=None, options=None, retrieval_scope="knowledge"):
        del query, filters, options, retrieval_scope
        generic = _imported_chunk("Imports/Web Imports/Generic/generic.md", "Generic", 0.10)
        techno = _imported_chunk("Imports/Web Imports/Techno/techno.md", "Techno", 0.11)
        progressive = _imported_chunk(
            "Imports/Web Imports/Progressive House/progressive.md",
            "Progressive House",
            0.12,
        )
        local = RetrievedChunk(
            text="Arrangement note",
            metadata={
                "source_path": "Knowledge/Arrangement/arrangement.md",
                "note_title": "Arrangement",
                "chunk_index": 0,
                "content_category": "curated_knowledge",
            },
            distance_or_score=0.09,
        )
        reranked = [local, generic, techno, progressive]
        return RetrievalDebugResult(
            initial_candidates=[local, generic, techno, progressive],
            reranked_candidates=reranked,
            primary_chunks=reranked[:2],
            final_chunks=reranked[:2],
            reranking_applied=False,
            reranking_changed=False,
        )


class StubVectorStore:
    def __init__(self, config: AppConfig) -> None:
        del config

    def is_index_compatible(self) -> bool:
        return True

    def count(self) -> int:
        return 4


class StubWebSearchService:
    def __init__(self, config: AppConfig) -> None:
        del config


def _imported_chunk(source_path: str, genre: str, score: float) -> RetrievedChunk:
    return RetrievedChunk(
        text=f"{genre} import",
        metadata={
            "source_path": source_path,
            "note_title": Path(source_path).stem,
            "chunk_index": 0,
            "content_category": "imported_knowledge",
            "import_genre": genre,
        },
        distance_or_score=score,
    )
