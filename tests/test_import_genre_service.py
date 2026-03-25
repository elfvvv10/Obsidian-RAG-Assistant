"""Tests for genre-aware imported content helpers."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from config import AppConfig
from services.import_genre_service import GENERIC_IMPORT_GENRE, ImportGenreService
from services.models import TrackContext


def make_config(root: Path) -> AppConfig:
    return AppConfig(
        obsidian_vault_path=root / "vault",
        obsidian_output_path=root / "output",
        chroma_db_path=root / "chroma",
        ollama_base_url="http://localhost:11434",
        ollama_chat_model="deepseek",
        ollama_embedding_model="nomic-embed-text",
        top_k_results=3,
        webpage_ingestion_folder="Imports/Web Imports",
        youtube_ingestion_folder="Imports/YouTube Imports",
    )


class ImportGenreServiceTests(unittest.TestCase):
    def test_available_genres_includes_generic_and_starter_list(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            service = ImportGenreService(make_config(Path(tmp_dir)))
            genres = service.available_genres()

        self.assertEqual(genres[0], GENERIC_IMPORT_GENRE)
        self.assertIn("Progressive House", genres)
        self.assertIn("UK Garage", genres)

    def test_available_genres_merges_existing_folders(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            config = make_config(root)
            (config.webpage_ingestion_path / "Organic House").mkdir(parents=True)
            (config.youtube_ingestion_path / "Dub Techno").mkdir(parents=True)

            service = ImportGenreService(config)
            genres = service.available_genres()

        self.assertIn("Organic House", genres)
        self.assertIn("Dub Techno", genres)

    def test_canonicalize_is_case_insensitive_for_known_values(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            service = ImportGenreService(make_config(Path(tmp_dir)))

        self.assertEqual(service.canonicalize("generic"), GENERIC_IMPORT_GENRE)
        self.assertEqual(service.canonicalize("progressive house"), "Progressive House")

    def test_canonicalize_preserves_new_genre_name_with_clean_spacing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            service = ImportGenreService(make_config(Path(tmp_dir)))

        self.assertEqual(service.canonicalize("  Leftfield   House  "), "Leftfield House")

    def test_eligible_genres_uses_generic_plus_track_genre(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            service = ImportGenreService(make_config(Path(tmp_dir)))

        eligible = service.eligible_genres(TrackContext(genre="techno"))
        self.assertEqual(eligible, (GENERIC_IMPORT_GENRE, "Techno"))

    def test_eligible_genres_falls_back_to_generic_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            service = ImportGenreService(make_config(Path(tmp_dir)))

        self.assertEqual(service.eligible_genres(None), (GENERIC_IMPORT_GENRE,))
