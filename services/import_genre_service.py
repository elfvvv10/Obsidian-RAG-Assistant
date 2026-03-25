"""Helpers for genre-aware imported content organization."""

from __future__ import annotations

from pathlib import Path

from config import AppConfig
from services.models import TrackContext


GENERIC_IMPORT_GENRE = "Generic"
STARTER_IMPORT_GENRES: tuple[str, ...] = (
    GENERIC_IMPORT_GENRE,
    "Progressive House",
    "Melodic House",
    "Melodic Techno",
    "Tech House",
    "Deep House",
    "Afro House",
    "Techno",
    "Trance",
    "Progressive Trance",
    "UK Garage",
    "Drum and Bass",
    "Breaks",
    "Future Rave",
)


class ImportGenreService:
    """Resolve import genres for UI choices, save paths, and retrieval eligibility."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config

    def available_genres(self) -> list[str]:
        discovered = {
            genre.casefold(): genre
            for genre in self._discover_source_genres(self.config.webpage_ingestion_path)
            + self._discover_source_genres(self.config.youtube_ingestion_path)
        }
        merged: list[str] = []
        seen: set[str] = set()
        for genre in STARTER_IMPORT_GENRES:
            seen.add(genre.casefold())
            merged.append(genre)
        extras = sorted(
            (
                genre
                for folded, genre in discovered.items()
                if folded not in seen
            ),
            key=str.casefold,
        )
        merged.extend(extras)
        return merged

    def canonicalize(self, value: str | None) -> str:
        normalized = " ".join((value or "").split()).strip()
        if not normalized:
            return GENERIC_IMPORT_GENRE
        for candidate in self.available_genres():
            if candidate.casefold() == normalized.casefold():
                return candidate
        return normalized

    def destination_for(self, base_path: Path, genre: str | None) -> Path:
        return base_path / self.canonicalize(genre)

    def eligible_genres(self, track_context: TrackContext | None) -> tuple[str, ...]:
        eligible = [GENERIC_IMPORT_GENRE]
        if track_context is None:
            return tuple(eligible)
        genre = self.canonicalize(track_context.genre)
        if genre and genre.casefold() != GENERIC_IMPORT_GENRE.casefold():
            eligible.append(genre)
        return tuple(eligible)

    def matches(self, candidate: str | None, eligible_genres: tuple[str, ...]) -> bool:
        if not candidate:
            return True
        folded = candidate.casefold()
        return any(genre.casefold() == folded for genre in eligible_genres)

    def _discover_source_genres(self, base_path: Path) -> list[str]:
        if not base_path.exists() or not base_path.is_dir():
            return []
        return [
            path.name
            for path in sorted(base_path.iterdir(), key=lambda item: item.name.casefold())
            if path.is_dir() and not path.name.startswith(".")
        ]
