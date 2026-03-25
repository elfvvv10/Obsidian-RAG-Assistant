"""Tests for legacy markdown track selection from Projects/."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from services.track_selector_service import TrackSelectorService, selected_track_index, selected_track_path


class TrackSelectorServiceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.service = TrackSelectorService()

    def test_missing_projects_folder_returns_empty_list(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            vault_path = Path(tmp_dir) / "vault"
            vault_path.mkdir()
            self.assertEqual(self.service.list_tracks(vault_path), [])

    def test_finds_valid_track_folders_and_ignores_invalid_ones(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            vault_path = Path(tmp_dir) / "vault"
            projects_path = vault_path / "Projects"
            (projects_path / "Moonlit Driver").mkdir(parents=True)
            (projects_path / "Moonlit Driver" / "track_context.md").write_text("", encoding="utf-8")
            (projects_path / "Ideas").mkdir(parents=True)
            tracks = self.service.list_tracks(vault_path)

            self.assertEqual(
                tracks,
                [{"name": "Moonlit Driver", "path": "Projects/Moonlit Driver/track_context.md"}],
            )

    def test_returns_relative_paths_sorted_by_name(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            vault_path = Path(tmp_dir) / "vault"
            projects_path = vault_path / "Projects"
            (projects_path / "Zulu Track").mkdir(parents=True)
            (projects_path / "Zulu Track" / "track_context.md").write_text("", encoding="utf-8")
            (projects_path / "Alpha Track").mkdir(parents=True)
            (projects_path / "Alpha Track" / "track_context.md").write_text("", encoding="utf-8")

            tracks = self.service.list_tracks(vault_path)

            self.assertEqual(
                tracks,
                [
                    {"name": "Alpha Track", "path": "Projects/Alpha Track/track_context.md"},
                    {"name": "Zulu Track", "path": "Projects/Zulu Track/track_context.md"},
                ],
            )

    def test_finds_nested_project_track_contexts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            vault_path = Path(tmp_dir) / "vault"
            nested_track = vault_path / "Projects" / "Current Tracks" / "Moonlit Driver"
            nested_track.mkdir(parents=True)
            (nested_track / "track_context.md").write_text("", encoding="utf-8")

            tracks = self.service.list_tracks(vault_path)

            self.assertEqual(
                tracks,
                [
                    {
                        "name": "Current Tracks / Moonlit Driver",
                        "path": "Projects/Current Tracks/Moonlit Driver/track_context.md",
                    }
                ],
            )

    def test_selected_track_path_and_index_support_ui_wiring(self) -> None:
        tracks = [
            {"name": "Alpha Track", "path": "Projects/Alpha Track/track_context.md"},
            {"name": "Zulu Track", "path": "Projects/Zulu Track/track_context.md"},
        ]

        self.assertEqual(
            selected_track_path("Zulu Track", tracks),
            "Projects/Zulu Track/track_context.md",
        )
        self.assertIsNone(selected_track_path("None", tracks))
        self.assertEqual(
            selected_track_index("Projects/Zulu Track/track_context.md", tracks),
            2,
        )

    def test_load_workflow_context_maps_legacy_markdown_fields(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            vault_path = Path(tmp_dir) / "vault"
            track_path = vault_path / "Projects" / "Current Tracks" / "Moonlit Driver"
            track_path.mkdir(parents=True)
            (track_path / "track_context.md").write_text(
                """---
type: track_context
primary_genre: progressive house
secondary_influences:
  - deep tech
  - tech house
bpm: 124
vibe:
  - driving
  - emotional
reference_artists:
  - Guy J
current_issues:
  - weak drop impact
priority_focus:
  - improve transitions
status: arranging first full draft
---

## Structure

- Intro
- Main Section

## Core Ideas

- Main hook: filtered melodic arp
- Bass: rolling

## Recent Decisions

- Shortened intro by 8 bars
""",
                encoding="utf-8",
            )

            mapped = self.service.load_workflow_context(
                vault_path,
                "Projects/Current Tracks/Moonlit Driver/track_context.md",
            )

            self.assertEqual(mapped["workflow_genre"], "progressive house")
            self.assertEqual(mapped["workflow_bpm"], "124")
            self.assertEqual(mapped["workflow_mood"], "driving, emotional")
            self.assertIn("Guy J", mapped["workflow_references"])
            self.assertIn("deep tech", mapped["workflow_references"])
            self.assertIn("Status:", mapped["workflow_arrangement_notes"])
            self.assertIn("Structure:", mapped["workflow_arrangement_notes"])
            self.assertIn("Current Issues:", mapped["workflow_arrangement_notes"])
            self.assertIn("Recent Decisions:", mapped["workflow_arrangement_notes"])
            self.assertIn("Main hook: filtered melodic arp", mapped["workflow_instrumentation"])
            self.assertIn("Vibe:", mapped["workflow_sound_palette"])
            self.assertIn("Influences:", mapped["workflow_sound_palette"])
