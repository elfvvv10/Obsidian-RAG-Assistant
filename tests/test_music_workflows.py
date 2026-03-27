"""Tests for music workflow routing, prompting, and save behavior."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from config import AppConfig
from saver import save_answer
from services.models import (
    CollaborationWorkflow,
    QueryRequest,
    ResearchRequest,
    TrackContext,
    WorkflowInput,
)
from services.music_workflow_service import MusicWorkflowService


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


class MusicWorkflowTests(unittest.TestCase):
    def test_query_plan_uses_workflow_specific_saved_outputs_folder(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            (root / "vault").mkdir()
            (root / "output").mkdir()
            service = MusicWorkflowService(make_config(root))

            plan = service.build_query_plan(
                QueryRequest(
                    question="Review this garage idea",
                    collaboration_workflow=CollaborationWorkflow.GENRE_FIT_REVIEW,
                    workflow_input=WorkflowInput(genre="UK garage", bpm="132"),
                )
            )

            self.assertIn("critiques/Genre Fit Reviews", str(plan.save_path))
            self.assertIn("Structured workflow context:", plan.prompt_text)
            self.assertIn("genre: UK garage", plan.prompt_text)

    def test_query_plan_uses_track_specific_subfolder_when_track_context_is_active(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            (root / "vault").mkdir()
            (root / "output").mkdir()
            service = MusicWorkflowService(make_config(root))

            plan = service.build_query_plan(
                QueryRequest(
                    question="Review this garage idea",
                    collaboration_workflow=CollaborationWorkflow.GENRE_FIT_REVIEW,
                    track_id="warehouse-hypnosis-01",
                    use_track_context=True,
                )
            )

            self.assertIn("critiques/Genre Fit Reviews/warehouse-hypnosis-01", str(plan.save_path))

    def test_research_plan_uses_saved_outputs_research_folder(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            (root / "vault").mkdir()
            (root / "output").mkdir()
            config = make_config(root)
            service = MusicWorkflowService(config)

            plan = service.build_research_plan(
                ResearchRequest(
                    goal="Compare melodic techno arrangement conventions",
                    workflow_input=WorkflowInput(genre="melodic techno"),
                )
            )

            self.assertEqual(plan.save_path, config.research_sessions_path)

    def test_research_plan_uses_track_specific_subfolder_when_track_context_is_active(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            (root / "vault").mkdir()
            (root / "output").mkdir()
            config = make_config(root)
            service = MusicWorkflowService(config)

            plan = service.build_research_plan(
                ResearchRequest(
                    goal="Compare melodic techno arrangement conventions",
                    track_id="warehouse-hypnosis-01",
                    use_track_context=True,
                )
            )

            self.assertEqual(plan.save_path, config.research_sessions_path / "warehouse-hypnosis-01")

    def test_save_answer_includes_workflow_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            output_path = root / "output"
            output_path.mkdir()

            destination = save_answer(
                output_path / "answers/Arrangement Plans",
                "Plan this progressive house arrangement",
                result=_answer_result(),
                source_type="saved_answer",
                status="draft",
                indexed=False,
                domain_profile="electronic_music",
                workflow_type="arrangement_planner",
                workflow_input={"genre": "progressive house", "track_length": "6:30"},
            )

            contents = destination.read_text(encoding="utf-8")
            self.assertIn('domain_profile: "electronic_music"', contents)
            self.assertIn('workflow_type: "arrangement_planner"', contents)
            self.assertIn("## Arrangement Plan", contents)
            self.assertIn("## Production Plan Notes", contents)
            self.assertIn("Genre: progressive house", contents)

    def test_save_answer_respects_track_specific_output_folder(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            output_path = root / "output"
            output_path.mkdir()

            destination = save_answer(
                output_path / "answers/Arrangement Plans/warehouse-hypnosis-01",
                "Plan this progressive house arrangement",
                result=_answer_result(),
                track_context=TrackContext(track_id="warehouse-hypnosis-01", track_name="Warehouse Hypnosis"),
            )

            self.assertIn("answers/Arrangement Plans/warehouse-hypnosis-01", str(destination))


def _answer_result():
    from utils import AnswerResult

    return AnswerResult(
        answer="[Local 1] Intro should stay sparse.\n\n[Inference] Build tension gradually into the first drop.",
        sources=["[Local 1] Arrangement Notes (arrangement.md)"],
        retrieved_chunks=[],
    )
