"""Tests for track context parsing and prompt injection."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from config import AppConfig
from services.models import AnswerMode, CollaborationWorkflow, RetrievalMode, WorkflowInput
from services.prompt_service import PromptService
from services.track_context_service import TrackContextService


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


class TrackContextServiceTests(unittest.TestCase):
    def test_parses_frontmatter_and_preserves_body(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            vault = root / "vault"
            project_dir = vault / "Projects" / "Moonlit Driver"
            project_dir.mkdir(parents=True)
            (root / "output").mkdir()
            context_path = project_dir / "track_context.md"
            context_path.write_text(
                "---\n"
                "track_title: Moonlit Driver\n"
                "primary_genre: progressive house\n"
                "current_issues:\n"
                "  - weak drop impact\n"
                "priority_focus:\n"
                "  - finish arrangement\n"
                "---\n\n"
                "## Structure\n\n"
                "- Intro\n",
                encoding="utf-8",
            )
            service = TrackContextService(make_config(root))

            result = service.get_track_context(
                CollaborationWorkflow.TRACK_CONCEPT_CRITIQUE,
                "Projects/Moonlit Driver",
            )

            self.assertTrue(result.found)
            self.assertEqual(result.frontmatter["track_title"], "Moonlit Driver")
            self.assertIn("## Structure", result.body)
            self.assertIn("Track Title: Moonlit Driver", result.prompt_block)
            self.assertIn("Current Issues: weak drop impact", result.prompt_block)

    def test_missing_or_partial_frontmatter_is_handled_gracefully(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            vault = root / "vault"
            project_dir = vault / "Projects" / "Moonlit Driver"
            project_dir.mkdir(parents=True)
            (root / "output").mkdir()
            (project_dir / "track_context.md").write_text(
                "## Core Ideas\n\n- Filtered arp\n",
                encoding="utf-8",
            )
            service = TrackContextService(make_config(root))

            result = service.get_track_context(
                CollaborationWorkflow.TRACK_CONCEPT_CRITIQUE,
                "Projects/Moonlit Driver/track_context.md",
            )

            self.assertTrue(result.found)
            self.assertEqual(result.frontmatter, {})
            self.assertIn("## Core Ideas", result.body)
            self.assertIn("Track context notes:", result.prompt_block)


class TrackContextPromptInjectionTests(unittest.TestCase):
    def test_critique_workflow_injects_track_context(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            vault = root / "vault"
            project_dir = vault / "Projects" / "Moonlit Driver"
            project_dir.mkdir(parents=True)
            (root / "output").mkdir()
            context_path = project_dir / "track_context.md"
            context_path.write_text(
                "---\n"
                "track_title: Moonlit Driver\n"
                "primary_genre: progressive house\n"
                "priority_focus:\n"
                "  - improve transitions\n"
                "---\n\n"
                "## Recent Decisions\n\n"
                "- Shortened intro by 8 bars\n",
                encoding="utf-8",
            )
            payload = PromptService(make_config(root)).build_prompt_payload(
                "Critique this track concept.",
                [],
                web_results=[],
                retrieval_mode=RetrievalMode.LOCAL_ONLY,
                answer_mode=AnswerMode.BALANCED,
                local_retrieval_weak=False,
                collaboration_workflow=CollaborationWorkflow.TRACK_CONCEPT_CRITIQUE,
                workflow_input=WorkflowInput(track_context_path="Projects/Moonlit Driver"),
            )

            self.assertIn("BEGIN INTERNAL TRACK CONTEXT", payload.system_prompt)
            self.assertIn("Track Title: Moonlit Driver", payload.system_prompt)
            self.assertIn("Priority Focus: improve transitions", payload.system_prompt)
            self.assertIn("## Recent Decisions", payload.system_prompt)
            self.assertEqual(payload.system_prompt.count("BEGIN INTERNAL TRACK CONTEXT"), 1)

    def test_critique_workflow_places_framework_before_track_context(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            vault = root / "vault"
            project_dir = vault / "Projects" / "Moonlit Driver"
            project_dir.mkdir(parents=True)
            (root / "output").mkdir()
            framework_path = root / "track_critique_framework_v1.md"
            framework_path.write_text("Framework guidance.", encoding="utf-8")
            context_path = project_dir / "track_context.md"
            context_path.write_text(
                "---\n"
                "track_title: Moonlit Driver\n"
                "status: arranging first full draft\n"
                "---\n\n"
                "## Structure\n\n"
                "- Intro\n",
                encoding="utf-8",
            )
            config = make_config(root)
            config.track_critique_framework_path = str(framework_path)
            payload = PromptService(config).build_prompt_payload(
                "Critique this track concept.",
                [],
                web_results=[],
                retrieval_mode=RetrievalMode.LOCAL_ONLY,
                answer_mode=AnswerMode.BALANCED,
                local_retrieval_weak=False,
                collaboration_workflow=CollaborationWorkflow.TRACK_CONCEPT_CRITIQUE,
                workflow_input=WorkflowInput(track_context_path="Projects/Moonlit Driver/track_context.md"),
            )

            self.assertIn("BEGIN INTERNAL CRITIQUE FRAMEWORK", payload.system_prompt)
            self.assertIn("BEGIN INTERNAL TRACK CONTEXT", payload.system_prompt)
            self.assertLess(
                payload.system_prompt.index("BEGIN INTERNAL CRITIQUE FRAMEWORK"),
                payload.system_prompt.index("BEGIN INTERNAL TRACK CONTEXT"),
            )

    def test_arrangement_workflow_injects_track_context(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            vault = root / "vault"
            project_dir = vault / "Projects" / "Moonlit Driver"
            project_dir.mkdir(parents=True)
            (root / "output").mkdir()
            context_path = project_dir / "track_context.md"
            context_path.write_text(
                "---\n"
                "track_title: Moonlit Driver\n"
                "status: arranging first full draft\n"
                "---\n\n"
                "## Structure\n\n"
                "- Intro\n",
                encoding="utf-8",
            )
            payload = PromptService(make_config(root)).build_prompt_payload(
                "Plan this arrangement.",
                [],
                web_results=[],
                retrieval_mode=RetrievalMode.LOCAL_ONLY,
                answer_mode=AnswerMode.BALANCED,
                local_retrieval_weak=False,
                collaboration_workflow=CollaborationWorkflow.ARRANGEMENT_PLANNER,
                workflow_input=WorkflowInput(track_context_path="Projects/Moonlit Driver/track_context.md"),
            )

            self.assertIn("BEGIN INTERNAL TRACK CONTEXT", payload.system_prompt)
            self.assertNotIn("BEGIN INTERNAL CRITIQUE FRAMEWORK", payload.system_prompt)

    def test_non_applicable_workflows_do_not_inject_track_context(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            vault = root / "vault"
            project_dir = vault / "Projects" / "Moonlit Driver"
            project_dir.mkdir(parents=True)
            (root / "output").mkdir()
            (project_dir / "track_context.md").write_text("## Structure\n\n- Intro\n", encoding="utf-8")

            payload = PromptService(make_config(root)).build_prompt_payload(
                "General question.",
                [],
                web_results=[],
                retrieval_mode=RetrievalMode.LOCAL_ONLY,
                answer_mode=AnswerMode.BALANCED,
                local_retrieval_weak=False,
                collaboration_workflow=CollaborationWorkflow.GENERAL_ASK,
                workflow_input=WorkflowInput(track_context_path="Projects/Moonlit Driver"),
            )

            self.assertNotIn("BEGIN INTERNAL TRACK CONTEXT", payload.system_prompt)

    def test_missing_track_context_falls_back_gracefully(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            (root / "vault").mkdir()
            (root / "output").mkdir()

            payload = PromptService(make_config(root)).build_prompt_payload(
                "Critique this track concept.",
                [],
                web_results=[],
                retrieval_mode=RetrievalMode.LOCAL_ONLY,
                answer_mode=AnswerMode.BALANCED,
                local_retrieval_weak=False,
                collaboration_workflow=CollaborationWorkflow.TRACK_CONCEPT_CRITIQUE,
                workflow_input=WorkflowInput(track_context_path="Projects/Missing Track"),
            )

            self.assertNotIn("BEGIN INTERNAL TRACK CONTEXT", payload.system_prompt)
            self.assertIn("Active collaboration workflow: track_concept_critique.", payload.system_prompt)
