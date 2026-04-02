"""Tests for answer-mode policy behavior."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from config import AppConfig
from llm import OpenAIChatClient
from services.models import CollaborationWorkflow, QueryRequest, SectionContext, TrackContext, WorkflowInput
from services.query_service import QueryService
from utils import RetrievedChunk
from web_search import WebSearchResult


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


def make_query_service(
    *,
    local_chunks: list[RetrievedChunk],
    web_results: list[WebSearchResult],
    answer_text: str,
) -> tuple[QueryService, dict[str, object]]:
    tracking: dict[str, object] = {
        "chat_calls": 0,
        "last_prompt": None,
        "last_model": None,
        "last_provider": None,
    }

    class StubEmbeddingClient:
        def __init__(self, config: AppConfig) -> None:
            pass

    class StubChatClient:
        def __init__(self, config: AppConfig, *, model_override: str | None = None) -> None:
            tracking["last_provider"] = config.chat_provider
            tracking["last_model"] = model_override or config.ollama_chat_model

        def answer_with_prompt(self, prompt_payload):
            tracking["chat_calls"] += 1
            tracking["last_prompt"] = prompt_payload
            return answer_text

    class StubRetriever:
        def __init__(self, config: AppConfig, embedding_client, vector_store) -> None:
            pass

        def retrieve(self, query: str, filters=None, options=None):
            return list(local_chunks)

    class StubVectorStore:
        def __init__(self, config: AppConfig) -> None:
            pass

        def is_index_compatible(self) -> bool:
            return True

        def count(self) -> int:
            return max(1, len(local_chunks))

    class StubWebSearchService:
        def __init__(self, config: AppConfig) -> None:
            pass

        def search(self, query: str) -> list[WebSearchResult]:
            return list(web_results)

    root = Path(tempfile.mkdtemp())
    (root / "vault").mkdir()
    (root / "output").mkdir()
    config = make_config(root)
    service = QueryService(
        config,
        embedding_client_cls=StubEmbeddingClient,
        chat_client_cls=StubChatClient,
        retriever_cls=StubRetriever,
        vector_store_cls=StubVectorStore,
        web_search_service_cls=StubWebSearchService,
        capture_debug_trace=False,
    )
    return service, tracking


class AnswerModePolicyTests(unittest.TestCase):
    def test_strict_mode_refuses_when_evidence_is_weak(self) -> None:
        service, tracking = make_query_service(
            local_chunks=[
                RetrievedChunk(
                    text="Possibly related note",
                    metadata={"note_title": "Agents", "source_path": "agents.md"},
                    distance_or_score=0.95,
                )
            ],
            web_results=[],
            answer_text="This should not be used.",
        )

        response = service.ask(QueryRequest(question="agents?", answer_mode="strict"))

        self.assertEqual(tracking["chat_calls"], 0)
        self.assertIn("Insufficient evidence", response.answer)
        self.assertIn("strict", response.answer_mode_used.value)
        self.assertTrue(any("Strict mode limited the answer" in warning for warning in response.warnings))

    def test_balanced_mode_allows_limited_synthesis(self) -> None:
        service, tracking = make_query_service(
            local_chunks=[
                RetrievedChunk(
                    text="Agents use retrieval to ground answers.",
                    metadata={"note_title": "Agents", "source_path": "agents.md"},
                    distance_or_score=0.1,
                )
            ],
            web_results=[],
            answer_text="Agents use retrieval to ground answers [Local 1].\n\n[Inference] This suggests grounding improves trust.",
        )

        response = service.ask(QueryRequest(question="agents?", answer_mode="balanced"))

        self.assertEqual(tracking["chat_calls"], 1)
        self.assertTrue(response.inference_used)
        self.assertEqual(response.answer_mode_used.value, "balanced")
        self.assertIn("[Local 1]", response.answer)

    def test_exploratory_mode_carries_inference_and_web_labels(self) -> None:
        service, tracking = make_query_service(
            local_chunks=[
                RetrievedChunk(
                    text="Agents use tools.",
                    metadata={"note_title": "Agents", "source_path": "agents.md"},
                    distance_or_score=0.1,
                )
            ],
            web_results=[WebSearchResult(title="Agents Update", url="https://example.com", snippet="Recent agents context")],
            answer_text="[Local 1] says agents use tools. [Web 1] adds recent agents context.\n\n[Inference] Together they suggest a broader workflow.",
        )

        response = service.ask(
            QueryRequest(question="agents?", answer_mode="exploratory", retrieval_mode="hybrid")
        )

        self.assertEqual(tracking["chat_calls"], 1)
        self.assertIn("[Local 1] Agents (agents.md)", response.sources)
        self.assertIn("[Web 1] Agents Update (https://example.com)", response.sources)
        self.assertEqual(response.debug.evidence_types_used, ("local_note", "web"))
        self.assertTrue(response.debug.inference_used)
        self.assertEqual(response.debug.answer_mode_used.value, "exploratory")

    def test_answer_mode_flows_into_prompt_payload(self) -> None:
        service, tracking = make_query_service(
            local_chunks=[
                RetrievedChunk(
                    text="Agents use retrieval.",
                    metadata={"note_title": "Agents", "source_path": "agents.md"},
                    distance_or_score=0.1,
                )
            ],
            web_results=[],
            answer_text="Grounded answer [Local 1].",
        )

        response = service.ask(QueryRequest(question="agents?", answer_mode="strict"))

        prompt_payload = tracking["last_prompt"]
        self.assertIsNotNone(prompt_payload)
        self.assertEqual(prompt_payload.answer_mode.value, "strict")
        self.assertIn("Strict mode instructions", prompt_payload.user_prompt)
        self.assertEqual(response.debug.answer_mode_requested.value, "strict")
        self.assertEqual(response.debug.answer_mode_used.value, "strict")

    def test_query_uses_chat_model_override_when_provided(self) -> None:
        service, tracking = make_query_service(
            local_chunks=[
                RetrievedChunk(
                    text="Agents use retrieval.",
                    metadata={"note_title": "Agents", "source_path": "agents.md"},
                    distance_or_score=0.1,
                )
            ],
            web_results=[],
            answer_text="Grounded answer [Local 1].",
        )

        response = service.ask(QueryRequest(question="agents?", chat_model_override="deepseek-r1"))

        self.assertEqual(tracking["last_model"], "deepseek-r1")
        self.assertEqual(response.debug.active_chat_model, "deepseek-r1")

    def test_query_uses_chat_provider_override_when_provided(self) -> None:
        service, tracking = make_query_service(
            local_chunks=[
                RetrievedChunk(
                    text="Agents use retrieval.",
                    metadata={"note_title": "Agents", "source_path": "agents.md"},
                    distance_or_score=0.1,
                )
            ],
            web_results=[],
            answer_text="Grounded answer [Local 1].",
        )

        response = service.ask(QueryRequest(question="agents?", chat_provider_override="openai"))

        self.assertEqual(tracking["last_provider"], "ollama")
        self.assertEqual(response.debug.active_chat_provider, "openai")

    def test_query_uses_openai_client_when_openai_provider_is_configured(self) -> None:
        root = Path(tempfile.mkdtemp())
        (root / "vault").mkdir()
        (root / "output").mkdir()
        config = make_config(root)
        config.chat_provider = "openai"
        config.openai_api_key = "test-key"
        config.openai_chat_model = "gpt-4.1-mini"

        class StubEmbeddingClient:
            def __init__(self, config: AppConfig) -> None:
                pass

        class StubRetriever:
            def __init__(self, config: AppConfig, embedding_client, vector_store) -> None:
                pass

            def retrieve(self, query: str, filters=None, options=None):
                return [
                    RetrievedChunk(
                        text="Agents use retrieval to ground answers.",
                        metadata={"note_title": "Agents", "source_path": "agents.md"},
                        distance_or_score=0.1,
                    )
                ]

        class StubVectorStore:
            def __init__(self, config: AppConfig) -> None:
                pass

            def is_index_compatible(self) -> bool:
                return True

            def count(self) -> int:
                return 1

        class StubWebSearchService:
            def __init__(self, config: AppConfig) -> None:
                pass

            def search(self, query: str) -> list[WebSearchResult]:
                return []

        with patch.object(OpenAIChatClient, "answer_with_prompt", return_value="Grounded answer [Ref 1]."):
            response = QueryService(
                config,
                embedding_client_cls=StubEmbeddingClient,
                retriever_cls=StubRetriever,
                vector_store_cls=StubVectorStore,
                web_search_service_cls=StubWebSearchService,
                capture_debug_trace=False,
            ).ask(QueryRequest(question="agents?"))

        self.assertEqual(response.debug.active_chat_provider, "openai")
        self.assertEqual(response.debug.active_chat_model, "gpt-4.1-mini")

    def test_music_workflow_flows_into_prompt_payload(self) -> None:
        service, tracking = make_query_service(
            local_chunks=[
                RetrievedChunk(
                    text="Breakbeat often relies on syncopated drums.",
                    metadata={"note_title": "Breakbeat Notes", "source_path": "breakbeat.md"},
                    distance_or_score=0.1,
                )
            ],
            web_results=[],
            answer_text="Grounded answer [Local 1].",
        )

        service.ask(
            QueryRequest(
                question="Does this idea fit breakbeat?",
                collaboration_workflow=CollaborationWorkflow.GENRE_FIT_REVIEW,
                workflow_input=WorkflowInput(genre="breakbeat", bpm="135"),
            )
        )

        prompt_payload = tracking["last_prompt"]
        self.assertIsNotNone(prompt_payload)
        self.assertEqual(prompt_payload.collaboration_workflow.value, "genre_fit_review")
        self.assertIn("Collaboration workflow: genre_fit_review", prompt_payload.user_prompt)
        self.assertIn("Assess likely genre or style fit", prompt_payload.user_prompt)
        self.assertIn("Genre: breakbeat", prompt_payload.user_prompt)
        self.assertIn("Producer-collaborator behavior", prompt_payload.system_prompt)
        self.assertIn("Do not open by summarizing sources", prompt_payload.system_prompt)

    def test_critique_workflow_prompt_encourages_implementation_coaching(self) -> None:
        service, tracking = make_query_service(
            local_chunks=[
                RetrievedChunk(
                    text="Track notes about weak transitions.",
                    metadata={"note_title": "Track Notes", "source_path": "track.md"},
                    distance_or_score=0.1,
                )
            ],
            web_results=[],
            answer_text="Grounded answer [Local 1].",
        )

        service.ask(
            QueryRequest(
                question="Critique this transition.",
                collaboration_workflow=CollaborationWorkflow.TRACK_CONCEPT_CRITIQUE,
                workflow_input=WorkflowInput(genre="progressive house"),
            )
        )

        prompt_payload = tracking["last_prompt"]
        self.assertIsNotNone(prompt_payload)
        self.assertIn("Start with a direct answer to the user's music question", prompt_payload.user_prompt)
        self.assertIn("Do not open with framing language", prompt_payload.user_prompt)
        self.assertIn("Every meaningful suggestion must include how to do it", prompt_payload.user_prompt)
        self.assertIn("Do not stop at abstract advice", prompt_payload.user_prompt)
        self.assertIn("prioritize genre-native techniques first", prompt_payload.user_prompt)
        self.assertIn("Ignore weak, tangential, or loosely related sources", prompt_payload.user_prompt)
        self.assertIn("Treat Track Context as long-term track identity and current production state", prompt_payload.user_prompt)
        self.assertIn("analyze it section by section", prompt_payload.user_prompt)
        self.assertIn("Overall Assessment, Arrangement / Energy Flow, Genre / Style Fit, Groove / Bass / Element Evolution, Priority Issues, Recommended Next Changes", prompt_payload.user_prompt)
        self.assertIn("how to implement the change", prompt_payload.user_prompt)
        self.assertIn("first pass", prompt_payload.user_prompt)
        self.assertIn("what to listen for afterward", prompt_payload.user_prompt)
        self.assertIn("why it matters", prompt_payload.user_prompt)
        self.assertIn("provide multiple concrete, usable ideas", prompt_payload.user_prompt)

    def test_arrangement_workflow_prompt_encourages_implementation_coaching(self) -> None:
        service, tracking = make_query_service(
            local_chunks=[
                RetrievedChunk(
                    text="Arrangement notes about low energy sections.",
                    metadata={"note_title": "Arrangement", "source_path": "arrangement.md"},
                    distance_or_score=0.1,
                )
            ],
            web_results=[],
            answer_text="Grounded answer [Local 1].",
        )

        service.ask(
            QueryRequest(
                question="Plan this arrangement.",
                collaboration_workflow=CollaborationWorkflow.ARRANGEMENT_PLANNER,
                workflow_input=WorkflowInput(track_length="6:00"),
            )
        )

        prompt_payload = tracking["last_prompt"]
        self.assertIsNotNone(prompt_payload)
        self.assertIn("Start with a direct answer to the user's music question", prompt_payload.user_prompt)
        self.assertIn("Use retrieved material to support, constrain, or refine the answer after the direct answer", prompt_payload.user_prompt)
        self.assertIn("how to implement it in practical production terms", prompt_payload.user_prompt)
        self.assertIn("minimal first pass", prompt_payload.user_prompt)
        self.assertIn("what to listen for afterward", prompt_payload.user_prompt)
        self.assertIn("Every meaningful suggestion must include how to do it", prompt_payload.user_prompt)
        self.assertIn("Include section-level actions", prompt_payload.user_prompt)
        self.assertIn("rough bar-count logic", prompt_payload.user_prompt)
        self.assertIn("tension and release logic", prompt_payload.user_prompt)

    def test_sound_design_workflow_gains_answer_first_and_genre_grounding(self) -> None:
        service, tracking = make_query_service(
            local_chunks=[
                RetrievedChunk(
                    text="Progressive house bass design notes.",
                    metadata={"note_title": "Bass Design", "source_path": "bass.md"},
                    distance_or_score=0.1,
                )
            ],
            web_results=[],
            answer_text="Grounded answer [Local 1].",
        )

        service.ask(
            QueryRequest(
                question="Give me some progressive house bassline ideas.",
                collaboration_workflow=CollaborationWorkflow.SOUND_DESIGN_BRAINSTORM,
                workflow_input=WorkflowInput(genre="progressive house"),
            )
        )

        prompt_payload = tracking["last_prompt"]
        self.assertIsNotNone(prompt_payload)
        self.assertIn("Start with a direct answer to the user's music question", prompt_payload.user_prompt)
        self.assertIn("Every meaningful suggestion must include how to do it", prompt_payload.user_prompt)
        self.assertIn("prioritize genre-native techniques first", prompt_payload.user_prompt)
        self.assertIn("Treat cross-genre or adjacent-genre ideas as optional variations", prompt_payload.user_prompt)
        self.assertIn("provide multiple concrete, usable ideas", prompt_payload.user_prompt)
        self.assertIn("PRIMARY IDEA", prompt_payload.user_prompt)
        self.assertIn("MIDI PATTERN", prompt_payload.user_prompt)
        self.assertIn("WHY IT WORKS", prompt_payload.user_prompt)
        self.assertIn("SOUND DESIGN", prompt_payload.user_prompt)
        self.assertIn("ONE VARIATION", prompt_payload.user_prompt)
        self.assertIn("FOLLOW-UP", prompt_payload.user_prompt)
        self.assertIn("give exactly one strong idea", prompt_payload.user_prompt)
        self.assertIn("relationship to the kick pattern", prompt_payload.user_prompt)
        self.assertIn("exact Ableton-style timing positions", prompt_payload.user_prompt)
        self.assertNotIn("Production Recipes", prompt_payload.user_prompt)
        self.assertNotIn("Provide at least 2 pattern examples", prompt_payload.user_prompt)

    def test_sound_design_patch_request_keeps_existing_broader_structure(self) -> None:
        service, tracking = make_query_service(
            local_chunks=[
                RetrievedChunk(
                    text="Warm pads benefit from slow attack and filtered upper harmonics.",
                    metadata={"note_title": "Pad Design", "source_path": "pads.md"},
                    distance_or_score=0.1,
                )
            ],
            web_results=[],
            answer_text="Grounded answer [Local 1].",
        )

        service.ask(
            QueryRequest(
                question="Design a warm breakdown pad patch",
                collaboration_workflow=CollaborationWorkflow.SOUND_DESIGN_BRAINSTORM,
                workflow_input=WorkflowInput(genre="progressive house"),
            )
        )

        prompt_payload = tracking["last_prompt"]
        self.assertIsNotNone(prompt_payload)
        self.assertIn("Quick Answer", prompt_payload.user_prompt)
        self.assertIn("Production Recipes", prompt_payload.user_prompt)
        self.assertIn("Groove / MIDI", prompt_payload.user_prompt)
        self.assertIn("How to Build It", prompt_payload.user_prompt)
        self.assertNotIn("PRIMARY IDEA", prompt_payload.user_prompt)
        self.assertNotIn("MIDI PATTERN", prompt_payload.user_prompt)
        self.assertNotIn("give exactly one strong idea", prompt_payload.user_prompt)

    def test_non_research_workflow_gains_shared_collaborator_contract(self) -> None:
        service, tracking = make_query_service(
            local_chunks=[
                RetrievedChunk(
                    text="Breakbeat note.",
                    metadata={"note_title": "Breakbeat", "source_path": "breakbeat.md"},
                    distance_or_score=0.1,
                )
            ],
            web_results=[],
            answer_text="Grounded answer [Local 1].",
        )

        service.ask(
            QueryRequest(
                question="Does this fit breakbeat?",
                collaboration_workflow=CollaborationWorkflow.GENRE_FIT_REVIEW,
                workflow_input=WorkflowInput(genre="breakbeat"),
            )
        )

        prompt_payload = tracking["last_prompt"]
        self.assertIsNotNone(prompt_payload)
        self.assertIn("Producer-collaborator behavior", prompt_payload.system_prompt)
        self.assertIn("Start with the answer, diagnosis, or suggestion", prompt_payload.system_prompt)
        self.assertIn("Do not open by summarizing sources", prompt_payload.system_prompt)
        self.assertNotIn("what to listen for afterward", prompt_payload.user_prompt)
        self.assertNotIn("how to implement the change", prompt_payload.user_prompt)

    def test_other_workflows_do_not_gain_sound_design_structure(self) -> None:
        service, tracking = make_query_service(
            local_chunks=[
                RetrievedChunk(
                    text="Arrangement note.",
                    metadata={"note_title": "Arrangement", "source_path": "arrangement.md"},
                    distance_or_score=0.1,
                )
            ],
            web_results=[],
            answer_text="Grounded answer [Local 1].",
        )

        service.ask(
            QueryRequest(
                question="Plan this arrangement.",
                collaboration_workflow=CollaborationWorkflow.ARRANGEMENT_PLANNER,
                workflow_input=WorkflowInput(track_length="6:00"),
            )
        )

        prompt_payload = tracking["last_prompt"]
        self.assertIsNotNone(prompt_payload)
        self.assertNotIn("Groove / MIDI", prompt_payload.user_prompt)
        self.assertNotIn("How to Build It", prompt_payload.user_prompt)
        self.assertNotIn("Production Recipes", prompt_payload.user_prompt)
        self.assertNotIn("Core recipes must be musically plausible for the requested genre or style", prompt_payload.user_prompt)
        self.assertNotIn("Weakly related or cross-genre retrieved material must not become core recommendations", prompt_payload.user_prompt)

    def test_midi_request_prompt_requires_two_daw_usable_patterns(self) -> None:
        service, tracking = make_query_service(
            local_chunks=[
                RetrievedChunk(
                    text="Progressive house basslines often lock to kick gaps and octave anchors.",
                    metadata={"note_title": "Bassline Notes", "source_path": "bassline.md"},
                    distance_or_score=0.1,
                )
            ],
            web_results=[],
            answer_text="Grounded answer [Local 1].",
        )

        service.ask(
            QueryRequest(
                question="Please share some progressive house bassline MIDI ideas",
                collaboration_workflow=CollaborationWorkflow.GENERAL_ASK,
                workflow_input=WorkflowInput(genre="progressive house"),
            )
        )

        prompt_payload = tracking["last_prompt"]
        self.assertIsNotNone(prompt_payload)
        self.assertIn("Provide at least 2 pattern examples", prompt_payload.user_prompt)
        self.assertIn("Each pattern must include: Bar Length, Timing / Step Grid, and Pitch", prompt_payload.user_prompt)
        self.assertIn("enter directly into a DAW without interpretation", prompt_payload.user_prompt)
        self.assertIn("Do not return only theory, explanation, or source commentary", prompt_payload.user_prompt)
        self.assertIn("Never substitute links, sources, videos, or reference summaries for actual output", prompt_payload.user_prompt)
        self.assertNotIn("careful research assistant for an Obsidian vault", prompt_payload.system_prompt)
        self.assertIn("careful, grounded collaborator for an Obsidian vault", prompt_payload.system_prompt)
        self.assertLess(
            prompt_payload.system_prompt.index("careful, grounded collaborator for an Obsidian vault"),
            prompt_payload.system_prompt.index("Producer-collaborator behavior:"),
        )
        self.assertNotIn("PRIMARY IDEA", prompt_payload.user_prompt)
        self.assertNotIn("FOLLOW-UP", prompt_payload.user_prompt)

    def test_drop_feels_flat_prompt_prioritizes_main_issue_and_next_steps(self) -> None:
        service, tracking = make_query_service(
            local_chunks=[
                RetrievedChunk(
                    text="Flat drops often lack contrast and bass evolution.",
                    metadata={"note_title": "Drop Notes", "source_path": "drop.md"},
                    distance_or_score=0.1,
                )
            ],
            web_results=[],
            answer_text="Grounded answer [Local 1].",
        )

        service.ask(
            QueryRequest(
                question="My drop feels flat, what should I change first?",
                collaboration_workflow=CollaborationWorkflow.TRACK_CONCEPT_CRITIQUE,
            )
        )

        prompt_payload = tracking["last_prompt"]
        self.assertIsNotNone(prompt_payload)
        self.assertIn("Identify the most important issue first", prompt_payload.user_prompt)
        self.assertIn("Explain why that issue matters", prompt_payload.user_prompt)
        self.assertIn("Give prioritized, concrete fixes", prompt_payload.user_prompt)

    def test_arp_request_requires_usable_pitch_and_rhythm(self) -> None:
        service, tracking = make_query_service(
            local_chunks=[
                RetrievedChunk(
                    text="Hypnotic arps work with tight rhythmic repetition and octave control.",
                    metadata={"note_title": "Arp Notes", "source_path": "arp.md"},
                    distance_or_score=0.1,
                )
            ],
            web_results=[],
            answer_text="Grounded answer [Local 1].",
        )

        service.ask(
            QueryRequest(
                question="Give me a dark hypnotic arp idea in A minor",
                collaboration_workflow=CollaborationWorkflow.GENERAL_ASK,
                workflow_input=WorkflowInput(mood="dark hypnotic"),
            )
        )

        prompt_payload = tracking["last_prompt"]
        self.assertIsNotNone(prompt_payload)
        self.assertIn("Provide at least 2 pattern examples", prompt_payload.user_prompt)
        self.assertIn("Timing / Step Grid", prompt_payload.user_prompt)
        self.assertIn("Pitch", prompt_payload.user_prompt)
        self.assertNotIn("refer the user elsewhere", prompt_payload.user_prompt)

    def test_weak_retrieval_still_requires_practical_output(self) -> None:
        service, tracking = make_query_service(
            local_chunks=[
                RetrievedChunk(
                    text="Possibly related note",
                    metadata={"note_title": "Weak Match", "source_path": "weak.md"},
                    distance_or_score=0.95,
                )
            ],
            web_results=[],
            answer_text="Grounded answer [Local 1].",
        )

        service.ask(
            QueryRequest(
                question="Please share some progressive house bassline MIDI ideas",
                collaboration_workflow=CollaborationWorkflow.GENERAL_ASK,
                workflow_input=WorkflowInput(genre="progressive house"),
            )
        )

        prompt_payload = tracking["last_prompt"]
        self.assertIsNotNone(prompt_payload)
        self.assertIn("Retrieval is weak or limited", prompt_payload.user_prompt)
        self.assertIn("must still generate practical output", prompt_payload.user_prompt)
        self.assertIn("Never substitute links, sources, videos, or reference summaries for actual output", prompt_payload.user_prompt)

    def test_track_aware_followup_prefers_direct_answer_when_context_is_sufficient(self) -> None:
        service, tracking = make_query_service(
            local_chunks=[
                RetrievedChunk(
                    text="Garage grooves often use tight hat interplay and shuffled support percussion.",
                    metadata={"note_title": "Garage Notes", "source_path": "garage.md"},
                    distance_or_score=0.1,
                )
            ],
            web_results=[],
            answer_text="Grounded answer [Local 1].",
        )

        response = service.ask(
            QueryRequest(
                question="Give me three better hat ideas for this groove",
                collaboration_workflow=CollaborationWorkflow.GENERAL_ASK,
                track_context=TrackContext(
                    track_id="warehouse-hypnosis-01",
                    genre="garage",
                    bpm=132,
                    vibe=["driving", "dark"],
                ),
                use_track_context=True,
            )
        )

        prompt_payload = tracking["last_prompt"]
        self.assertIsNotNone(prompt_payload)
        self.assertEqual(prompt_payload.response_mode, "direct_answer")
        self.assertEqual(response.debug.response_mode_selected, "direct_answer")
        self.assertFalse(response.debug.followup_triggered)
        self.assertIn("Answer directly and concretely", prompt_payload.system_prompt)

    def test_track_aware_followup_uses_answer_plus_followup_for_missing_drop_context(self) -> None:
        service, tracking = make_query_service(
            local_chunks=[
                RetrievedChunk(
                    text="Hypnotic techno drops need contrast and a clear new anchor element.",
                    metadata={"note_title": "Drop Notes", "source_path": "drop.md"},
                    distance_or_score=0.1,
                )
            ],
            web_results=[],
            answer_text="Grounded answer [Local 1].",
        )

        response = service.ask(
            QueryRequest(
                question="How do I make this drop hit harder?",
                collaboration_workflow=CollaborationWorkflow.TRACK_CONCEPT_CRITIQUE,
                track_context=TrackContext(
                    track_id="warehouse-hypnosis-01",
                    genre="hypnotic techno",
                    bpm=132,
                    vibe=["dark", "driving"],
                ),
                use_track_context=True,
            )
        )

        prompt_payload = tracking["last_prompt"]
        self.assertIsNotNone(prompt_payload)
        self.assertEqual(prompt_payload.response_mode, "answer_plus_followup")
        self.assertEqual(prompt_payload.missing_dimension, "energy_problem_type")
        self.assertEqual(response.debug.response_mode_selected, "answer_plus_followup")
        self.assertTrue(response.debug.followup_triggered)
        self.assertEqual(response.debug.missing_dimension, "energy_problem_type")
        self.assertIn("give brief provisional diagnosis or directional advice first", prompt_payload.system_prompt.lower())

    def test_track_aware_followup_does_not_ask_for_known_identity_fields(self) -> None:
        service, tracking = make_query_service(
            local_chunks=[
                RetrievedChunk(
                    text="Techno breaks often work by holding tension instead of fully resetting.",
                    metadata={"note_title": "Break Notes", "source_path": "break.md"},
                    distance_or_score=0.1,
                )
            ],
            web_results=[],
            answer_text="Grounded answer [Local 1].",
        )

        service.ask(
            QueryRequest(
                question="The break feels like it kills momentum",
                collaboration_workflow=CollaborationWorkflow.TRACK_CONCEPT_CRITIQUE,
                track_context=TrackContext(
                    track_id="warehouse-hypnosis-01",
                    genre="hypnotic techno",
                    bpm=132,
                    reference_tracks=["boris-brejcha-gravity"],
                ),
                use_track_context=True,
            )
        )

        prompt_payload = tracking["last_prompt"]
        self.assertIsNotNone(prompt_payload)
        self.assertIn("Do not ask for details that are already known", prompt_payload.system_prompt)
        self.assertNotIn("What genre is this?", prompt_payload.system_prompt)
        self.assertNotIn("What BPM is this?", prompt_payload.system_prompt)

    def test_track_aware_followup_can_choose_followup_only_for_under_specified_break_issue(self) -> None:
        service, tracking = make_query_service(
            local_chunks=[
                RetrievedChunk(
                    text="Arrangement notes mention momentum problems in breakdowns.",
                    metadata={"note_title": "Arrangement", "source_path": "arrangement.md"},
                    distance_or_score=0.1,
                )
            ],
            web_results=[],
            answer_text="Grounded answer [Local 1].",
        )

        response = service.ask(
            QueryRequest(
                question="The break feels like it kills momentum",
                collaboration_workflow=CollaborationWorkflow.GENERAL_ASK,
            )
        )

        prompt_payload = tracking["last_prompt"]
        self.assertIsNotNone(prompt_payload)
        self.assertEqual(prompt_payload.response_mode, "followup_only")
        self.assertEqual(prompt_payload.missing_dimension, "section_role")
        self.assertEqual(response.debug.response_mode_selected, "followup_only")
        self.assertIn("ask 1 short producer-aware follow-up question only", prompt_payload.system_prompt.lower())

    def test_research_workflow_keeps_direct_research_behavior(self) -> None:
        service, tracking = make_query_service(
            local_chunks=[
                RetrievedChunk(
                    text="Boris Brejcha often uses long-form pacing with controlled introductions of new motifs.",
                    metadata={"note_title": "Boris Notes", "source_path": "boris.md"},
                    distance_or_score=0.1,
                )
            ],
            web_results=[],
            answer_text="Grounded answer [Local 1].",
        )

        response = service.ask(
            QueryRequest(
                question="What does Boris Brejcha usually do with arrangement pacing?",
                collaboration_workflow=CollaborationWorkflow.RESEARCH_SESSION,
            )
        )

        prompt_payload = tracking["last_prompt"]
        self.assertIsNotNone(prompt_payload)
        self.assertEqual(prompt_payload.response_mode, "direct_answer")
        self.assertEqual(response.debug.response_mode_selected, "direct_answer")
        self.assertNotIn("Track-aware follow-up behavior", prompt_payload.system_prompt)
        self.assertIn("careful research assistant for an Obsidian vault", prompt_payload.system_prompt)

    def test_non_research_prompt_keeps_collaborator_identity_ahead_of_behavior_block(self) -> None:
        service, tracking = make_query_service(
            local_chunks=[
                RetrievedChunk(
                    text="Arrangement note.",
                    metadata={"note_title": "Arrangement", "source_path": "arrangement.md"},
                    distance_or_score=0.1,
                )
            ],
            web_results=[],
            answer_text="Grounded answer [Local 1].",
        )

        service.ask(
            QueryRequest(
                question="How do I make this drop hit harder?",
                collaboration_workflow=CollaborationWorkflow.TRACK_CONCEPT_CRITIQUE,
            )
        )

        prompt_payload = tracking["last_prompt"]
        self.assertIsNotNone(prompt_payload)
        identity_index = prompt_payload.system_prompt.index(
            "careful, grounded collaborator for an Obsidian vault"
        )
        collaborator_block_index = prompt_payload.system_prompt.index("Producer-collaborator behavior:")
        self.assertLess(identity_index, collaborator_block_index)
        self.assertNotIn("careful research assistant for an Obsidian vault", prompt_payload.system_prompt)

    def test_arrangement_metadata_prevents_unnecessary_drop_followup(self) -> None:
        service, tracking = make_query_service(
            local_chunks=[
                RetrievedChunk(
                    text=(
                        "# S2 - First Drop\n\n"
                        "Track: Warehouse Hypnosis\n"
                        "Bars: 33-48\n"
                        "Energy: 7\n"
                        "Purpose: release pressure with a stronger bass anchor\n"
                    ),
                    metadata={
                        "note_title": "Warehouse Hypnosis Arrangement",
                        "source_path": "Projects/Warehouse Hypnosis/arrangement.md",
                        "source_type": "track_arrangement",
                        "heading_context": "S2 - First Drop",
                        "arrangement_track_name": "Warehouse Hypnosis",
                        "arrangement_section_name": "First Drop",
                        "arrangement_energy": 7,
                    },
                    distance_or_score=0.1,
                )
            ],
            web_results=[],
            answer_text="Grounded answer [Local 1].",
        )

        response = service.ask(
            QueryRequest(
                question="How do I make this drop hit harder?",
                collaboration_workflow=CollaborationWorkflow.TRACK_CONCEPT_CRITIQUE,
                track_context=TrackContext(
                    track_id="warehouse-hypnosis-01",
                    genre="hypnotic techno",
                    bpm=132,
                    vibe=["dark", "driving"],
                ),
                use_track_context=True,
            )
        )

        prompt_payload = tracking["last_prompt"]
        self.assertIsNotNone(prompt_payload)
        self.assertEqual(prompt_payload.response_mode, "direct_answer")
        self.assertEqual(response.debug.response_mode_selected, "direct_answer")
        self.assertEqual(response.debug.missing_dimension, "")

    def test_arrangement_section_purpose_prevents_unnecessary_break_followup(self) -> None:
        service, tracking = make_query_service(
            local_chunks=[
                RetrievedChunk(
                    text=(
                        "# S3 - Breakdown\n\n"
                        "Track: Warehouse Hypnosis\n"
                        "Bars: 49-64\n"
                        "Energy: 4\n"
                        "Purpose: keep tension alive while stripping low-end weight\n"
                    ),
                    metadata={
                        "note_title": "Warehouse Hypnosis Arrangement",
                        "source_path": "Projects/Warehouse Hypnosis/arrangement.md",
                        "source_type": "track_arrangement",
                        "heading_context": "S3 - Breakdown",
                        "arrangement_track_name": "Warehouse Hypnosis",
                        "arrangement_section_name": "Breakdown",
                        "arrangement_energy": 4,
                    },
                    distance_or_score=0.1,
                )
            ],
            web_results=[],
            answer_text="Grounded answer [Local 1].",
        )

        response = service.ask(
            QueryRequest(
                question="The break feels like it kills momentum",
                collaboration_workflow=CollaborationWorkflow.GENERAL_ASK,
            )
        )

        prompt_payload = tracking["last_prompt"]
        self.assertIsNotNone(prompt_payload)
        self.assertEqual(prompt_payload.response_mode, "direct_answer")
        self.assertEqual(response.debug.response_mode_selected, "direct_answer")
        self.assertEqual(response.debug.missing_dimension, "")

    def test_explicit_section_reference_flows_into_prompt_and_debug(self) -> None:
        service, tracking = make_query_service(
            local_chunks=[
                RetrievedChunk(
                    text="Drop notes about impact and contrast.",
                    metadata={"note_title": "Drop Notes", "source_path": "drop.md"},
                    distance_or_score=0.1,
                )
            ],
            web_results=[],
            answer_text="Grounded answer [Local 1].",
        )

        response = service.ask(
            QueryRequest(
                question="How do I make this drop hit harder?",
                collaboration_workflow=CollaborationWorkflow.TRACK_CONCEPT_CRITIQUE,
                track_context=TrackContext(
                    track_id="warehouse-hypnosis-01",
                    sections={
                        "drop": SectionContext(
                            name="drop",
                            role="main groove",
                            energy_level="high",
                            elements=["kick", "bass"],
                        )
                    },
                ),
                use_track_context=True,
            )
        )

        prompt_payload = tracking["last_prompt"]
        self.assertIsNotNone(prompt_payload)
        self.assertEqual(prompt_payload.active_section, "drop")
        self.assertEqual(response.debug.active_section, "drop")
        self.assertIn("Active section: drop.", prompt_payload.system_prompt)
        self.assertIn("Known section role: main groove.", prompt_payload.system_prompt)
        self.assertIn("If the section is known, name it directly", prompt_payload.system_prompt)

    def test_session_section_focus_can_drive_active_section_without_explicit_question_label(self) -> None:
        service, tracking = make_query_service(
            local_chunks=[
                RetrievedChunk(
                    text="Break notes about tension and momentum.",
                    metadata={"note_title": "Break Notes", "source_path": "break.md"},
                    distance_or_score=0.1,
                )
            ],
            web_results=[],
            answer_text="Grounded answer [Local 1].",
        )

        response = service.ask(
            QueryRequest(
                question="How do I keep the momentum up here?",
                collaboration_workflow=CollaborationWorkflow.TRACK_CONCEPT_CRITIQUE,
                track_context=TrackContext(
                    track_id="warehouse-hypnosis-01",
                    sections={
                        "break": SectionContext(
                            name="break",
                            role="keep tension alive while stripping weight",
                            energy_level="medium",
                            elements=["perc loop", "arp tail"],
                        )
                    },
                ),
                use_track_context=True,
                section_focus="break",
            )
        )

        prompt_payload = tracking["last_prompt"]
        self.assertIsNotNone(prompt_payload)
        self.assertEqual(prompt_payload.active_section, "break")
        self.assertEqual(response.debug.active_section, "break")
        self.assertIn("The active section in play is: break.", prompt_payload.system_prompt)
        self.assertIn("Known section role: keep tension alive while stripping weight.", prompt_payload.system_prompt)
