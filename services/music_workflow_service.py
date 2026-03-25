"""Music collaboration workflow routing and save-path helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from config import AppConfig
from services.models import CollaborationWorkflow, QueryRequest, ResearchRequest, WorkflowInput


@dataclass(slots=True)
class WorkflowExecutionPlan:
    """Normalized workflow execution details."""

    prompt_text: str
    save_path: Path


class MusicWorkflowService:
    """Keep music workflow routing and save conventions out of the UI."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config

    def build_query_plan(self, request: QueryRequest) -> WorkflowExecutionPlan:
        return WorkflowExecutionPlan(
            prompt_text=self._build_prompt_text(request.question, request.workflow_input),
            save_path=self.default_save_path(request.collaboration_workflow),
        )

    def build_research_plan(self, request: ResearchRequest) -> WorkflowExecutionPlan:
        return WorkflowExecutionPlan(
            prompt_text=self._build_prompt_text(request.goal, request.workflow_input),
            save_path=self.default_save_path(request.collaboration_workflow),
        )

    def default_save_path(self, workflow: CollaborationWorkflow) -> Path:
        if workflow == CollaborationWorkflow.RESEARCH_SESSION:
            return self.config.research_sessions_path
        return self.config.draft_answers_path / _draft_folder_name(workflow)

    def _build_prompt_text(self, prompt_text: str, workflow_input: WorkflowInput) -> str:
        values = workflow_input.as_dict()
        if not values:
            return prompt_text
        lines = [prompt_text, "", "Structured workflow context:"]
        lines.extend(f"- {key.replace('_', ' ')}: {value}" for key, value in values.items())
        return "\n".join(lines)


def _draft_folder_name(workflow: CollaborationWorkflow) -> str:
    return {
        CollaborationWorkflow.GENERAL_ASK: "answers/General Asks",
        CollaborationWorkflow.GENRE_FIT_REVIEW: "critiques/Genre Fit Reviews",
        CollaborationWorkflow.TRACK_CONCEPT_CRITIQUE: "critiques/Track Concept Critiques",
        CollaborationWorkflow.ARRANGEMENT_PLANNER: "answers/Arrangement Plans",
        CollaborationWorkflow.SOUND_DESIGN_BRAINSTORM: "answers/Sound Design Brainstorms",
    }.get(workflow, "answers/General Asks")
