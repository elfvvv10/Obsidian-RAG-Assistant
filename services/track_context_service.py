"""Track-context loading and formatting for workflow-aware prompt injection."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from config import AppConfig
from metadata_parser import parse_markdown_metadata
from services.models import CollaborationWorkflow
from utils import get_logger


logger = get_logger()


_TRACK_CONTEXT_FIELDS: tuple[str, ...] = (
    "type",
    "project_id",
    "track_title",
    "primary_genre",
    "secondary_influences",
    "bpm",
    "key",
    "time_signature",
    "vibe",
    "energy_profile",
    "reference_artists",
    "reference_tracks",
    "listener_goal",
    "status",
    "current_section",
    "completion_estimate",
    "last_major_change",
    "current_issues",
    "priority_focus",
    "notes",
    "tags",
)


@dataclass(slots=True)
class TrackContextResult:
    """Parsed track-context payload and prompt-ready block."""

    resolved_path: Path | None = None
    frontmatter: dict[str, object] | None = None
    body: str = ""
    prompt_block: str = ""
    found: bool = False


class TrackContextService:
    """Resolve, parse, and format per-track context documents."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config

    def get_track_context(
        self,
        workflow: CollaborationWorkflow,
        track_context_path: str | None,
    ) -> TrackContextResult:
        """Return prompt-ready context for supported workflows, or an empty result."""
        if workflow not in {
            CollaborationWorkflow.TRACK_CONCEPT_CRITIQUE,
            CollaborationWorkflow.ARRANGEMENT_PLANNER,
        }:
            return TrackContextResult()

        resolved_path = self._resolve_track_context_path(track_context_path)
        if resolved_path is None:
            self._debug_log(
                "Track context lookup: no track_context_path provided for workflow=%s.",
                workflow.value,
            )
            return TrackContextResult()

        if not resolved_path.exists() or not resolved_path.is_file():
            self._debug_log(
                "Track context lookup: path missing for workflow=%s at %s.",
                workflow.value,
                resolved_path,
            )
            return TrackContextResult(resolved_path=resolved_path)

        raw_content = resolved_path.read_text(encoding="utf-8")
        frontmatter, body = parse_markdown_metadata(raw_content)
        prompt_block = self._format_prompt_block(frontmatter, body)
        self._debug_log(
            "Track context lookup: loaded context for workflow=%s from %s.",
            workflow.value,
            resolved_path,
        )
        return TrackContextResult(
            resolved_path=resolved_path,
            frontmatter=frontmatter,
            body=body,
            prompt_block=prompt_block,
            found=bool(prompt_block),
        )

    def _resolve_track_context_path(self, track_context_path: str | None) -> Path | None:
        raw_path = (track_context_path or "").strip()
        if not raw_path:
            return None

        candidate = Path(raw_path).expanduser()
        if not candidate.is_absolute():
            candidate = (self.config.obsidian_vault_path / candidate).resolve()

        if candidate.suffix.lower() == ".md":
            return candidate
        return (candidate / "track_context.md").resolve()

    def _format_prompt_block(self, frontmatter: dict[str, object], body: str) -> str:
        summary_lines = [
            f"- {field.replace('_', ' ').title()}: {self._format_field_value(frontmatter[field])}"
            for field in _TRACK_CONTEXT_FIELDS
            if field in frontmatter and self._format_field_value(frontmatter[field])
        ]
        body = body.strip()
        body_block = f"\n\nTrack context notes:\n{body}" if body else ""
        if not summary_lines and not body:
            return ""
        summary_block = "\n".join(summary_lines)
        if summary_block:
            summary_block = f"Track context summary:\n{summary_block}"
        instruction = (
            "Use this as internal track-state guidance for continuity, prioritization, and finish-oriented advice. "
            "Do not treat it as evidence or a citation source."
        )
        content_parts = [instruction]
        if summary_block:
            content_parts.append(summary_block)
        if body_block:
            content_parts.append(body_block.lstrip("\n"))
        return "\n\n".join(content_parts)

    def _format_field_value(self, value: object) -> str:
        if isinstance(value, list):
            return ", ".join(str(item).strip() for item in value if str(item).strip())
        return str(value).strip()

    def _debug_log(self, message: str, *args: object) -> None:
        if self.config.framework_debug:
            logger.info(message, *args)
