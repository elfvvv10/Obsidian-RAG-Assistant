"""Structured Track Context update proposal parsing and apply helpers."""

from __future__ import annotations

from dataclasses import asdict
import json
import re
from typing import Any

from services.models import TrackContext, TrackContextUpdateProposal
from services.track_context_utils import normalize_track_context

_UPDATE_BLOCK_PATTERN = re.compile(
    r"```track_context_update\s*(\{.*?\})\s*```",
    re.IGNORECASE | re.DOTALL,
)

_SCALAR_FIELD_ALIASES = {
    "title": "track_name",
    "track_name": "track_name",
    "genre": "genre",
    "bpm": "bpm",
    "key": "key",
    "status": "current_stage",
    "current_stage": "current_stage",
    "current_problem": "current_problem",
}

_LIST_FIELD_ALIASES = {
    "vibe": "vibe",
    "references": "reference_tracks",
    "reference_tracks": "reference_tracks",
    "current_issues": "known_issues",
    "known_issues": "known_issues",
    "next_actions": "goals",
    "goals": "goals",
}

_VALID_CONFIDENCE = {"low", "medium", "high"}
_VALID_SECTION_FIELDS = {"name", "bars", "role", "energy_level", "elements", "issues", "notes"}


class TrackContextUpdateService:
    """Handle structured Track Context update proposal extraction and application."""

    def extract(
        self,
        answer: str,
        track_context: TrackContext | None,
    ) -> tuple[str, TrackContextUpdateProposal | None]:
        """Return the answer without any proposal block plus an optional parsed proposal."""
        if track_context is None:
            return answer, None

        match = _UPDATE_BLOCK_PATTERN.search(answer)
        if not match:
            return answer, None

        cleaned_answer = self._strip_update_block(answer)
        try:
            payload = json.loads(match.group(1))
        except json.JSONDecodeError:
            return cleaned_answer, None

        proposal = self._normalize_proposal(payload, track_context.track_id)
        return cleaned_answer, proposal

    def request_structured_proposal(
        self,
        chat_client: object,
        *,
        question: str,
        answer: str,
        track_context: TrackContext | None,
        structured_output_supported: bool,
    ) -> TrackContextUpdateProposal | None:
        """Best-effort provider-aware structured proposal request."""
        if track_context is None or not structured_output_supported:
            return None

        request_fn = getattr(chat_client, "answer_with_json_schema", None)
        if not callable(request_fn):
            return None

        try:
            payload = request_fn(
                system_prompt=_STRUCTURED_PROPOSAL_SYSTEM_PROMPT,
                user_prompt=_structured_proposal_user_prompt(
                    question=question,
                    answer=answer,
                    track_context=track_context,
                ),
                schema_name="track_context_update_proposal",
                json_schema=_TRACK_CONTEXT_UPDATE_JSON_SCHEMA,
            )
        except Exception:
            return None

        return self._normalize_proposal(payload, track_context.track_id)

    def preview(
        self,
        track_context: TrackContext,
        proposal: TrackContextUpdateProposal | None,
    ) -> TrackContext:
        """Return a preview of Track Context after applying the proposal."""
        return self.apply(track_context, proposal)

    def apply(
        self,
        track_context: TrackContext,
        proposal: TrackContextUpdateProposal | None,
    ) -> TrackContext:
        """Apply a valid proposal onto an existing Track Context in memory."""
        if proposal is None or proposal.is_empty():
            return track_context
        if proposal.track_id and proposal.track_id != track_context.track_id:
            return track_context

        merged = asdict(track_context)
        for key, value in proposal.set_fields.items():
            normalized_key = _SCALAR_FIELD_ALIASES.get(key)
            if not normalized_key:
                continue
            cleaned_value = _clean_scalar_value(value)
            if cleaned_value is None:
                continue
            merged[normalized_key] = cleaned_value

        for key, items in proposal.add_to_lists.items():
            normalized_key = _LIST_FIELD_ALIASES.get(key)
            if not normalized_key:
                continue
            merged[normalized_key] = _merge_unique_list(
                list(merged.get(normalized_key, []) or []),
                items,
            )

        for key, items in proposal.remove_from_lists.items():
            normalized_key = _LIST_FIELD_ALIASES.get(key)
            if not normalized_key:
                continue
            merged[normalized_key] = _remove_items_from_list(
                list(merged.get(normalized_key, []) or []),
                items,
            )

        merged["sections"] = self._apply_section_updates(
            merged.get("sections"),
            set_sections=proposal.set_sections,
            add_section_issues=proposal.add_section_issues,
            remove_section_issues=proposal.remove_section_issues,
            add_section_elements=proposal.add_section_elements,
            add_section_notes=proposal.add_section_notes,
        )

        merged["track_id"] = track_context.track_id
        return normalize_track_context(merged)

    def _normalize_proposal(
        self,
        payload: object,
        active_track_id: str,
    ) -> TrackContextUpdateProposal | None:
        if not isinstance(payload, dict):
            return None

        raw_track_id = str(payload.get("track_id", "")).strip()
        if raw_track_id and raw_track_id != active_track_id:
            return None

        proposal = TrackContextUpdateProposal(
            track_id=active_track_id,
            summary=str(payload.get("summary", "")).strip(),
            set_fields=self._normalize_scalar_updates(payload.get("set_fields")),
            add_to_lists=self._normalize_list_updates(payload.get("add_to_lists")),
            remove_from_lists=self._normalize_list_updates(payload.get("remove_from_lists")),
            set_sections=self._normalize_section_updates(payload.get("set_sections")),
            add_section_issues=self._normalize_section_issue_updates(payload.get("add_section_issues")),
            remove_section_issues=self._normalize_section_issue_updates(payload.get("remove_section_issues")),
            add_section_elements=self._normalize_section_list_updates(payload.get("add_section_elements")),
            add_section_notes=self._normalize_section_list_updates(payload.get("add_section_notes")),
            section_focus=self._normalize_section_focus(
                payload.get("section_focus", payload.get("set_active_section"))
            ),
            confidence=self._normalize_confidence(payload.get("confidence")),
            source_reasoning=str(payload.get("source_reasoning", "")).strip(),
        )
        return None if proposal.is_empty() else proposal

    def _normalize_scalar_updates(self, value: object) -> dict[str, object]:
        if not isinstance(value, dict):
            return {}
        normalized: dict[str, object] = {}
        for key, raw_value in value.items():
            normalized_key = str(key).strip()
            if normalized_key not in _SCALAR_FIELD_ALIASES:
                continue
            cleaned_value = _clean_scalar_value(raw_value)
            if cleaned_value is None:
                continue
            normalized[normalized_key] = cleaned_value
        return normalized

    def _normalize_list_updates(self, value: object) -> dict[str, list[str]]:
        if not isinstance(value, dict):
            return {}
        normalized: dict[str, list[str]] = {}
        for key, raw_items in value.items():
            normalized_key = str(key).strip()
            if normalized_key not in _LIST_FIELD_ALIASES:
                continue
            cleaned_items = _clean_string_list(raw_items)
            if cleaned_items:
                normalized[normalized_key] = cleaned_items
        return normalized

    def _normalize_confidence(self, value: object) -> str:
        cleaned = str(value or "").strip().lower()
        return cleaned if cleaned in _VALID_CONFIDENCE else ""

    def _normalize_section_focus(self, value: object) -> str:
        cleaned = str(value or "").strip().lower()
        return cleaned

    def _normalize_section_updates(self, value: object) -> dict[str, dict[str, object]]:
        if not isinstance(value, dict):
            return {}
        normalized: dict[str, dict[str, object]] = {}
        for raw_section_key, raw_section_payload in value.items():
            section_key = str(raw_section_key).strip().lower()
            if not section_key or not isinstance(raw_section_payload, dict):
                continue
            cleaned_section: dict[str, object] = {}
            for raw_field, raw_field_value in raw_section_payload.items():
                field_name = str(raw_field).strip()
                if field_name not in _VALID_SECTION_FIELDS:
                    continue
                if field_name in {"elements", "issues"}:
                    cleaned_items = _clean_string_list(raw_field_value)
                    if cleaned_items:
                        cleaned_section[field_name] = cleaned_items
                    continue
                cleaned_value = _clean_scalar_value(raw_field_value)
                if cleaned_value is not None:
                    cleaned_section[field_name] = cleaned_value
            if cleaned_section:
                normalized[section_key] = cleaned_section
        return normalized

    def _normalize_section_issue_updates(self, value: object) -> dict[str, list[str]]:
        return self._normalize_section_list_updates(value)

    def _normalize_section_list_updates(self, value: object) -> dict[str, list[str]]:
        if not isinstance(value, dict):
            return {}
        normalized: dict[str, list[str]] = {}
        for raw_section_key, raw_items in value.items():
            section_key = str(raw_section_key).strip().lower()
            if not section_key:
                continue
            cleaned_items = _clean_string_list(raw_items)
            if cleaned_items:
                normalized[section_key] = cleaned_items
        return normalized

    def _apply_section_updates(
        self,
        current_sections: object,
        *,
        set_sections: dict[str, dict[str, object]],
        add_section_issues: dict[str, list[str]],
        remove_section_issues: dict[str, list[str]],
        add_section_elements: dict[str, list[str]],
        add_section_notes: dict[str, list[str]],
    ) -> dict[str, object]:
        normalized_sections: dict[str, object] = {}
        if isinstance(current_sections, dict):
            normalized_sections = {
                str(key): (asdict(value) if hasattr(value, "__dataclass_fields__") else dict(value))
                for key, value in current_sections.items()
            }

        for section_key, section_updates in set_sections.items():
            existing = normalized_sections.get(section_key, {"name": section_key})
            if not isinstance(existing, dict):
                existing = {"name": section_key}
            merged_section = dict(existing)
            merged_section.setdefault("name", section_key)
            for field_name, field_value in section_updates.items():
                if field_name in {"elements", "issues"}:
                    merged_section[field_name] = _merge_unique_list(
                        list(merged_section.get(field_name, []) or []),
                        list(field_value if isinstance(field_value, list) else [field_value]),
                    )
                else:
                    merged_section[field_name] = field_value
            normalized_sections[section_key] = merged_section

        for section_key, issues in add_section_issues.items():
            existing = normalized_sections.get(section_key, {"name": section_key})
            if not isinstance(existing, dict):
                existing = {"name": section_key}
            merged_section = dict(existing)
            merged_section.setdefault("name", section_key)
            merged_section["issues"] = _merge_unique_list(
                list(merged_section.get("issues", []) or []),
                issues,
            )
            normalized_sections[section_key] = merged_section

        for section_key, issues in remove_section_issues.items():
            existing = normalized_sections.get(section_key)
            if not isinstance(existing, dict):
                continue
            merged_section = dict(existing)
            merged_section["issues"] = _remove_items_from_list(
                list(merged_section.get("issues", []) or []),
                issues,
            )
            normalized_sections[section_key] = merged_section

        for section_key, elements in add_section_elements.items():
            existing = normalized_sections.get(section_key, {"name": section_key})
            if not isinstance(existing, dict):
                existing = {"name": section_key}
            merged_section = dict(existing)
            merged_section.setdefault("name", section_key)
            merged_section["elements"] = _merge_unique_list(
                list(merged_section.get("elements", []) or []),
                elements,
            )
            normalized_sections[section_key] = merged_section

        for section_key, notes in add_section_notes.items():
            existing = normalized_sections.get(section_key, {"name": section_key})
            if not isinstance(existing, dict):
                existing = {"name": section_key}
            merged_section = dict(existing)
            merged_section.setdefault("name", section_key)
            current_notes = str(merged_section.get("notes", "") or "").strip()
            note_items = [note for note in notes if note.strip()]
            if not note_items:
                normalized_sections[section_key] = merged_section
                continue
            if not current_notes:
                merged_section["notes"] = "\n".join(note_items)
            else:
                existing_lines = [line.strip() for line in current_notes.splitlines() if line.strip()]
                merged_section["notes"] = "\n".join(_merge_unique_list(existing_lines, note_items))
            normalized_sections[section_key] = merged_section

        return normalized_sections

    def _strip_update_block(self, answer: str) -> str:
        cleaned = _UPDATE_BLOCK_PATTERN.sub("", answer).strip()
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
        return cleaned


def _clean_scalar_value(value: object) -> object | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        if isinstance(value, float) and value.is_integer():
            return int(value)
        return value
    cleaned = str(value).strip()
    if not cleaned:
        return None
    if cleaned.replace(".", "", 1).isdigit():
        try:
            return int(float(cleaned))
        except ValueError:
            return cleaned
    return cleaned


def _clean_string_list(value: object) -> list[str]:
    if value is None:
        return []
    items = value if isinstance(value, list) else [value]
    cleaned: list[str] = []
    seen: set[str] = set()
    for item in items:
        normalized = str(item).strip()
        if not normalized:
            continue
        lowered = normalized.lower()
        if lowered in seen:
            continue
        seen.add(lowered)
        cleaned.append(normalized)
    return cleaned


def _merge_unique_list(existing: list[str], additions: list[str]) -> list[str]:
    merged = list(existing)
    seen = {item.strip().lower() for item in existing if item.strip()}
    for item in additions:
        cleaned = item.strip()
        if not cleaned or cleaned.lower() in seen:
            continue
        seen.add(cleaned.lower())
        merged.append(cleaned)
    return merged


def _remove_items_from_list(existing: list[str], removals: list[str]) -> list[str]:
    removal_set = {item.strip().lower() for item in removals if item.strip()}
    if not removal_set:
        return existing
    return [item for item in existing if item.strip().lower() not in removal_set]


def _structured_proposal_user_prompt(
    *,
    question: str,
    answer: str,
    track_context: TrackContext,
) -> str:
    return (
        f"Active track_id: {track_context.track_id}\n"
        f"Active title: {track_context.track_name or ''}\n"
        f"Current genre: {track_context.genre or ''}\n"
        f"Current bpm: {track_context.bpm or ''}\n"
        f"Current key: {track_context.key or ''}\n"
        f"Current vibe: {', '.join(track_context.vibe)}\n"
        f"Current references: {', '.join(track_context.reference_tracks)}\n"
        f"Current issues: {', '.join(track_context.known_issues)}\n"
        f"Current next actions: {', '.join(track_context.goals)}\n\n"
        "Current sections:\n"
        f"{_format_sections_for_prompt(track_context)}\n\n"
        f"User question:\n{question}\n\n"
        f"Assistant answer:\n{answer}\n\n"
        "Return a structured proposal only if the question/answer pair clearly supports a conservative Track Context update. "
        "Prefer no changes over weak or speculative changes."
    )


_STRUCTURED_PROPOSAL_SYSTEM_PROMPT = (
    "You extract optional Track Context update proposals for an active in-progress music track. "
    "Return only a conservative JSON object that matches the schema. "
    "Do not invent updates. "
    "Use set_fields only for clear corrections or explicit user-provided values. "
    "Use add_to_lists for grounded current issues, references, vibe tags, or next actions. "
    "Use set_sections and add_section_issues when the conversation clearly establishes section-specific track state. "
    "Use add_section_elements and add_section_notes for additive section memory that helps track what is present or what should change. "
    "Use section_focus when the conversation clearly shifts to a specific section that should stay in focus for the next turn or two. "
    "If no meaningful update is warranted, return an empty proposal."
)

_TRACK_CONTEXT_UPDATE_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "track_id": {"type": "string"},
        "summary": {"type": "string"},
        "set_fields": {
            "type": "object",
            "properties": {
                "title": {"type": "string"},
                "genre": {"type": "string"},
                "bpm": {"type": "integer"},
                "key": {"type": "string"},
                "status": {"type": "string"},
                "current_stage": {"type": "string"},
                "current_problem": {"type": "string"},
            },
            "additionalProperties": False,
        },
        "add_to_lists": {
            "type": "object",
            "properties": {
                "vibe": {"type": "array", "items": {"type": "string"}},
                "references": {"type": "array", "items": {"type": "string"}},
                "current_issues": {"type": "array", "items": {"type": "string"}},
                "next_actions": {"type": "array", "items": {"type": "string"}},
            },
            "additionalProperties": False,
        },
        "remove_from_lists": {
            "type": "object",
            "properties": {
                "vibe": {"type": "array", "items": {"type": "string"}},
                "references": {"type": "array", "items": {"type": "string"}},
                "current_issues": {"type": "array", "items": {"type": "string"}},
                "next_actions": {"type": "array", "items": {"type": "string"}},
            },
            "additionalProperties": False,
        },
        "confidence": {"type": "string", "enum": ["", "low", "medium", "high"]},
        "source_reasoning": {"type": "string"},
        "set_sections": {
            "type": "object",
            "additionalProperties": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "bars": {"type": "string"},
                    "role": {"type": "string"},
                    "energy_level": {"type": "string"},
                    "elements": {"type": "array", "items": {"type": "string"}},
                    "issues": {"type": "array", "items": {"type": "string"}},
                    "notes": {"type": "string"},
                },
                "additionalProperties": False,
            },
        },
        "add_section_issues": {
            "type": "object",
            "additionalProperties": {"type": "array", "items": {"type": "string"}},
        },
        "remove_section_issues": {
            "type": "object",
            "additionalProperties": {"type": "array", "items": {"type": "string"}},
        },
        "add_section_elements": {
            "type": "object",
            "additionalProperties": {"type": "array", "items": {"type": "string"}},
        },
        "add_section_notes": {
            "type": "object",
            "additionalProperties": {"type": "array", "items": {"type": "string"}},
        },
        "section_focus": {"type": "string"},
    },
    "required": [
        "track_id",
        "summary",
        "set_fields",
        "add_to_lists",
        "remove_from_lists",
        "confidence",
        "source_reasoning",
        "set_sections",
        "add_section_issues",
        "remove_section_issues",
        "add_section_elements",
        "add_section_notes",
        "section_focus",
    ],
    "additionalProperties": False,
}


def _format_sections_for_prompt(track_context: TrackContext) -> str:
    if not track_context.sections:
        return "None"
    lines: list[str] = []
    for section_key, section in track_context.sections.items():
        parts = [section.name or section_key]
        if section.role:
            parts.append(f"role={section.role}")
        if section.energy_level:
            parts.append(f"energy={section.energy_level}")
        if section.bars:
            parts.append(f"bars={section.bars}")
        if section.issues:
            parts.append(f"issues={', '.join(section.issues)}")
        lines.append(f"- {section_key}: " + " | ".join(parts))
    return "\n".join(lines)
