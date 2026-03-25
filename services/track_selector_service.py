"""Helpers for selecting legacy markdown track contexts from the vault."""

from __future__ import annotations

from pathlib import Path

from metadata_parser import parse_markdown_metadata


_WORKFLOW_AUTOFILL_KEYS: tuple[str, ...] = (
    "workflow_genre",
    "workflow_references",
    "workflow_bpm",
    "workflow_mood",
    "workflow_energy_goal",
    "workflow_track_length",
    "workflow_role_of_key_elements",
    "workflow_arrangement_notes",
    "workflow_instrumentation",
    "workflow_sound_palette",
)


class TrackSelectorService:
    """Discover track folders that expose a legacy markdown track context."""

    def list_tracks(self, vault_path: Path) -> list[dict[str, str]]:
        """Return sorted relative markdown track-context paths under Projects/."""
        projects_path = vault_path / "Projects"
        if not projects_path.exists() or not projects_path.is_dir():
            return []

        tracks: list[dict[str, str]] = []
        for track_context_path in projects_path.rglob("track_context.md"):
            if not track_context_path.is_file():
                continue
            track_folder = track_context_path.parent
            try:
                relative_folder = track_folder.relative_to(projects_path)
            except ValueError:
                continue
            display_name = " / ".join(relative_folder.parts) or track_folder.name
            tracks.append(
                {
                    "name": display_name,
                    "path": track_context_path.relative_to(vault_path).as_posix(),
                }
            )
        return sorted(tracks, key=lambda item: item["name"].lower())

    def load_workflow_context(self, vault_path: Path, track_context_path: str) -> dict[str, str]:
        """Map a legacy markdown track context into workflow form fields."""
        resolved_path = _resolve_track_context_path(vault_path, track_context_path)
        if resolved_path is None or not resolved_path.exists() or not resolved_path.is_file():
            return {key: "" for key in _WORKFLOW_AUTOFILL_KEYS}

        frontmatter, body = parse_markdown_metadata(resolved_path.read_text(encoding="utf-8"))
        core_ideas = _extract_markdown_section(body, "Core Ideas")
        structure = _extract_markdown_section(body, "Structure")
        recent_decisions = _extract_markdown_section(body, "Recent Decisions")

        references = _join_items(
            _as_list(frontmatter.get("reference_tracks")),
            _as_list(frontmatter.get("reference_artists")),
            _as_list(frontmatter.get("secondary_influences")),
        )
        arrangement_notes = _join_blocks(
            _labeled_block("Status", _clean_text(frontmatter.get("status"))),
            _labeled_block("Structure", structure),
            _labeled_block("Current Issues", _join_lines(_as_list(frontmatter.get("current_issues")))),
            _labeled_block("Priority Focus", _join_lines(_as_list(frontmatter.get("priority_focus")))),
            _labeled_block("Recent Decisions", recent_decisions),
        )
        sound_palette = _join_blocks(
            _labeled_block("Vibe", _join_items(_as_list(frontmatter.get("vibe")))),
            _labeled_block("Influences", _join_items(_as_list(frontmatter.get("secondary_influences")))),
        )

        return {
            "workflow_genre": _clean_text(frontmatter.get("primary_genre")),
            "workflow_references": references,
            "workflow_bpm": _clean_text(frontmatter.get("bpm")),
            "workflow_mood": _join_items(_as_list(frontmatter.get("vibe"))),
            "workflow_energy_goal": _first_non_empty(
                _clean_text(frontmatter.get("listener_goal")),
                _clean_text(frontmatter.get("energy_profile")),
            ),
            "workflow_track_length": _first_non_empty(
                _clean_text(frontmatter.get("track_length")),
                _clean_text(frontmatter.get("duration")),
            ),
            "workflow_role_of_key_elements": core_ideas,
            "workflow_arrangement_notes": arrangement_notes,
            "workflow_instrumentation": core_ideas,
            "workflow_sound_palette": sound_palette,
        }


def selected_track_path(selected_track: str, tracks: list[dict[str, str]]) -> str | None:
    """Resolve a selected track name to its relative markdown path."""
    selected_name = (selected_track or "").strip()
    if selected_name == "None" or not selected_name:
        return None
    for track in tracks:
        if track["name"] == selected_name:
            return track["path"]
    return None


def selected_track_index(current_path: str, tracks: list[dict[str, str]]) -> int:
    """Return the selectbox index for an existing markdown track-context path."""
    normalized_path = (current_path or "").strip()
    if not normalized_path:
        return 0
    for index, track in enumerate(tracks, start=1):
        if track["path"] == normalized_path:
            return index
    return 0


def _resolve_track_context_path(vault_path: Path, track_context_path: str) -> Path | None:
    raw_path = (track_context_path or "").strip()
    if not raw_path:
        return None

    candidate = Path(raw_path)
    if not candidate.is_absolute():
        candidate = vault_path / candidate
    if candidate.suffix.lower() != ".md":
        candidate = candidate / "track_context.md"
    return candidate.resolve()


def _extract_markdown_section(body: str, heading: str) -> str:
    lines = body.splitlines()
    collected: list[str] = []
    in_section = False
    expected_heading = heading.strip().lower()

    for raw_line in lines:
        stripped = raw_line.strip()
        if stripped.startswith("## "):
            current_heading = stripped[3:].strip().lower()
            if in_section:
                break
            in_section = current_heading == expected_heading
            continue
        if in_section:
            collected.append(raw_line.rstrip())

    return "\n".join(line for line in collected if line.strip()).strip()


def _as_list(value: object) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    cleaned = _clean_text(value)
    return [cleaned] if cleaned else []


def _clean_text(value: object) -> str:
    return str(value).strip() if value is not None else ""


def _join_items(*groups: list[str]) -> str:
    items: list[str] = []
    seen: set[str] = set()
    for group in groups:
        for item in group:
            cleaned = item.strip()
            if not cleaned:
                continue
            lowered = cleaned.lower()
            if lowered in seen:
                continue
            seen.add(lowered)
            items.append(cleaned)
    return ", ".join(items)


def _join_lines(items: list[str]) -> str:
    return "\n".join(f"- {item}" for item in items if item.strip())


def _labeled_block(label: str, value: str) -> str:
    cleaned = value.strip()
    if not cleaned:
        return ""
    return f"{label}:\n{cleaned}"


def _join_blocks(*blocks: str) -> str:
    return "\n\n".join(block.strip() for block in blocks if block.strip())


def _first_non_empty(*values: str) -> str:
    for value in values:
        if value.strip():
            return value
    return ""
