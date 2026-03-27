"""Normalization helpers for YAML-backed track context."""

from __future__ import annotations

from collections.abc import Mapping

from services.models import SectionContext, TrackContext

VALID_CURRENT_STAGES = {
    "idea",
    "sketch",
    "writing",
    "arrangement",
    "sound_design",
    "production",
    "mixing",
    "mastering",
    "finalizing",
}


def _clean_str(value: object) -> str | None:
    """Return a stripped string when present, otherwise None."""
    if value is None:
        return None
    cleaned = str(value).strip()
    return cleaned or None


def _clean_list(value: object) -> list[str]:
    """Normalize a list-like value into a compact string list."""
    if value is None:
        return []
    if isinstance(value, list):
        items = value
    else:
        items = [value]
    cleaned: list[str] = []
    for item in items:
        normalized = _clean_str(item)
        if normalized:
            cleaned.append(normalized)
    return cleaned


def _clean_dict_str(value: object) -> dict[str, str]:
    """Normalize a mapping into a string-to-string dictionary."""
    if not isinstance(value, Mapping):
        return {}
    cleaned: dict[str, str] = {}
    for key, raw_value in value.items():
        cleaned_key = _clean_str(key)
        cleaned_value = _clean_str(raw_value)
        if cleaned_key and cleaned_value:
            cleaned[cleaned_key] = cleaned_value
    return cleaned


def _clean_section_mapping(value: object) -> dict[str, SectionContext]:
    """Normalize a section mapping into lightweight SectionContext objects."""
    if not isinstance(value, Mapping):
        return {}
    cleaned: dict[str, SectionContext] = {}
    for raw_key, raw_section in value.items():
        section_key = _clean_str(raw_key)
        if not section_key:
            continue
        section_payload = raw_section if isinstance(raw_section, Mapping) else {}
        name = _clean_str(section_payload.get("name")) or section_key
        cleaned[section_key] = SectionContext(
            name=name,
            bars=_clean_str(section_payload.get("bars")) or "",
            role=_clean_str(section_payload.get("role")) or "",
            energy_level=_clean_str(section_payload.get("energy_level")) or "",
            elements=_clean_list(section_payload.get("elements")),
            issues=_clean_list(section_payload.get("issues")),
            notes=_clean_str(section_payload.get("notes")) or "",
        )
    return cleaned


def _coerce_bpm(value: object) -> int | None:
    """Convert BPM values to an integer when possible."""
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value if value > 0 else None
    if isinstance(value, float):
        return int(value) if value > 0 else None

    cleaned = _clean_str(value)
    if not cleaned:
        return None
    try:
        bpm = int(float(cleaned))
    except ValueError:
        return None
    return bpm if bpm > 0 else None


def normalize_track_context(raw: dict) -> TrackContext:
    """Normalize raw YAML data into the TrackContext dataclass."""
    track_id = _clean_str(raw.get("track_id")) or "default_track"
    track_title = _clean_str(raw.get("title")) or _clean_str(raw.get("track_name"))
    references = _clean_list(raw.get("references")) or _clean_list(raw.get("reference_tracks"))

    current_stage = _clean_str(raw.get("current_stage"))
    if current_stage is not None:
        current_stage = current_stage.lower()
        if current_stage not in VALID_CURRENT_STAGES:
            current_stage = None

    return TrackContext(
        track_id=track_id,
        track_name=track_title,
        genre=_clean_str(raw.get("genre")),
        bpm=_coerce_bpm(raw.get("bpm")),
        key=_clean_str(raw.get("key")),
        vibe=_clean_list(raw.get("vibe")),
        reference_tracks=references,
        current_stage=current_stage,
        current_problem=_clean_str(raw.get("current_problem")),
        known_issues=_clean_list(raw.get("known_issues")),
        goals=_clean_list(raw.get("goals")),
        sections=_clean_section_mapping(raw.get("sections")),
    )
