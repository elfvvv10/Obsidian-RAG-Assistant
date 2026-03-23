"""Helpers for parsing markdown frontmatter and tags."""

from __future__ import annotations

import re


FRONTMATTER_DELIMITER = "---"


def parse_markdown_metadata(content: str) -> tuple[dict[str, object], str]:
    """Parse a minimal YAML-like frontmatter block and return body content."""
    normalized = content.replace("\r\n", "\n")
    if not normalized.startswith(f"{FRONTMATTER_DELIMITER}\n"):
        return {}, normalized

    lines = normalized.splitlines()
    end_index = None
    for index in range(1, len(lines)):
        if lines[index].strip() in {FRONTMATTER_DELIMITER, "..."}:
            end_index = index
            break

    if end_index is None:
        return {}, normalized

    frontmatter_lines = lines[1:end_index]
    body = "\n".join(lines[end_index + 1 :]).lstrip("\n")
    return _parse_frontmatter_lines(frontmatter_lines), body


def extract_tags(frontmatter: dict[str, object], content: str) -> tuple[str, ...]:
    """Extract normalized tags from frontmatter and inline markdown tags."""
    tags: list[str] = []

    for key in ("tags", "tag"):
        if key not in frontmatter:
            continue
        tags.extend(_normalize_tag_values(frontmatter[key]))

    for line in content.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            continue
        for match in re.findall(r"(?<!\w)#([A-Za-z0-9_/-]+)", line):
            tags.append(_normalize_tag(match))

    deduped: list[str] = []
    seen: set[str] = set()
    for tag in tags:
        if not tag or tag in seen:
            continue
        seen.add(tag)
        deduped.append(tag)
    return tuple(deduped)


def _parse_frontmatter_lines(lines: list[str]) -> dict[str, object]:
    metadata: dict[str, object] = {}
    current_list_key: str | None = None
    current_list_values: list[str] = []

    def flush_current_list() -> None:
        nonlocal current_list_key, current_list_values
        if current_list_key is not None:
            metadata[current_list_key] = list(current_list_values)
        current_list_key = None
        current_list_values = []

    for raw_line in lines:
        line = raw_line.rstrip()
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue

        if current_list_key and stripped.startswith("- "):
            current_list_values.append(stripped[2:].strip().strip("'\""))
            continue

        flush_current_list()

        if ":" not in stripped:
            continue

        key, raw_value = stripped.split(":", 1)
        key = key.strip()
        value = raw_value.strip()

        if not value:
            current_list_key = key
            current_list_values = []
            continue

        if value.startswith("[") and value.endswith("]"):
            items = [item.strip().strip("'\"") for item in value[1:-1].split(",") if item.strip()]
            metadata[key] = items
            continue

        metadata[key] = value.strip("'\"")

    flush_current_list()
    return metadata


def _normalize_tag_values(value: object) -> list[str]:
    if isinstance(value, list):
        return [_normalize_tag(str(item)) for item in value]
    if isinstance(value, str):
        if value.startswith("[") and value.endswith("]"):
            value = value[1:-1]
        parts = [part.strip() for part in value.split(",")] if "," in value else [value]
        return [_normalize_tag(part) for part in parts]
    return []


def _normalize_tag(value: str) -> str:
    return value.strip().lstrip("#").strip().lower()
