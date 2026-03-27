"""Pure formatting helpers for reviewing Track Context update proposals."""

from __future__ import annotations

from services.models import TrackContextUpdateProposal


def proposal_groups(
    proposal: TrackContextUpdateProposal | None,
) -> list[tuple[str, list[str]]]:
    """Return grouped, human-readable proposal changes."""
    if proposal is None or proposal.is_empty():
        return []

    groups: list[tuple[str, list[str]]] = []
    if proposal.set_fields:
        groups.append(
            (
                "Set Fields",
                [f"{key}: {value}" for key, value in proposal.set_fields.items()],
            )
        )
    if proposal.add_to_lists:
        groups.append(
            (
                "Add To Lists",
                [f"{key}: {', '.join(values)}" for key, values in proposal.add_to_lists.items()],
            )
        )
    if proposal.remove_from_lists:
        groups.append(
            (
                "Remove From Lists",
                [f"{key}: {', '.join(values)}" for key, values in proposal.remove_from_lists.items()],
            )
        )
    if proposal.set_sections:
        groups.append(
            (
                "Set Sections",
                [
                    f"{section_key}: {_format_section_payload(section_value)}"
                    for section_key, section_value in proposal.set_sections.items()
                ],
            )
        )
    if proposal.add_section_issues:
        groups.append(
            (
                "Add Section Issues",
                [
                    f"{section_key}: {', '.join(values)}"
                    for section_key, values in proposal.add_section_issues.items()
                ],
            )
        )
    if proposal.remove_section_issues:
        groups.append(
            (
                "Remove Section Issues",
                [
                    f"{section_key}: {', '.join(values)}"
                    for section_key, values in proposal.remove_section_issues.items()
                ],
            )
        )
    if proposal.add_section_elements:
        groups.append(
            (
                "Add Section Elements",
                [
                    f"{section_key}: {', '.join(values)}"
                    for section_key, values in proposal.add_section_elements.items()
                ],
            )
        )
    if proposal.add_section_notes:
        groups.append(
            (
                "Add Section Notes",
                [
                    f"{section_key}: {' | '.join(values)}"
                    for section_key, values in proposal.add_section_notes.items()
                ],
            )
        )
    if proposal.section_focus.strip():
        groups.append(("Section Focus", [proposal.section_focus.strip()]))
    return groups


def proposal_markdown_block(
    proposal: TrackContextUpdateProposal | None,
) -> str:
    """Render a saved-output markdown block for a Track Context update proposal."""
    if proposal is None or proposal.is_empty():
        return ""

    lines = ["## Suggested Track Context Update", ""]
    if proposal.summary.strip():
        lines.append(f"- Summary: {proposal.summary.strip()}")
    if proposal.confidence.strip():
        lines.append(f"- Confidence: {proposal.confidence.strip()}")
    if proposal.source_reasoning.strip():
        lines.append(f"- Why Suggested: {proposal.source_reasoning.strip()}")
    for heading, items in proposal_groups(proposal):
        lines.append(f"### {heading}")
        lines.append("")
        lines.extend(f"- {item}" for item in items)
        lines.append("")
    return "\n".join(lines).rstrip() + "\n\n"


def _format_section_payload(section_value: dict[str, object]) -> str:
    parts: list[str] = []
    for key, value in section_value.items():
        if isinstance(value, list):
            cleaned = ", ".join(str(item) for item in value if str(item).strip())
        else:
            cleaned = str(value).strip()
        if cleaned:
            parts.append(f"{key}={cleaned}")
    return " | ".join(parts) if parts else "No section fields provided"
