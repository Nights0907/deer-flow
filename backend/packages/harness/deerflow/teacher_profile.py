from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from deerflow.config import get_paths

_MAX_WEAK_ITEMS = 10
_MAX_RECENT_SESSIONS = 8
_BASIC_INFO_HEADER = "## Basic Info"
_WEAK_KNOWLEDGE_HEADER = "## Weak Knowledge"
_WEAK_ABILITY_HEADER = "## Weak Ability"
_PREFERENCES_HEADER = "## Learning Preferences"
_RECENT_SESSIONS_HEADER = "## Recent Sessions"


def get_student_profile_md_path(student_id: str) -> Path:
    return get_paths().student_profile_md_file(student_id)


def _normalize_items(items: list[str] | None) -> list[str]:
    if not items:
        return []
    normalized: list[str] = []
    seen: set[str] = set()
    for item in items:
        text = str(item).strip()
        if not text:
            continue
        key = text.casefold()
        if key in seen:
            continue
        seen.add(key)
        normalized.append(text)
    return normalized


def _parse_datetime(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def _coerce_datetime(value: datetime | str | None) -> datetime:
    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=UTC)
        return value.astimezone(UTC)
    parsed = _parse_datetime(value)
    if parsed is not None:
        if parsed.tzinfo is None:
            return parsed.replace(tzinfo=UTC)
        return parsed.astimezone(UTC)
    return datetime.now(UTC)


def _format_datetime(value: datetime | str | None) -> str:
    return _coerce_datetime(value).isoformat(timespec="seconds")


def empty_student_profile(student_id: str) -> dict[str, Any]:
    return {
        "student_id": student_id,
        "student_name": None,
        "grade": None,
        "subject": None,
        "weak_knowledge": [],
        "weak_ability": [],
        "preferences": [],
        "recent_sessions": [],
    }


def _parse_ranked_section(lines: list[str]) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for line in lines:
        stripped = line.strip()
        if not stripped.startswith("- "):
            continue
        body = stripped[2:].strip()
        if not body:
            continue
        parts = [part.strip() for part in body.split("|")]
        name = parts[0]
        count = 1
        last_seen = None
        for part in parts[1:]:
            if part.startswith("count="):
                try:
                    count = max(1, int(part.removeprefix("count=").strip()))
                except ValueError:
                    count = 1
            elif part.startswith("last_seen="):
                last_seen = part.removeprefix("last_seen=").strip() or None
        items.append({"name": name, "count": count, "last_seen": last_seen})
    return items


def _parse_simple_section(lines: list[str]) -> list[str]:
    items: list[str] = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("- "):
            items.append(stripped[2:].strip())
    return _normalize_items(items)


def _parse_session_like_section(lines: list[str], first_key: str) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("- "):
            if current is not None:
                items.append(current)
            current = {first_key: stripped[2:].strip()}
            continue
        if current is None or ":" not in stripped:
            continue
        key, value = stripped.split(":", 1)
        current[key.strip().lower().replace(" ", "_")] = value.strip()
    if current is not None:
        items.append(current)
    return items


def _parse_recent_sessions(lines: list[str]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for session in _parse_session_like_section(lines, "when"):
        normalized.append(
            {
                "when": session.get("when") or _format_datetime(None),
                "problem_id": session.get("problem_id") or None,
                "subject": session.get("subject") or None,
                "grade": session.get("grade") or None,
                "knowledges": _normalize_items((session.get("knowledges") or "").split(",") if session.get("knowledges") else []),
                "weak_knowledge": _normalize_items((session.get("weak_knowledge") or "").split(",") if session.get("weak_knowledge") else []),
                "weak_ability": _normalize_items((session.get("weak_ability") or "").split(",") if session.get("weak_ability") else []),
                "summary": session.get("summary") or "",
            }
        )
    return normalized


def parse_student_profile_markdown(student_id: str, markdown: str) -> dict[str, Any]:
    profile = empty_student_profile(student_id)
    text = markdown.strip()
    if not text or text == "No profile summary available yet.":
        return profile

    lines = text.splitlines()
    if lines and lines[0].startswith("# Student Profile:"):
        header_student_id = lines[0].split(":", 1)[1].strip()
        if header_student_id:
            profile["student_id"] = header_student_id
        lines = lines[1:]

    sections: dict[str, list[str]] = {}
    current_header: str | None = None
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("## "):
            current_header = stripped
            sections.setdefault(current_header, [])
            continue
        if current_header is not None:
            sections[current_header].append(line)

    basic_info_lines = sections.get(_BASIC_INFO_HEADER, [])
    weak_knowledge_lines = sections.get(_WEAK_KNOWLEDGE_HEADER, [])
    weak_ability_lines = sections.get(_WEAK_ABILITY_HEADER, [])
    recent_session_lines = sections.get(_RECENT_SESSIONS_HEADER, [])
    preference_lines = sections.get(_PREFERENCES_HEADER, [])

    for line in basic_info_lines:
        stripped = line.strip()
        if not stripped or ":" not in stripped:
            continue
        key, value = stripped.split(":", 1)
        normalized_key = key.strip().lower().replace(" ", "_")
        normalized_value = value.strip() or None
        if normalized_key in {"student_name", "grade", "subject"}:
            profile[normalized_key] = normalized_value

    profile["weak_knowledge"] = _truncate_ranked_items(_parse_ranked_section(weak_knowledge_lines))
    if not profile["weak_knowledge"] and weak_knowledge_lines:
        profile["weak_knowledge"] = _truncate_ranked_items([{"name": item, "count": 1, "last_seen": None} for item in _parse_simple_section(weak_knowledge_lines)])

    profile["weak_ability"] = _truncate_ranked_items(_parse_ranked_section(weak_ability_lines))
    if not profile["weak_ability"] and weak_ability_lines:
        profile["weak_ability"] = _truncate_ranked_items([{"name": item, "count": 1, "last_seen": None} for item in _parse_simple_section(weak_ability_lines)])

    profile["preferences"] = _parse_simple_section(preference_lines)

    if recent_session_lines:
        profile["recent_sessions"] = _truncate_recent_sessions(_parse_recent_sessions(recent_session_lines))
    else:
        legacy_summary = sections.get("## Recent Summary", [])
        summary = "\n".join(line.strip() for line in legacy_summary if line.strip()).strip()
        if summary:
            profile["recent_sessions"] = [
                {
                    "when": _format_datetime(None),
                    "problem_id": None,
                    "subject": None,
                    "grade": None,
                    "knowledges": [],
                    "weak_knowledge": [item["name"] for item in profile["weak_knowledge"]],
                    "weak_ability": [item["name"] for item in profile["weak_ability"]],
                    "summary": summary,
                }
            ]

    return profile


def _truncate_ranked_items(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(items, key=lambda item: (-int(item.get("count") or 0), -(int(_coerce_datetime(item.get("last_seen")).timestamp())), item.get("name", "")))[:_MAX_WEAK_ITEMS]


def _truncate_recent_sessions(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(items, key=lambda item: _coerce_datetime(item.get("when")), reverse=True)[:_MAX_RECENT_SESSIONS]


def render_student_profile_markdown(profile: dict[str, Any]) -> str:
    student_id = profile["student_id"]
    lines = [f"# Student Profile: {student_id}", ""]

    lines.append(_BASIC_INFO_HEADER)
    if profile.get("student_name"):
        lines.append(f"student_name: {profile['student_name']}")
    if profile.get("grade"):
        lines.append(f"grade: {profile['grade']}")
    if profile.get("subject"):
        lines.append(f"subject: {profile['subject']}")
    lines.append("")

    lines.append(_WEAK_KNOWLEDGE_HEADER)
    for item in _truncate_ranked_items(profile.get("weak_knowledge") or []):
        lines.append(f"- {item['name']} | count={int(item.get('count') or 1)} | last_seen={_format_datetime(item.get('last_seen'))}")
    lines.append("")

    lines.append(_WEAK_ABILITY_HEADER)
    for item in _truncate_ranked_items(profile.get("weak_ability") or []):
        lines.append(f"- {item['name']} | count={int(item.get('count') or 1)} | last_seen={_format_datetime(item.get('last_seen'))}")
    lines.append("")

    lines.append(_PREFERENCES_HEADER)
    for item in _normalize_items(profile.get("preferences") or []):
        lines.append(f"- {item}")
    lines.append("")

    lines.append(_RECENT_SESSIONS_HEADER)
    for session in _truncate_recent_sessions(profile.get("recent_sessions") or []):
        lines.append(f"- {_format_datetime(session.get('when'))}")
        if session.get("problem_id") is not None:
            lines.append(f"  problem_id: {session['problem_id']}")
        if session.get("subject"):
            lines.append(f"  subject: {session['subject']}")
        if session.get("grade"):
            lines.append(f"  grade: {session['grade']}")
        if session.get("knowledges"):
            lines.append(f"  knowledges: {', '.join(_normalize_items(session['knowledges']))}")
        if session.get("weak_knowledge"):
            lines.append(f"  weak_knowledge: {', '.join(_normalize_items(session['weak_knowledge']))}")
        if session.get("weak_ability"):
            lines.append(f"  weak_ability: {', '.join(_normalize_items(session['weak_ability']))}")
        if session.get("summary"):
            lines.append(f"  summary: {str(session['summary']).strip()}")
    lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def read_student_profile_summary(student_id: str) -> str:
    profile_path = get_student_profile_md_path(student_id)
    if not profile_path.exists():
        return ""
    return profile_path.read_text(encoding="utf-8").strip()


def write_student_profile_summary(student_id: str, summary: str) -> Path:
    profile_path = get_student_profile_md_path(student_id)
    profile_path.parent.mkdir(parents=True, exist_ok=True)
    profile_path.write_text(summary.strip() + "\n" if summary.strip() else "", encoding="utf-8")
    return profile_path


def _merge_ranked_items(existing: list[dict[str, Any]], additions: list[str], seen_at: datetime) -> list[dict[str, Any]]:
    by_name: dict[str, dict[str, Any]] = {}
    for item in existing:
        name = str(item.get("name") or "").strip()
        if not name:
            continue
        by_name[name.casefold()] = {
            "name": name,
            "count": max(1, int(item.get("count") or 1)),
            "last_seen": item.get("last_seen") or _format_datetime(seen_at),
        }
    for name in _normalize_items(additions):
        key = name.casefold()
        if key in by_name:
            by_name[key]["count"] += 1
            by_name[key]["last_seen"] = _format_datetime(seen_at)
        else:
            by_name[key] = {"name": name, "count": 1, "last_seen": _format_datetime(seen_at)}
    return _truncate_ranked_items(list(by_name.values()))


def merge_student_profile_observation(
    profile: dict[str, Any],
    *,
    observed_at: datetime | str | None,
    problem_id: int | str | None = None,
    subject: str | None = None,
    grade: str | None = None,
    knowledges: list[str] | None = None,
    weak_knowledge: list[str] | None = None,
    weak_ability: list[str] | None = None,
    summary: str | None = None,
    problem_type: str | None = None,
    difficulty: str | None = None,
    error_analysis: str | None = None,
) -> dict[str, Any]:
    merged = empty_student_profile(profile["student_id"])
    merged["student_name"] = profile.get("student_name")
    merged["grade"] = grade or profile.get("grade")
    merged["subject"] = subject or profile.get("subject")
    merged["preferences"] = _normalize_items(profile.get("preferences") or [])
    seen_at = _coerce_datetime(observed_at)
    merged["weak_knowledge"] = _merge_ranked_items(profile.get("weak_knowledge") or [], weak_knowledge or [], seen_at)
    merged["weak_ability"] = _merge_ranked_items(profile.get("weak_ability") or [], weak_ability or [], seen_at)
    merged["recent_sessions"] = _truncate_recent_sessions(
        [
            {
                "when": _format_datetime(seen_at),
                "problem_id": str(problem_id) if problem_id is not None else None,
                "subject": subject,
                "grade": grade,
                "knowledges": _normalize_items(knowledges),
                "weak_knowledge": _normalize_items(weak_knowledge),
                "weak_ability": _normalize_items(weak_ability),
                "summary": (summary or "").strip(),
            },
            *(profile.get("recent_sessions") or []),
        ]
    )
    return merged


def update_student_profile_from_observation(
    student_id: str,
    *,
    observed_at: datetime | str | None,
    problem_id: int | str | None = None,
    subject: str | None = None,
    grade: str | None = None,
    knowledges: list[str] | None = None,
    weak_knowledge: list[str] | None = None,
    weak_ability: list[str] | None = None,
    summary: str | None = None,
    problem_type: str | None = None,
    difficulty: str | None = None,
    error_analysis: str | None = None,
) -> Path:
    current = parse_student_profile_markdown(student_id, read_student_profile_summary(student_id))
    merged = merge_student_profile_observation(
        current,
        observed_at=observed_at,
        problem_id=problem_id,
        subject=subject,
        grade=grade,
        knowledges=knowledges,
        weak_knowledge=weak_knowledge,
        weak_ability=weak_ability,
        summary=summary,
        problem_type=problem_type,
        difficulty=difficulty,
        error_analysis=error_analysis,
    )
    return write_student_profile_summary(student_id, render_student_profile_markdown(merged))


def update_student_profile_manual(
    student_id: str,
    *,
    student_name: str | None = None,
    grade: str | None = None,
    subject: str | None = None,
    weak_knowledge: list[str] | None = None,
    weak_ability: list[str] | None = None,
    preferences: list[str] | None = None,
    recent_summary: str | None = None,
) -> Path:
    current = parse_student_profile_markdown(student_id, read_student_profile_summary(student_id))
    observed_at = datetime.now(UTC)
    merged = merge_student_profile_observation(
        current,
        observed_at=observed_at,
        weak_knowledge=weak_knowledge,
        weak_ability=weak_ability,
        summary=recent_summary,
    )
    merged["student_name"] = student_name or current.get("student_name")
    merged["grade"] = grade or current.get("grade")
    merged["subject"] = subject or current.get("subject")
    if preferences is not None:
        merged["preferences"] = _normalize_items(preferences)
    else:
        merged["preferences"] = _normalize_items(current.get("preferences") or [])
    return write_student_profile_summary(student_id, render_student_profile_markdown(merged))


def build_student_profile_markdown(
    student_id: str,
    *,
    student_name: str | None = None,
    grade: str | None = None,
    subject: str | None = None,
    weak_knowledge: list[str] | None = None,
    weak_ability: list[str] | None = None,
    preferences: list[str] | None = None,
    recent_summary: str | None = None,
) -> str:
    profile = empty_student_profile(student_id)
    profile["student_name"] = student_name
    profile["grade"] = grade
    profile["subject"] = subject
    if preferences is not None:
        profile["preferences"] = _normalize_items(preferences)
    profile = merge_student_profile_observation(
        profile,
        observed_at=datetime.now(UTC),
        weak_knowledge=weak_knowledge,
        weak_ability=weak_ability,
        summary=recent_summary,
    )
    return render_student_profile_markdown(profile)
