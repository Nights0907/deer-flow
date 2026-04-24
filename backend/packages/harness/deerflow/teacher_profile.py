from pathlib import Path

from deerflow.config import get_paths


def get_student_profile_md_path(student_id: str) -> Path:
    """Return the markdown summary path for a student profile."""
    return get_paths().student_profile_md_file(student_id)


def read_student_profile_summary(student_id: str) -> str:
    """Read the student profile markdown summary.

    Returns an empty string when the profile does not exist yet so callers can
    degrade gracefully to non-personalized tutoring.
    """
    profile_path = get_student_profile_md_path(student_id)
    if not profile_path.exists():
        return ""
    return profile_path.read_text(encoding="utf-8").strip()


def write_student_profile_summary(student_id: str, summary: str) -> Path:
    """Write the student profile markdown summary.

    This is intentionally minimal for the first integration stage. Later we can
    replace the caller-side summary assembly with data returned from the external
    profile system while keeping the same file contract.
    """
    profile_path = get_student_profile_md_path(student_id)
    profile_path.parent.mkdir(parents=True, exist_ok=True)
    profile_path.write_text(summary.strip() + "\n" if summary.strip() else "", encoding="utf-8")
    return profile_path


def build_student_profile_markdown(
    student_id: str,
    *,
    weak_knowledge: list[str] | None = None,
    weak_ability: list[str] | None = None,
    preferences: list[str] | None = None,
    recent_summary: str | None = None,
) -> str:
    """Build a minimal markdown summary for a student profile.

    Placeholder helper for the initial integration. If your external profile
    service later returns a richer profile document, replace this formatter or
    bypass it and write the returned markdown directly.
    """
    lines = [f"# Student Profile: {student_id}", ""]

    if weak_knowledge:
        lines.append("## Weak Knowledge")
        lines.extend(f"- {item}" for item in weak_knowledge)
        lines.append("")

    if weak_ability:
        lines.append("## Weak Ability")
        lines.extend(f"- {item}" for item in weak_ability)
        lines.append("")

    if preferences:
        lines.append("## Learning Preferences")
        lines.extend(f"- {item}" for item in preferences)
        lines.append("")

    if recent_summary:
        lines.append("## Recent Summary")
        lines.append(recent_summary.strip())
        lines.append("")

    if len(lines) == 2:
        lines.append("No profile summary available yet.")

    return "\n".join(lines).rstrip() + "\n"
