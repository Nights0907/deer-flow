from deerflow.config.paths import Paths
from deerflow.teacher_profile import (
    build_student_profile_markdown,
    get_student_profile_md_path,
    read_student_profile_summary,
    write_student_profile_summary,
)


def test_get_student_profile_md_path_uses_students_directory(monkeypatch, tmp_path):
    monkeypatch.setattr("deerflow.teacher_profile.get_paths", lambda: Paths(base_dir=tmp_path))

    path = get_student_profile_md_path("student-001")

    assert path == tmp_path / "students" / "student-001" / "PROFILE.md"


def test_read_student_profile_summary_returns_empty_when_missing(monkeypatch, tmp_path):
    monkeypatch.setattr("deerflow.teacher_profile.get_paths", lambda: Paths(base_dir=tmp_path))

    assert read_student_profile_summary("student-001") == ""


def test_write_and_read_student_profile_summary(monkeypatch, tmp_path):
    monkeypatch.setattr("deerflow.teacher_profile.get_paths", lambda: Paths(base_dir=tmp_path))

    write_student_profile_summary("student-001", "# Profile\n\n- weak point")

    assert read_student_profile_summary("student-001") == "# Profile\n\n- weak point"


def test_build_student_profile_markdown_contains_sections():
    markdown = build_student_profile_markdown(
        "student-001",
        weak_knowledge=["quadratic equation"],
        weak_ability=["careless calculation"],
        preferences=["likes hint-first teaching"],
        recent_summary="Needs more practice on factorization.",
    )

    assert "# Student Profile: student-001" in markdown
    assert "## Weak Knowledge" in markdown
    assert "quadratic equation" in markdown
    assert "## Weak Ability" in markdown
    assert "## Learning Preferences" in markdown
    assert "## Recent Summary" in markdown
