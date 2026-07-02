from datetime import UTC, datetime, timedelta

from deerflow.config.paths import Paths
from deerflow.teacher_profile import (
    build_student_profile_markdown,
    get_student_profile_md_path,
    parse_student_profile_markdown,
    read_student_profile_summary,
    render_student_profile_markdown,
    update_student_profile_from_observation,
    update_student_profile_manual,
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


def test_build_student_profile_markdown_contains_structured_sections():
    markdown = build_student_profile_markdown(
        "student-001",
        student_name="Alice",
        grade="grade-7",
        subject="math",
        weak_knowledge=["quadratic equation"],
        weak_ability=["careless calculation"],
        preferences=["likes hint-first teaching"],
        recent_summary="Needs more practice on factorization.",
    )

    assert "# Student Profile: student-001" in markdown
    assert "## Basic Info" in markdown
    assert "student_name: Alice" in markdown
    assert "grade: grade-7" in markdown
    assert "subject: math" in markdown
    assert "## Weak Knowledge" in markdown
    assert "quadratic equation | count=1 | last_seen=" in markdown
    assert "## Weak Ability" in markdown
    assert "careless calculation | count=1 | last_seen=" in markdown
    assert "## Learning Preferences" in markdown
    assert "- likes hint-first teaching" in markdown
    assert "## Recent Sessions" in markdown
    assert "summary: Needs more practice on factorization." in markdown


def test_parse_student_profile_markdown_migrates_legacy_markdown():
    profile = parse_student_profile_markdown(
        "student-001",
        """# Student Profile: student-001

## Weak Knowledge
- quadratic equation
- factoring

## Weak Ability
- careful checking

## Recent Summary
Needs more practice on factorization.
""",
    )

    assert {item["name"] for item in profile["weak_knowledge"]} == {"quadratic equation", "factoring"}
    assert profile["weak_knowledge"][0]["count"] == 1
    assert [item["name"] for item in profile["weak_ability"]] == ["careful checking"]
    assert profile["recent_sessions"][0]["summary"] == "Needs more practice on factorization."


def test_render_round_trip_preserves_structured_profile():
    profile = {
        "student_id": "student-001",
        "student_name": "Alice",
        "grade": "grade-7",
        "subject": "math",
        "weak_knowledge": [{"name": "equation", "count": 3, "last_seen": "2026-05-07T10:00:00+00:00"}],
        "weak_ability": [{"name": "checking", "count": 2, "last_seen": "2026-05-07T09:00:00+00:00"}],
        "preferences": ["use diagrams"],
        "recent_sessions": [
            {
                "when": "2026-05-07T10:00:00+00:00",
                "problem_id": "1",
                "subject": "math",
                "grade": "grade-7",
                "knowledges": ["equation"],
                "weak_knowledge": ["equation"],
                "weak_ability": ["checking"],
                "summary": "reviewed equation solving",
            }
        ],
    }

    reparsed = parse_student_profile_markdown("student-001", render_student_profile_markdown(profile))

    assert reparsed == profile


def test_update_student_profile_from_observation_accumulates_and_truncates(monkeypatch, tmp_path):
    monkeypatch.setattr("deerflow.teacher_profile.get_paths", lambda: Paths(base_dir=tmp_path))

    start = datetime(2026, 5, 1, tzinfo=UTC)
    for index in range(12):
        update_student_profile_from_observation(
            "student-001",
            observed_at=start + timedelta(days=index),
            problem_id=index,
            subject="math",
            grade="grade-7",
            knowledges=[f"k{index}"],
            weak_knowledge=[f"wk{index}", "shared knowledge"],
            weak_ability=[f"wa{index}", "shared ability"],
            summary=f"summary {index}",
        )

    profile = parse_student_profile_markdown("student-001", read_student_profile_summary("student-001"))

    assert len(profile["weak_knowledge"]) == 10
    assert len(profile["weak_ability"]) == 10
    assert profile["weak_knowledge"][0]["name"] == "shared knowledge"
    assert profile["weak_knowledge"][0]["count"] == 12
    assert profile["weak_ability"][0]["name"] == "shared ability"
    assert profile["weak_ability"][0]["count"] == 12
    assert len(profile["recent_sessions"]) == 8
    assert profile["recent_sessions"][0]["problem_id"] == "11"
    assert profile["recent_sessions"][-1]["problem_id"] == "4"


def test_update_student_profile_manual_preserves_preferences_across_auto_updates(monkeypatch, tmp_path):
    monkeypatch.setattr("deerflow.teacher_profile.get_paths", lambda: Paths(base_dir=tmp_path))

    update_student_profile_manual(
        "student-001",
        student_name="Alice",
        grade="grade-7",
        subject="math",
        preferences=["use diagrams", "slower pace"],
        recent_summary="manual note",
    )
    update_student_profile_from_observation(
        "student-001",
        observed_at="2026-05-07T10:00:00+00:00",
        problem_id=123,
        weak_knowledge=["equation"],
        weak_ability=["checking"],
        summary="auto note",
    )

    profile = parse_student_profile_markdown("student-001", read_student_profile_summary("student-001"))

    assert profile["student_name"] == "Alice"
    assert profile["grade"] == "grade-7"
    assert profile["subject"] == "math"
    assert profile["preferences"] == ["use diagrams", "slower pace"]
    summaries = {session["summary"] for session in profile["recent_sessions"]}
    assert summaries == {"auto note", "manual note"}


def test_update_student_profile_from_observation_keeps_l0_profile_only(monkeypatch, tmp_path):
    monkeypatch.setattr("deerflow.teacher_profile.get_paths", lambda: Paths(base_dir=tmp_path))

    update_student_profile_from_observation(
        "student-001",
        observed_at="2026-05-07T10:00:00+00:00",
        problem_id=123,
        subject="math",
        grade="grade-7",
        knowledges=["equation"],
        weak_knowledge=["equation application"],
        weak_ability=["checking"],
        summary="auto note",
        problem_type="大题",
        difficulty="中等",
        error_analysis="sign mistake",
    )

    markdown = read_student_profile_summary("student-001")
    profile = parse_student_profile_markdown("student-001", markdown)

    assert "## Math Archive Summary" not in markdown
    assert "## Archived Math Problems" not in markdown
    assert "difficulty:" not in markdown
    assert "error_analysis:" not in markdown
    assert profile["recent_sessions"][0]["problem_id"] == "123"
    assert profile["recent_sessions"][0]["summary"] == "auto note"
