from app.gateway.services import build_run_config
from deerflow.agents.lead_agent.prompt import get_student_profile_prompt_section
from deerflow.config.agents_config import load_agent_config, load_agent_soul
from deerflow.config.paths import Paths


def test_build_run_config_injects_student_id_into_context():
    config = build_run_config(
        "thread-1",
        request_config=None,
        metadata={"student_id": "stu-1"},
        assistant_id="digital-teacher",
    )

    assert config["context"]["thread_id"] == "thread-1"
    assert config["context"]["student_id"] == "stu-1"
    assert config["metadata"]["student_id"] == "stu-1"


def test_build_run_config_preserves_existing_context_and_adds_student_id():
    config = build_run_config(
        "thread-1",
        request_config={"context": {"thread_id": "thread-1", "foo": "bar"}},
        metadata={"student_id": "stu-1"},
        assistant_id="digital-teacher",
    )

    assert config["context"]["foo"] == "bar"
    assert config["context"]["student_id"] == "stu-1"


def test_student_profile_prompt_section_only_for_digital_teacher():
    assert get_student_profile_prompt_section("other-agent") == ""
    assert "read_student_profile" in get_student_profile_prompt_section("digital-teacher")


def test_paths_support_student_profile_markdown(tmp_path):
    paths = Paths(base_dir=tmp_path)

    assert paths.student_profile_md_file("stu-1") == tmp_path / "students" / "stu-1" / "PROFILE.md"


def test_digital_teacher_agent_config_loads():
    config = load_agent_config("digital-teacher")

    assert config is not None
    assert config.name == "digital-teacher"
    assert "teacher" in (config.tool_groups or [])
    assert "digital-teacher-guided-questioning" in (config.skills or [])
    assert "digital-teacher-feynman" in (config.skills or [])


def test_digital_teacher_soul_loads():
    soul = load_agent_soul("digital-teacher")

    assert soul is not None
    assert "digital teacher" in soul.lower()
    assert "Every new problem-solving request must call `solve_problem`" in soul
    assert "read the student's profile first" in soul
    assert "use the tool result as the default source of truth" in soul
