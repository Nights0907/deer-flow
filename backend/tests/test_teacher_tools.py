import pytest

from deerflow.tools import teacher_tools


class FakeResponse:
    def __init__(self, content):
        self.content = content


class FakeModel:
    def __init__(self, content):
        self._content = content

    async def ainvoke(self, _messages):
        return FakeResponse(self._content)


class SequenceModelFactory:
    def __init__(self, contents):
        self._contents = list(contents)
        self.calls = 0

    def __call__(self, name=None, thinking_enabled=False):
        content = self._contents[self.calls]
        self.calls += 1
        return FakeModel(content)


def test_normalize_knowledges_returns_list_for_scalar():
    assert teacher_tools._normalize_knowledges("quadratic") == ["quadratic"]


def test_normalize_knowledges_returns_empty_for_none():
    assert teacher_tools._normalize_knowledges(None) == []


def test_normalize_knowledges_keeps_list_unchanged():
    assert teacher_tools._normalize_knowledges(["a", "b"]) == ["a", "b"]


def test_extract_json_object_handles_fenced_json():
    payload = teacher_tools._extract_json_object("```json\n{\"answer\": \"42\"}\n```")

    assert payload == {"answer": "42"}


@pytest.mark.anyio
async def test_solve_problem_tool_returns_structured_result(monkeypatch):
    factory = SequenceModelFactory(
        [
            '{"answer":"42","steps":["step 1"],"explanation":"done"}',
            '{"knowledges":["algebra"]}',
            '{"error_analysis":null}',
            '{"weak_knowledge_candidates":["equation"],"weak_ability_candidates":["careless calculation"]}',
        ]
    )
    monkeypatch.setattr(teacher_tools, "create_chat_model", factory)

    result = await teacher_tools.solve_problem_tool.ainvoke({"question": "1+1?"})

    assert result["status"] == "ok"
    assert result["answer"] == "42"
    assert result["steps"] == ["step 1"]
    assert result["explanation"] == "done"
    assert result["knowledges"] == ["algebra"]
    assert result["error_analysis"] is None
    assert result["weak_knowledge_candidates"] == ["equation"]
    assert result["weak_ability_candidates"] == ["careless calculation"]
    assert result["raw"]["core"] == {"answer": "42", "steps": ["step 1"], "explanation": "done"}
    assert result["raw"]["knowledges"] == {"knowledges": ["algebra"]}
    assert result["raw"]["weak_points"] == {
        "weak_knowledge_candidates": ["equation"],
        "weak_ability_candidates": ["careless calculation"],
    }


@pytest.mark.anyio
async def test_solve_problem_tool_degrades_when_diagnostics_fail(monkeypatch):
    factory = SequenceModelFactory(
        [
            '{"answer":"42","steps":["step 1"],"explanation":"done"}',
            'not json',
            'still not json',
            'broken too',
        ]
    )
    monkeypatch.setattr(teacher_tools, "create_chat_model", factory)

    result = await teacher_tools.solve_problem_tool.ainvoke({"question": "1+1?"})

    assert result["status"] == "ok"
    assert result["answer"] == "42"
    assert result["knowledges"] == []
    assert result["error_analysis"] is None
    assert result["weak_knowledge_candidates"] == []
    assert result["weak_ability_candidates"] == []
    assert result["raw"]["core"] == {"answer": "42", "steps": ["step 1"], "explanation": "done"}
    assert result["raw"]["knowledges"] is None
    assert result["raw"]["error_analysis"] is None
    assert result["raw"]["weak_points"] is None


@pytest.mark.anyio
async def test_recommend_similar_problems_tool_returns_items(monkeypatch):
    monkeypatch.setattr(
        teacher_tools,
        "create_chat_model",
        lambda name=None, thinking_enabled=False: FakeModel(
            '{"items":[{"title":"Variant 1","question":"Q","practice_objective":"practice factoring","similarity":"same pattern"}],"message":"focus on factoring"}'
        ),
    )

    result = await teacher_tools.recommend_similar_problems_tool.ainvoke({"question": "x^2+2x+1=0"})

    assert result["status"] == "ok"
    assert result["items"][0]["title"] == "Variant 1"


@pytest.mark.anyio
async def test_ocr_problem_image_tool_requires_vision_model(monkeypatch):
    monkeypatch.setattr(teacher_tools, "_model_supports_vision", lambda model_name: False)

    result = await teacher_tools.ocr_problem_image_tool.ainvoke({"image_url": "https://example.com/problem.png"})

    assert result["status"] == "error"
    assert "vision-capable" in result["message"]


@pytest.mark.anyio
async def test_evaluate_student_explanation_tool_returns_structured_result(monkeypatch):
    monkeypatch.setattr(
        teacher_tools,
        "create_chat_model",
        lambda name=None, thinking_enabled=False: FakeModel(
            '{"understood":["knows substitution"],"misconception":"confused sign handling","gap_type":"procedure_gap","remediation":"redo the sign change slowly","followup_question":"Why does the sign flip here?","should_update_profile":true,"weak_knowledge_candidates":["integer sign rules"],"weak_ability_candidates":["step checking"]}'
        ),
    )

    result = await teacher_tools.evaluate_student_explanation_tool.ainvoke(
        {"question": "Solve x-3=5", "student_explanation": "I move 3 to the other side."}
    )

    assert result["status"] == "ok"
    assert result["understood"] == ["knows substitution"]
    assert result["gap_type"] == "procedure_gap"
    assert result["should_update_profile"] is True
    assert result["weak_knowledge_candidates"] == ["integer sign rules"]
    assert result["weak_ability_candidates"] == ["step checking"]


@pytest.mark.anyio
async def test_sync_student_profile_tool_returns_empty_without_profile(monkeypatch):
    monkeypatch.setattr(teacher_tools, "read_student_profile_summary", lambda student_id: "")

    result = await teacher_tools.sync_student_profile_tool.ainvoke({"student_id": "stu-1"})

    assert result == {"status": "empty", "message": "No local student profile markdown exists yet."}


@pytest.mark.anyio
async def test_sync_student_profile_tool_rewrites_existing_profile(monkeypatch, tmp_path):
    target = tmp_path / "PROFILE.md"
    monkeypatch.setattr(teacher_tools, "read_student_profile_summary", lambda student_id: "# Profile")
    monkeypatch.setattr(teacher_tools, "write_student_profile_summary", lambda student_id, summary: target)

    result = await teacher_tools.sync_student_profile_tool.ainvoke({"student_id": "stu-1"})

    assert result == {"status": "ok", "path": str(target)}
