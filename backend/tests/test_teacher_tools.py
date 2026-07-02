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


async def fake_persist_ok(**kwargs):
    return {"problem_id": 1, "problem_detail_id": "abc", "student_profile_path": "/tmp/PROFILE.md", "created_at": "2026-04-27T00:00:00+00:00"}, None


async def fake_persist_none(**kwargs):
    return None, None


async def fake_persist_error(**kwargs):
    return None, "db down"


def test_normalize_knowledges_returns_list_for_scalar():
    assert teacher_tools._normalize_knowledges("quadratic") == ["quadratic"]


def test_normalize_knowledges_returns_empty_for_none():
    assert teacher_tools._normalize_knowledges(None) == []


def test_normalize_knowledges_keeps_list_unchanged():
    assert teacher_tools._normalize_knowledges(["a", "b"]) == ["a", "b"]


def test_extract_json_object_handles_fenced_json():
    payload = teacher_tools._extract_json_object("```json\n{\"answer\": \"42\"}\n```")

    assert payload == {"answer": "42"}


def test_solve_classification_system_instruction_includes_fixed_taxonomy():
    instruction = teacher_tools._solve_classification_system_instruction()

    assert "You must choose knowledge_type and knowledge_detail from the following fixed Chinese taxonomy." in instruction
    assert "函数" in instruction
    assert "函数的单调性" in instruction
    assert "解析几何" in instruction
    assert "椭圆" in instruction


@pytest.mark.anyio
async def test_solve_problem_tool_returns_structured_result(monkeypatch):
    factory = SequenceModelFactory(
        [
            '{"answer":"42","steps":["step 1"],"explanation":"done"}',
            '{"knowledges":["函数的单调性"]}',
            '{"error_analysis":null}',
            '{"subject":"数学"}',
            '{"weak_knowledge_candidates":["equation"],"weak_ability_candidates":["careless calculation"]}',
            '{"problem_type":"大题","difficulty":"简单","knowledge_type":"函数","knowledge_detail":"函数的单调性","stage":"高中","ability_tags":["数形结合"],"method_tags":["构造函数"],"error_tags":["公式误用"]}',
        ]
    )
    monkeypatch.setattr(teacher_tools, "create_chat_model", factory)
    monkeypatch.setattr(teacher_tools, "persist_safely_async", fake_persist_ok)

    result = await teacher_tools.solve_problem_tool.ainvoke({"question": "1+1?"})

    assert result["status"] == "ok"
    assert result["answer"] == "42"
    assert result["steps"] == ["step 1"]
    assert result["explanation"] == "done"
    assert result["knowledges"] == ["函数的单调性"]
    assert result["error_analysis"] is None
    assert result["problem_type"] == "大题"
    assert result["difficulty"] == "简单"
    assert result["knowledge_type"] == "函数"
    assert result["knowledge_detail"] == "函数的单调性"
    assert result["stage"] == "高中"
    assert result["ability_tags"] == ["数形结合"]
    assert result["method_tags"] == ["构造函数"]
    assert result["error_tags"] == ["公式误用"]
    assert result["weak_knowledge_candidates"] == ["equation"]
    assert result["weak_ability_candidates"] == ["careless calculation"]
    assert result["raw"]["core"] == {"answer": "42", "steps": ["step 1"], "explanation": "done"}
    assert result["raw"]["knowledges"] == {"knowledges": ["函数的单调性"]}
    assert result["raw"]["weak_points"] == {
        "weak_knowledge_candidates": ["equation"],
        "weak_ability_candidates": ["careless calculation"],
    }
    assert result["raw"]["classification"]["problem_type"] == "大题"
    assert result["raw"]["classification"]["knowledge_detail"] == "函数的单调性"
    assert result["raw"]["classification"]["ability_tags"] == ["数形结合"]
    assert result["persistence"]["problem_id"] == 1
    assert result["persistence"]["problem_detail_id"] == "abc"


@pytest.mark.anyio
async def test_solve_problem_tool_passes_basic_record_fields_to_persistence(monkeypatch):
    factory = SequenceModelFactory(
        [
            '{"answer":"42","steps":["step 1"],"explanation":"done"}',
            '{"knowledges":["函数的单调性"]}',
            '{"error_analysis":null}',
            '{"subject":"数学"}',
            '{"weak_knowledge_candidates":["equation"],"weak_ability_candidates":["careless calculation"]}',
            '{"problem_type":"大题","difficulty":"简单","knowledge_type":"函数","knowledge_detail":"函数的单调性","stage":"高中","ability_tags":["数形结合"],"method_tags":["构造函数"],"error_tags":["公式误用"]}',
        ]
    )
    captured = {}

    async def fake_persist(**kwargs):
        captured.update(kwargs)
        return {"problem_id": 1, "problem_detail_id": "abc", "student_profile_path": "/tmp/PROFILE.md", "created_at": "2026-04-27T00:00:00+00:00"}, None

    monkeypatch.setattr(teacher_tools, "create_chat_model", factory)
    monkeypatch.setattr(teacher_tools, "persist_safely_async", fake_persist)

    await teacher_tools.solve_problem_tool.ainvoke(
        {
            "question": "1+1?",
            "student_id": "stu-1",
            "image_url": "https://example.com/problem.png",
            "subject": "数学",
            "grade": "grade-1",
        }
    )

    assert captured["question"] == "1+1?"
    assert captured["student_id"] == "stu-1"
    assert captured["image_url"] == "https://example.com/problem.png"
    assert captured["subject"] == "数学"
    assert captured["grade"] == "grade-1"
    assert captured["result"]["answer"] == "42"
    assert captured["result"]["problem_type"] == "大题"
    assert captured["result"]["difficulty"] == "简单"
    assert captured["result"]["knowledge_type"] == "函数"
    assert captured["result"]["knowledge_detail"] == "函数的单调性"


@pytest.mark.anyio
async def test_solve_problem_tool_logs_structured_fields(monkeypatch, caplog):
    factory = SequenceModelFactory(
        [
            '{"answer":"42","steps":["step 1"],"explanation":"done"}',
            '{"knowledges":["algebra"]}',
            '{"error_analysis":"sign mistake"}',
            '{"subject":"数学"}',
            '{"weak_knowledge_candidates":["equation"],"weak_ability_candidates":["careless calculation"]}',
            '{"problem_type":"大题","difficulty":"简单"}',
        ]
    )
    monkeypatch.setattr(teacher_tools, "create_chat_model", factory)
    monkeypatch.setattr(teacher_tools, "persist_safely_async", fake_persist_none)

    with caplog.at_level("INFO"):
        await teacher_tools.solve_problem_tool.ainvoke({"question": "1+1?", "student_id": "stu-1", "subject": "数学", "grade": "grade-7"})

    assert "solve_problem generated result" in caplog.text
    assert 'answer=' in caplog.text
    assert 'steps=["step 1"]' in caplog.text
    assert 'knowledges=[]' in caplog.text
    assert "sign mistake" in caplog.text
    assert 'weak_knowledge_candidates=["equation"]' in caplog.text
    assert 'weak_ability_candidates=["careless calculation"]' in caplog.text


@pytest.mark.anyio
async def test_solve_problem_tool_normalizes_english_knowledges_to_fixed_chinese_taxonomy(monkeypatch):
    factory = SequenceModelFactory(
        [
            '{"answer":"42","steps":["step 1"],"explanation":"done"}',
            '{"knowledges":["monotonicity","函数单调综合"]}',
            '{"error_analysis":null}',
            '{"subject":"math"}',
            '{"weak_knowledge_candidates":[],"weak_ability_candidates":[]}',
            '{"problem_type":"大题","difficulty":"easy","knowledge_type":"function","knowledge_detail":"单调"}',
        ]
    )
    monkeypatch.setattr(teacher_tools, "create_chat_model", factory)
    monkeypatch.setattr(teacher_tools, "persist_safely_async", fake_persist_none)

    result = await teacher_tools.solve_problem_tool.ainvoke({"question": "1+1?"})

    assert result["knowledge_type"] == "函数"
    assert result["knowledge_detail"] == "函数的单调性"
    assert result["knowledges"] == ["函数的单调性"]


@pytest.mark.anyio
async def test_solve_problem_tool_infers_subject_when_missing(monkeypatch):
    factory = SequenceModelFactory(
        [
            '{"answer":"42","steps":["step 1"],"explanation":"done"}',
            '{"knowledges":["函数的单调性"]}',
            '{"error_analysis":null}',
            '{"subject":"math"}',
            '{"weak_knowledge_candidates":[],"weak_ability_candidates":[]}',
            '{"problem_type":"大题","difficulty":"简单","knowledge_type":"函数","knowledge_detail":"函数的单调性","stage":"高中","ability_tags":["数形结合"],"method_tags":["构造函数"],"error_tags":["公式误用"]}',
        ]
    )
    captured = {}

    async def fake_persist(**kwargs):
        captured.update(kwargs)
        return None, None

    monkeypatch.setattr(teacher_tools, "create_chat_model", factory)
    monkeypatch.setattr(teacher_tools, "persist_safely_async", fake_persist)

    await teacher_tools.solve_problem_tool.ainvoke({"question": "解方程 x+1=2"})

    assert captured["subject"] == "数学"


@pytest.mark.anyio
async def test_solve_problem_tool_degrades_when_diagnostics_fail(monkeypatch):
    factory = SequenceModelFactory(
        [
            '{"answer":"42","steps":["step 1"],"explanation":"done"}',
            'not json',
            'still not json',
            'broken too',
            'also broken',
        ]
    )
    monkeypatch.setattr(teacher_tools, "create_chat_model", factory)
    monkeypatch.setattr(teacher_tools, "persist_safely_async", fake_persist_error)

    result = await teacher_tools.solve_problem_tool.ainvoke({"question": "1+1?"})

    assert result["status"] == "ok"
    assert result["answer"] == "42"
    assert result["knowledges"] == []
    assert result["error_analysis"] is None
    assert result["problem_type"] is None
    assert result["difficulty"] is None
    assert result["weak_knowledge_candidates"] == []
    assert result["weak_ability_candidates"] == []
    assert result["raw"]["core"] == {"answer": "42", "steps": ["step 1"], "explanation": "done"}
    assert result["raw"]["knowledges"] is None
    assert result["raw"]["error_analysis"] is None
    assert result["raw"]["subject"] is None
    assert result["raw"]["weak_points"] is None
    assert result["raw"]["classification"] is None
    assert result["persistence_error"] == "db down"


@pytest.mark.anyio
async def test_recommend_similar_problems_tool_prefers_problem_bank(monkeypatch):
    monkeypatch.setattr(
        teacher_tools,
        "retrieve_similar_problems",
        lambda **kwargs: [
            {
                "qid": 101,
                "question": "已存题目",
                "problem_type": "equation",
                "knowledge_type": "函数",
                "knowledge_detail": "函数单调性",
                "difficulty": "easy",
                "recommend_reason": "同考函数单调性；方法相近：构造函数",
            }
        ],
    )

    result = await teacher_tools.recommend_similar_problems_tool.ainvoke({"question": "x^2+2x+1=0", "subject": "数学"})

    assert result["status"] == "ok"
    assert result["raw"]["source"] == "problem_bank"
    assert result["items"][0]["question"] == "已存题目"
    assert result["items"][0]["similarity"] == "同考函数单调性；方法相近：构造函数"


@pytest.mark.anyio
async def test_recommend_similar_problems_tool_falls_back_to_model(monkeypatch):
    monkeypatch.setattr(teacher_tools, "retrieve_similar_problems", lambda **kwargs: [])
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


def test_build_student_profile_context_does_not_inline_full_markdown(monkeypatch):
    monkeypatch.setattr(teacher_tools, "read_student_profile_summary", lambda student_id: "# Student Profile: stu-1")

    context = teacher_tools._build_student_profile_context("stu-1")

    assert "student_profile_l0_is_injected_in_system_prompt: true" in context
    assert "# Student Profile" not in context


def test_update_student_profile_tool_uses_manual_update_entry(monkeypatch, tmp_path):
    target = tmp_path / "PROFILE.md"
    captured = {}

    def fake_update(student_id, **kwargs):
        captured["student_id"] = student_id
        captured.update(kwargs)
        return target

    monkeypatch.setattr(teacher_tools, "update_student_profile_manual", fake_update)

    result = teacher_tools.update_student_profile_tool.invoke(
        {
            "student_id": "stu-1",
            "student_name": "Alice",
            "grade": "grade-7",
            "subject": "数学",
            "weak_knowledge": ["equation"],
            "weak_ability": ["checking"],
            "preferences": ["use diagrams"],
            "recent_summary": "manual note",
        }
    )

    assert result == {"status": "ok", "path": str(target)}
    assert captured == {
        "student_id": "stu-1",
        "student_name": "Alice",
        "grade": "grade-7",
        "subject": "数学",
        "weak_knowledge": ["equation"],
        "weak_ability": ["checking"],
        "preferences": ["use diagrams"],
        "recent_summary": "manual note",
    }


@pytest.mark.anyio
async def test_sync_student_profile_tool_rewrites_existing_profile(monkeypatch, tmp_path):
    target = tmp_path / "PROFILE.md"
    monkeypatch.setattr(teacher_tools, "read_student_profile_summary", lambda student_id: "# Student Profile: stu-1\n\n## Weak Knowledge\n- equation\n")
    monkeypatch.setattr(
        teacher_tools,
        "parse_student_profile_markdown",
        lambda student_id, markdown: {
            "student_id": student_id,
            "weak_knowledge": [{"name": "equation", "count": 1, "last_seen": "2026-05-07T10:00:00+00:00"}],
            "weak_ability": [],
            "preferences": [],
            "recent_sessions": [],
        },
    )
    monkeypatch.setattr(teacher_tools, "render_student_profile_markdown", lambda profile: "# canonical")
    monkeypatch.setattr(teacher_tools, "write_student_profile_summary", lambda student_id, summary: target)

    result = await teacher_tools.sync_student_profile_tool.ainvoke({"student_id": "stu-1"})

    assert result == {"status": "ok", "path": str(target)}
