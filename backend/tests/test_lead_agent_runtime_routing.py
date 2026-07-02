from types import SimpleNamespace

from langchain_core.messages import HumanMessage

from deerflow.agents.lead_agent import agent as lead_agent_module
from deerflow.agents.middlewares.runtime_agent_routing_middleware import RuntimeAgentRoutingMiddleware


def test_extract_latest_user_text_from_human_messages():
    text = lead_agent_module._extract_latest_user_text(
        [
            HumanMessage(content="你好"),
            HumanMessage(content="请讲解这道题的步骤"),
        ]
    )

    assert text == "请讲解这道题的步骤"


def test_looks_like_teacher_request_for_problem_text():
    assert lead_agent_module._looks_like_teacher_request("请讲解这道题的步骤") is True
    assert lead_agent_module._looks_like_teacher_request("2x + 3 = 9") is True


def test_looks_like_teacher_request_false_for_regular_chat():
    assert lead_agent_module._looks_like_teacher_request("今天上海天气怎么样") is False


def test_resolve_runtime_agent_name_auto_routes_teacher_request():
    config = {
        "configurable": {},
        "input": {"messages": [HumanMessage(content="请帮我讲解这道题")]},
    }

    assert lead_agent_module._resolve_runtime_agent_name(config) == "digital-teacher"


def test_resolve_runtime_agent_name_preserves_explicit_agent():
    config = {
        "configurable": {"agent_name": "code-reviewer"},
        "input": {"messages": [HumanMessage(content="请帮我讲解这道题")]},
    }

    assert lead_agent_module._resolve_runtime_agent_name(config) == "code-reviewer"


def test_resolve_runtime_agent_name_uses_configurable_latest_user_text_fallback():
    config = {
        "configurable": {"_latest_user_text": "请讲解这道题"},
    }

    assert lead_agent_module._resolve_runtime_agent_name(config) == "digital-teacher"


def test_make_lead_agent_logs_and_uses_runtime_routed_agent(monkeypatch):
    captured: dict[str, object] = {}
    captured_prompt_kwargs: dict[str, object] = {}
    captured_tool_kwargs: dict[str, object] = {}

    monkeypatch.setattr(lead_agent_module, "get_app_config", lambda: SimpleNamespace(get_model_config=lambda _name: SimpleNamespace(supports_thinking=False)))
    monkeypatch.setattr(lead_agent_module, "_resolve_model_name", lambda _name=None: "safe-model")
    monkeypatch.setattr(lead_agent_module, "load_agent_config", lambda name: SimpleNamespace(model=None, tool_groups=["teacher"], skills=[] if name == "digital-teacher" else None))
    monkeypatch.setattr(lead_agent_module, "create_chat_model", lambda **kwargs: object())

    def _fake_get_available_tools(**kwargs):
        captured_tool_kwargs.update(kwargs)
        return []

    monkeypatch.setattr("deerflow.tools.get_available_tools", _fake_get_available_tools)
    monkeypatch.setattr(lead_agent_module, "_build_middlewares", lambda *args, **kwargs: [])

    def _fake_apply_prompt_template(**kwargs):
        captured_prompt_kwargs.update(kwargs)
        return "prompt"

    monkeypatch.setattr(lead_agent_module, "apply_prompt_template", _fake_apply_prompt_template)

    def _fake_create_agent(**kwargs):
        captured.update(kwargs)
        return kwargs

    monkeypatch.setattr(lead_agent_module, "create_agent", _fake_create_agent)

    result = lead_agent_module.make_lead_agent(
        {
            "configurable": {"thinking_enabled": True, "subagent_enabled": False},
            "input": {"messages": [HumanMessage(content="请讲解这道题")]},
        }
    )

    assert result is not None
    assert captured["system_prompt"] == "prompt"
    assert captured_prompt_kwargs["agent_name"] == "digital-teacher"
    assert captured_tool_kwargs["groups"] == ["teacher"]



def test_build_middlewares_includes_runtime_agent_routing_middleware():
    middlewares = lead_agent_module._build_middlewares({"configurable": {}}, model_name=None)

    assert any(isinstance(middleware, RuntimeAgentRoutingMiddleware) for middleware in middlewares)
