from types import SimpleNamespace

from langchain_core.messages import HumanMessage

from deerflow.agents.middlewares.runtime_agent_routing_middleware import RuntimeAgentRoutingMiddleware


def _runtime(agent_name=None):
    return SimpleNamespace(context={} if agent_name is None else {"agent_name": agent_name})


def test_runtime_agent_routing_middleware_exposes_latest_user_text():
    middleware = RuntimeAgentRoutingMiddleware()
    runtime = _runtime()

    result = middleware.before_model(
        {"messages": [HumanMessage(content="请讲解这道题的步骤")]},
        runtime,
    )

    assert result == {"configurable": {"_latest_user_text": "请讲解这道题的步骤"}}
    assert runtime.context == {}


def test_runtime_agent_routing_middleware_preserves_existing_context():
    middleware = RuntimeAgentRoutingMiddleware()
    runtime = _runtime("code-reviewer")

    result = middleware.before_model(
        {"messages": [HumanMessage(content="请讲解这道题的步骤")]},
        runtime,
    )

    assert result == {"configurable": {"_latest_user_text": "请讲解这道题的步骤"}}
    assert runtime.context["agent_name"] == "code-reviewer"
