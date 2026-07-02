from __future__ import annotations

from typing import Any

from langchain.agents import AgentState
from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import HumanMessage
from langgraph.runtime import Runtime



def _extract_latest_user_text(messages: list[Any] | None) -> str:
    if not isinstance(messages, list):
        return ""
    for message in reversed(messages):
        if not isinstance(message, HumanMessage):
            continue
        content = message.content
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            parts: list[str] = []
            for block in content:
                if isinstance(block, str):
                    parts.append(block)
                elif isinstance(block, dict) and block.get("type") == "text":
                    text = block.get("text")
                    if isinstance(text, str):
                        parts.append(text)
            return "\n".join(parts).strip()
    return ""


class RuntimeAgentRoutingMiddleware(AgentMiddleware[AgentState]):
    def before_model(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        latest_user_text = _extract_latest_user_text(state.get("messages"))
        if latest_user_text:
            return {"configurable": {"_latest_user_text": latest_user_text}}
        return None

    async def abefore_model(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        return self.before_model(state, runtime)
