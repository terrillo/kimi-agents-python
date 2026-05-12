"""Server-side `builtin_function` tools.

Builtin tools differ from :class:`~kimi_agents_python.tools.KimiTool` in that
the *server* runs the implementation. The client only declares the tool
(``{"type": "builtin_function", "function": {"name": "$web_search"}}``) and,
when the model returns ``finish_reason="tool_calls"``, echoes the model's
``arguments`` back as the ``role=tool`` result. The server then completes the
search and returns the final assistant message.

See ``.platform/guide/use-web-search.md`` for the wire-level protocol.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from .types import BuiltinFunctionRef, BuiltinToolDef


@dataclass(slots=True, frozen=True)
class BuiltinTool:
    """A Moonshot server-side tool. Drop-in for :class:`KimiTool` in tool loops."""

    name: str
    read_only: bool = True
    can_parallel: bool = True

    def to_tool_def(self) -> BuiltinToolDef:
        return BuiltinToolDef(function=BuiltinFunctionRef(name=self.name))

    def invoke(self, arguments: str | dict[str, Any]) -> str:
        """Echo the model's arguments back verbatim.

        The server has already prepared the call; we just need to return the
        arguments as the tool result so the model can complete the loop.
        """
        if isinstance(arguments, str):
            return arguments
        return json.dumps(arguments)

    async def ainvoke(self, arguments: str | dict[str, Any]) -> str:
        return self.invoke(arguments)


web_search: BuiltinTool = BuiltinTool(name="$web_search")
"""Server-side web search.

Usage requires disabling thinking — pass
``thinking={"type": "disabled"}`` to ``chat.create`` /
``Session.send``. The token cost of the retrieved pages is reflected in the
subsequent turn's ``usage.prompt_tokens``; the per-call fee ($0.005) is billed
by Moonshot separately (see :data:`kimi_agents_python.WEB_SEARCH_CALL_USD`).
"""


__all__ = ["BuiltinTool", "web_search"]
