from __future__ import annotations

import pytest

from kimi_agents_python import (
    KimiAgent,
    LoopGuards,
    Model,
    RunCancelledError,
    RunContext,
    kimi_tool,
    web_search,
)
from kimi_agents_python.agent import KimiAgent  # noqa: F811 – confirm internal import path


# -- KimiAgent ----------------------------------------------------------------


def test_kimiagent_minimal():
    agent = KimiAgent(name="bot", model=Model.KIMI_K2_6)
    assert agent.name == "bot"
    assert agent.model == Model.KIMI_K2_6
    assert agent.instructions is None
    assert agent.tools == []
    assert agent.handoffs == []
    assert agent.output_type is None
    assert agent.max_steps == 10
    assert agent.guards is None
    assert agent.model_settings == {}


def test_kimiagent_with_all_fields():
    @kimi_tool
    def lookup(q: str) -> str:
        """Search."""
        return q

    sub = KimiAgent(name="sub", model=Model.KIMI_K2_5)
    guards = LoopGuards(max_tokens=10_000, repeat_threshold=3)
    agent = KimiAgent(
        name="main",
        model=Model.KIMI_K2_6,
        instructions="Be helpful.",
        tools=[lookup, web_search],
        handoffs=[sub],
        output_type=dict,
        max_steps=20,
        guards=guards,
        model_settings={"thinking": {"type": "disabled"}},
    )

    assert agent.instructions == "Be helpful."
    assert len(agent.tools) == 2
    assert len(agent.handoffs) == 1
    assert agent.handoffs[0] is sub
    assert agent.output_type is dict
    assert agent.max_steps == 20
    assert agent.guards is guards
    assert agent.model_settings == {"thinking": {"type": "disabled"}}


def test_kimiagent_model_as_string():
    agent = KimiAgent(name="x", model="kimi-k2.6")
    assert agent.model == "kimi-k2.6"


def test_kimiagent_empty_handoffs_and_tools_are_independent_lists():
    a = KimiAgent(name="a", model=Model.KIMI_K2_6)
    b = KimiAgent(name="b", model=Model.KIMI_K2_6)
    a.tools.append(web_search)
    assert b.tools == []


# -- RunContext ---------------------------------------------------------------


def test_runcontext_defaults():
    ctx = RunContext()
    assert isinstance(ctx.run_id, str)
    assert len(ctx.run_id) == 32  # uuid4().hex
    assert not ctx.cancelled()
    assert ctx.metadata == {}


def test_runcontext_cancel():
    ctx = RunContext()
    assert not ctx.cancelled()
    ctx.cancel.set()
    assert ctx.cancelled()


# -- RunCancelledError --------------------------------------------------------


def test_runcancellederror_is_kimiexception():
    from kimi_agents_python import KimiError

    err = RunCancelledError("cancelled")
    assert isinstance(err, KimiError)
    assert str(err) == "cancelled"
