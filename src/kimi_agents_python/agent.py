from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass, field
from decimal import Decimal
from typing import TYPE_CHECKING, Any

from ._enums import Model
from ._stats import TokenStats
from .exceptions import KimiError
from .tools import Compactor, LoopGuards, ToolLike

if TYPE_CHECKING:
    from .types import Message


@dataclass(slots=True)
class RunContext:
    """Per-run state: unique id, cancellation token, and caller-defined metadata."""

    run_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    cancel: asyncio.Event = field(default_factory=asyncio.Event)
    metadata: dict[str, Any] = field(default_factory=dict)

    def cancelled(self) -> bool:
        return self.cancel.is_set()


@dataclass(slots=True)
class KimiAgent:
    """A configured Kimi agent. Pass to :class:`Runner`.

    All fields are optional except ``name`` and ``model``. Tools and handoffs
    are expanded into the wire-format tool list by :class:`Runner` at call time;
    ``handoffs`` agents are converted to :class:`~kimi_agents_python.KimiTool`
    instances automatically so the parent model can delegate via function calls.
    """

    name: str
    model: Model | str
    instructions: str | None = None
    tools: list[ToolLike] = field(default_factory=list)
    handoffs: list[KimiAgent] = field(default_factory=list)
    output_type: type | None = None  # reserved — Phase 2 structured output
    max_steps: int = 10
    guards: LoopGuards | None = None
    compactor: Compactor | None = None
    model_settings: dict[str, Any] = field(default_factory=dict)
    graceful_caps: bool = False
    """If ``True``, hitting ``max_steps`` or any :class:`LoopGuards` cap
    returns a :class:`RunResult` with ``truncated=True`` and the partial
    transcript instead of raising :class:`KimiToolLoopError`. Use for
    bounded-by-design agents (a builder with a token budget, a planner
    with a step cap) where partial output is the contract, not an error.
    """


@dataclass(slots=True)
class RunResult:
    """Outcome of a :meth:`Runner.run` call.

    ``truncated`` is ``True`` when the agent had ``graceful_caps=True`` and
    a cap fired — ``messages`` and ``usage`` reflect the partial state at
    the trip point. ``final_output`` is the content of the last assistant
    message, which may be empty when the cap fired before the model
    produced any prose.
    """

    final_output: str
    messages: list[Message]
    last_agent: KimiAgent
    usage: TokenStats
    cost_usd: Decimal
    truncated: bool = False


class RunCancelledError(KimiError):
    """Raised by :meth:`Runner.run` when :attr:`RunContext.cancel` is set before the run starts."""
