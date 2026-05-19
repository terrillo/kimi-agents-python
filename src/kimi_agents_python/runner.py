import asyncio
from collections.abc import Sequence

from pydantic import create_model

from ._enums import Role
from ._stats import TokenStats
from .agent import KimiAgent, RunCancelledError, RunContext, RunResult
from .client import AsyncKimiClient
from .session import _coerce_messages, _message_from_assistant
from .tools import KimiTool, ToolLike, _arun_tools_inner
from .types import Message


class Runner:
    """Executes :class:`KimiAgent` instances.

    All execution state lives in :class:`RunContext` and the returned
    :class:`RunResult`; ``Runner`` itself is stateless.
    """

    @classmethod
    async def run(
        cls,
        agent: KimiAgent,
        prompt: str,
        *,
        client: AsyncKimiClient,
        context: RunContext | None = None,
    ) -> RunResult:
        """Run ``agent`` on ``prompt`` and return a :class:`RunResult`.

        ``agent.handoffs`` are automatically converted to tools so the model
        can delegate via function calls. Raises :class:`RunCancelledError` if
        ``context.cancel`` is already set when the call begins.

        .. note::
            When tools are used, ``result.usage`` reflects only the final LLM
            response (the turn that produced ``final_output``), not the sum of
            all intermediate tool-call turns. This matches
            :meth:`~kimi_agents_python.Session.send` behaviour.
        """
        ctx = context or RunContext()
        if ctx.cancelled():
            raise RunCancelledError(
                f"Run cancelled before starting agent '{agent.name}'"
            )

        handoff_names = [h.name for h in agent.handoffs]
        duplicates = {n for n in handoff_names if handoff_names.count(n) > 1}
        if duplicates:
            raise ValueError(
                f"Agent '{agent.name}' has duplicate handoff names: {sorted(duplicates)}"
            )

        handoff_tools = [cls._build_handoff_tool(h, client, ctx) for h in agent.handoffs]
        all_tools: list[ToolLike] = list(agent.tools) + handoff_tools

        messages: list[Message] = []
        if agent.instructions:
            messages.append(Message(role=Role.SYSTEM, content=agent.instructions))
        messages.append(Message(role=Role.USER, content=prompt))

        chat_kwargs = dict(agent.model_settings)

        if all_tools:
            # usage is summed across every turn of the tool loop, not just
            # the terminal response (see _arun_tools_inner).
            response, transcript, usage = await _arun_tools_inner(
                client,
                model=agent.model,
                messages=messages,
                tools=all_tools,
                max_steps=agent.max_steps,
                guards=agent.guards,
                compactor=agent.compactor,
                **chat_kwargs,
            )
            final_messages = _coerce_messages(transcript)
            final_output = response.choices[0].message.content or ""
        else:
            response = await client.chat._create(
                model=agent.model, messages=messages, **chat_kwargs
            )
            msg = response.choices[0].message
            final_messages = messages + [_message_from_assistant(msg)]
            final_output = msg.content or ""
            usage = TokenStats()
            usage.record(response.usage, model=agent.model)

        return RunResult(
            final_output=final_output,
            messages=final_messages,
            last_agent=agent,
            usage=usage,
            cost_usd=usage.cost_usd,
        )

    @classmethod
    def run_sync(
        cls,
        agent: KimiAgent,
        prompt: str,
        *,
        client: AsyncKimiClient,
        context: RunContext | None = None,
    ) -> RunResult:
        """Sync convenience wrapper around :meth:`run`. Do not call from inside an async context."""
        return asyncio.run(cls.run(agent, prompt, client=client, context=context))

    @classmethod
    async def run_parallel(
        cls,
        runs: Sequence[tuple[KimiAgent, str]],
        *,
        client: AsyncKimiClient,
        context: RunContext | None = None,
    ) -> list[RunResult]:
        """Run multiple ``(agent, prompt)`` pairs concurrently.

        Returns results in the same order as ``runs``. A shared ``context``
        means a single ``context.cancel.set()`` call signals all agents.
        Each :class:`RunResult` carries independent usage and cost tracking.
        """
        ctx = context or RunContext()
        return list(
            await asyncio.gather(
                *[cls.run(agent, inp, client=client, context=ctx) for agent, inp in runs]
            )
        )

    @classmethod
    def _build_handoff_tool(
        cls,
        agent: KimiAgent,
        client: AsyncKimiClient,
        ctx: RunContext,
    ) -> KimiTool:
        tool_name = f"handoff_to_{agent.name}"
        description = f"Delegate this task to the {agent.name} agent."
        param_description = f"Full task description for the {agent.name} agent"

        async def _execute(task: str) -> str:
            result = await cls.run(agent, task, client=client, context=ctx)
            return result.final_output

        _execute.__name__ = tool_name

        # Build the JSON schema and args model directly to avoid annotation
        # resolution issues with closure variables.
        schema = {
            "type": "object",
            "properties": {"task": {"type": "string", "description": param_description}},
            "required": ["task"],
        }
        args_model = create_model(f"{tool_name}_Args", task=(str, ...))

        return KimiTool(
            name=tool_name,
            description=description,
            params_json_schema=schema,
            func=_execute,
            is_async=True,
            strict=True,
            can_parallel=False,
            read_only=False,
            failure_error_function=lambda e: f"Handoff to '{agent.name}' failed: {e}",
            args_model=args_model,
        )


def handoff(
    agent: KimiAgent,
    client: AsyncKimiClient,
    context: RunContext | None = None,
) -> KimiTool:
    """Build a :class:`KimiTool` that delegates to ``agent`` when called.

    Use this to add a sub-agent directly to a parent's ``tools`` list when
    you need explicit control. For automatic conversion, set
    ``KimiAgent.handoffs`` instead — :class:`Runner` handles it at run time.

    Examples::

        # Pattern A — recommended: Runner wires client automatically
        parent = KimiAgent(name="orchestrator", handoffs=[researcher], ...)

        # Pattern B — manual: explicit tool injection
        parent = KimiAgent(
            name="orchestrator",
            tools=[handoff(researcher, client=client)],
            ...
        )
    """
    return Runner._build_handoff_tool(agent, client, context or RunContext())
