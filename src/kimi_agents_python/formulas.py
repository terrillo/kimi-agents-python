"""Tool wrappers for Moonshot Formula API tools.

A :class:`FormulaTool` (or :class:`AsyncFormulaTool`) is a duck-typed
:class:`~kimi_agents_python.tools.ToolLike` whose ``invoke()`` dispatches to
the Formula API rather than running local Python.

Create via ``client.formulas.load("moonshot/web-search:latest")`` —
the resource fetches the tool defs from ``/formulas/{uri}/tools`` and
returns a wrapper per published function.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from .types import FunctionDef, ToolDef

if TYPE_CHECKING:
    from .resources.formulas import AsyncFormulas, Formulas


def _default_failure(tool_name: str) -> Callable[[Exception], str]:
    def fmt(e: Exception) -> str:
        return f"Formula tool '{tool_name}' raised {type(e).__name__}: {e}"

    return fmt


def _split_def(d: dict[str, Any]) -> tuple[str, str, dict[str, Any], bool]:
    """Pull (name, description, parameters, strict) out of a `/formulas/.../tools` entry."""
    fn = d.get("function") or {}
    name = fn.get("name") or d.get("name") or ""
    description = fn.get("description") or ""
    parameters = fn.get("parameters") or {"type": "object", "properties": {}}
    strict = bool(fn.get("strict") or False)
    return name, description, parameters, strict


@dataclass(slots=True)
class FormulaTool:
    """Sync Formula tool wrapper. Drop directly into a tool list."""

    formula_uri: str
    name: str
    description: str
    params_json_schema: dict[str, Any] = field(repr=False)
    _formulas: Formulas = field(repr=False)
    strict: bool = False
    read_only: bool = False
    can_parallel: bool = True
    failure_error_function: Callable[[Exception], str] | None = None

    @classmethod
    def from_def(
        cls, formulas: Formulas, formula_uri: str, definition: dict[str, Any]
    ) -> FormulaTool:
        name, description, parameters, strict = _split_def(definition)
        return cls(
            formula_uri=formula_uri,
            name=name,
            description=description,
            params_json_schema=parameters,
            _formulas=formulas,
            strict=strict,
            failure_error_function=_default_failure(name),
        )

    def to_tool_def(self) -> ToolDef:
        return ToolDef(
            function=FunctionDef(
                name=self.name,
                description=self.description or None,
                parameters=self.params_json_schema,
                strict=self.strict or None,
            )
        )

    def invoke(self, arguments: str | dict[str, Any]) -> str:
        try:
            fiber = self._formulas.invoke(
                self.formula_uri, name=self.name, arguments=arguments
            )
            return fiber.output_str
        except Exception as e:
            return self._handle_failure(e)

    async def ainvoke(self, arguments: str | dict[str, Any]) -> str:
        raise RuntimeError(
            f"FormulaTool '{self.name}' is sync; use AsyncFormulaTool from "
            f"AsyncKimiClient.formulas.load() inside an async tool loop."
        )

    def _handle_failure(self, exc: Exception) -> str:
        if self.failure_error_function is None:
            raise exc
        return self.failure_error_function(exc)


@dataclass(slots=True)
class AsyncFormulaTool:
    """Async Formula tool wrapper. Use with :class:`AsyncSession`."""

    formula_uri: str
    name: str
    description: str
    params_json_schema: dict[str, Any] = field(repr=False)
    _formulas: AsyncFormulas = field(repr=False)
    strict: bool = False
    read_only: bool = False
    can_parallel: bool = True
    failure_error_function: Callable[[Exception], str] | None = None

    @classmethod
    def from_def(
        cls,
        formulas: AsyncFormulas,
        formula_uri: str,
        definition: dict[str, Any],
    ) -> AsyncFormulaTool:
        name, description, parameters, strict = _split_def(definition)
        return cls(
            formula_uri=formula_uri,
            name=name,
            description=description,
            params_json_schema=parameters,
            _formulas=formulas,
            strict=strict,
            failure_error_function=_default_failure(name),
        )

    def to_tool_def(self) -> ToolDef:
        return ToolDef(
            function=FunctionDef(
                name=self.name,
                description=self.description or None,
                parameters=self.params_json_schema,
                strict=self.strict or None,
            )
        )

    def invoke(self, arguments: str | dict[str, Any]) -> str:
        raise RuntimeError(
            f"AsyncFormulaTool '{self.name}' must be awaited; "
            f"call ainvoke() or pass it to AsyncSession.send()."
        )

    async def ainvoke(self, arguments: str | dict[str, Any]) -> str:
        try:
            fiber = await self._formulas.invoke(
                self.formula_uri, name=self.name, arguments=arguments
            )
            return fiber.output_str
        except Exception as e:
            return self._handle_failure(e)

    def _handle_failure(self, exc: Exception) -> str:
        if self.failure_error_function is None:
            raise exc
        return self.failure_error_function(exc)


__all__ = ["AsyncFormulaTool", "FormulaTool"]
