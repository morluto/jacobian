"""Narrow runtime adapter protocols shared by registration and dispatch."""

from __future__ import annotations

from typing import Literal, Protocol, runtime_checkable

from jacobian.contracts.capabilities import (
    CapabilityDescriptor,
    CapabilityRequest,
    CapabilityResult,
)


class CapabilityAdapter(Protocol):
    """Operator-installed adapter; registration requires no MCP changes."""

    @property
    def descriptor(self) -> CapabilityDescriptor: ...

    def invoke(self, request: CapabilityRequest) -> CapabilityResult: ...


@runtime_checkable
class TypedInputAdapter(Protocol):
    """Adapter that owns one typed input parse and needs no schema execution."""

    typed_input: Literal[True]


__all__ = ["CapabilityAdapter", "TypedInputAdapter"]
