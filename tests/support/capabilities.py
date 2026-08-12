"""Capability test builders shared across semantic test lanes."""

from __future__ import annotations

from typing import Any, Protocol

from jacobian.contracts.capabilities import (
    CapabilityRequest,
    CapabilityResult,
)
from jacobian.runtime.services import CoreServices


class CapabilityRuntime(Protocol):
    """The minimal public invocation surface used by capability tests."""

    @property
    def core(self) -> CoreServices: ...


def invoke_capability(
    runtime: CapabilityRuntime,
    capability_id: str,
    payload: dict[str, Any],
) -> CapabilityResult:
    """Invoke one capability through the public request envelope."""

    return runtime.core.capabilities.invoke(
        CapabilityRequest(capability_id=capability_id, input=payload)
    )
