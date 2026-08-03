"""Capability test builders shared across semantic test lanes."""

from __future__ import annotations

from typing import Any

from jacobian.contracts.capabilities import (
    CapabilityMode,
    CapabilityRequest,
    CapabilityResult,
)
from jacobian.runtime.model import JacobianRuntime


def invoke_capability(
    runtime: JacobianRuntime,
    capability_id: str,
    payload: dict[str, Any],
    *,
    mode: CapabilityMode = CapabilityMode.EXPLORE,
) -> CapabilityResult:
    """Invoke one capability through the public request envelope."""

    return runtime.core.capabilities.invoke(
        CapabilityRequest(capability_id=capability_id, mode=mode, input=payload)
    )
