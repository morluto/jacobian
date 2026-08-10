"""Small constructors for operator-authored invocation examples."""

from typing import Any

from jacobian.contracts.capabilities import CapabilityInvocationExample, CapabilityMode


def example(
    name: str,
    description: str,
    payload: dict[str, Any],
    *,
    mode: CapabilityMode = CapabilityMode.EXPLORE,
) -> CapabilityInvocationExample:
    return CapabilityInvocationExample(
        name=name,
        description=description,
        mode=mode,
        input=payload,
    )
