"""Small constructors for operator-authored invocation examples."""

from typing import Any

from jacobian.contracts.capabilities import CapabilityInvocationExample


def example(
    name: str,
    description: str,
    payload: dict[str, Any],
) -> CapabilityInvocationExample:
    return CapabilityInvocationExample(
        name=name,
        description=description,
        input=payload,
    )
