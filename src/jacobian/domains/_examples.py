"""Small constructors for operator-authored invocation examples."""

from typing import Any

from jacobian.contracts.operations import OperationExample


def example(
    name: str,
    description: str,
    payload: dict[str, Any],
) -> OperationExample:
    return OperationExample(
        name=name,
        description=description,
        input=payload,
    )
