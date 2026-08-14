"""Domain-owned exact rational-linear operations."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from jacobian.operation_declarations import OperationDeclarations


def rational_linear_operations() -> OperationDeclarations:
    from jacobian.domains.rational_linear.domain_declarations import (
        rational_linear_operations as build,
    )

    return build()


__all__ = ["rational_linear_operations"]
