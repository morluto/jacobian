"""Domain-owned exact rational-linear operations."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from jacobian.operation_declarations import OperationDeclarations


def build_rational_linear_bundle() -> OperationDeclarations:
    from jacobian.domains.rational_linear.bundle import (
        build_rational_linear_bundle as build,
    )

    return build()


__all__ = ["build_rational_linear_bundle"]
