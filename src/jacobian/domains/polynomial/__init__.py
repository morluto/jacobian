"""Exact rational polynomial operations."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from jacobian.operation_declarations import OperationDeclarations

__all__ = ["polynomial_operations"]


def polynomial_operations() -> OperationDeclarations:
    """Construct the optional operations without polluting native imports."""

    from jacobian.domains.polynomial.domain_declarations import (
        polynomial_operations as _build,
    )

    return _build()
