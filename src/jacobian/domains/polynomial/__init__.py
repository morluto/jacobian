"""Exact rational polynomial operations."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from jacobian.operation_declarations import OperationDeclarations

__all__ = ["build_polynomial_bundle"]


def build_polynomial_bundle() -> OperationDeclarations:
    """Construct the optional operation bundle without polluting native imports."""

    from jacobian.domains.polynomial.bundle import build_polynomial_bundle as _build

    return _build()
