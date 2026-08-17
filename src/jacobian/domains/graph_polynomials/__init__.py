"""Exact graph polynomial operations."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from jacobian.math_tools import MathTools

__all__ = ["graph_polynomial_operations"]


def graph_polynomial_operations() -> MathTools:
    from jacobian.domains.graph_polynomials.math_tools import (
        GRAPH_POLYNOMIAL_OPERATIONS,
    )

    return GRAPH_POLYNOMIAL_OPERATIONS
