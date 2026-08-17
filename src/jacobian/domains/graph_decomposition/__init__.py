"""Exact structural graph decomposition operations."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from jacobian.math_tools import MathTools

__all__ = ["graph_decomposition_operations"]


def graph_decomposition_operations() -> MathTools:
    from jacobian.domains.graph_decomposition.math_tools import (
        GRAPH_DECOMPOSITION_OPERATIONS,
    )

    return GRAPH_DECOMPOSITION_OPERATIONS
