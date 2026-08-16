"""Exact graph coloring and independent set operations."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from jacobian.math_tools import MathTools

__all__ = ["graph_coloring_operations"]


def graph_coloring_operations() -> MathTools:
    from jacobian.domains.graph_coloring_ops.math_tools import (
        GRAPH_COLORING_OPERATIONS,
    )

    return GRAPH_COLORING_OPERATIONS
