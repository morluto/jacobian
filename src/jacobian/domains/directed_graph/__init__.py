"""Exact directed graph operations."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from jacobian.math_tools import MathTools

__all__ = ["directed_graph_operations"]


def directed_graph_operations() -> MathTools:
    from jacobian.domains.directed_graph.math_tools import (
        DIRECTED_GRAPH_OPERATIONS,
    )

    return DIRECTED_GRAPH_OPERATIONS
