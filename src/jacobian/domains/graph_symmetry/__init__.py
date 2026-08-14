"""Exact declared graph-symmetry operations."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from jacobian.math_tools import MathTools

__all__ = ["graph_symmetry_operations"]


def graph_symmetry_operations() -> MathTools:
    from jacobian.domains.graph_symmetry.operations import GRAPH_SYMMETRY_OPERATIONS

    return GRAPH_SYMMETRY_OPERATIONS
