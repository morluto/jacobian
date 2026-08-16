"""Exact graph flow and cut operations."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from jacobian.math_tools import MathTools

__all__ = ["graph_flow_operations"]


def graph_flow_operations() -> MathTools:
    from jacobian.domains.graph_flow.math_tools import GRAPH_FLOW_OPERATIONS

    return GRAPH_FLOW_OPERATIONS
