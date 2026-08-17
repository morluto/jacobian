"""Exact graph isomorphism decision operations."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from jacobian.math_tools import MathTools

__all__ = ["graph_isomorphism_operations"]


def graph_isomorphism_operations() -> MathTools:
    from jacobian.domains.graph_isomorphism.math_tools import (
        GRAPH_ISOMORPHISM_OPERATIONS,
    )

    return GRAPH_ISOMORPHISM_OPERATIONS
