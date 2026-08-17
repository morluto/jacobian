"""Graph realization operations."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from jacobian.math_tools import MathTools

__all__ = ["graph_realization_operations"]


def graph_realization_operations() -> MathTools:
    from jacobian.domains.graph_realization.math_tools import (
        GRAPH_REALIZATION_OPERATIONS,
    )

    return GRAPH_REALIZATION_OPERATIONS
