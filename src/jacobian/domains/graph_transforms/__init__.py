"""Exact graph transform operations."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from jacobian.math_tools import MathTools

__all__ = ["graph_transform_operations"]


def graph_transform_operations() -> MathTools:
    from jacobian.domains.graph_transforms.math_tools import (
        GRAPH_TRANSFORM_OPERATIONS,
    )

    return GRAPH_TRANSFORM_OPERATIONS
