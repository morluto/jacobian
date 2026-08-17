"""Exact finite metric space operations."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from jacobian.math_tools import MathTools

__all__ = ["finite_metric_space_operations"]


def finite_metric_space_operations() -> MathTools:
    from jacobian.domains.finite_metric_spaces.math_tools import (
        FINITE_METRIC_SPACE_OPERATIONS,
    )

    return FINITE_METRIC_SPACE_OPERATIONS
