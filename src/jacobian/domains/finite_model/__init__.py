"""Bounded finite-model finding operations."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from jacobian.math_tools import MathTools

__all__ = ["finite_model_operations"]


def finite_model_operations() -> MathTools:
    from jacobian.domains.finite_model.math_tools import FINITE_MODEL_OPERATIONS

    return FINITE_MODEL_OPERATIONS
