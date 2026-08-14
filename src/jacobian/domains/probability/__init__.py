"""Exact finite-probability operations."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from jacobian.math_tools import MathTools

__all__ = ["finite_probability_operations"]


def finite_probability_operations() -> MathTools:
    from jacobian.domains.probability.operations import (
        finite_probability_operations as build,
    )

    return build()
