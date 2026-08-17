"""Exact polynomial interpolation operations."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from jacobian.math_tools import MathTools

__all__ = ["polynomial_interpolation_operations"]


def polynomial_interpolation_operations() -> MathTools:
    from jacobian.domains.polynomial_interpolation.math_tools import (
        POLYNOMIAL_INTERPOLATION_OPERATIONS,
    )

    return POLYNOMIAL_INTERPOLATION_OPERATIONS
