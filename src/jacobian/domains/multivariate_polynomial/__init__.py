"""Exact multivariate polynomial operations."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from jacobian.math_tools import MathTools

__all__ = ["multivariate_polynomial_operations"]


def multivariate_polynomial_operations() -> MathTools:
    from jacobian.domains.multivariate_polynomial.math_tools import (
        MULTIVARIATE_POLYNOMIAL_OPERATIONS,
    )

    return MULTIVARIATE_POLYNOMIAL_OPERATIONS
