"""Exact numerical semigroup operations."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from jacobian.math_tools import MathTools

__all__ = ["numerical_semigroup_operations"]


def numerical_semigroup_operations() -> MathTools:
    from jacobian.domains.numerical_semigroups.math_tools import (
        NUMERICAL_SEMIGROUP_OPERATIONS,
    )

    return NUMERICAL_SEMIGROUP_OPERATIONS
