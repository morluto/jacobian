"""Bounded finite-instance claim testing operations."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from jacobian.math_tools import MathTools

__all__ = ["finite_instance_testing_operations"]


def finite_instance_testing_operations() -> MathTools:
    from jacobian.domains.finite_instance_testing.math_tools import (
        FINITE_INSTANCE_TESTING_OPERATIONS,
    )

    return FINITE_INSTANCE_TESTING_OPERATIONS
