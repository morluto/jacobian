"""Domain-owned exact rational-linear operations."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from jacobian.math_tools import MathTools

__all__ = ["rational_linear_operations"]


def rational_linear_operations() -> MathTools:
    from jacobian.domains.rational_linear.operations import (
        rational_linear_operations as build,
    )

    return build()
