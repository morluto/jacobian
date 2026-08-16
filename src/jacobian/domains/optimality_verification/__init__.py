"""Exact optimality verification operations."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from jacobian.math_tools import MathTools

__all__ = ["optimality_verification_operations"]


def optimality_verification_operations() -> MathTools:
    from jacobian.domains.optimality_verification.math_tools import (
        OPTIMALITY_VERIFICATION_OPERATIONS,
    )

    return OPTIMALITY_VERIFICATION_OPERATIONS
