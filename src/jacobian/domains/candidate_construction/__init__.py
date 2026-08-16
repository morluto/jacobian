"""Bounded constraint-satisfaction object construction."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from jacobian.math_tools import MathTools

__all__ = ["candidate_construction_operations"]


def candidate_construction_operations() -> MathTools:
    from jacobian.domains.candidate_construction.math_tools import (
        CANDIDATE_CONSTRUCTION_OPERATIONS,
    )

    return CANDIDATE_CONSTRUCTION_OPERATIONS
