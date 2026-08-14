"""Transformation-certified Smith normal forms."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from jacobian.math_tools import MathTools

__all__ = ["certified_snf_operations"]


def certified_snf_operations() -> MathTools:
    from jacobian.domains.certified_snf.operations import CERTIFIED_SNF_OPERATIONS

    return CERTIFIED_SNF_OPERATIONS
