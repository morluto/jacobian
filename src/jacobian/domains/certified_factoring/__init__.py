"""Certified integer factoring operations."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from jacobian.math_tools import MathTools
__all__ = ["certified_factoring_operations"]


def certified_factoring_operations() -> MathTools:
    from jacobian.domains.certified_factoring.math_tools import (
        CERTIFIED_FACTORING_OPERATIONS,
    )

    return CERTIFIED_FACTORING_OPERATIONS
