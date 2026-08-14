"""Exact matrix operation declarations."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from jacobian.math_tools import MathTools

__all__ = ["matrix_operations"]


def matrix_operations() -> MathTools:
    from jacobian.domains.matrices.math_tools import MATRIX_OPERATIONS

    return MATRIX_OPERATIONS
