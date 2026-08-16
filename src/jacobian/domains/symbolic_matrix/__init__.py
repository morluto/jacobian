"""Exact symbolic matrix operations over QQ(t_1, ..., t_n)."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from jacobian.math_tools import MathTools

__all__ = ["symbolic_matrix_operations"]


def symbolic_matrix_operations() -> MathTools:
    from jacobian.domains.symbolic_matrix.math_tools import (
        SYMBOLIC_MATRIX_OPERATIONS,
    )

    return SYMBOLIC_MATRIX_OPERATIONS
