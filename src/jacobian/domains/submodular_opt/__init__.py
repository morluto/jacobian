"""Submodular optimization operations."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from jacobian.math_tools import MathTools

__all__ = ["submodular_opt_operations"]


def submodular_opt_operations() -> MathTools:
    from jacobian.domains.submodular_opt.math_tools import (
        SUBMODULAR_OPT_OPERATIONS,
    )

    return SUBMODULAR_OPT_OPERATIONS
