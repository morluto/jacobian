"""Exact multiple testing operations."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from jacobian.math_tools import MathTools

__all__ = ["multiple_testing_operations"]


def multiple_testing_operations() -> MathTools:
    from jacobian.domains.multiple_testing.math_tools import (
        MULTIPLE_TESTING_OPERATIONS,
    )

    return MULTIPLE_TESTING_OPERATIONS
