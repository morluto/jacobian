"""Exact regular language operations."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from jacobian.math_tools import MathTools

__all__ = ["regular_language_operations"]


def regular_language_operations() -> MathTools:
    from jacobian.domains.regular_languages.math_tools import (
        REGULAR_LANGUAGE_OPERATIONS,
    )

    return REGULAR_LANGUAGE_OPERATIONS
