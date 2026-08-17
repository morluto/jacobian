"""Exact linear canonical-form operations."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from jacobian.math_tools import MathTools

__all__ = ["linear_canonical_form_operations"]


def linear_canonical_form_operations() -> MathTools:
    from jacobian.domains.linear_canonical_forms.math_tools import (
        LINEAR_CANONICAL_FORM_OPERATIONS,
    )

    return LINEAR_CANONICAL_FORM_OPERATIONS
