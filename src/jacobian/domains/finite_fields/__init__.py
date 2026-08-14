"""Finite-field domain installation."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from jacobian.math_tools import MathTools

__all__ = ["finite_field_operations"]


def finite_field_operations() -> MathTools:
    from jacobian.domains.finite_fields.operations import (
        finite_field_operations as build,
    )

    return build()
