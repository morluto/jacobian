"""Number field operations."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from jacobian.math_tools import MathTools
__all__ = ["number_field_operations"]


def number_field_operations() -> MathTools:
    from jacobian.domains.number_field.math_tools import NUMBER_FIELD_OPERATIONS

    return NUMBER_FIELD_OPERATIONS
