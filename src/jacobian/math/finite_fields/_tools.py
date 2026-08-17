"""Finite-field tool declarations."""

from jacobian.catalog.models import MathTools
from jacobian.math.finite_fields._operations import (
    finite_field_operations as _build_tools,
)

__all__ = ["TOOLS"]

TOOLS: MathTools = _build_tools()
