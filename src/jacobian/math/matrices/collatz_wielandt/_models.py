"""Typed contracts for the Collatz-Wielandt quotient profile."""

from typing import Any

from pydantic import Field, ValidationInfo, field_validator
from pydantic_core import PydanticCustomError

from jacobian._exact import CanonicalRational
from jacobian._models import StrictModel
from jacobian.math.matrices._analysis_input import (
    AnalysisRationalInput,
    AnalysisRationalMatrixInput,
)
from jacobian.math.matrices.values import MAX_RATIONAL_MATRIX_AXIS, RationalMatrix


class CollatzWielandtRequest(StrictModel):
    """Request the Collatz-Wielandt quotient profile."""

    matrix: AnalysisRationalMatrixInput
    vector: tuple[AnalysisRationalInput, ...] = Field(
        min_length=1,
        max_length=MAX_RATIONAL_MATRIX_AXIS,
        description="Positive rational coordinates aligned with the matrix axes; "
        "JSON ratios normalize after checking each raw component's 32768-digit bound.",
    )

    @field_validator("vector", mode="before")
    @classmethod
    def bound_raw_vector(cls, value: Any, info: ValidationInfo) -> Any:
        # No admitted matrix can have more axes. Bound count before scalar gcds.
        if isinstance(value, (list, tuple)) and len(value) > MAX_RATIONAL_MATRIX_AXIS:
            raise PydanticCustomError(
                "matrix.budget_exceeded", "vector exceeds matrix axis bound"
            )
        return (
            tuple(value) if info.mode == "json" and isinstance(value, list) else value
        )


class CollatzWielandtResult(StrictModel):
    """The Collatz-Wielandt quotient profile."""

    matrix: RationalMatrix
    vector: tuple[CanonicalRational, ...]
    quotients: tuple[CanonicalRational, ...]
    max_quotient: CanonicalRational


__all__ = [
    "CollatzWielandtRequest",
    "CollatzWielandtResult",
]
