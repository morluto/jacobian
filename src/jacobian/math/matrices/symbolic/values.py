"""Canonical polynomial-ring and rational-function matrix values."""

from typing import Annotated, Literal, Self

from pydantic import Field, model_validator

from jacobian._models import StrictModel

# The operation request models remain in ``_models`` for the symbolic catalog,
# while these names provide the stable value-owned import path for producers
# and consumers.
from jacobian.math.matrices.symbolic._models import (
    RationalFunctionMatrix,
    RationalFunctionVector,
    RationalFunctionVectorBasis,
)
from jacobian.math.polynomials.values import PolynomialVariable, RationalPolynomial


class RationalPolynomialMatrix(StrictModel):
    """A rectangular matrix over QQ[t], including its empty ordered axes.

    Entries reuse the polynomial owner's canonical values. This carrier never
    silently embeds the coefficient ring in its field of fractions.
    """

    domain: Literal["QQ"] = "QQ"
    variables: tuple[PolynomialVariable, ...] = Field(min_length=1, max_length=1)
    row_count: int = Field(ge=0, le=8192)
    column_count: int = Field(ge=0, le=8192)
    entries: tuple[
        Annotated[tuple[RationalPolynomial, ...], Field(max_length=8192)], ...
    ] = Field(max_length=8192)

    @model_validator(mode="after")
    def require_shape_and_ring(self) -> Self:
        if len(self.entries) != self.row_count or any(
            len(row) != self.column_count for row in self.entries
        ):
            raise ValueError("polynomial entries must match the declared matrix axes")
        if any(
            entry.variables != self.variables for row in self.entries for entry in row
        ):
            raise ValueError("every polynomial must belong to the matrix's QQ[t] ring")
        return self


__all__ = [
    "RationalFunctionMatrix",
    "RationalFunctionVector",
    "RationalFunctionVectorBasis",
    "RationalPolynomialMatrix",
]
