"""Complete source-bound differential matrices of rational coordinate maps."""

from typing import Annotated, Self

from pydantic import Field, model_validator

from jacobian._models import StrictModel
from jacobian.math.polynomials.rational_functions.values import (
    MAX_RATIONAL_MAP_COMPONENTS,
    RationalFunctionMap,
)
from jacobian.math.polynomials.values import (
    MAX_POLYNOMIAL_VARIABLES,
    PolynomialVariable,
    RationalFunction,
)


class RationalMapJacobianRequest(StrictModel):
    source: RationalFunctionMap


class RationalFunctionMapJacobian(StrictModel):
    """The entire Jacobian on the common regular locus of the retained source.

    Entry [a,i] is the source component a differentiated in source coordinate
    i. Original component denominators remain in ``source`` even if entries
    simplify to polynomials or zero; no larger pointwise domain is claimed.
    """

    source: RationalFunctionMap
    row_axis: tuple[PolynomialVariable, ...] = Field(
        max_length=MAX_RATIONAL_MAP_COMPONENTS
    )
    column_axis: tuple[PolynomialVariable, ...] = Field(
        max_length=MAX_POLYNOMIAL_VARIABLES
    )
    entries: tuple[
        Annotated[
            tuple[RationalFunction, ...], Field(max_length=MAX_POLYNOMIAL_VARIABLES)
        ],
        ...,
    ] = Field(max_length=MAX_RATIONAL_MAP_COMPONENTS)

    @model_validator(mode="after")
    def require_source_axes(self) -> Self:
        if self.row_axis != self.source.target_coordinates:
            raise ValueError(
                "Jacobian row axis must equal the source target coordinates"
            )
        if self.column_axis != self.source.source_variables:
            raise ValueError("Jacobian column axis must equal the source variables")
        if len(self.entries) != len(self.row_axis) or any(
            len(row) != len(self.column_axis) for row in self.entries
        ):
            raise ValueError("Jacobian entries must form the complete declared matrix")
        if any(
            value.variables != self.column_axis for row in self.entries for value in row
        ):
            raise ValueError("every Jacobian entry must retain the ordered source axis")
        return self
