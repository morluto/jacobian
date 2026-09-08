"""Complete scalar field gradients on an unchanged coordinate axis."""

from typing import Self

from pydantic import Field, model_validator

from jacobian._models import StrictModel
from jacobian.math.polynomials.values import PolynomialVariable, RationalFunction


class RationalGradientRequest(StrictModel):
    function: RationalFunction


class RationalFunctionGradient(StrictModel):
    """The complete coordinate partial derivatives of a retained field element."""

    variables: tuple[PolynomialVariable, ...] = Field(max_length=8)
    partial_derivatives: tuple[RationalFunction, ...] = Field(max_length=8)

    @model_validator(mode="after")
    def require_axes(self) -> Self:
        if len(self.partial_derivatives) != len(self.variables):
            raise ValueError("gradient must have one component per coordinate")
        if any(value.variables != self.variables for value in self.partial_derivatives):
            raise ValueError("every partial derivative must retain the declared axis")
        return self
