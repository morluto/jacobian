"""Canonical elementary symmetric polynomial families."""

from itertools import combinations

from pydantic import Field, model_validator

from jacobian._exact import CanonicalRational
from jacobian._models import StrictModel
from jacobian.math.polynomials.values import (
    MAX_POLYNOMIAL_VARIABLES,
    PolynomialVariable,
    RationalPolynomial,
    RationalPolynomialTerm,
    SparseRationalPolynomial,
)


class ElementarySymmetricFamilyRequest(StrictModel):
    variables: tuple[PolynomialVariable, ...] = Field(
        min_length=0, max_length=MAX_POLYNOMIAL_VARIABLES
    )
    maximum_degree: int = Field(ge=0, le=MAX_POLYNOMIAL_VARIABLES)

    @model_validator(mode="after")
    def require_degree_and_axis(self) -> "ElementarySymmetricFamilyRequest":
        if len(set(self.variables)) != len(self.variables):
            raise ValueError("elementary symmetric variables must be unique")
        if self.maximum_degree > len(self.variables):
            raise ValueError("maximum_degree cannot exceed the variable count")
        return self


class ElementarySymmetricFamilyResult(StrictModel):
    variables: tuple[PolynomialVariable, ...]
    maximum_degree: int
    polynomials: tuple[RationalPolynomial, ...]


def elementary_symmetric_family(
    request: ElementarySymmetricFamilyRequest,
) -> ElementarySymmetricFamilyResult:
    variable_count = len(request.variables)
    one = CanonicalRational(num=1, den=1)
    polynomials = []
    for degree in range(request.maximum_degree + 1):
        terms = tuple(
            RationalPolynomialTerm(
                coefficient=one,
                exponents=tuple(
                    int(index in selected) for index in range(variable_count)
                ),
            )
            for selected in combinations(range(variable_count), degree)
        )
        polynomials.append(
            RationalPolynomial(
                variables=request.variables,
                polynomial=SparseRationalPolynomial(terms=terms),
            )
        )
    return ElementarySymmetricFamilyResult(
        variables=request.variables,
        maximum_degree=request.maximum_degree,
        polynomials=tuple(polynomials),
    )
