"""Contracts and exact replay for multivariate Sylvester resultants."""

from __future__ import annotations

from typing import Annotated, Literal, Self

from pydantic import Field, model_validator

from jacobian._exact import CanonicalRational
from jacobian._models import StrictModel
from jacobian.math.polynomials._conversions import (
    rational_polynomial_from_sympy,
    rational_polynomial_to_sympy,
    symbols_for_variables,
)
from jacobian.math.polynomials.values import (
    PolynomialVariable,
    RationalPolynomial,
    require_polynomial_budget,
)

from ._models import (
    _MAX_ELIMINATION_DEGREE_SUM,
    _MAX_MULTIVARIATE_COEFFICIENT_DIGITS,
    _MAX_MULTIVARIATE_EXPONENT,
    _MAX_MULTIVARIATE_TERMS,
    _validate_multivariate_pair,
    _validation_error,
)

# The result converter rejects sparse outputs above this size.  The request
# validator uses the same bound to reject large possible supports before
# SymPy expands the Sylvester determinant.
_MAX_RESULTANT_TERMS = 1_024


def _resultant_support_bound(
    left: RationalPolynomial,
    right: RationalPolynomial,
    elimination_index: int,
) -> int:
    """Bound the number of possible monomials in a multivariate resultant.

    If ``f`` and ``g`` have elimination degrees ``m`` and ``n``, and total
    remaining-variable degrees ``d_f`` and ``d_g``, every resultant monomial
    has total degree at most ``n*d_f + m*d_g``.  The returned binomial is the
    number of monomials up to that degree in the remaining variables.
    """
    from math import comb

    remaining_variable_count = len(left.variables) - 1
    if remaining_variable_count == 0:
        return 1

    def degree(polynomial: RationalPolynomial, *, in_remaining: bool) -> int:
        return max(
            (
                sum(
                    exponent
                    for index, exponent in enumerate(term.exponents)
                    if (index != elimination_index) == in_remaining
                )
                for term in polynomial.polynomial.terms
            ),
            default=0,
        )

    left_elimination_degree = degree(left, in_remaining=False)
    right_elimination_degree = degree(right, in_remaining=False)
    left_remaining_degree = degree(left, in_remaining=True)
    right_remaining_degree = degree(right, in_remaining=True)
    resultant_degree_bound = (
        right_elimination_degree * left_remaining_degree
        + left_elimination_degree * right_remaining_degree
    )
    return comb(
        resultant_degree_bound + remaining_variable_count,
        remaining_variable_count,
    )


class MultivariateResultantRequest(StrictModel):
    """Compute a bounded resultant with respect to one variable.

    The eliminated-variable domain follows the Sylvester determinant exactly,
    including its degenerate rows: an input that is a nonzero constant in the
    elimination variable contributes the standard power rule
    ``Res_x(f, c) = c ^ deg_x(f)`` (and symmetrically ``Res_x(c, g) =
    c ^ deg_x(g)``), two inputs both constant in the eliminated variable give
    the empty-determinant value 1, and a zero input gives 0.  The request
    rejects inputs whose degree envelope can produce more terms than the
    exact sparse result contract can represent.
    """

    left: RationalPolynomial
    right: RationalPolynomial
    elimination_variable: PolynomialVariable = Field(
        description="Variable eliminated by the Sylvester resultant.",
    )

    @model_validator(mode="after")
    def require_multivariate_ring(self) -> Self:
        _validate_multivariate_pair(self.left, self.right)
        if self.elimination_variable not in self.left.variables:
            raise _validation_error(
                "elimination variable must belong to the declared ring"
            )
        for polynomial in (self.left, self.right):
            require_polynomial_budget(
                polynomial,
                maximum_terms=_MAX_MULTIVARIATE_TERMS,
                maximum_exponent=_MAX_MULTIVARIATE_EXPONENT,
                maximum_coefficient_digits=_MAX_MULTIVARIATE_COEFFICIENT_DIGITS,
            )
        variable_index = self.left.variables.index(self.elimination_variable)
        degree_sum = max(
            (term.exponents[variable_index] for term in self.left.polynomial.terms),
            default=0,
        ) + max(
            (term.exponents[variable_index] for term in self.right.polynomial.terms),
            default=0,
        )
        if degree_sum > _MAX_ELIMINATION_DEGREE_SUM:
            raise _validation_error("Sylvester degree exceeds the resultant budget")
        if (
            _resultant_support_bound(self.left, self.right, variable_index)
            > _MAX_RESULTANT_TERMS
        ):
            raise _validation_error("resultant output exceeds the term budget")
        return self


class MultivariateScalarValue(StrictModel):
    kind: Literal["SCALAR"] = "SCALAR"
    value: CanonicalRational


class MultivariatePolynomialValue(StrictModel):
    kind: Literal["POLYNOMIAL"] = "POLYNOMIAL"
    value: RationalPolynomial


MultivariateInvariantValue = Annotated[
    MultivariateScalarValue | MultivariatePolynomialValue,
    Field(discriminator="kind"),
]


def _sylvester_resultant_value(
    request: MultivariateResultantRequest,
) -> MultivariateScalarValue | MultivariatePolynomialValue:
    """Replay the exact resultant value of an admitted request.

    The shared ``polynomial_resultant`` backend helper owns SymPy's
    swap-sign restoration (upstream sympy/sympy#10666), so no further
    orientation compensation happens here.
    """
    from sympy import QQ, Poly

    from jacobian.math.polynomials._sympy import polynomial_resultant

    variables = request.left.variables
    elimination_index = variables.index(request.elimination_variable)
    generator = symbols_for_variables(variables)[elimination_index]

    left = rational_polynomial_to_sympy(request.left)
    right = rational_polynomial_to_sympy(request.right)
    value = polynomial_resultant(left, right, generator)

    remaining_variables = tuple(
        variable for variable in variables if variable != request.elimination_variable
    )
    if not remaining_variables:
        from jacobian.math.polynomials._conversions import rational_from_sympy

        return MultivariateScalarValue(value=rational_from_sympy(value))
    resultant_poly = Poly(value, *symbols_for_variables(remaining_variables), domain=QQ)
    return MultivariatePolynomialValue(
        value=rational_polynomial_from_sympy(
            resultant_poly,
            remaining_variables,
            maximum_terms=_MAX_RESULTANT_TERMS,
        ),
    )


class MultivariateResultantResult(StrictModel):
    """The exact resultant bound to its source pair.

    Retains both source polynomials and the eliminated variable so
    validation replays the exact Sylvester determinant instead of trusting
    an independently authored value.  The replay runs inside the same
    admitted degree envelope as the producer.
    """

    left: RationalPolynomial
    right: RationalPolynomial
    elimination_variable: PolynomialVariable
    resultant: MultivariateInvariantValue
    convention: Literal["SYLVESTER_DETERMINANT"] = "SYLVESTER_DETERMINANT"

    @model_validator(mode="after")
    def require_source_bound(self) -> Self:
        request = MultivariateResultantRequest(
            left=self.left,
            right=self.right,
            elimination_variable=self.elimination_variable,
        )
        if self.resultant != _sylvester_resultant_value(request):
            raise _validation_error(
                "resultant must equal the Sylvester determinant of the "
                "retained source polynomials"
            )
        return self


__all__ = [
    "MultivariateInvariantValue",
    "MultivariatePolynomialValue",
    "MultivariateResultantRequest",
    "MultivariateResultantResult",
    "MultivariateScalarValue",
]
