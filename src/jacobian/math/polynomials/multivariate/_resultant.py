"""Contracts and bounded execution for multivariate Sylvester resultants."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import Field, model_validator

from jacobian._models import StrictModel
from jacobian.math.polynomials._conversions import (
    rational_polynomial_from_sympy,
    rational_polynomial_to_sympy,
    symbols_for_variables,
)
from jacobian.math.polynomials._models import (
    PolynomialInvariantValue,
    PolynomialScalarValue,
    PolynomialValue,
)
from jacobian.math.polynomials.values import PolynomialVariable, RationalPolynomial

from ._models import (
    _degree_in_variable,
    _remaining_total_degree,
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
    has total degree at most ``n*d_f + m*d_g``.  In each remaining axis ``j``
    its degree is also at most ``n*deg_j(f) + m*deg_j(g)``: every Sylvester
    determinant term takes ``n`` coefficients of ``f`` and ``m`` of ``g``.

    Bound support by the smaller of the total-degree simplex on axes with a
    positive degree bound and the per-axis degree box.  An inactive axis has
    only exponent zero, even when retained in the canonical output ring.
    These are output-support bounds, not coefficient-height or backend
    intermediate-work bounds.
    """
    from math import comb, prod

    left_elimination_degree = _degree_in_variable(left, elimination_index)
    right_elimination_degree = _degree_in_variable(right, elimination_index)
    axis_degree_bounds = tuple(
        right_elimination_degree * _degree_in_variable(left, index)
        + left_elimination_degree * _degree_in_variable(right, index)
        for index in range(len(left.variables))
        if index != elimination_index
    )
    active_variable_count = sum(degree > 0 for degree in axis_degree_bounds)
    left_remaining_degree = _remaining_total_degree(left, elimination_index)
    right_remaining_degree = _remaining_total_degree(right, elimination_index)
    resultant_degree_bound = (
        right_elimination_degree * left_remaining_degree
        + left_elimination_degree * right_remaining_degree
    )
    return min(
        comb(resultant_degree_bound + active_variable_count, active_variable_count),
        prod(degree + 1 for degree in axis_degree_bounds),
    )


class MultivariateResultantRequest(StrictModel):
    """Compute a bounded resultant with respect to one variable.

    The eliminated-variable domain follows the Sylvester determinant exactly,
    including its degenerate rows: an input that is a nonzero constant in the
    elimination variable contributes the standard power rule
    ``Res_x(f, c) = c ^ deg_x(f)`` (and symmetrically ``Res_x(c, g) =
    c ^ deg_x(g)``), two inputs both constant in the eliminated variable give
    the empty-determinant value 1, and a zero input gives 0.  The request
    rejects inputs whose active coefficient degree envelope exceeds the
    1,024-term output budget.  It uses the smaller of the active total-degree
    and per-axis degree bounds; inactive remaining variables are retained in
    the result ring without increasing either support bound.
    """

    left: RationalPolynomial
    right: RationalPolynomial
    elimination_variable: PolynomialVariable = Field(
        description="Variable eliminated by the Sylvester resultant.",
    )


def _sylvester_resultant_value(
    left: RationalPolynomial,
    right: RationalPolynomial,
    elimination_variable: str,
) -> PolynomialScalarValue | PolynomialValue:
    """Replay the exact resultant value of an admitted request.

    The shared ``polynomial_resultant`` backend helper owns SymPy's
    swap-sign restoration (upstream sympy/sympy#10666), so no further
    orientation compensation happens here.
    """
    from sympy import QQ, Poly

    from jacobian.math.polynomials._sympy import polynomial_resultant

    variables = left.variables
    elimination_index = variables.index(elimination_variable)
    generator = symbols_for_variables(variables)[elimination_index]

    left_value = rational_polynomial_to_sympy(left)
    right_value = rational_polynomial_to_sympy(right)
    value = polynomial_resultant(left_value, right_value, generator)

    remaining_variables = tuple(
        variable for variable in variables if variable != elimination_variable
    )
    if not remaining_variables:
        from jacobian.math.polynomials._conversions import rational_from_sympy

        return PolynomialScalarValue(value=rational_from_sympy(value))
    resultant_poly = Poly(value, *symbols_for_variables(remaining_variables), domain=QQ)
    return PolynomialValue(
        value=rational_polynomial_from_sympy(
            resultant_poly,
            remaining_variables,
            maximum_terms=_MAX_RESULTANT_TERMS,
        ),
    )


class MultivariateResultantResult(StrictModel):
    """The exact resultant produced by one admitted Sylvester computation."""

    left: RationalPolynomial
    right: RationalPolynomial
    elimination_variable: PolynomialVariable
    resultant: PolynomialInvariantValue
    convention: Literal["SYLVESTER_DETERMINANT"] = "SYLVESTER_DETERMINANT"

    @model_validator(mode="after")
    def require_structural_resultant(self) -> Self:
        _validate_multivariate_pair(self.left, self.right)
        if self.elimination_variable not in self.left.variables:
            raise _validation_error(
                "elimination variable must belong to the declared ring"
            )
        remaining_variables = tuple(
            variable
            for variable in self.left.variables
            if variable != self.elimination_variable
        )
        if isinstance(self.resultant, PolynomialScalarValue):
            if remaining_variables:
                raise _validation_error(
                    "a multivariate resultant must retain its remaining-variable ring"
                )
        elif self.resultant.value.variables != remaining_variables:
            raise _validation_error(
                "resultant polynomial must use the remaining declared variables"
            )
        return self

    @classmethod
    def _from_kernel(
        cls,
        left: RationalPolynomial,
        right: RationalPolynomial,
        elimination_variable: str,
        *,
        resultant: PolynomialInvariantValue,
    ) -> Self:
        """Build a result after the admitted Sylvester kernel established it."""

        return cls.model_construct(
            left=left,
            right=right,
            elimination_variable=elimination_variable,
            resultant=resultant,
        )


__all__ = [
    "MultivariateResultantRequest",
    "MultivariateResultantResult",
]
