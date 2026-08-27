"""Exact Newton interpolation kernels over canonical rationals."""

from __future__ import annotations

from fractions import Fraction

from jacobian._exact import CanonicalRational
from jacobian.math.polynomials.interpolation._kernel import (
    divided_difference_coefficients,
    evaluate_newton_form,
    hermite_interpolation_coefficients,
    ordinary_derivative_value,
)
from jacobian.math.polynomials.interpolation._models import (
    DividedDifferencesRequest,
    DividedDifferencesResult,
    HermiteConstraintReplay,
    HermiteInterpolationRequest,
    HermiteInterpolationResult,
    NewtonEvaluateRequest,
    NewtonEvaluateResult,
    NewtonForm,
    NewtonFormRequest,
    OrdinaryDerivativeJetTable,
)
from jacobian.math.polynomials.values import (
    RationalPolynomial,
    RationalPolynomialTerm,
    SparseRationalPolynomial,
)


def _canonical(values: tuple[Fraction, ...]) -> tuple[CanonicalRational, ...]:
    return tuple(CanonicalRational.from_fraction(value) for value in values)


def compute_divided_differences(
    request: DividedDifferencesRequest,
) -> DividedDifferencesResult:
    coefficients = divided_difference_coefficients(
        request.samples.nodes,
        request.samples.values,
    )
    return DividedDifferencesResult(coefficients=_canonical(coefficients))


def compute_newton_form(request: NewtonFormRequest) -> NewtonForm:
    coefficients = divided_difference_coefficients(
        request.samples.nodes,
        request.samples.values,
    )
    return NewtonForm(
        coefficients=_canonical(coefficients),
        nodes=request.samples.nodes,
    )


def compute_newton_evaluate(request: NewtonEvaluateRequest) -> NewtonEvaluateResult:
    return NewtonEvaluateResult(
        result=CanonicalRational.from_fraction(
            evaluate_newton_form(
                request.newton_form.nodes,
                request.newton_form.coefficients,
                request.evaluation_point,
            )
        )
    )


def hermite_interpolation(
    table: OrdinaryDerivativeJetTable,
) -> HermiteInterpolationResult:
    """Return the unique degree-``< M`` polynomial matching one jet table."""

    coefficients = hermite_interpolation_coefficients(table)
    nonzero_degrees = tuple(
        degree for degree, coefficient in enumerate(coefficients) if coefficient
    )
    degree = max(nonzero_degrees) if nonzero_degrees else None
    polynomial = RationalPolynomial(
        variables=(table.variable,),
        polynomial=SparseRationalPolynomial(
            terms=tuple(
                RationalPolynomialTerm(
                    coefficient=CanonicalRational.from_fraction(
                        coefficients[term_degree]
                    ),
                    exponents=(term_degree,),
                )
                for term_degree in reversed(nonzero_degrees)
            )
        ),
    )
    replay = tuple(
        HermiteConstraintReplay(
            node=CanonicalRational.from_integer_ratio(*jet.node.as_integer_ratio()),
            derivative_order=derivative.derivative_order,
            expected=CanonicalRational.from_integer_ratio(
                *derivative.value.as_integer_ratio()
            ),
            computed=CanonicalRational.from_fraction(
                ordinary_derivative_value(
                    coefficients,
                    jet.node.as_fraction(),
                    derivative.derivative_order,
                )
            ),
        )
        for jet in sorted(table.jets, key=lambda item: item.node.as_fraction())
        for derivative in jet.derivatives
    )
    return HermiteInterpolationResult._from_kernel(
        source=table,
        polynomial=polynomial,
        total_multiplicity=len(coefficients),
        degree=degree,
        leading_coefficient=CanonicalRational.from_fraction(
            Fraction(0) if degree is None else coefficients[degree]
        ),
        replay=replay,
    )


def _verify_hermite_interpolation_result(result: HermiteInterpolationResult) -> bool:
    """Recompute an independently supplied Hermite claim inside its admission envelope."""

    try:
        request = HermiteInterpolationRequest.model_validate(
            {"table": result.source.model_dump(mode="json")}
        )
        return hermite_interpolation(request.table) == result
    except ValueError:
        return False


def compute_hermite_interpolation(
    request: HermiteInterpolationRequest,
) -> HermiteInterpolationResult:
    return hermite_interpolation(request.table)


__all__ = [
    "compute_divided_differences",
    "compute_hermite_interpolation",
    "compute_newton_evaluate",
    "compute_newton_form",
    "hermite_interpolation",
]
