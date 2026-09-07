"""Exact Newton interpolation kernels over canonical rationals."""

from __future__ import annotations

from collections.abc import Callable
from fractions import Fraction

from pydantic_core import PydanticCustomError

from jacobian._exact import MAX_CANONICAL_RATIONAL_DIGITS, CanonicalRational
from jacobian.canonical import format_canonical_integer
from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.polynomials.interpolation._kernel import (
    divided_difference_coefficients,
    evaluate_newton_form,
    hermite_interpolation_coefficients,
    ordinary_derivative_value,
)
from jacobian.math.polynomials.interpolation._models import (
    _MAX_RATIONAL_DIGITS,
    DividedDifferencesResult,
    HermiteConstraintReplay,
    HermiteInterpolationResult,
    InterpolationSamples,
    NewtonEvaluateResult,
    NewtonForm,
    OrdinaryDerivativeJetTable,
    _require_distinct,
    _require_hermite_preflight,
    _validation_error,
)
from jacobian.math.polynomials.values import (
    RationalPolynomial,
    RationalPolynomialTerm,
    SparseRationalPolynomial,
)


def _canonical(values: tuple[Fraction, ...]) -> tuple[CanonicalRational, ...]:
    return tuple(CanonicalRational.from_fraction(value) for value in values)


def _run_admission[ResultT](admission: Callable[[], ResultT]) -> ResultT:
    try:
        return admission()
    except OperationDomainValidationError:
        raise
    except PydanticCustomError as exc:
        raise OperationDomainValidationError(
            location=(), code=exc.type, message=exc.message()
        ) from exc
    except ValueError as exc:
        raise OperationDomainValidationError(
            location=(), code="polynomial.interpolation_admission", message=str(exc)
        ) from exc


def _admit_samples(samples: InterpolationSamples) -> tuple[CanonicalRational, ...]:
    _require_distinct(samples.nodes)
    coefficients = _canonical(
        divided_difference_coefficients(samples.nodes, samples.values)
    )
    for coefficient in coefficients:
        if (
            len(format_canonical_integer(abs(coefficient.num)))
            > MAX_CANONICAL_RATIONAL_DIGITS
            or len(format_canonical_integer(coefficient.den))
            > MAX_CANONICAL_RATIONAL_DIGITS
        ):
            raise _validation_error(
                "derived Newton coefficient exceeds the canonical digit bound"
            )
    return coefficients


def _admit_hermite(table: OrdinaryDerivativeJetTable) -> None:
    _require_hermite_preflight(table)


def _admit_newton_evaluate(evaluation_point: CanonicalRational) -> None:
    if (
        len(format_canonical_integer(abs(evaluation_point.num))) > _MAX_RATIONAL_DIGITS
        or len(format_canonical_integer(evaluation_point.den)) > _MAX_RATIONAL_DIGITS
    ):
        raise _validation_error(
            f"evaluation point exceeds the {_MAX_RATIONAL_DIGITS}-digit bound"
        )


def divided_differences(samples: InterpolationSamples) -> DividedDifferencesResult:
    return DividedDifferencesResult(
        samples=samples,
        coefficients=_admit_samples(samples),
    )


def newton_form(samples: InterpolationSamples) -> NewtonForm:
    coefficients = _admit_samples(samples)
    return NewtonForm(
        coefficients=coefficients,
        nodes=samples.nodes,
    )


def evaluate_newton(
    form: NewtonForm, evaluation_point: CanonicalRational
) -> NewtonEvaluateResult:
    _run_admission(lambda: _admit_newton_evaluate(evaluation_point))
    return NewtonEvaluateResult(
        newton_form=form,
        evaluation_point=evaluation_point,
        result=CanonicalRational.from_fraction(
            evaluate_newton_form(
                form.nodes,
                form.coefficients,
                evaluation_point,
            )
        ),
    )


def verify_divided_differences(claim: DividedDifferencesResult) -> bool:
    """Verify divided-difference coefficients against retained samples."""

    try:
        return divided_differences(claim.samples) == claim
    except (OperationDomainValidationError, ValueError, TypeError):
        return False


def verify_newton_evaluation(claim: NewtonEvaluateResult) -> bool:
    """Verify a Newton evaluation against its retained form and point."""

    try:
        return evaluate_newton(claim.newton_form, claim.evaluation_point) == claim
    except (OperationDomainValidationError, ValueError, TypeError):
        return False


def verify_hermite_interpolation(claim: HermiteInterpolationResult) -> bool:
    """Verify all Hermite polynomial, degree, leading, and replay claims."""

    try:
        return hermite_interpolation(claim.source) == claim
    except (OperationDomainValidationError, ValueError, TypeError):
        return False


def hermite_interpolation(
    table: OrdinaryDerivativeJetTable,
) -> HermiteInterpolationResult:
    """Return the unique degree-``< M`` polynomial matching one jet table."""

    _run_admission(lambda: _admit_hermite(table))

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


__all__ = [
    "divided_differences",
    "evaluate_newton",
    "hermite_interpolation",
    "newton_form",
    "verify_divided_differences",
    "verify_hermite_interpolation",
    "verify_newton_evaluation",
]
