"""Admitted complete field gradients, preserving sparse Laurent structure."""

import time
from fractions import Fraction
from typing import NoReturn

from pydantic_core import PydanticCustomError

from jacobian._exact import CanonicalRational
from jacobian._execution import (
    bind_request_deadline,
    current_request_execution,
    request_checkpoint,
    request_execution,
)
from jacobian.canonical import format_canonical_integer
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.polynomials._conversions import sparse_rational_polynomial_to_sympy
from jacobian.math.polynomials.rational_functions._bounds import (
    BoundsLedger,
    BoundWorkCategory,
    FractionBound,
    RationalFunctionBoundLimits,
    _fraction_bound,
    _polynomial_backend_conversion_work_units,
    _recognition_work_units,
    _remove_exact_common_factor,
    _remove_guaranteed_common_monomial,
    _validate_canonical_result_bound,
)
from jacobian.math.polynomials.rational_functions._bounds import (
    _differentiate_fraction as _derivative_bound,
)
from jacobian.math.polynomials.rational_functions.gradient._gcd_process import (
    DerivativeGcdFactor,
    forced_denominator_derivative_gcds,
    source_is_coprime,
)
from jacobian.math.polynomials.rational_functions.gradient._kernel import (
    _differentiate_fraction,
    _normalize_fraction,
)
from jacobian.math.polynomials.rational_functions.gradient._models import (
    RationalFunctionGradient,
)
from jacobian.math.polynomials.values import (
    MAX_RATIONAL_FUNCTION_COEFFICIENT_DIGITS,
    MAX_RATIONAL_FUNCTION_EXPONENT,
    MAX_RATIONAL_FUNCTION_TERMS,
    RationalFunction,
    RationalPolynomialTerm,
    SparseRationalPolynomial,
    _require_rational_function_shapes,
    _require_rational_function_structural_normal_form,
    require_sparse_polynomial_budget,
)


def _reject(reason: str, message: str) -> NoReturn:
    raise OperationResourceAdmissionError(
        location=("function",),
        code=f"rational_function.gradient.{reason}",
        message=message,
    )


class _Ledger:
    def __init__(self) -> None:
        self.work = 0
        self.limits = RationalFunctionBoundLimits(
            raw_terms=4096,
            raw_digits=4096,
            result_exponent=64,
            result_terms=256,
            result_digits=128,
            label="Rational gradient",
            reject=_reject,
        )

    def charge(self, category: BoundWorkCategory, amount: int) -> None:
        request_checkpoint(f"during rational gradient {category}")
        self.work += amount
        if self.work > 25_000_000:
            _reject(
                "work_budget",
                "rational gradient exceeds its exact coefficient-work envelope",
            )


type _MonomialGradientPlan = tuple[
    tuple[tuple[int, ...], tuple[tuple[Fraction, tuple[int, ...]], ...]], ...
]


def _recognize_source(source: RationalFunction) -> None:
    """Recognize the authored field presentation at the admitted owner boundary."""
    try:
        _require_rational_function_shapes(source)
        _require_rational_function_structural_normal_form(source)
        for part, polynomial in (
            ("numerator", source.numerator),
            ("denominator", source.denominator),
        ):
            require_sparse_polynomial_budget(
                polynomial,
                maximum_terms=MAX_RATIONAL_FUNCTION_TERMS,
                maximum_exponent=MAX_RATIONAL_FUNCTION_EXPONENT,
                maximum_coefficient_digits=MAX_RATIONAL_FUNCTION_COEFFICIENT_DIGITS,
                label=f"rational function {part}",
            )
    except (PydanticCustomError, ValueError) as exc:
        if isinstance(exc, PydanticCustomError):
            raise OperationDomainValidationError(
                location=(), code=exc.type, message=exc.message()
            ) from exc
        raise OperationDomainValidationError(
            location=(), code="polynomial.budget", message=str(exc)
        ) from exc
    if not source.numerator.terms or not source.variables:
        return
    if len(source.denominator.terms) == 1:
        powers = source.denominator.terms[0].exponents
        if any(
            exponent and all(term.exponents[axis] for term in source.numerator.terms)
            for axis, exponent in enumerate(powers)
        ):
            raise OperationDomainValidationError(
                location=(),
                code="polynomial.not_coprime",
                message="rational-function numerator and denominator must be coprime",
            )
        return
    if not source_is_coprime(source):
        raise OperationDomainValidationError(
            location=(),
            code="polynomial.not_coprime",
            message="rational-function numerator and denominator must be coprime",
        )


def _prepare_monomial_gradient(source: RationalFunction) -> _MonomialGradientPlan:
    """Differentiate finite Laurent support without a dense polynomial expansion.

    The canonical carrier bounds this phase by 8*256 coefficient scalings of
    at most 128-digit rationals by integers of magnitude <=64. Exact valuation
    normalization only shifts the unchanged surviving support. All component
    supports, exponents and scalar sizes are checked before result construction.
    """
    _recognize_source(source)
    variables = source.variables
    denominator_powers = source.denominator.terms[0].exponents
    plans: list[
        tuple[tuple[int, ...], tuple[tuple[Fraction, tuple[int, ...]], ...]]
    ] = []
    for axis in range(len(variables)):
        raw = tuple(
            (
                term.coefficient.as_fraction()
                * (term.exponents[axis] - denominator_powers[axis]),
                tuple(
                    exponent - power - int(j == axis)
                    for j, (exponent, power) in enumerate(
                        zip(term.exponents, denominator_powers, strict=True)
                    )
                ),
            )
            for term in source.numerator.terms
            if term.exponents[axis] != denominator_powers[axis]
        )
        powers = tuple(
            max(0, -min((e[j] for _, e in raw), default=0))
            for j in range(len(variables))
        )
        normalized = tuple(
            (coefficient, tuple(e + p for e, p in zip(exponents, powers, strict=True)))
            for coefficient, exponents in raw
        )
        if any(p > 64 for p in powers) or any(
            any(e > 64 for e in exponents) for _, exponents in normalized
        ):
            _reject(
                "result_exponent",
                "Laurent derivative exceeds the canonical exponent bound",
            )
        if any(
            len(format_canonical_integer(abs(value.numerator))) > 128
            or len(format_canonical_integer(value.denominator)) > 128
            for value, _ in normalized
        ):
            _reject(
                "result_height",
                "Laurent derivative exceeds the canonical coefficient bound",
            )
        plans.append((powers, normalized))
    request_checkpoint("after sparse Laurent gradient admission")
    return tuple(plans)


def _build_monomial_gradient(
    variables: tuple[str, ...], plans: _MonomialGradientPlan
) -> tuple[RationalFunction, ...]:
    """Construct the already computed Laurent derivatives after aggregate admission."""
    return tuple(
        RationalFunction._from_kernel(
            variables=variables,
            numerator=SparseRationalPolynomial(
                terms=tuple(
                    RationalPolynomialTerm(
                        coefficient=CanonicalRational.from_fraction(value),
                        exponents=exponents,
                    )
                    for value, exponents in terms
                )
            ),
            denominator=SparseRationalPolynomial(
                terms=(
                    RationalPolynomialTerm(
                        coefficient=CanonicalRational(num=1, den=1),
                        exponents=powers,
                    ),
                )
            ),
        )
        for powers, terms in plans
    )


def _admit_general_gradient(
    function: RationalFunction, ledger: BoundsLedger
) -> tuple[tuple[tuple[FractionBound, int], ...], tuple[DerivativeGcdFactor, ...]]:
    """Admit one row into a caller-owned complete scalar or matrix ledger."""
    source_bound = _fraction_bound(function, ledger)
    ledger.charge("recognition", _recognition_work_units(source_bound))
    # Source recognition constructs its own exact pair; account for the
    # additional conversion used to retain the pair through differentiation.
    ledger.charge(
        "source_conversion",
        sum(
            _polynomial_backend_conversion_work_units(bound)
            for bound in (source_bound.numerator, source_bound.denominator)
        ),
    )
    _recognize_source(function)
    variable_count = len(function.variables)
    if function.denominator.terms:
        ledger.charge(
            "normalization",
            (len(function.denominator.terms) + 1)
            * (
                max(
                    max(term.exponents, default=0)
                    for term in function.denominator.terms
                )
                + 1
            )
            * max(variable_count, 1),
        )
    factors = forced_denominator_derivative_gcds(function.denominator, variable_count)
    components = []
    for axis in range(variable_count):
        bound = _remove_guaranteed_common_monomial(
            _remove_exact_common_factor(
                _derivative_bound(function, source_bound, axis, ledger),
                factors[axis].bound,
            )
        )
        digits = _validate_canonical_result_bound(bound, ledger)
        components.append((bound, digits))
    return tuple(components), factors


def _general_gradient_admitted(
    function: RationalFunction,
    factors: tuple[DerivativeGcdFactor, ...],
) -> tuple[RationalFunction, ...]:
    """Recognize and differentiate after the caller's whole-profile admission."""
    request_checkpoint("after rational gradient source recognition")
    numerator = sparse_rational_polynomial_to_sympy(
        function.numerator, function.variables
    )
    denominator = sparse_rational_polynomial_to_sympy(
        function.denominator, function.variables
    )
    return tuple(
        _normalize_fraction(
            *_differentiate_fraction(numerator, denominator, axis),
            function.variables,
            factors[axis].records,
        )
        for axis in range(len(function.variables))
    )


def gradient(function: RationalFunction) -> RationalFunctionGradient:
    """Return every exact coordinate partial derivative with its retained source."""
    execution = current_request_execution()
    if execution is None:
        with request_execution(time.monotonic()):
            return gradient(function)
    deadline = execution.started_at + 60
    if execution.deadline is not None:
        deadline = min(deadline, execution.deadline)
    bind_request_deadline(deadline)
    request_checkpoint("before rational gradient admission")
    if len(function.denominator.terms) == 1:
        derivatives = _build_monomial_gradient(
            function.variables, _prepare_monomial_gradient(function)
        )
    else:
        ledger = _Ledger()
        _, factors = _admit_general_gradient(function, ledger)
        derivatives = _general_gradient_admitted(function, factors)
    result = RationalFunctionGradient(
        source=function, variables=function.variables, partial_derivatives=derivatives
    )
    request_checkpoint("after rational gradient construction")
    return result
