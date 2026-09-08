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
from jacobian.math.geometry.differential._recognition_process import (
    RationalFunctionRecognitionCandidate,
    recognize_canonical_rational_functions,
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
    _remove_guaranteed_linear_power_factor,
    _validate_canonical_result_bound,
)
from jacobian.math.polynomials.rational_functions._bounds import (
    _differentiate_fraction as _derivative_bound,
)
from jacobian.math.polynomials.rational_functions.gradient._gcd_process import (
    DerivativeGcdFactor,
    forced_denominator_derivative_gcds,
)
from jacobian.math.polynomials.rational_functions.gradient._kernel import (
    _differentiate_fraction,
    _normalize_fraction,
)
from jacobian.math.polynomials.rational_functions.gradient._models import (
    RationalFunctionGradient,
)
from jacobian.math.polynomials.values import (
    RationalFunction,
    RationalPolynomialTerm,
    SparseRationalPolynomial,
    _rational_function_one,
    require_canonical_rational_function,
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


def _source_deadline() -> float:
    execution = current_request_execution()
    if execution is None:
        return time.monotonic() + 60
    deadline = execution.started_at + 60
    if execution.deadline is not None:
        deadline = min(deadline, execution.deadline)
    return deadline


def _recognize_source(source: RationalFunction, deadline: float | None = None) -> None:
    """Recognize the authored field presentation at the admitted owner boundary."""
    if deadline is None:
        deadline = _source_deadline()
    if (
        not source.numerator.terms
        or not source.variables
        or len(source.denominator.terms) == 1
    ):
        try:
            require_canonical_rational_function(source)
        except PydanticCustomError as exc:
            raise OperationDomainValidationError(
                location=(), code=exc.type, message=exc.message()
            ) from exc
        return
    recognition = recognize_canonical_rational_functions(
        (
            RationalFunctionRecognitionCandidate(
                owner="tensor",
                component=0,
                value=source,
            ),
        ),
        deadline=deadline,
    )
    if recognition.non_coprime is not None:
        raise OperationDomainValidationError(
            location=(),
            code="not_coprime",
            message="rational-function numerator and denominator must be coprime",
        )


def _prepare_monomial_gradient(
    source: RationalFunction, deadline: float | None = None
) -> _MonomialGradientPlan:
    """Differentiate finite Laurent support without a dense polynomial expansion.

    The canonical carrier bounds this phase by 8*256 coefficient scalings of
    at most 128-digit rationals by integers of magnitude <=64. Exact valuation
    normalization only shifts the unchanged surviving support. All component
    supports, exponents and scalar sizes are checked before result construction.
    """
    _recognize_source(source, deadline)
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
) -> tuple[FractionBound, ...]:
    """Admit one row's derivative bounds without launching GCD workers."""
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
    components = []
    for axis in range(variable_count):
        raw = _remove_guaranteed_common_monomial(
            _derivative_bound(function, source_bound, axis, ledger)
        )
        components.append(_remove_guaranteed_linear_power_factor(raw, function))
    return tuple(components)


def _admit_general_factors(
    function: RationalFunction,
    bounds: tuple[FractionBound, ...],
) -> tuple[DerivativeGcdFactor, ...]:
    """Run source coprimality and denominator-derivative GCDs after admission."""
    _recognize_source(function)
    return forced_denominator_derivative_gcds(
        function.denominator,
        len(function.variables),
        axes=tuple(axis for axis, bound in enumerate(bounds) if not bound.is_zero),
    )


def _reduced_admitted_bound(
    bound: FractionBound, factor: DerivativeGcdFactor
) -> FractionBound:
    """Apply the admitted exact gcd after guaranteed linear-power cancellation."""

    if (
        bound.numerator.proven_cancellation_support
        or bound.denominator.proven_cancellation_support
    ):
        return bound
    return _remove_guaranteed_common_monomial(
        _remove_exact_common_factor(bound, factor.bound)
    )


def _canonical_zero(variables: tuple[str, ...]) -> RationalFunction:
    return RationalFunction._from_kernel(
        variables=variables,
        numerator=SparseRationalPolynomial(terms=()),
        denominator=_rational_function_one(len(variables)),
    )


def _validate_admitted_factors(
    bounds: tuple[FractionBound, ...],
    factors: tuple[DerivativeGcdFactor, ...],
    ledger: BoundsLedger,
) -> None:
    for bound, factor in zip(bounds, factors, strict=True):
        reduced = _reduced_admitted_bound(bound, factor)
        _validate_canonical_result_bound(reduced, ledger, work_bound=reduced)


def _general_gradient_admitted(
    function: RationalFunction,
    factors: tuple[DerivativeGcdFactor, ...],
    bounds: tuple[FractionBound, ...],
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
        _canonical_zero(function.variables)
        if bounds[axis].is_zero
        else _normalize_fraction(
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
            function.variables, _prepare_monomial_gradient(function, deadline)
        )
    else:
        ledger = _Ledger()
        bounds = _admit_general_gradient(function, ledger)
        factors = _admit_general_factors(function, bounds)
        _validate_admitted_factors(bounds, factors, ledger)
        derivatives = _general_gradient_admitted(function, factors, bounds)
    result = RationalFunctionGradient(
        source=function, variables=function.variables, partial_derivatives=derivatives
    )
    request_checkpoint("after rational gradient construction")
    return result
