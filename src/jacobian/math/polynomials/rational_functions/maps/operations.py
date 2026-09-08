"""Whole-map admission followed by shared exact scalar differentiation."""

import time
from typing import NoReturn

from jacobian._execution import (
    bind_request_deadline,
    current_request_execution,
    request_checkpoint,
    request_execution,
)
from jacobian.canonical import encode_strict_json
from jacobian.catalog.models import OperationResourceAdmissionError
from jacobian.math.polynomials.rational_functions._bounds import (
    BoundWorkCategory,
    FractionBound,
    RationalFunctionBoundLimits,
    _canonical_coefficient_digits,
    _total_degree_term_bound,
)
from jacobian.math.polynomials.rational_functions.gradient._gcd_process import (
    DerivativeGcdFactor,
)
from jacobian.math.polynomials.rational_functions.gradient.operations import (
    _admit_general_factors,
    _admit_general_gradient,
    _build_monomial_gradient,
    _general_gradient_admitted,
    _MonomialGradientPlan,
    _prepare_monomial_gradient,
    _reduced_admitted_bound,
    _validate_admitted_factors,
)
from jacobian.math.polynomials.rational_functions.maps._models import (
    RationalFunctionMapJacobian,
)
from jacobian.math.polynomials.rational_functions.values import RationalFunctionMap
from jacobian.math.polynomials.values import RationalFunction


def _reject(reason: str, message: str) -> NoReturn:
    raise OperationResourceAdmissionError(
        location=("source",),
        code=f"rational_function_map.jacobian.{reason}",
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
            label="Rational map Jacobian",
            reject=_reject,
        )

    def charge(self, category: BoundWorkCategory, amount: int) -> None:
        request_checkpoint(f"during rational map Jacobian {category}")
        self.work += amount
        if self.work > 25_000_000:
            _reject(
                "work_budget", "Jacobian exceeds its aggregate coefficient-work bound"
            )


class _Allocation:
    """Finite sparse support and exact coefficient storage across the entire map."""

    def __init__(self) -> None:
        self.terms = 0
        self.bits = 0

    def charge(self, terms: int, bits: int) -> None:
        self.terms += terms
        self.bits += bits
        if self.terms > 65_536 or self.bits > 8_388_608:
            _reject(
                "allocation",
                "Jacobian source or complete output exceeds 65,536 polynomial terms "
                "or 8,388,608 numerator/denominator coefficient bits",
            )


def _general_allocation(bound: FractionBound, digits: int) -> tuple[int, int]:
    if bound.is_zero:
        return 1, 2
    denominator_is_unit = not any(bound.denominator.degrees)
    terms = sum(
        polynomial.terms
        if denominator_is_unit
        else min(polynomial.terms, _total_degree_term_bound(polynomial))
        if polynomial.proven_cancellation_support
        else _total_degree_term_bound(polynomial)
        for polynomial in (bound.numerator, bound.denominator)
    )
    # A rational coefficient has two integers; log2(10) < 4.
    return terms, terms * 8 * digits


def _charge_and_build_general_row(
    component: RationalFunction,
    admitted_bounds: tuple[FractionBound, ...],
    ledger: _Ledger,
    output_allocation: _Allocation,
    factor_cache: dict[bytes, tuple[DerivativeGcdFactor, ...]],
    general_rows: dict[bytes, tuple[RationalFunction, ...]],
    key: bytes,
) -> tuple[RationalFunction, ...]:
    factors = factor_cache.get(key)
    if factors is None:
        factors = _admit_general_factors(component, admitted_bounds)
        factor_cache[key] = factors
    _validate_admitted_factors(admitted_bounds, factors, ledger)
    for bound, factor in zip(admitted_bounds, factors, strict=True):
        reduced = _reduced_admitted_bound(bound, factor)
        digits = 1 if reduced.is_zero else _canonical_coefficient_digits(reduced)
        output_allocation.charge(*_general_allocation(reduced, digits))
    row = general_rows.get(key)
    if row is None:
        row = _general_gradient_admitted(component, factors, admitted_bounds)
        general_rows[key] = row
    return row


def jacobian_matrix(source: RationalFunctionMap) -> RationalFunctionMapJacobian:
    """Differentiate every component on the source map's common regular locus."""
    execution = current_request_execution()
    if execution is None:
        with request_execution(time.monotonic()):
            return jacobian_matrix(source)
    deadline = execution.started_at + 60
    if execution.deadline is not None:
        deadline = min(deadline, execution.deadline)
    bind_request_deadline(deadline)
    request_checkpoint("before rational map Jacobian admission")
    ledger = _Ledger()
    source_allocation = _Allocation()
    for component in source.components:
        for polynomial in (component.numerator, component.denominator):
            source_allocation.charge(
                len(polynomial.terms),
                sum(
                    abs(term.coefficient.num).bit_length()
                    + term.coefficient.den.bit_length()
                    for term in polynomial.terms
                ),
            )
        ledger.charge(
            "source_conversion",
            (len(component.numerator.terms) + len(component.denominator.terms))
            * (len(source.source_variables) + 1),
        )
    output_allocation = source_allocation
    plans: list[_MonomialGradientPlan | None] = []
    general_bounds: list[tuple[FractionBound, ...] | None] = []
    for component in source.components:
        if len(component.denominator.terms) == 1:
            # Bounded sparse presolve: at most n*s rational scalings and n*n*s
            # exponent shifts, with source rationals already at most128 digits.
            ledger.charge(
                "differentiation",
                (len(source.source_variables) + 1) ** 2
                * (len(component.numerator.terms) + 1)
                * 4,
            )
            try:
                plan = _prepare_monomial_gradient(component)
            except OperationResourceAdmissionError as exc:
                error = exc.errors()[0]
                _reject(str(error["type"]).rsplit(".", 1)[-1], str(error["msg"]))
            for _, terms in plan:
                output_allocation.charge(
                    len(terms) + 1,
                    2
                    + sum(
                        abs(value.numerator).bit_length()
                        + value.denominator.bit_length()
                        for value, _ in terms
                    ),
                )
            plans.append(plan)
            general_bounds.append(None)
        else:
            bounds = _admit_general_gradient(component, ledger)
            plans.append(None)
            general_bounds.append(bounds)
    entries = []
    general_rows: dict[bytes, tuple[RationalFunction, ...]] = {}
    monomial_rows: dict[bytes, tuple[RationalFunction, ...]] = {}
    factor_cache: dict[bytes, tuple[DerivativeGcdFactor, ...]] = {}
    for component, monomial_plan, admitted_bounds in zip(
        source.components, plans, general_bounds, strict=True
    ):
        request_checkpoint("during rational map Jacobian row construction")
        key = encode_strict_json(component.model_dump(mode="json"))
        if monomial_plan is None:
            if admitted_bounds is None:
                raise RuntimeError("admitted Jacobian row is missing derivative bounds")
            entries.append(
                _charge_and_build_general_row(
                    component,
                    admitted_bounds,
                    ledger,
                    output_allocation,
                    factor_cache,
                    general_rows,
                    key,
                )
            )
        else:
            row = monomial_rows.get(key)
            if row is None:
                row = _build_monomial_gradient(source.source_variables, monomial_plan)
                monomial_rows[key] = row
            entries.append(row)
    result = RationalFunctionMapJacobian(
        source=source,
        row_axis=source.target_coordinates,
        column_axis=source.source_variables,
        entries=tuple(entries),
    )
    request_checkpoint("after rational map Jacobian construction")
    return result
