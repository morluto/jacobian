"""Whole-map admission followed by shared exact scalar differentiation."""

import time
from typing import NoReturn

from jacobian._execution import (
    bind_request_deadline,
    current_request_execution,
    request_checkpoint,
    request_execution,
)
from jacobian.catalog.models import OperationResourceAdmissionError
from jacobian.math.polynomials.rational_functions._bounds import (
    BoundWorkCategory,
    FractionBound,
    RationalFunctionBoundLimits,
    _dense_term_bound,
)
from jacobian.math.polynomials.rational_functions.gradient.operations import (
    _admit_general_gradient,
    _build_monomial_gradient,
    _general_gradient_admitted,
    _MonomialGradientPlan,
    _prepare_monomial_gradient,
)
from jacobian.math.polynomials.rational_functions.maps._models import (
    RationalFunctionMapJacobian,
)
from jacobian.math.polynomials.rational_functions.values import RationalFunctionMap


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
        else _dense_term_bound(polynomial.degrees)
        for polynomial in (bound.numerator, bound.denominator)
    )
    # A rational coefficient has two integers; log2(10) < 4.
    return terms, terms * 8 * digits


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
    output_allocation = _Allocation()
    plans: list[_MonomialGradientPlan | None] = []
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
        else:
            for bound, digits in _admit_general_gradient(component, ledger):
                output_allocation.charge(*_general_allocation(bound, digits))
            plans.append(None)
    # No result construction or general polynomial backend work starts until
    # every row's arithmetic and the complete matrix allocation are admitted.
    entries = []
    for component, plan in zip(source.components, plans, strict=True):
        request_checkpoint("during rational map Jacobian row construction")
        entries.append(
            _general_gradient_admitted(component)
            if plan is None
            else _build_monomial_gradient(source.source_variables, plan)
        )
    result = RationalFunctionMapJacobian(
        source=source,
        row_axis=source.target_coordinates,
        column_axis=source.source_variables,
        entries=tuple(entries),
    )
    request_checkpoint("after rational map Jacobian construction")
    return result
