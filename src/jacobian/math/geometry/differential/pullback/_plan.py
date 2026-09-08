"""Backend-free whole-request admission for metric pullbacks."""

from dataclasses import dataclass, replace
from typing import Any, NoReturn

from jacobian._execution import request_checkpoint
from jacobian.catalog.models import OperationResourceAdmissionError
from jacobian.math.geometry.differential.metrics._models import RationalCoordinateMetric
from jacobian.math.polynomials.rational_functions._bounds import (
    BoundWorkCategory,
    FractionBound,
    RationalFunctionBoundLimits,
    _add_polynomials,
    _differentiate_polynomial,
    _fraction_bound,
    _multiply_polynomials,
    _one_polynomial,
    _polynomial_bound,
    _remove_guaranteed_common_monomial,
    _validate_canonical_result_bound,
)
from jacobian.math.polynomials.rational_functions.composition.operations import (
    _substitute_bound,
)
from jacobian.math.polynomials.rational_functions.values import RationalFunctionMap


def reject(reason: str, message: str) -> NoReturn:
    raise OperationResourceAdmissionError(
        location=("metric", "map"),
        code=f"differential_geometry.rational_metric.pullback.{reason}",
        message=message,
    )


class Ledger:
    limits = RationalFunctionBoundLimits(
        raw_terms=4096,
        raw_digits=4096,
        result_exponent=64,
        result_terms=256,
        result_digits=128,
        label="rational metric pullback",
        reject=reject,
    )

    def __init__(self) -> None:
        self.work = 0

    def charge(self, category: BoundWorkCategory, amount: int) -> None:
        request_checkpoint(f"during rational metric pullback admission {category}")
        self.work += amount
        if self.work > 100_000_000:
            reject("work", "complete metric pullback exceeds its exact work envelope")


@dataclass(frozen=True)
class Plan:
    metric: RationalCoordinateMetric
    map: RationalFunctionMap
    metric_substitutions: tuple[FractionBound, ...]
    jacobian_entries: tuple[FractionBound, ...]


def _add_fraction(
    left: FractionBound, right: FractionBound, ledger: Ledger
) -> FractionBound:
    return FractionBound(
        _add_polynomials(
            _multiply_polynomials(left.numerator, right.denominator, ledger),
            _multiply_polynomials(right.numerator, left.denominator, ledger),
            ledger,
        ),
        _multiply_polynomials(left.denominator, right.denominator, ledger),
    )


def _determinant_bound(
    entries: tuple[FractionBound, ...], dimension: int, ledger: Ledger
) -> FractionBound:
    from itertools import permutations

    result: FractionBound | None = None
    for permutation in permutations(range(dimension)):
        term = FractionBound(_one_polynomial(dimension), _one_polynomial(dimension))
        for row, column in enumerate(permutation):
            entry = entries[row * dimension + column]
            term = FractionBound(
                _multiply_polynomials(term.numerator, entry.numerator, ledger),
                _multiply_polynomials(term.denominator, entry.denominator, ledger),
            )
        result = term if result is None else _add_fraction(result, term, ledger)
    assert result is not None
    return result


def _substitute_bound_only(
    polynomial: Any, inner_bounds: tuple[FractionBound, ...], ledger: Ledger
) -> FractionBound:
    variable_count = len(inner_bounds[0].numerator.degrees) if inner_bounds else 0
    numerator = _one_polynomial(variable_count)
    denominator = _one_polynomial(variable_count)
    for degree, bound in zip(polynomial.degrees, inner_bounds, strict=True):
        # Clearing denominators for every monomial of P is bounded by
        # (N_i + D_i)^degree; this retains both numerator and denominator
        # support, including terms whose exponent is not the maximum degree.
        base = _add_polynomials(bound.numerator, bound.denominator, ledger)
        for _ in range(degree):
            numerator = _multiply_polynomials(numerator, base, ledger)
            denominator = _multiply_polynomials(denominator, bound.denominator, ledger)
    numerator = replace(
        numerator,
        coefficient_digits=(
            numerator.coefficient_digits
            + polynomial.coefficient_digits
            + polynomial.terms.bit_length()
        ),
        rational_content=numerator.rational_content,
    )
    return FractionBound(numerator, denominator)


def _derivative_bound(value: FractionBound, axis: int, ledger: Ledger) -> FractionBound:
    n, d = value.numerator, value.denominator
    dn = _differentiate_polynomial(
        n,
        axis,
        ledger,
        active_terms=n.terms if n.degrees[axis] else 0,
        maximum_axis_exponent=max(1, n.degrees[axis]),
        minimum_exponents=tuple(
            max(0, e - int(i == axis)) for i, e in enumerate(n.minimum_exponents)
        ),
    )
    dd = _differentiate_polynomial(
        d,
        axis,
        ledger,
        active_terms=d.terms if d.degrees[axis] else 0,
        maximum_axis_exponent=max(1, d.degrees[axis]),
        minimum_exponents=tuple(
            max(0, e - int(i == axis)) for i, e in enumerate(d.minimum_exponents)
        ),
    )
    numerator = _add_polynomials(
        _multiply_polynomials(dn, d, ledger),
        _multiply_polynomials(n, dd, ledger),
        ledger,
    )
    return FractionBound(numerator, _multiply_polynomials(d, d, ledger))


def build_plan(
    metric: RationalCoordinateMetric, map_value: RationalFunctionMap
) -> Plan:
    n, m = len(map_value.source_variables), len(metric.tensor.coordinate_axis)
    if n < 1 or n > 4 or n**2 > 256 or m**2 > 256 or m * n > 4096:
        reject(
            "shape",
            "complete metric pullback exceeds the tensor or Jacobian shape budget",
        )
    ledger = Ledger()
    metric_bounds = tuple(
        _fraction_bound(value, ledger) for value in metric.tensor.components
    )
    map_bounds = tuple(_fraction_bound(value, ledger) for value in map_value.components)
    substitutions = []
    for value in metric.tensor.components:
        substituted = _substitute_bound(value.numerator, map_bounds, ledger)
        denominator = _substitute_bound(value.denominator, map_bounds, ledger)
        substitutions.append(
            FractionBound(
                _multiply_polynomials(
                    substituted.numerator, denominator.denominator, ledger
                ),
                _multiply_polynomials(
                    substituted.denominator, denominator.numerator, ledger
                ),
            )
        )
    determinant_bound = _determinant_bound(metric_bounds, m, ledger)
    determinant_num = _substitute_bound_only(
        determinant_bound.numerator, map_bounds, ledger
    )
    determinant_den = _substitute_bound_only(
        determinant_bound.denominator, map_bounds, ledger
    )
    _multiply_polynomials(
        determinant_num.numerator, determinant_den.denominator, ledger
    )
    _multiply_polynomials(
        determinant_num.denominator, determinant_den.numerator, ledger
    )
    guard_terms = sum(bound.denominator.terms for bound in map_bounds)
    guard_terms += sum(value.denominator.terms for value in substitutions)
    guard_terms += determinant_num.numerator.terms
    for guard in metric.tensor.retained_nonzero_denominators:
        guard_bound = _substitute_bound_only(
            _polynomial_bound(guard), map_bounds, ledger
        )
        _validate_canonical_result_bound(guard_bound, ledger)
        guard_terms += guard_bound.numerator.terms
    jacobian = []
    for bound in map_bounds:
        for axis in range(n):
            jacobian.append(_derivative_bound(bound, axis, ledger))
    # Reserve the complete contraction using raw cleared fractions. This is
    # deliberately conservative and happens before any nested executor runs.
    output_terms = 0
    output_digits = 0
    for a in range(n):
        for b in range(n):
            contraction: FractionBound | None = None
            for i in range(m):
                for j in range(m):
                    left = jacobian[i * n + a]
                    right = jacobian[j * n + b]
                    middle = substitutions[i * m + j]
                    term = FractionBound(
                        _multiply_polynomials(
                            _multiply_polynomials(
                                left.numerator, middle.numerator, ledger
                            ),
                            right.numerator,
                            ledger,
                        ),
                        _multiply_polynomials(
                            _multiply_polynomials(
                                left.denominator, middle.denominator, ledger
                            ),
                            right.denominator,
                            ledger,
                        ),
                    )
                    contraction = (
                        term
                        if contraction is None
                        else FractionBound(
                            _add_polynomials(
                                _multiply_polynomials(
                                    contraction.numerator, term.denominator, ledger
                                ),
                                _multiply_polynomials(
                                    term.numerator, contraction.denominator, ledger
                                ),
                                ledger,
                            ),
                            _multiply_polynomials(
                                contraction.denominator, term.denominator, ledger
                            ),
                        )
                    )
            assert contraction is not None
            canonical = _remove_guaranteed_common_monomial(contraction)
            digits = _validate_canonical_result_bound(canonical, ledger)
            output_terms += canonical.numerator.terms + canonical.denominator.terms
            output_digits += (
                8 * digits * (canonical.numerator.terms + canonical.denominator.terms)
            )
    guard_count = (
        len(map_value.components)
        + len(metric.tensor.components)
        + len(metric.tensor.retained_nonzero_denominators)
        + 1
        + n * n
    )
    if guard_count > 768:
        reject("locus", "complete pullback locus exceeds 768 guards")
    output_terms += guard_terms
    output_digits += guard_terms * 8 * 128
    if output_terms > 65_536 or output_digits > 8_388_608:
        reject(
            "allocation", "complete pullback output exceeds its exact allocation budget"
        )
    return Plan(metric, map_value, tuple(substitutions), tuple(jacobian))


__all__ = ["Plan", "build_plan"]
