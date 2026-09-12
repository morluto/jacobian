"""Whole request admission for rational metric pullbacks."""

from dataclasses import dataclass, replace
from fractions import Fraction
from itertools import permutations
from typing import NoReturn

from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.geometry.differential.metrics._dag import ONE, ZERO, Dag, Expression
from jacobian.math.geometry.differential.metrics._models import RationalCoordinateMetric
from jacobian.math.polynomials.rational_functions._bounds import (
    FractionBound,
    _polynomial_admission_work_units,
    _polynomial_backend_conversion_work_units,
    _polynomial_bound,
    _recognition_work_units,
)
from jacobian.math.polynomials.rational_functions.values import RationalFunctionMap
from jacobian.math.polynomials.values import SparseRationalPolynomial


def reject(reason: str, message: str) -> NoReturn:
    raise OperationResourceAdmissionError(
        location=("metric", "map"),
        code=f"differential_geometry.rational_metric.pullback.{reason}",
        message=message,
    )


@dataclass(frozen=True)
class Plan:
    dag: Dag
    metric: RationalCoordinateMetric
    map: RationalFunctionMap
    substitutions: tuple[Expression, ...]
    jacobian: tuple[Expression, ...]
    output: tuple[Expression, ...]
    determinant: Expression
    guards: tuple[Expression, ...]
    fractions: dict[Expression, tuple[int, int]]


def _substitute(
    dag: Dag, value: SparseRationalPolynomial, inner: tuple[Expression, ...]
) -> Expression:
    result = ZERO
    powers: dict[tuple[int, int], Expression] = {}
    for term in value.terms:
        part = Expression(term.coefficient.as_fraction())
        for axis, exponent in enumerate(term.exponents):
            power = ONE
            for degree in range(1, exponent + 1):
                key = (axis, degree)
                if key not in powers:
                    powers[key] = dag.multiply(power, inner[axis])
                power = powers[key]
            part = dag.multiply(part, power)
        result = dag.add(result, part)
    return result


def _determinant(dag: Dag, entries: tuple[Expression, ...], n: int) -> Expression:
    terms = []
    for permutation in permutations(range(n)):
        inversions = sum(
            permutation[i] > permutation[j] for i in range(n) for j in range(i + 1, n)
        )
        terms.append(
            dag.multiply(
                Expression(Fraction((-1) ** inversions)),
                *(entries[i * n + j] for i, j in enumerate(permutation)),
            )
        )
    return dag.add(*terms)


def build_plan(
    metric: RationalCoordinateMetric, map_value: RationalFunctionMap
) -> Plan:
    n, m = len(map_value.source_variables), len(metric.tensor.coordinate_axis)
    if not 1 <= n <= 4 or not 1 <= m <= 4:
        reject("shape", "source and target dimensions must be between 1 and 4")
    dag = Dag(n, reject=reject, label="rational metric pullback")
    # Metric components are authored on the target axis. Reserve their raw
    # parsing, backend conversion, and canonical recognition before any
    # substituted DAG node can be evaluated.
    for value in dict.fromkeys(metric.tensor.components):
        for polynomial in (value.numerator, value.denominator):
            dag.ledger.charge(
                "source_conversion", _polynomial_admission_work_units(polynomial, m)
            )
            dag.ledger.charge(
                "source_conversion",
                _polynomial_backend_conversion_work_units(
                    _polynomial_bound(polynomial)
                ),
            )
        nb, db = (
            _polynomial_bound(value.numerator),
            _polynomial_bound(value.denominator),
        )
        width = max(len(nb.degrees), len(db.degrees))
        if len(nb.degrees) != width:
            nb = replace(
                nb,
                degrees=nb.degrees + (0,) * (width - len(nb.degrees)),
                minimum_exponents=nb.minimum_exponents
                + (0,) * (width - len(nb.minimum_exponents)),
            )
        if len(db.degrees) != width:
            db = replace(
                db,
                degrees=db.degrees + (0,) * (width - len(db.degrees)),
                minimum_exponents=db.minimum_exponents
                + (0,) * (width - len(db.minimum_exponents)),
            )
        dag.ledger.charge("recognition", _recognition_work_units(FractionBound(nb, db)))
    for guard in metric.tensor.retained_nonzero_denominators:
        dag.ledger.charge(
            "source_conversion", _polynomial_admission_work_units(guard, m)
        )
    maps = tuple(dag.fraction(value) for value in map_value.components)
    map_denominators = tuple(
        dag.source(value.denominator) for value in map_value.components
    )
    component_guards = tuple(
        _substitute(dag, value.denominator, maps) for value in metric.tensor.components
    )
    if any(not value.scalar for value in component_guards):
        raise OperationDomainValidationError(
            location=("metric", "map"),
            code="differential_geometry.rational_metric.pullback.undefined_metric_locus",
            message="a required metric denominator vanishes identically after substitution",
        )
    substitutions = tuple(
        _substitute(dag, value.numerator, maps) for value in metric.tensor.components
    )
    substitutions = tuple(
        dag.multiply(
            value,
            dag.inverse(component_guards[i]),
        )
        for i, value in enumerate(substitutions)
    )
    jacobian = tuple(dag.derivative(value, axis) for value in maps for axis in range(n))
    determinant = _determinant(dag, substitutions, m)
    output = tuple(
        dag.add(
            *(
                dag.multiply(
                    jacobian[i * n + a], substitutions[i * m + j], jacobian[j * n + b]
                )
                for i in range(m)
                for j in range(m)
            )
        )
        for a in range(n)
        for b in range(n)
    )
    inherited = tuple(
        _substitute(dag, guard, maps)
        for guard in metric.tensor.retained_nonzero_denominators
    )
    guards = tuple(map_denominators) + component_guards + inherited + (determinant,)
    values = tuple(dict.fromkeys((*output, *guards)))
    sizes = {value: dag.admit_output(value) for value in values}
    fractions = {
        value: (
            dag.polynomial(value.numerator, value.scalar),
            dag.polynomial(value.denominator),
        )
        for value in values
    }
    # Cancellation may change each denominator polynomial. Count distinct full
    # expressions, not merely shared raw denominator factors, as in curvature.
    guard_values = {value for value in guards if value.numerator}
    output_denominators = {value for value in output if value.denominator}
    if len(guard_values | output_denominators) > 768:
        reject("locus", "complete pullback locus exceeds 768 guards")

    def source_allocation(
        polynomial: SparseRationalPolynomial, dimension: int
    ) -> tuple[int, int, int]:
        return (
            len(polynomial.terms),
            sum(
                abs(term.coefficient.num).bit_length()
                + term.coefficient.den.bit_length()
                for term in polynomial.terms
            ),
            dimension * (len(polynomial.terms) + 1),
        )

    source = [
        source_allocation(polynomial, m)
        for value in metric.tensor.components
        for polynomial in (value.numerator, value.denominator)
    ]
    source += [
        source_allocation(guard, m)
        for guard in metric.tensor.retained_nonzero_denominators
    ]
    source += [
        source_allocation(polynomial, n)
        for value in map_value.components
        for polynomial in (value.numerator, value.denominator)
    ]
    # The result repeats its complete locus in the tensor and profile fields.
    # A whole admitted fraction bounds its monic numerator/denominator guard.
    locus = [sizes[value] for value in guard_values] + [
        sizes[value] for value in output_denominators
    ]
    allocations = source + [sizes[value] for value in output] + 2 * locus
    terms, bits, coordinates = (sum(row[i] for row in allocations) for i in range(3))
    coordinates += (n + m) * (
        len(output) + len(metric.tensor.components) + len(map_value.components) + 6
    )
    if terms > 65536 or bits > 8388608 or coordinates > 1_048_576:
        reject(
            "allocation", "complete pullback output exceeds its exact allocation budget"
        )
    return Plan(
        dag,
        metric,
        map_value,
        substitutions,
        jacobian,
        output,
        determinant,
        guards,
        fractions,
    )


__all__ = ["Plan", "build_plan"]
