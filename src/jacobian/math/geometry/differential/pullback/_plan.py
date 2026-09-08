"""Whole request admission for rational metric pullbacks."""
from dataclasses import dataclass, replace
from fractions import Fraction
from itertools import permutations

from jacobian.catalog.models import OperationResourceAdmissionError
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


def reject(reason: str, message: str) -> None:
    raise OperationResourceAdmissionError(location=("metric", "map"), code=f"differential_geometry.rational_metric.pullback.{reason}", message=message)
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

def _substitute(dag: Dag, value: SparseRationalPolynomial, inner: tuple[Expression, ...]) -> Expression:
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
        inversions = sum(permutation[i] > permutation[j] for i in range(n) for j in range(i + 1, n))
        terms.append(dag.multiply(Expression(Fraction((-1) ** inversions)), *(entries[i * n + j] for i, j in enumerate(permutation))))
    return dag.add(*terms)

def build_plan(metric: RationalCoordinateMetric, map_value: RationalFunctionMap) -> Plan:
    n, m = len(map_value.source_variables), len(metric.tensor.coordinate_axis)
    if not 1 <= n <= 4 or not 1 <= m <= 4:
        reject("shape", "source and target dimensions must be between 1 and 4")
    dag = Dag(n)
    # Metric components are authored on the target axis. Reserve their raw
    # parsing, backend conversion, and canonical recognition before any
    # substituted DAG node can be evaluated.
    for value in metric.tensor.components:
        for polynomial in (value.numerator, value.denominator):
            dag.ledger.charge("source_conversion", _polynomial_admission_work_units(polynomial, m))
            dag.ledger.charge("source_conversion", _polynomial_backend_conversion_work_units(_polynomial_bound(polynomial)))
        nb, db = _polynomial_bound(value.numerator), _polynomial_bound(value.denominator)
        width = max(len(nb.degrees), len(db.degrees))
        if len(nb.degrees) != width:
            nb = replace(nb, degrees=nb.degrees + (0,) * (width - len(nb.degrees)), minimum_exponents=nb.minimum_exponents + (0,) * (width - len(nb.minimum_exponents)))
        if len(db.degrees) != width:
            db = replace(db, degrees=db.degrees + (0,) * (width - len(db.degrees)), minimum_exponents=db.minimum_exponents + (0,) * (width - len(db.minimum_exponents)))
        dag.ledger.charge("recognition", _recognition_work_units(FractionBound(nb, db)))
    for guard in metric.tensor.retained_nonzero_denominators:
        dag.ledger.charge("source_conversion", _polynomial_admission_work_units(guard, m))
    maps = tuple(dag.fraction(value) for value in map_value.components)
    map_denominators = tuple(dag.source(value.denominator) for value in map_value.components)
    substitutions = tuple(_substitute(dag, value.numerator, maps) for value in metric.tensor.components)
    substitutions = tuple(dag.multiply(value, dag.inverse(_substitute(dag, metric.tensor.components[i].denominator, maps))) for i, value in enumerate(substitutions))
    jacobian = tuple(dag.derivative(value, axis) for value in maps for axis in range(n))
    determinant = _determinant(dag, substitutions, m)
    output = tuple(dag.add(*(dag.multiply(jacobian[i * n + a], substitutions[i * m + j], jacobian[j * n + b]) for i in range(m) for j in range(m))) for a in range(n) for b in range(n))
    inherited = tuple(_substitute(dag, guard, maps) for guard in metric.tensor.retained_nonzero_denominators)
    component_guards = tuple(_substitute(dag, value.denominator, maps) for value in metric.tensor.components)
    guards = tuple(map_denominators) + component_guards + inherited + (determinant,)
    values = tuple(dict.fromkeys((*output, *guards)))
    sizes = {value: dag.admit_output(value) for value in values}
    if len(guards) > 768:
        reject("locus", "complete pullback locus exceeds 768 guards")
    terms = sum(sizes[value][0] for value in sizes) + len(output) * n * (len(values) + 2)
    bits = sum(sizes[value][1] for value in sizes)
    if terms > 65536 or bits > 8388608:
        reject("allocation", "complete pullback output exceeds its exact allocation budget")
    return Plan(dag, metric, map_value, substitutions, jacobian, output, determinant, guards)
__all__ = ["Plan", "build_plan"]
