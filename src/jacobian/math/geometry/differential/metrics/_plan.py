"""Complete metric, inverse, connection and curvature DAG admission."""

from dataclasses import dataclass
from fractions import Fraction
from itertools import permutations, product

from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.geometry.differential.metrics._dag import (
    ONE,
    ZERO,
    Dag,
    Expression,
    reject,
)
from jacobian.math.geometry.differential.metrics._models import RationalCoordinateMetric
from jacobian.math.polynomials.values import SparseRationalPolynomial


def singular() -> OperationDomainValidationError:
    return OperationDomainValidationError(
        location=("metric",),
        code="differential_geometry.curvature.singular_metric",
        message="metric determinant is identically zero",
    )


def _has_nonconstant_denominator(dag: Dag, value: Expression) -> bool:
    """Return whether an expression carries a genuine polynomial denominator.

    Guard admission works on complete expressions, since distinct numerators
    can canonicalize to distinct denominator factors even when the DAG shares
    one denominator node.
    """

    return any(
        any(degree for degree in dag.nodes[index].bound.degrees)
        for index in value.denominator
    )


@dataclass(frozen=True)
class ConnectionPlan:
    """Admitted metric inverse and Levi-Civita connection DAG."""

    dag: Dag
    entries: tuple[Expression, ...]
    determinant: Expression
    inverse: tuple[Expression, ...]
    connection: tuple[Expression, ...]


@dataclass(frozen=True)
class Plan:
    dag: Dag
    fractions: dict[Expression, tuple[int, int]]
    determinant: Expression
    inverse: tuple[Expression, ...]
    connection: tuple[Expression, ...]
    riemann: tuple[Expression, ...]
    ricci: tuple[Expression, ...]
    scalar: Expression


def _determinant(
    dag: Dag,
    entries: tuple[Expression, ...],
    dimension: int,
    rows: tuple[int, ...],
    columns: tuple[int, ...],
) -> Expression:
    terms = []
    for permutation in permutations(columns):
        sign = (-1) ** sum(
            permutation[i] > permutation[j]
            for i in range(len(permutation))
            for j in range(i + 1, len(permutation))
        )
        terms.append(
            dag.multiply(
                Expression(Fraction(sign)),
                *(
                    entries[i * dimension + j]
                    for i, j in zip(rows, permutation, strict=True)
                ),
            )
        )
    return dag.add(*terms) if terms else ONE


def build_connection_plan(metric: RationalCoordinateMetric) -> ConnectionPlan:
    n = len(metric.tensor.coordinate_axis)
    dag = Dag(n)
    entries = tuple(dag.fraction(value) for value in metric.tensor.components)
    axes = tuple(range(n))
    det = _determinant(dag, entries, n, axes, axes)
    if not det.scalar:
        raise singular()
    inverse = tuple(
        dag.multiply(
            Expression(Fraction((-1) ** (i + j))),
            _determinant(
                dag,
                entries,
                n,
                tuple(k for k in axes if k != j),
                tuple(k for k in axes if k != i),
            ),
            dag.inverse(det),
        )
        for i, j in product(axes, repeat=2)
    )
    derivatives = {
        (index, axis): dag.derivative(value, axis)
        for index, value in enumerate(entries)
        for axis in axes
    }
    connection_list = [ZERO] * n**3
    for k in axes:
        for i in axes:
            for j in range(i, n):
                value = dag.multiply(
                    Expression(Fraction(1, 2)),
                    dag.add(
                        *(
                            dag.multiply(
                                inverse[k * n + upper],
                                dag.add(
                                    derivatives[upper * n + j, i],
                                    derivatives[upper * n + i, j],
                                    dag.multiply(
                                        Expression(Fraction(-1)),
                                        derivatives[i * n + j, upper],
                                    ),
                                ),
                            )
                            for upper in axes
                        )
                    ),
                )
                connection_list[(k * n + i) * n + j] = value
                connection_list[(k * n + j) * n + i] = value
    return ConnectionPlan(dag, entries, det, inverse, tuple(connection_list))


def build_plan(metric: RationalCoordinateMetric) -> Plan:
    connection_plan = build_connection_plan(metric)
    n = len(metric.tensor.coordinate_axis)
    dag = connection_plan.dag
    det = connection_plan.determinant
    inverse = connection_plan.inverse
    connection = connection_plan.connection
    axes = tuple(range(n))

    def gamma(k: int, i: int, j: int) -> Expression:
        return connection[(k * n + i) * n + j]

    riemann_list = [ZERO] * n**4
    for upper, k, i in product(axes, repeat=3):
        for j in range(i + 1, n):
            value = dag.add(
                dag.derivative(gamma(upper, j, k), i),
                dag.multiply(
                    Expression(Fraction(-1)), dag.derivative(gamma(upper, i, k), j)
                ),
                *(dag.multiply(gamma(upper, i, m), gamma(m, j, k)) for m in axes),
                *(
                    dag.multiply(
                        Expression(Fraction(-1)), gamma(upper, j, m), gamma(m, i, k)
                    )
                    for m in axes
                ),
            )
            riemann_list[((upper * n + k) * n + i) * n + j] = value
            riemann_list[((upper * n + k) * n + j) * n + i] = dag.multiply(
                Expression(Fraction(-1)), value
            )
    riemann = tuple(riemann_list)
    ricci = tuple(
        dag.add(*(riemann[((i * n + k) * n + i) * n + j] for i in axes))
        for k, j in product(axes, repeat=2)
    )
    scalar = dag.add(*(dag.multiply(a, b) for a, b in zip(inverse, ricci, strict=True)))
    # Retain the determinant numerator as its already-shared factors. Their
    # conjunction equals det(g)!=0 on the source denominator locus. This
    # avoids expanding an unnecessary determinant product in diagonal charts.
    determinant_allocations: list[tuple[int, int, int]] = []
    for index in set(det.numerator):
        bound = dag.nodes[index].bound
        if (
            bound.terms > 256
            or max(bound.degrees) > 64
            or bound.coefficient_digits > 128
        ):
            reject(
                "determinant_locus",
                "determinant locus factors exceed canonical polynomial bounds",
            )
        dag.ledger.charge("normalization", bound.terms * bound.coefficient_digits)
        determinant_allocations.append(
            (
                bound.terms,
                8 * bound.coefficient_digits * bound.terms,
                n * (bound.terms + 1),
            )
        )
    outputs = (*inverse, *connection, *riemann, *ricci, scalar)
    sizes = {value: dag.admit_output(value) for value in dict.fromkeys(outputs)}
    potential_guard_expressions = {
        value for value in outputs if _has_nonconstant_denominator(dag, value)
    }
    potential_guards = (
        len(metric.tensor.retained_nonzero_denominators)
        + len(set(det.numerator))
        + len(potential_guard_expressions)
    )
    if potential_guards > 768:
        reject("locus", "complete retained curvature locus exceeds 768 guards")

    # Count mathematical storage, including every occurrence of the common
    # locus in the five output coordinate objects. A whole fraction bounds its
    # denominator guard even when cancellation creates a different denominator.
    def source_allocation(polynomial: SparseRationalPolynomial) -> tuple[int, int, int]:
        return (
            len(polynomial.terms),
            sum(
                abs(term.coefficient.num).bit_length()
                + term.coefficient.den.bit_length()
                for term in polynomial.terms
            ),
            n * (len(polynomial.terms) + 1),
        )

    inherited = [
        source_allocation(guard)
        for guard in metric.tensor.retained_nonzero_denominators
    ]
    source = [
        source_allocation(polynomial)
        for component in metric.tensor.components
        for polynomial in (component.numerator, component.denominator)
    ]
    guards = (
        inherited
        + determinant_allocations
        + [sizes[value] for value in sizes if value.denominator]
    )
    allocations = source + inherited + [sizes[value] for value in outputs] + 5 * guards
    terms, coefficient_bits, coordinate_slots = (
        sum(allocation[index] for allocation in allocations) for index in range(3)
    )
    # Include tensor/connection component axes and their variance/index slots.
    coordinate_slots += n * (len(outputs) + 6) + 4 * len(outputs)
    if (
        terms > 262_144
        or coefficient_bits > 268_435_456
        or coordinate_slots > 1_048_576
    ):
        reject(
            "output",
            "complete curvature values exceed polynomial term, coefficient-bit, "
            "or coordinate allocation bounds",
        )
    fractions = {
        value: (
            dag.polynomial(value.numerator, value.scalar),
            dag.polynomial(value.denominator),
        )
        for value in sizes
    }
    return Plan(dag, fractions, det, inverse, connection, riemann, ricci, scalar)
