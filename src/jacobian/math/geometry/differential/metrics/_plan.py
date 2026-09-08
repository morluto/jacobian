"""Complete metric, inverse, connection and curvature DAG admission."""

from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, replace
from fractions import Fraction
from itertools import permutations, product
from typing import NoReturn

from jacobian.canonical import format_canonical_integer
from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.geometry.differential.metrics._dag import (
    ONE,
    ZERO,
    Dag,
    Expression,
    reject,
)
from jacobian.math.geometry.differential.metrics._models import RationalCoordinateMetric
from jacobian.math.geometry.differential.values import _polynomial_key
from jacobian.math.polynomials.rational_functions._bounds import (
    _remove_guaranteed_common_monomial,
)
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


def _node_guard_key(dag: Dag, index: int) -> object:
    """Identify one DAG polynomial node for locus-cap accounting.

    Generated nodes such as ``xy-1`` keep one identity whether they appear as
    a determinant factor or as an inverse denominator.
    """

    source = dag.nodes[index].source
    if source is not None:
        return _polynomial_key(source)
    return ("dag-node", index)


def _denominator_guard_identity(dag: Dag, value: Expression) -> object | None:
    """Identify one retained output denominator after guaranteed monomial cancel.

    A single sourced or generated factor matches the determinant or inherited
    key only when guaranteed cancellation leaves the same degree and valuation
    envelope. A proper factor remaining after cancel, such as ``x-y`` from
    det ``y(x-y)``, is counted separately. Shared raw ``D^2`` identities still
    split when axis-specific numerators can cancel non-monomial factors.
    Constant remaining numerators keep one identity so cheap cases such as
    ``1/(x+y)`` stay inside the cap.
    """

    if not _has_nonconstant_denominator(dag, value):
        return None
    cancelled = _remove_guaranteed_common_monomial(dag.bound(value))
    remaining = cancelled.denominator
    if not any(remaining.degrees):
        return None
    numerators = Counter(value.numerator)
    denominators = Counter(value.denominator)
    factors: list[tuple[int, int]] = []
    remaining_numerator: list[tuple[int, int]] = []
    cancelled_shared = False
    for index, multiplicity in sorted(denominators.items()):
        if not any(dag.nodes[index].bound.degrees):
            continue
        shared = min(multiplicity, numerators.get(index, 0))
        leftover = multiplicity - shared
        if shared:
            cancelled_shared = True
        if leftover:
            factors.append((index, leftover))
    for index, multiplicity in sorted(numerators.items()):
        if not any(dag.nodes[index].bound.degrees):
            continue
        leftover = multiplicity - min(multiplicity, denominators.get(index, 0))
        if leftover:
            remaining_numerator.append((index, leftover))
    if not factors:
        return None
    remaining_numerator_identity = tuple(remaining_numerator)
    if len(factors) == 1 and factors[0][1] == 1:
        index = factors[0][0]
        raw = dag.nodes[index].bound
        if (
            remaining.degrees == raw.degrees
            and remaining.minimum_exponents == raw.minimum_exponents
            and not cancelled_shared
        ):
            return _node_guard_key(dag, index)
        return (
            "canonical-result-denominator",
            (_node_guard_key(dag, index), 1),
            remaining.minimum_exponents,
            remaining.degrees,
            remaining_numerator_identity,
        )
    identity: tuple[object, ...] = (
        "canonical-result-denominator",
        tuple(
            (_node_guard_key(dag, index), multiplicity)
            for index, multiplicity in factors
        ),
        remaining.minimum_exponents,
        remaining.degrees,
    )
    if remaining_numerator_identity:
        identity = (*identity, remaining_numerator_identity)
    return identity


def potential_locus_guard_keys(
    dag: Dag,
    inherited: tuple[SparseRationalPolynomial, ...],
    determinant: Expression,
    outputs: tuple[Expression, ...],
) -> set[object]:
    """Return the polynomial identities counted against the 768-guard cap."""

    keys: set[object] = {_polynomial_key(guard) for guard in inherited}
    for index in set(determinant.numerator):
        keys.add(_node_guard_key(dag, index))
    for value in outputs:
        identity = _denominator_guard_identity(dag, value)
        if identity is not None:
            keys.add(identity)
    return keys


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
        # The permutation sign is relative to the ordered column subset.
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


def build_connection_plan(
    metric: RationalCoordinateMetric,
    *,
    reject: Callable[[str, str], NoReturn] | None = None,
    admission_reject: Callable[[str, str], NoReturn] | None = None,
    label: str | None = None,
    singular_metric: Callable[[], OperationDomainValidationError] | None = None,
) -> ConnectionPlan:
    n = len(metric.tensor.coordinate_axis)
    dag = Dag(n)
    owner_reject = admission_reject or reject
    if owner_reject is not None:
        dag.ledger.limits = replace(
            dag.ledger.limits,
            reject=owner_reject,
            label=label or dag.ledger.limits.label,
        )
    entries = tuple(dag.fraction(value) for value in metric.tensor.components)
    axes = tuple(range(n))
    det = _determinant(dag, entries, n, axes, axes)
    if not det.scalar:
        raise (singular_metric or singular)()
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
    inherited_keys = {
        _polynomial_key(guard) for guard in metric.tensor.retained_nonzero_denominators
    }
    unique_determinant: dict[object, tuple[int, int, int]] = {}
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
        key = _node_guard_key(dag, index)
        if key in inherited_keys:
            continue
        unique_determinant.setdefault(
            key,
            (
                bound.terms,
                8 * bound.coefficient_digits * bound.terms,
                n * (bound.terms + 1),
            ),
        )
    outputs = (*inverse, *connection, *riemann, *ricci, scalar)
    sizes = {value: dag.admit_output(value) for value in dict.fromkeys(outputs)}
    potential_guards = len(_potential_locus_keys(dag, metric, det, outputs))
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
        + list(unique_determinant.values())
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


def _remaining_denominator_factors(
    dag: Dag, value: Expression
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    numerators = Counter(value.numerator)
    remaining: list[tuple[int, int]] = []
    remaining_numerator: list[tuple[int, int]] = []
    for index, multiplicity in sorted(Counter(value.denominator).items()):
        if not any(dag.nodes[index].bound.degrees):
            continue
        leftover = multiplicity - min(multiplicity, numerators.get(index, 0))
        if leftover:
            remaining.append((index, leftover))
    for index, multiplicity in sorted(numerators.items()):
        if not any(dag.nodes[index].bound.degrees):
            continue
        leftover = multiplicity - min(
            multiplicity, Counter(value.denominator).get(index, 0)
        )
        if leftover:
            remaining_numerator.append((index, leftover))
    return remaining, remaining_numerator


def _powered_monomial_key(
    source: SparseRationalPolynomial, multiplicity: int
) -> object | None:
    if len(source.terms) != 1 or multiplicity < 1:
        return None
    exponents = tuple(degree * multiplicity for degree in source.terms[0].exponents)
    coefficient = source.terms[0].coefficient.as_fraction() ** multiplicity
    return (
        (
            exponents,
            format_canonical_integer(coefficient.numerator),
            format_canonical_integer(coefficient.denominator),
        ),
    )


def _complete_result_denominator_keys(dag: Dag, value: Expression) -> set[object]:
    """Identities that may already appear in an inherited locus family."""

    identity = _denominator_guard_identity(dag, value)
    if identity is None:
        return set()
    keys: set[object] = {identity}
    factors, _remaining_numerator = _remaining_denominator_factors(dag, value)
    if len(factors) != 1:
        return keys
    index, multiplicity = factors[0]
    source = dag.nodes[index].source
    if source is None:
        return keys
    for power in (multiplicity, multiplicity - 1):
        powered = _powered_monomial_key(source, power)
        if powered is not None:
            keys.add(powered)
    return keys


def _monic_polynomial_key(
    polynomial: SparseRationalPolynomial,
) -> tuple[tuple[tuple[int, ...], str, str], ...]:
    if not polynomial.terms:
        return _polynomial_key(polynomial)
    leading = polynomial.terms[0].coefficient.as_fraction()
    return tuple(
        (
            term.exponents,
            format_canonical_integer(
                (term.coefficient.as_fraction() / leading).numerator
            ),
            format_canonical_integer(
                (term.coefficient.as_fraction() / leading).denominator
            ),
        )
        for term in polynomial.terms
    )


def _source_guard_keys(
    source: SparseRationalPolynomial, multiplicity: int = 1
) -> set[object]:
    monic = _monic_polynomial_key(source)
    keys: set[object] = {monic if multiplicity == 1 else (monic, multiplicity)}
    if len(source.terms) != 1:
        return keys
    exponents = source.terms[0].exponents
    for axis_index, degree in enumerate(exponents):
        if degree:
            axis_exponents = tuple(int(i == axis_index) for i in range(len(exponents)))
            keys.add(((axis_exponents, "1", "1"),))
    return keys


def _potential_locus_keys(
    dag: Dag,
    metric: RationalCoordinateMetric,
    determinant: Expression,
    outputs: tuple[Expression, ...],
) -> set[object]:
    keys: set[object] = {
        _polynomial_key(guard) for guard in metric.tensor.retained_nonzero_denominators
    }
    for index in set(determinant.numerator):
        source = dag.nodes[index].source
        keys.add(
            _monic_polynomial_key(source)
            if source is not None
            else ("determinant", index)
        )
    for value in outputs:
        if not _has_nonconstant_denominator(dag, value):
            continue
        for index, multiplicity in Counter(value.denominator).items():
            source = dag.nodes[index].source
            if source is None:
                keys.add(("canonical-result-denominator", index, multiplicity))
            else:
                keys.update(_source_guard_keys(source, multiplicity))
    return keys

