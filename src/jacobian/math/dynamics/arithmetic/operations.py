"""Exact bounded native arithmetic-dynamics kernels."""

from __future__ import annotations

from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass
from fractions import Fraction
from typing import Any, Literal

from jacobian._exact import CanonicalRational
from jacobian.math._rational_height import RationalHeight
from jacobian.math.dynamics.arithmetic._models import (
    MAX_COEFFICIENT_DIGITS,
    MAX_DEGREE,
    MAX_DYNATOMIC_DEGREE,
    MAX_FIELD_PRIME,
    MAX_ITERATE,
    MAX_ITERATE_DEGREE,
    MAX_ORBIT_STEPS,
    MAX_POLYNOMIAL_OUTPUT_DIGITS,
    CoefficientHeight,
    _add_heights,
    _divide_height_polynomials,
    _fraction_height,
    _iterate_heights,
    _multiply_height_polynomials,
    _require_polynomial_height,
)
from jacobian.math.polynomials._conversions import (
    rational_polynomial_from_sympy,
    rational_polynomial_to_sympy,
)
from jacobian.math.polynomials.values import (
    RationalPolynomial,
    RationalPolynomialTerm,
    SparseRationalPolynomial,
)


@dataclass(frozen=True, slots=True)
class RepeatEvidence:
    first_seen_index: int
    repeated_at_index: int
    preperiod: int
    period: int


@dataclass(frozen=True, slots=True)
class OrbitComputation:
    orbit: tuple[Fraction, ...]
    requested_steps: int
    termination: Literal["REPEAT_FOUND", "STEP_BOUND_REACHED", "OUTPUT_BOUND_REACHED"]
    repeat: RepeatEvidence | None


@dataclass(frozen=True, slots=True)
class FunctionalGraph:
    edges: tuple[tuple[int, int], ...]
    cycles: tuple[tuple[int, ...], ...]
    tail_lengths: tuple[int, ...]


def _canonical_polynomial(values: Sequence[Fraction]) -> RationalPolynomial:
    terms = tuple(
        RationalPolynomialTerm(
            coefficient=CanonicalRational.from_fraction(value), exponents=(index,)
        )
        for index, value in reversed(tuple(enumerate(values)))
        if value
    )
    return RationalPolynomial(
        variables=("x",), polynomial=SparseRationalPolynomial(terms=terms)
    )


def polynomial_from_coefficients(
    coefficients: Sequence[Fraction | int],
) -> RationalPolynomial:
    """Build a canonical univariate ``QQ`` polynomial from low-to-high values."""
    values = tuple(Fraction(value) for value in coefficients)
    if not 1 <= len(values) <= MAX_DEGREE + 1:
        raise ValueError(
            f"polynomial must have between 1 and {MAX_DEGREE + 1} coefficients"
        )
    if any(_fraction_digits(value) > MAX_COEFFICIENT_DIGITS for value in values):
        raise ValueError("polynomial coefficient exceeds the input digit bound")
    if len(values) > 1 and values[-1] == 0:
        raise ValueError("polynomial coefficients must omit trailing zeros")
    return _canonical_polynomial(values)


def polynomial_coefficients(polynomial: RationalPolynomial) -> tuple[Fraction, ...]:
    """Return low-to-high exact coefficients of a univariate ``QQ`` polynomial."""

    _require_polynomial(polynomial)
    exponents = {
        term.exponents[0]: term.coefficient.as_fraction()
        for term in polynomial.polynomial.terms
    }
    degree = max(exponents, default=0)
    if degree > MAX_ITERATE_DEGREE:
        raise ValueError("polynomial degree exceeds the extraction bound")
    values = tuple(exponents.get(index, Fraction(0)) for index in range(degree + 1))
    _require_bounded_output_coefficients(values)
    if not polynomial.polynomial.terms:
        return (Fraction(0),)
    return values


def iterate_polynomial(polynomial: RationalPolynomial, n: int) -> RationalPolynomial:
    """Return the exact n-th compositional iterate, with iterate zero the identity."""

    import sympy

    if not 0 <= n <= MAX_ITERATE:
        raise ValueError(f"iterate count must be between 0 and {MAX_ITERATE}")
    source_coefficients = polynomial_coefficients(polynomial)
    source_degree = 0 if len(source_coefficients) == 1 else len(source_coefficients) - 1
    output_degree = 1 if n == 0 else source_degree**n
    if output_degree > MAX_ITERATE_DEGREE:
        raise ValueError("iterate output degree exceeds bound")
    _require_polynomial_height(
        _iterate_heights(
            tuple(_fraction_height(value) for value in source_coefficients), n
        ),
        "iterate",
    )
    source = _require_input_polynomial(polynomial)
    result = sympy.Poly(source.gens[0], source.gens[0], domain=sympy.QQ)
    for _ in range(n):
        result = source.compose(result)
    return _from_sympy(result, MAX_ITERATE_DEGREE + 1)


def fixed_point_equation(polynomial: RationalPolynomial, n: int) -> RationalPolynomial:
    """Return ``f^n(x) - x`` as a native polynomial projection."""

    import sympy

    source = _require_input_polynomial(polynomial)
    if n < 1:
        raise ValueError("fixed-point iterate must be positive")
    identity = sympy.Poly(source.gens[0], source.gens[0], domain=sympy.QQ)
    result = _to_sympy(iterate_polynomial(polynomial, n)) - identity
    _require_bounded_output_coefficients(result)
    return _from_sympy(result, MAX_ITERATE_DEGREE + 1)


def dynatomic_polynomial(polynomial: RationalPolynomial, n: int) -> RationalPolynomial:
    """Return the exact n-th Möbius-normalized formal-period polynomial."""

    import sympy

    if n < 1:
        raise ValueError("dynatomic index must be positive")
    source_coefficients = polynomial_coefficients(polynomial)
    source_degree = 0 if len(source_coefficients) == 1 else len(source_coefficients) - 1
    if source_degree < 2:
        raise ValueError("dynatomic polynomial requires map degree at least two")
    if source_degree**n > MAX_DYNATOMIC_DEGREE:
        raise ValueError("dynatomic output degree exceeds bound")
    source_heights = tuple(_fraction_height(value) for value in source_coefficients)
    numerator_heights: tuple[CoefficientHeight, ...] = (RationalHeight(1, 1),)
    denominator_heights: tuple[CoefficientHeight, ...] = (RationalHeight(1, 1),)
    for divisor in sympy.divisors(n):
        term_heights = list(_iterate_heights(source_heights, int(divisor)))
        if len(term_heights) < 2:
            term_heights.extend([None] * (2 - len(term_heights)))
        term_heights[1] = _add_heights(term_heights[1], RationalHeight(1, 1))
        if (mobius := int(sympy.mobius(n // divisor))) == 1:
            numerator_heights = _multiply_height_polynomials(
                numerator_heights, tuple(term_heights)
            )
        elif mobius == -1:
            denominator_heights = _multiply_height_polynomials(
                denominator_heights, tuple(term_heights)
            )
    _require_polynomial_height(
        _divide_height_polynomials(numerator_heights, denominator_heights),
        "dynatomic quotient",
    )
    source = _require_input_polynomial(polynomial)
    numerator = sympy.Poly(1, source.gens[0], domain=sympy.QQ)
    denominator = sympy.Poly(1, source.gens[0], domain=sympy.QQ)
    for divisor in sympy.divisors(n):
        term = _to_sympy(fixed_point_equation(polynomial, int(divisor)))
        mobius = int(sympy.mobius(n // divisor))
        if mobius == 1:
            numerator *= term
        elif mobius == -1:
            denominator *= term
    quotient, remainder = numerator.div(denominator)
    if not remainder.is_zero:
        raise RuntimeError("dynatomic quotient was not an exact polynomial")
    _require_bounded_output_coefficients(quotient)
    return _from_sympy(quotient, MAX_DYNATOMIC_DEGREE + 1)


def orbit_prefix(
    polynomial: RationalPolynomial,
    start: CanonicalRational,
    max_steps: int,
    *,
    max_value_digits: int = 2_048,
) -> OrbitComputation:
    """Iterate until a first repeat or an explicit step/output bound."""

    source = _require_input_polynomial(polynomial)
    if not 0 <= max_steps <= MAX_ORBIT_STEPS:
        raise ValueError(f"orbit step bound must be between 0 and {MAX_ORBIT_STEPS}")
    if max_value_digits < 1:
        raise ValueError("orbit value digit bound must be positive")
    initial = start.as_fraction()
    if _fraction_digits(initial) > MAX_COEFFICIENT_DIGITS:
        raise ValueError("orbit start exceeds the input digit bound")
    values = [initial]
    seen = {values[0]: 0}
    for step in range(1, max_steps + 1):
        next_value = Fraction(source.eval(values[-1]))
        if _fraction_digits(next_value) > max_value_digits:
            return OrbitComputation(
                orbit=tuple(values),
                requested_steps=max_steps,
                termination="OUTPUT_BOUND_REACHED",
                repeat=None,
            )
        values.append(next_value)
        if next_value in seen:
            first_seen = seen[next_value]
            return OrbitComputation(
                orbit=tuple(values),
                requested_steps=max_steps,
                termination="REPEAT_FOUND",
                repeat=RepeatEvidence(
                    first_seen_index=first_seen,
                    repeated_at_index=step,
                    preperiod=first_seen,
                    period=step - first_seen,
                ),
            )
        seen[next_value] = step
    return OrbitComputation(
        orbit=tuple(values),
        requested_steps=max_steps,
        termination="STEP_BOUND_REACHED",
        repeat=None,
    )


def validate_cycle(
    polynomial: RationalPolynomial, cycle: Sequence[CanonicalRational]
) -> None:
    """Reject a sequence that is not one exact ordered periodic cycle."""

    source = _require_input_polynomial(polynomial)
    points = tuple(point.as_fraction() for point in cycle)
    if not 1 <= len(points) <= MAX_ORBIT_STEPS:
        raise ValueError(f"cycle must contain between 1 and {MAX_ORBIT_STEPS} points")
    if any(_fraction_digits(point) > MAX_COEFFICIENT_DIGITS for point in points):
        raise ValueError("cycle point exceeds the input digit bound")
    if len(set(points)) != len(points):
        raise ValueError("cycle must contain distinct points")
    for index, point in enumerate(points):
        if Fraction(source.eval(point)) != points[(index + 1) % len(points)]:
            raise ValueError("cycle points do not follow the polynomial map")


def cycle_multiplier(
    polynomial: RationalPolynomial, cycle: Sequence[CanonicalRational]
) -> CanonicalRational:
    """Return the exact derivative product around a validated periodic cycle."""

    source = _require_input_polynomial(polynomial)
    points = tuple(point.as_fraction() for point in cycle)
    validate_cycle(polynomial, cycle)
    derivative = source.diff()
    multiplier = Fraction(1)
    for point in points:
        multiplier *= Fraction(derivative.eval(point))
        if _fraction_digits(multiplier) > MAX_POLYNOMIAL_OUTPUT_DIGITS:
            raise ValueError("cycle multiplier exceeds the output digit bound")
    return CanonicalRational.from_fraction(multiplier)


def finite_field_functional_graph(
    coefficients: Sequence[int],
    prime: int,
) -> FunctionalGraph:
    """Enumerate the complete functional graph of a polynomial over ``GF(p)``."""

    import sympy

    if not 2 <= prime <= MAX_FIELD_PRIME or not sympy.isprime(prime):
        raise ValueError(
            f"prime must be a prime number between 2 and {MAX_FIELD_PRIME}"
        )
    values = tuple(int(value) for value in coefficients)
    if not 1 <= len(values) <= MAX_DEGREE + 1:
        raise ValueError(
            f"polynomial must have between 1 and {MAX_DEGREE + 1} coefficients"
        )
    if any(len(str(abs(value))) > MAX_COEFFICIENT_DIGITS for value in values):
        raise ValueError("polynomial coefficient exceeds the input digit bound")
    normalized = tuple(value % prime for value in values)
    if len(normalized) > 1 and normalized[-1] == 0:
        raise ValueError("polynomial coefficients must omit trailing zeros modulo p")

    def evaluate(point: int) -> int:
        value = 0
        for coefficient in reversed(normalized):
            value = (value * point + coefficient) % prime
        return value

    targets = tuple(evaluate(source) for source in range(prime))
    cycles = _functional_graph_cycles(targets)
    tail_lengths = _tail_lengths(targets, cycles)
    return FunctionalGraph(
        edges=tuple(enumerate(targets)),
        cycles=cycles,
        tail_lengths=tail_lengths,
    )


def _functional_graph_cycles(targets: tuple[int, ...]) -> tuple[tuple[int, ...], ...]:
    processed: set[int] = set()
    cycles: list[tuple[int, ...]] = []
    for start in range(len(targets)):
        if start in processed:
            continue
        path: list[int] = []
        path_positions: dict[int, int] = {}
        current = start
        while current not in processed and current not in path_positions:
            path_positions[current] = len(path)
            path.append(current)
            current = targets[current]
        if current in path_positions:
            cycle = path[path_positions[current] :]
            least_index = cycle.index(min(cycle))
            cycles.append(tuple(cycle[least_index:] + cycle[:least_index]))
        processed.update(path)
    return tuple(sorted(cycles))


def _tail_lengths(
    targets: tuple[int, ...],
    cycles: tuple[tuple[int, ...], ...],
) -> tuple[int, ...]:
    reverse_edges: list[list[int]] = [[] for _ in targets]
    for source, target in enumerate(targets):
        reverse_edges[target].append(source)
    distances = [-1] * len(targets)
    queue: deque[int] = deque()
    for node in (node for cycle in cycles for node in cycle):
        distances[node] = 0
        queue.append(node)
    while queue:
        target = queue.popleft()
        for source in reverse_edges[target]:
            if distances[source] < 0:
                distances[source] = distances[target] + 1
                queue.append(source)
    if any(distance < 0 for distance in distances):
        raise RuntimeError("functional graph traversal did not reach every vertex")
    return tuple(distances)


def _require_polynomial(polynomial: RationalPolynomial) -> RationalPolynomial:
    if polynomial.domain != "QQ" or polynomial.variables != ("x",):
        raise ValueError("polynomial must be univariate over QQ in variable x")
    return polynomial


def _to_sympy(polynomial: RationalPolynomial) -> Any:
    return rational_polynomial_to_sympy(_require_polynomial(polynomial))


def _from_sympy(polynomial: Any, maximum_terms: int) -> RationalPolynomial:
    return rational_polynomial_from_sympy(
        polynomial, ("x",), maximum_terms=maximum_terms
    )


def _require_input_polynomial(polynomial: RationalPolynomial) -> Any:
    _require_polynomial(polynomial)
    source = _to_sympy(polynomial)
    if not source.is_zero and source.degree() > MAX_DEGREE:
        raise ValueError("polynomial degree exceeds the input bound")
    if any(
        _fraction_digits(Fraction(coefficient)) > MAX_COEFFICIENT_DIGITS
        for coefficient in source.all_coeffs()
    ):
        raise ValueError("polynomial coefficient exceeds the input digit bound")
    return source


def _fraction_digits(value: Fraction) -> int:
    return max(len(str(abs(value.numerator))), len(str(value.denominator)))


def _require_bounded_output_coefficients(polynomial: Any) -> None:
    if isinstance(polynomial, tuple):
        coefficients = polynomial
    else:
        coefficients = tuple(
            Fraction(coefficient) for coefficient in polynomial.all_coeffs()
        )
    if any(
        _fraction_digits(Fraction(coefficient)) > MAX_POLYNOMIAL_OUTPUT_DIGITS
        for coefficient in coefficients
    ):
        raise ValueError("polynomial coefficient exceeds the output digit bound")


__all__ = [
    "FunctionalGraph",
    "OrbitComputation",
    "RepeatEvidence",
    "cycle_multiplier",
    "dynatomic_polynomial",
    "finite_field_functional_graph",
    "fixed_point_equation",
    "iterate_polynomial",
    "orbit_prefix",
    "polynomial_coefficients",
    "polynomial_from_coefficients",
    "validate_cycle",
]
