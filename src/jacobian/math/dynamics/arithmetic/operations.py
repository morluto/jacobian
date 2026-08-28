"""Exact bounded native arithmetic-dynamics kernels."""

from __future__ import annotations

from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass
from fractions import Fraction
from typing import Any, Literal

from jacobian.math.dynamics.arithmetic._models import MAX_FIELD_PRIME

_MAX_INPUT_DEGREE = 30
_MAX_INPUT_DIGITS = 128
_MAX_ITERATE_DEGREE = 1_024
_MAX_DYNATOMIC_DEGREE = 512
_MAX_ORBIT_STEPS = 1_000
_MAX_OUTPUT_DIGITS = 32_768


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


def polynomial_from_coefficients(coefficients: Sequence[Fraction | int]) -> Any:
    """Build a canonical univariate ``QQ`` polynomial from low-to-high values."""

    import sympy

    values = tuple(Fraction(value) for value in coefficients)
    if not 1 <= len(values) <= _MAX_INPUT_DEGREE + 1:
        raise ValueError("polynomial must have between 1 and 31 coefficients")
    if any(_fraction_digits(value) > _MAX_INPUT_DIGITS for value in values):
        raise ValueError("polynomial coefficient exceeds the input digit bound")
    if len(values) > 1 and values[-1] == 0:
        raise ValueError("polynomial coefficients must omit trailing zeros")
    x = sympy.Symbol("x")
    expression = sum(
        sympy.Rational(value.numerator, value.denominator) * x**i
        for i, value in enumerate(values)
    )
    return sympy.Poly(expression, x, domain=sympy.QQ)


def polynomial_coefficients(polynomial: Any) -> tuple[Fraction, ...]:
    """Return low-to-high exact coefficients of a univariate ``QQ`` polynomial."""

    source = _require_polynomial(polynomial)
    if not source.is_zero and source.degree() > _MAX_ITERATE_DEGREE:
        raise ValueError("polynomial degree exceeds the extraction bound")
    _require_bounded_output_coefficients(source)
    if source.is_zero:
        return (Fraction(0),)
    return tuple(Fraction(source.nth(index)) for index in range(source.degree() + 1))


def iterate_polynomial(polynomial: Any, n: int) -> Any:
    """Return the exact n-th compositional iterate, with iterate zero the identity."""

    import sympy

    source = _require_input_polynomial(polynomial)
    if not 0 <= n <= 20:
        raise ValueError("iterate count must be between 0 and 20")
    source_degree = 0 if source.is_zero else max(0, int(source.degree()))
    output_degree = 1 if n == 0 else source_degree**n
    if output_degree > _MAX_ITERATE_DEGREE:
        raise ValueError("iterate output degree exceeds bound")
    result = sympy.Poly(source.gens[0], source.gens[0], domain=sympy.QQ)
    for _ in range(n):
        result = source.compose(result)
        _require_bounded_output_coefficients(result)
    return result


def fixed_point_equation(polynomial: Any, n: int) -> Any:
    """Return ``f^n(x) - x`` as a native polynomial projection."""

    import sympy

    source = _require_input_polynomial(polynomial)
    if n < 1:
        raise ValueError("fixed-point iterate must be positive")
    identity = sympy.Poly(source.gens[0], source.gens[0], domain=sympy.QQ)
    return iterate_polynomial(source, n) - identity


def dynatomic_polynomial(polynomial: Any, n: int) -> Any:
    """Return the exact n-th Möbius-normalized formal-period polynomial."""

    import sympy

    source = _require_input_polynomial(polynomial)
    if source.degree() < 2:
        raise ValueError("dynatomic polynomial requires degree at least two")
    if n < 1:
        raise ValueError("dynatomic index must be positive")
    if int(source.degree()) ** n > _MAX_DYNATOMIC_DEGREE:
        raise ValueError("dynatomic output degree exceeds bound")
    numerator = sympy.Poly(1, source.gens[0], domain=sympy.QQ)
    denominator = sympy.Poly(1, source.gens[0], domain=sympy.QQ)
    for divisor in sympy.divisors(n):
        term = fixed_point_equation(source, int(divisor))
        mobius = int(sympy.mobius(n // divisor))
        if mobius == 1:
            numerator *= term
            _require_bounded_output_coefficients(numerator)
        elif mobius == -1:
            denominator *= term
            _require_bounded_output_coefficients(denominator)
    quotient, remainder = numerator.div(denominator)
    if not remainder.is_zero:
        raise RuntimeError("dynatomic quotient was not an exact polynomial")
    _require_bounded_output_coefficients(quotient)
    return quotient


def orbit_prefix(
    polynomial: Any,
    start: Fraction,
    max_steps: int,
    *,
    max_value_digits: int = 2_048,
) -> OrbitComputation:
    """Iterate until a first repeat or an explicit step/output bound."""

    source = _require_input_polynomial(polynomial)
    if not 0 <= max_steps <= _MAX_ORBIT_STEPS:
        raise ValueError("orbit step bound must be between 0 and 1000")
    if max_value_digits < 1:
        raise ValueError("orbit value digit bound must be positive")
    initial = Fraction(start)
    if _fraction_digits(initial) > _MAX_INPUT_DIGITS:
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


def validate_cycle(polynomial: Any, cycle: Sequence[Fraction]) -> None:
    """Reject a sequence that is not one exact ordered periodic cycle."""

    source = _require_input_polynomial(polynomial)
    points = tuple(Fraction(point) for point in cycle)
    if not 1 <= len(points) <= _MAX_ORBIT_STEPS:
        raise ValueError("cycle must contain between 1 and 1000 points")
    if any(_fraction_digits(point) > _MAX_INPUT_DIGITS for point in points):
        raise ValueError("cycle point exceeds the input digit bound")
    if len(set(points)) != len(points):
        raise ValueError("cycle must contain distinct points")
    for index, point in enumerate(points):
        if Fraction(source.eval(point)) != points[(index + 1) % len(points)]:
            raise ValueError("cycle points do not follow the polynomial map")


def cycle_multiplier(polynomial: Any, cycle: Sequence[Fraction]) -> Fraction:
    """Return the exact derivative product around a validated periodic cycle."""

    source = _require_input_polynomial(polynomial)
    points = tuple(Fraction(point) for point in cycle)
    validate_cycle(source, points)
    derivative = source.diff()
    multiplier = Fraction(1)
    for point in points:
        multiplier *= Fraction(derivative.eval(point))
        if _fraction_digits(multiplier) > _MAX_OUTPUT_DIGITS:
            raise ValueError("cycle multiplier exceeds the output digit bound")
    return multiplier


def finite_field_functional_graph(
    coefficients: Sequence[int],
    prime: int,
) -> FunctionalGraph:
    """Enumerate the complete functional graph of a polynomial over ``GF(p)``."""

    import sympy

    if not 2 <= prime <= MAX_FIELD_PRIME or not sympy.isprime(prime):
        raise ValueError("prime must be a prime number between 2 and 10000")
    values = tuple(int(value) for value in coefficients)
    if not 1 <= len(values) <= _MAX_INPUT_DEGREE + 1:
        raise ValueError("polynomial must have between 1 and 31 coefficients")
    if any(len(str(abs(value))) > _MAX_INPUT_DIGITS for value in values):
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


def _require_polynomial(polynomial: Any) -> Any:
    import sympy

    if not isinstance(polynomial, sympy.Poly):
        raise TypeError("polynomial must be a SymPy Poly")
    if len(polynomial.gens) != 1 or not polynomial.domain.is_QQ:
        raise ValueError("polynomial must be univariate over QQ")
    return polynomial


def _require_input_polynomial(polynomial: Any) -> Any:
    source = _require_polynomial(polynomial)
    if not source.is_zero and source.degree() > _MAX_INPUT_DEGREE:
        raise ValueError("polynomial degree exceeds the input bound")
    if any(
        _fraction_digits(Fraction(coefficient)) > _MAX_INPUT_DIGITS
        for coefficient in source.all_coeffs()
    ):
        raise ValueError("polynomial coefficient exceeds the input digit bound")
    return source


def _fraction_digits(value: Fraction) -> int:
    return max(len(str(abs(value.numerator))), len(str(value.denominator)))


def _require_bounded_output_coefficients(polynomial: Any) -> None:
    if any(
        _fraction_digits(Fraction(coefficient)) > _MAX_OUTPUT_DIGITS
        for coefficient in polynomial.all_coeffs()
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
