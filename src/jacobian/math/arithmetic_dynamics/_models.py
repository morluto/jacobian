"""Typed wire contracts for exact bounded arithmetic dynamics."""

from __future__ import annotations

from fractions import Fraction
from itertools import pairwise
from typing import Literal, Self

from pydantic import Field, model_validator

from jacobian._models import StrictModel

MAX_COEFFICIENT_DIGITS = 128
MAX_DEGREE = 30
MAX_ITERATE = 20
MAX_ITERATE_DEGREE = 1024
MAX_DYNATOMIC_DEGREE = 512
MAX_ORBIT_STEPS = 1_000
MAX_ORBIT_VALUE_DIGITS = 2_048
MAX_POLYNOMIAL_OUTPUT_DIGITS = 32_768
MAX_FIELD_PRIME = 10_000


def parse_canonical_rational(value: str, *, label: str) -> Fraction:
    if len(value) > 2 * MAX_COEFFICIENT_DIGITS + 2:
        raise ValueError(f"{label} exceeds the rational digit bound")
    try:
        parsed = Fraction(value)
    except (ValueError, ZeroDivisionError):
        raise ValueError(f"{label} must be a canonical rational") from None
    if str(parsed) != value:
        raise ValueError(f"{label} must be a reduced canonical rational")
    if (
        len(str(abs(parsed.numerator))) > MAX_COEFFICIENT_DIGITS
        or len(str(parsed.denominator)) > MAX_COEFFICIENT_DIGITS
    ):
        raise ValueError(f"{label} exceeds the rational digit bound")
    return parsed


def parse_polynomial_coefficients(values: tuple[str, ...]) -> tuple[Fraction, ...]:
    coefficients = tuple(
        parse_canonical_rational(value, label="coefficient") for value in values
    )
    if len(coefficients) > 1 and coefficients[-1] == 0:
        raise ValueError("polynomial coefficients must omit trailing zeros")
    return coefficients


class PolynomialCoefficientRequest(StrictModel):
    coefficients: tuple[str, ...] = Field(min_length=1, max_length=MAX_DEGREE + 1)

    @model_validator(mode="after")
    def require_canonical_coefficients(self) -> Self:
        parse_polynomial_coefficients(self.coefficients)
        return self

    def coefficient_values(self) -> tuple[Fraction, ...]:
        return parse_polynomial_coefficients(self.coefficients)

    def polynomial_degree(self) -> int:
        values = self.coefficient_values()
        return 0 if values == (Fraction(0),) else len(values) - 1


class MapIterateRequest(PolynomialCoefficientRequest):
    """Compute one exact polynomial iterate within an output-degree bound."""

    n: int = Field(ge=0, le=MAX_ITERATE)

    @model_validator(mode="after")
    def require_bounded_iterate_degree(self) -> Self:
        degree = self.polynomial_degree()
        output_degree = 1 if self.n == 0 else degree**self.n
        if output_degree > MAX_ITERATE_DEGREE:
            raise ValueError("iterate output degree exceeds bound")
        return self


class OrbitPrefixRequest(PolynomialCoefficientRequest):
    """Compute until a first repeat or an explicit finite/output bound."""

    start: str
    max_steps: int = Field(ge=0, le=MAX_ORBIT_STEPS)

    @model_validator(mode="after")
    def require_canonical_start(self) -> Self:
        parse_canonical_rational(self.start, label="start")
        return self


class DynatomicPolynomialRequest(PolynomialCoefficientRequest):
    """Compute the n-th dynatomic polynomial of a degree-at-least-two map."""

    n: int = Field(ge=1, le=MAX_ITERATE)

    @model_validator(mode="after")
    def require_bounded_dynatomic_degree(self) -> Self:
        degree = self.polynomial_degree()
        if degree < 2:
            raise ValueError("dynatomic polynomial requires map degree at least two")
        if degree**self.n > MAX_DYNATOMIC_DEGREE:
            raise ValueError("dynatomic output degree exceeds bound")
        return self


class CycleMultiplierRequest(PolynomialCoefficientRequest):
    """Compute the multiplier of a supplied, validated exact rational cycle."""

    cycle: tuple[str, ...] = Field(min_length=1, max_length=MAX_ORBIT_STEPS)

    @model_validator(mode="after")
    def require_exact_cycle(self) -> Self:
        points = tuple(
            parse_canonical_rational(value, label="cycle point") for value in self.cycle
        )
        if len(set(points)) != len(points):
            raise ValueError("cycle points must be distinct")
        coefficients = self.coefficient_values()
        for index, point in enumerate(points):
            expected = points[(index + 1) % len(points)]
            if _evaluate(coefficients, point) != expected:
                raise ValueError("cycle points must follow the polynomial map in order")
        return self


class FiniteFieldMapRequest(StrictModel):
    """A canonical polynomial map over the prime field GF(p)."""

    prime: int = Field(ge=2, le=MAX_FIELD_PRIME)
    coefficients: tuple[str, ...] = Field(min_length=1, max_length=MAX_DEGREE + 1)

    @model_validator(mode="after")
    def require_canonical_prime_field_map(self) -> Self:
        if not _is_prime(self.prime):
            raise ValueError("prime must be a prime number")
        values = tuple(_parse_canonical_integer(value) for value in self.coefficients)
        if len(values) > 1 and values[-1] % self.prime == 0:
            raise ValueError("coefficients must omit trailing zeros modulo the prime")
        return self


class MapIterateResult(StrictModel):
    source_coefficients: tuple[str, ...] = Field(
        min_length=1, max_length=MAX_DEGREE + 1
    )
    n: int = Field(ge=0, le=MAX_ITERATE)
    coefficients: tuple[str, ...] = Field(
        min_length=1, max_length=MAX_ITERATE_DEGREE + 1
    )
    degree: int = Field(ge=0, le=MAX_ITERATE_DEGREE)
    complete: Literal[True] = True
    method: Literal["EXACT_POLYNOMIAL_COMPOSITION"] = "EXACT_POLYNOMIAL_COMPOSITION"

    @model_validator(mode="after")
    def bind_degree_and_coefficients(self) -> Self:
        parse_polynomial_coefficients(self.source_coefficients)
        values = tuple(Fraction(value) for value in self.coefficients)
        if any(
            str(Fraction(value)) != value
            or len(value) > MAX_POLYNOMIAL_OUTPUT_DIGITS * 2 + 2
            for value in self.coefficients
        ):
            raise ValueError("iterate coefficients must be bounded and canonical")
        expected_degree = 0 if values == (Fraction(0),) else len(values) - 1
        if self.degree != expected_degree:
            raise ValueError("degree must match the canonical coefficient tuple")
        return self


class OrbitRepeatEvidence(StrictModel):
    first_seen_index: int = Field(ge=0)
    repeated_at_index: int = Field(ge=1)
    preperiod: int = Field(ge=0)
    period: int = Field(ge=1)

    @model_validator(mode="after")
    def bind_indices(self) -> Self:
        if self.preperiod != self.first_seen_index:
            raise ValueError("preperiod must equal first-seen index")
        if self.period != self.repeated_at_index - self.first_seen_index:
            raise ValueError("period must equal the repeat-index difference")
        return self


class OrbitPrefixResult(StrictModel):
    source_coefficients: tuple[str, ...] = Field(
        min_length=1, max_length=MAX_DEGREE + 1
    )
    start: str
    orbit: tuple[str, ...] = Field(min_length=1, max_length=MAX_ORBIT_STEPS + 1)
    requested_steps: int = Field(ge=0, le=MAX_ORBIT_STEPS)
    computed_steps: int = Field(ge=0, le=MAX_ORBIT_STEPS)
    termination: Literal["REPEAT_FOUND", "STEP_BOUND_REACHED", "OUTPUT_BOUND_REACHED"]
    repeat: OrbitRepeatEvidence | None = None
    eventual_behavior_complete: bool
    truncated: bool

    @model_validator(mode="after")
    def bind_termination_evidence(self) -> Self:
        _require_bound_orbit(self.source_coefficients, self.start, self.orbit)
        if len(self.orbit) != self.computed_steps + 1:
            raise ValueError("orbit length must equal computed steps plus one")
        if self.computed_steps > self.requested_steps:
            raise ValueError("computed steps cannot exceed the request bound")
        if self.termination == "REPEAT_FOUND":
            if (
                self.repeat is None
                or not self.eventual_behavior_complete
                or self.truncated
            ):
                raise ValueError("repeat termination requires complete repeat evidence")
            if self.repeat.repeated_at_index != self.computed_steps:
                raise ValueError("repeat evidence must bind the final orbit value")
            if self.orbit[self.repeat.first_seen_index] != self.orbit[-1]:
                raise ValueError("repeat evidence must bind equal orbit values")
        elif (
            self.repeat is not None
            or self.eventual_behavior_complete
            or not self.truncated
        ):
            raise ValueError("bounded termination cannot imply eventual behavior")
        if self.termination == "STEP_BOUND_REACHED" and (
            self.computed_steps != self.requested_steps
        ):
            raise ValueError("step-bound termination must exhaust the requested prefix")
        return self


class DynatomicPolynomialResult(StrictModel):
    source_coefficients: tuple[str, ...] = Field(
        min_length=1, max_length=MAX_DEGREE + 1
    )
    coefficients: tuple[str, ...] = Field(
        min_length=1, max_length=MAX_DYNATOMIC_DEGREE + 1
    )
    degree: int = Field(ge=0, le=MAX_DYNATOMIC_DEGREE)
    n: int = Field(ge=1, le=MAX_ITERATE)
    complete: Literal[True] = True
    method: Literal["MOBIUS_EXACT_POLYNOMIAL_DIVISION"] = (
        "MOBIUS_EXACT_POLYNOMIAL_DIVISION"
    )

    @model_validator(mode="after")
    def bind_degree_and_coefficients(self) -> Self:
        parse_polynomial_coefficients(self.source_coefficients)
        values = tuple(Fraction(value) for value in self.coefficients)
        if any(
            str(Fraction(value)) != value
            or len(value) > MAX_POLYNOMIAL_OUTPUT_DIGITS * 2 + 2
            for value in self.coefficients
        ):
            raise ValueError("dynatomic coefficients must be bounded and canonical")
        expected_degree = 0 if values == (Fraction(0),) else len(values) - 1
        if self.degree != expected_degree:
            raise ValueError("degree must match the canonical coefficient tuple")
        return self


class CycleMultiplierResult(StrictModel):
    source_coefficients: tuple[str, ...] = Field(
        min_length=1, max_length=MAX_DEGREE + 1
    )
    multiplier: str
    cycle: tuple[str, ...] = Field(min_length=1, max_length=MAX_ORBIT_STEPS)
    period: int = Field(ge=1, le=MAX_ORBIT_STEPS)
    validated_cycle: Literal[True] = True
    complete: Literal[True] = True

    @model_validator(mode="after")
    def bind_period_and_multiplier(self) -> Self:
        source = parse_polynomial_coefficients(self.source_coefficients)
        points = tuple(
            parse_canonical_rational(value, label="cycle point") for value in self.cycle
        )
        if len(set(points)) != len(points) or any(
            _evaluate(source, point) != points[(index + 1) % len(points)]
            for index, point in enumerate(points)
        ):
            raise ValueError("cycle must be a distinct ordered cycle of the bound map")
        if self.period != len(self.cycle):
            raise ValueError("period must match cycle length")
        value = Fraction(self.multiplier)
        if (
            str(value) != self.multiplier
            or len(str(abs(value.numerator))) > MAX_POLYNOMIAL_OUTPUT_DIGITS
            or len(str(value.denominator)) > MAX_POLYNOMIAL_OUTPUT_DIGITS
        ):
            raise ValueError("multiplier must be a bounded canonical rational")
        return self


class FiniteFieldMapResult(StrictModel):
    prime: int = Field(ge=2, le=MAX_FIELD_PRIME)
    coefficients: tuple[str, ...] = Field(min_length=1, max_length=MAX_DEGREE + 1)
    edges: tuple[tuple[int, int], ...]
    cycles: tuple[tuple[int, ...], ...]
    tail_lengths: tuple[int, ...]
    complete: Literal[True] = True
    method: Literal["COMPLETE_FUNCTIONAL_GRAPH_ENUMERATION"] = (
        "COMPLETE_FUNCTIONAL_GRAPH_ENUMERATION"
    )

    @model_validator(mode="after")
    def bind_complete_graph(self) -> Self:
        if not _is_prime(self.prime):
            raise ValueError("prime must be a prime number")
        coefficients = tuple(
            _parse_canonical_integer(value) for value in self.coefficients
        )
        if len(coefficients) > 1 and coefficients[-1] % self.prime == 0:
            raise ValueError("coefficients must omit trailing zeros modulo the prime")
        self._require_complete_edges()
        if any(
            target != _evaluate_mod_prime(coefficients, source, self.prime)
            for source, target in self.edges
        ):
            raise ValueError(
                "functional graph edges must evaluate the bound polynomial"
            )
        cycle_set = self._require_canonical_cycles()
        self._require_tail_evidence(cycle_set)
        return self

    def _require_complete_edges(self) -> None:
        if len(self.edges) != self.prime or len(self.tail_lengths) != self.prime:
            raise ValueError("functional graph must cover every field element")
        if tuple(source for source, _ in self.edges) != tuple(range(self.prime)):
            raise ValueError("functional graph edges must be source ordered")
        if any(not 0 <= target < self.prime for _, target in self.edges):
            raise ValueError("functional graph edge target out of range")
        if any(length < 0 for length in self.tail_lengths):
            raise ValueError("tail lengths must be nonnegative")

    def _require_canonical_cycles(self) -> set[int]:
        if self.cycles != tuple(sorted(self.cycles)):
            raise ValueError("cycles must be canonical and sorted")
        if any(not cycle or cycle[0] != min(cycle) for cycle in self.cycles):
            raise ValueError("each cycle must start at its least element")
        cycle_nodes = [node for cycle in self.cycles for node in cycle]
        if len(cycle_nodes) != len(set(cycle_nodes)):
            raise ValueError("functional graph cycles must be disjoint")
        targets = tuple(target for _, target in self.edges)
        for cycle in self.cycles:
            for index, node in enumerate(cycle):
                if targets[node] != cycle[(index + 1) % len(cycle)]:
                    raise ValueError("cycle must follow functional graph edges")
        return set(cycle_nodes)

    def _require_tail_evidence(self, cycle_set: set[int]) -> None:
        if cycle_set != {
            node for node, length in enumerate(self.tail_lengths) if length == 0
        }:
            raise ValueError("zero tail lengths must identify exactly the cycle nodes")
        if any(
            self.tail_lengths[source] != self.tail_lengths[target] + 1
            for source, target in self.edges
            if source not in cycle_set
        ):
            raise ValueError("tail lengths must decrease by one along every tail edge")


def _evaluate(coefficients: tuple[Fraction, ...], point: Fraction) -> Fraction:
    value = Fraction(0)
    for coefficient in reversed(coefficients):
        value = value * point + coefficient
    return value


def _require_bound_orbit(
    source_coefficients: tuple[str, ...],
    start: str,
    orbit_values: tuple[str, ...],
) -> None:
    source = parse_polynomial_coefficients(source_coefficients)
    initial = parse_canonical_rational(start, label="start")
    orbit = tuple(
        parse_canonical_rational(value, label="orbit value") for value in orbit_values
    )
    if orbit[0] != initial:
        raise ValueError("orbit must begin at the bound start point")
    if any(_evaluate(source, point) != target for point, target in pairwise(orbit)):
        raise ValueError("orbit values must follow the bound polynomial map")


def _evaluate_mod_prime(coefficients: tuple[int, ...], point: int, prime: int) -> int:
    value = 0
    for coefficient in reversed(coefficients):
        value = (value * point + coefficient) % prime
    return value


def _parse_canonical_integer(value: str) -> int:
    if len(value) > MAX_COEFFICIENT_DIGITS + 1:
        raise ValueError("coefficient exceeds the integer digit bound")
    try:
        parsed = int(value)
    except ValueError:
        raise ValueError("coefficient must be a canonical integer") from None
    if str(parsed) != value:
        raise ValueError("coefficient must be a canonical integer")
    return parsed


def _is_prime(n: int) -> bool:
    if n < 2:
        return False
    if n < 4:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6
    return True


__all__ = [
    "CycleMultiplierRequest",
    "CycleMultiplierResult",
    "DynatomicPolynomialRequest",
    "DynatomicPolynomialResult",
    "FiniteFieldMapRequest",
    "FiniteFieldMapResult",
    "MapIterateRequest",
    "MapIterateResult",
    "OrbitPrefixRequest",
    "OrbitPrefixResult",
    "OrbitRepeatEvidence",
]
