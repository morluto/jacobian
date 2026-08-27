"""Typed wire contracts for exact bounded arithmetic dynamics."""

from __future__ import annotations

from collections.abc import Callable
from fractions import Fraction
from functools import wraps
from itertools import pairwise
from typing import Any, Literal, Self

from pydantic import Field
from pydantic import model_validator as _pydantic_model_validator
from pydantic_core import PydanticCustomError

from jacobian._exact import CanonicalRational, require_bounded_rational
from jacobian._models import StrictModel
from jacobian.math._rational_height import RationalHeight, sum_heights

MAX_COEFFICIENT_DIGITS = 128
MAX_DEGREE = 30
MAX_ITERATE = 20
MAX_ITERATE_DEGREE = 1024
MAX_DYNATOMIC_DEGREE = 512
MAX_ORBIT_STEPS = 1_000
MAX_ORBIT_VALUE_DIGITS = 2_048
MAX_POLYNOMIAL_OUTPUT_DIGITS = 32_768
MAX_FIELD_PRIME = 10_000

CoefficientHeight = RationalHeight | None


_VALIDATION_CODES = {
    "iterate output degree exceeds bound": "iterate_degree_exceeds_bound",
    "iterate coefficient growth exceeds the 32768-digit output bound": "iterate_coefficient_growth_exceeds_bound",
    "dynatomic polynomial requires map degree at least two": "dynatomic_degree_too_small",
    "dynatomic output degree exceeds bound": "dynatomic_degree_exceeds_bound",
    "cycle points must be distinct": "cycle_points_not_distinct",
    "cycle points must follow the polynomial map in order": "cycle_map_mismatch",
    "prime must be a prime number": "prime_not_prime",
    "coefficients must omit trailing zeros modulo the prime": "trailing_zero_coefficients",
    "degree must match the canonical coefficient tuple": "degree_mismatch",
    "preperiod must equal first-seen index": "preperiod_mismatch",
    "period must equal the repeat-index difference": "period_mismatch",
    "orbit length must equal computed steps plus one": "orbit_length_mismatch",
    "computed steps cannot exceed the request bound": "computed_steps_exceed_bound",
    "repeat termination requires complete repeat evidence": "repeat_evidence_incomplete",
    "repeat evidence must bind the final orbit value": "repeat_final_value_mismatch",
    "repeat evidence must bind equal orbit values": "repeat_value_mismatch",
    "bounded termination cannot imply eventual behavior": "bounded_termination_claims_eventual_behavior",
    "step-bound termination must exhaust the requested prefix": "step_bound_incomplete",
    "cycle must be a distinct ordered cycle of the bound map": "cycle_map_mismatch",
    "period must match cycle length": "period_mismatch",
    "functional graph must cover every field element": "functional_graph_incomplete",
    "functional graph edges must be source ordered": "functional_graph_edges_unordered",
    "functional graph edge target out of range": "functional_graph_target_out_of_range",
    "tail lengths must be nonnegative": "negative_tail_length",
    "cycles must be canonical and sorted": "cycles_not_canonical",
    "each cycle must start at its least element": "cycle_not_canonical",
    "functional graph cycles must be disjoint": "cycles_overlap",
    "functional graph edges must evaluate the bound polynomial": "functional_graph_edge_mismatch",
    "cycle must follow functional graph edges": "cycle_edge_mismatch",
    "zero tail lengths must identify exactly the cycle nodes": "cycle_tail_mismatch",
    "tail lengths must decrease by one along every tail edge": "tail_length_mismatch",
    "orbit must begin at the bound start point": "orbit_start_mismatch",
    "orbit values must follow the bound polynomial map": "orbit_map_mismatch",
    "polynomial coefficients must omit trailing zeros": "trailing_zero_coefficients",
}

_VALIDATION_FRAGMENTS = (
    ("coefficient exceeds the integer digit bound", "coefficient_digit_bound"),
    ("coefficient must be a canonical integer", "coefficient_not_canonical"),
    ("exceeds the 128-digit bound", "rational_digit_bound"),
    ("exceeds the 2048-digit bound", "orbit_value_digit_bound"),
    ("exceeds the 32768-digit bound", "result_digit_bound"),
)


def _validation_error(message: str) -> PydanticCustomError:
    code = _VALIDATION_CODES.get(message)
    if code is None:
        for fragment, fragment_code in _VALIDATION_FRAGMENTS:
            if fragment in message:
                code = fragment_code
                break
    if code is None and "coefficient growth exceeds" in message:
        code = (
            "iterate_coefficient_growth_exceeds_bound"
            if message.startswith("iterate")
            else "dynatomic_coefficient_growth_exceeds_bound"
        )
    if code is None:
        code = "contract_invariant"
    return PydanticCustomError(f"arithmetic_dynamics.{code}", message)


def model_validator(
    *, mode: Literal["before", "after", "wrap"]
) -> Callable[[Callable[..., Any]], Any]:
    """Convert owner-local model invariant failures to stable error codes."""

    def decorate(function: Callable[..., Any]) -> Any:
        @wraps(function)
        def wrapped(self: Any, *inner_args: Any, **inner_kwargs: Any) -> Any:
            try:
                return function(self, *inner_args, **inner_kwargs)
            except ValueError as exc:
                raise _validation_error(str(exc)) from exc

        return _pydantic_model_validator(mode=mode)(wrapped)

    return decorate


def _fraction_height(value: Fraction) -> CoefficientHeight:
    if value == 0:
        return None
    return RationalHeight(len(str(abs(value.numerator))), len(str(value.denominator)))


def _add_heights(
    left: CoefficientHeight, right: CoefficientHeight
) -> CoefficientHeight:
    if left is None:
        return right
    if right is None:
        return left
    return sum_heights((left, right))


def _multiply_height_polynomials(
    left: tuple[CoefficientHeight, ...], right: tuple[CoefficientHeight, ...]
) -> tuple[CoefficientHeight, ...]:
    result: list[CoefficientHeight] = [None] * (len(left) + len(right) - 1)
    for left_index, left_height in enumerate(left):
        if left_height is None:
            continue
        for right_index, right_height in enumerate(right):
            if right_height is None:
                continue
            index = left_index + right_index
            result[index] = _add_heights(
                result[index], left_height.product(right_height)
            )
    return _trim_heights(tuple(result))


def _compose_height_polynomials(
    outer: tuple[CoefficientHeight, ...], inner: tuple[CoefficientHeight, ...]
) -> tuple[CoefficientHeight, ...]:
    result: tuple[CoefficientHeight, ...] = (None,)
    for coefficient in reversed(outer):
        result = _multiply_height_polynomials(result, inner)
        result = (_add_heights(result[0], coefficient), *result[1:])
    return result


def _require_polynomial_height(
    coefficients: tuple[CoefficientHeight, ...], operation: str
) -> None:
    if any(
        height is not None and height.exceeds(MAX_POLYNOMIAL_OUTPUT_DIGITS)
        for height in coefficients
    ):
        raise ValueError(
            f"{operation} coefficient growth exceeds the "
            f"{MAX_POLYNOMIAL_OUTPUT_DIGITS}-digit output bound"
        )


def _iterate_heights(
    source: tuple[CoefficientHeight, ...], count: int
) -> tuple[CoefficientHeight, ...]:
    result: tuple[CoefficientHeight, ...] = (None, RationalHeight(1, 1))
    for _ in range(count):
        result = _compose_height_polynomials(source, result)
        _require_polynomial_height(result, "iterate")
    return result


def _trim_heights(
    coefficients: tuple[CoefficientHeight, ...],
) -> tuple[CoefficientHeight, ...]:
    end = len(coefficients)
    while end > 1 and coefficients[end - 1] is None:
        end -= 1
    return coefficients[:end]


def _divide_height_polynomials(
    numerator: tuple[CoefficientHeight, ...],
    denominator: tuple[CoefficientHeight, ...],
) -> tuple[CoefficientHeight, ...]:
    remainder = list(_trim_heights(numerator))
    divisor = _trim_heights(denominator)
    divisor_degree = len(divisor) - 1
    divisor_lead = divisor[-1]
    if divisor_lead is None:
        raise RuntimeError("height preflight received a zero polynomial divisor")
    quotient: list[CoefficientHeight] = [None] * max(1, len(remainder) - divisor_degree)
    while len(remainder) - 1 >= divisor_degree and remainder[-1] is not None:
        offset = len(remainder) - 1 - divisor_degree
        coefficient = remainder[-1].quotient(divisor_lead)
        quotient[offset] = coefficient
        _require_polynomial_height((coefficient,), "dynatomic quotient")
        for index, divisor_height in enumerate(divisor):
            if divisor_height is None:
                continue
            target = offset + index
            remainder[target] = _add_heights(
                remainder[target], coefficient.product(divisor_height)
            )
        remainder[-1] = None  # exact leading-term cancellation
        remainder = list(_trim_heights(tuple(remainder)))
        _require_polynomial_height(tuple(remainder), "dynatomic division")
    return _trim_heights(tuple(quotient))


def _mobius(value: int) -> int:
    factors = 0
    candidate = 2
    remaining = value
    while candidate * candidate <= remaining:
        if remaining % candidate == 0:
            remaining //= candidate
            factors += 1
            if remaining % candidate == 0:
                return 0
            while remaining % candidate == 0:
                remaining //= candidate
        candidate += 1
    if remaining > 1:
        factors += 1
    return -1 if factors % 2 else 1


def _bounded_fraction(
    value: CanonicalRational, *, max_digits: int, label: str
) -> Fraction:
    require_bounded_rational(value, max_digits=max_digits, label=label)
    return value.as_fraction()


def parse_polynomial_coefficients(
    values: tuple[CanonicalRational, ...],
) -> tuple[Fraction, ...]:
    coefficients = tuple(
        _bounded_fraction(value, max_digits=MAX_COEFFICIENT_DIGITS, label="coefficient")
        for value in values
    )
    if len(coefficients) > 1 and coefficients[-1] == 0:
        raise ValueError("polynomial coefficients must omit trailing zeros")
    return coefficients


class PolynomialCoefficientRequest(StrictModel):
    coefficients: tuple[CanonicalRational, ...] = Field(
        min_length=1, max_length=MAX_DEGREE + 1
    )

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
            raise _validation_error("iterate output degree exceeds bound")
        source = tuple(_fraction_height(value) for value in self.coefficient_values())
        _iterate_heights(source, self.n)
        return self


class OrbitPrefixRequest(PolynomialCoefficientRequest):
    """Compute until a first repeat or an explicit finite/output bound."""

    start: CanonicalRational
    max_steps: int = Field(ge=0, le=MAX_ORBIT_STEPS)

    @model_validator(mode="after")
    def require_canonical_start(self) -> Self:
        _bounded_fraction(self.start, max_digits=MAX_COEFFICIENT_DIGITS, label="start")
        return self


class DynatomicPolynomialRequest(PolynomialCoefficientRequest):
    """Compute the n-th dynatomic polynomial of a degree-at-least-two map."""

    n: int = Field(ge=1, le=MAX_ITERATE)

    @model_validator(mode="after")
    def require_bounded_dynatomic_degree(self) -> Self:
        degree = self.polynomial_degree()
        if degree < 2:
            raise _validation_error(
                "dynatomic polynomial requires map degree at least two"
            )
        if degree**self.n > MAX_DYNATOMIC_DEGREE:
            raise _validation_error("dynatomic output degree exceeds bound")
        source = tuple(_fraction_height(value) for value in self.coefficient_values())
        numerator: tuple[CoefficientHeight, ...] = (RationalHeight(1, 1),)
        denominator: tuple[CoefficientHeight, ...] = (RationalHeight(1, 1),)
        for divisor in range(1, self.n + 1):
            if self.n % divisor != 0:
                continue
            term = list(_iterate_heights(source, divisor))
            if len(term) < 2:
                term.extend([None] * (2 - len(term)))
            term[1] = _add_heights(term[1], RationalHeight(1, 1))
            mobius = _mobius(self.n // divisor)
            if mobius == 1:
                numerator = _multiply_height_polynomials(numerator, tuple(term))
                _require_polynomial_height(numerator, "dynatomic numerator")
            elif mobius == -1:
                denominator = _multiply_height_polynomials(denominator, tuple(term))
                _require_polynomial_height(denominator, "dynatomic denominator")
        quotient = _divide_height_polynomials(numerator, denominator)
        _require_polynomial_height(quotient, "dynatomic quotient")
        return self


class CycleMultiplierRequest(PolynomialCoefficientRequest):
    """Compute the multiplier of a supplied, validated exact rational cycle."""

    cycle: tuple[CanonicalRational, ...] = Field(
        min_length=1, max_length=MAX_ORBIT_STEPS
    )

    @model_validator(mode="after")
    def require_exact_cycle(self) -> Self:
        points = tuple(
            _bounded_fraction(
                value, max_digits=MAX_COEFFICIENT_DIGITS, label="cycle point"
            )
            for value in self.cycle
        )
        if len(set(points)) != len(points):
            raise _validation_error("cycle points must be distinct")
        coefficients = self.coefficient_values()
        for index, point in enumerate(points):
            expected = points[(index + 1) % len(points)]
            if _evaluate(coefficients, point) != expected:
                raise _validation_error(
                    "cycle points must follow the polynomial map in order"
                )
        return self


class FiniteFieldMapRequest(StrictModel):
    """A canonical polynomial map over the prime field GF(p)."""

    prime: int = Field(ge=2, le=MAX_FIELD_PRIME)
    coefficients: tuple[str, ...] = Field(min_length=1, max_length=MAX_DEGREE + 1)

    @model_validator(mode="after")
    def require_canonical_prime_field_map(self) -> Self:
        if not _is_prime(self.prime):
            raise _validation_error("prime must be a prime number")
        values = tuple(_parse_canonical_integer(value) for value in self.coefficients)
        if len(values) > 1 and values[-1] % self.prime == 0:
            raise _validation_error(
                "coefficients must omit trailing zeros modulo the prime"
            )
        return self


class MapIterateResult(StrictModel):
    source_coefficients: tuple[CanonicalRational, ...] = Field(
        min_length=1, max_length=MAX_DEGREE + 1
    )
    n: int = Field(ge=0, le=MAX_ITERATE)
    coefficients: tuple[CanonicalRational, ...] = Field(
        min_length=1, max_length=MAX_ITERATE_DEGREE + 1
    )
    degree: int = Field(ge=0, le=MAX_ITERATE_DEGREE)
    complete: Literal[True] = True
    method: Literal["EXACT_POLYNOMIAL_COMPOSITION"] = "EXACT_POLYNOMIAL_COMPOSITION"

    @model_validator(mode="after")
    def bind_degree_and_coefficients(self) -> Self:
        parse_polynomial_coefficients(self.source_coefficients)
        values = tuple(
            _bounded_fraction(
                value,
                max_digits=MAX_POLYNOMIAL_OUTPUT_DIGITS,
                label="iterate coefficient",
            )
            for value in self.coefficients
        )
        expected_degree = 0 if values == (Fraction(0),) else len(values) - 1
        if self.degree != expected_degree:
            raise _validation_error("degree must match the canonical coefficient tuple")
        return self


class OrbitRepeatEvidence(StrictModel):
    first_seen_index: int = Field(ge=0)
    repeated_at_index: int = Field(ge=1)
    preperiod: int = Field(ge=0)
    period: int = Field(ge=1)

    @model_validator(mode="after")
    def bind_indices(self) -> Self:
        if self.preperiod != self.first_seen_index:
            raise _validation_error("preperiod must equal first-seen index")
        if self.period != self.repeated_at_index - self.first_seen_index:
            raise _validation_error("period must equal the repeat-index difference")
        return self


class OrbitPrefixResult(StrictModel):
    source_coefficients: tuple[CanonicalRational, ...] = Field(
        min_length=1, max_length=MAX_DEGREE + 1
    )
    start: CanonicalRational
    orbit: tuple[CanonicalRational, ...] = Field(
        min_length=1, max_length=MAX_ORBIT_STEPS + 1
    )
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
            raise _validation_error("orbit length must equal computed steps plus one")
        if self.computed_steps > self.requested_steps:
            raise _validation_error("computed steps cannot exceed the request bound")
        if self.termination == "REPEAT_FOUND":
            if (
                self.repeat is None
                or not self.eventual_behavior_complete
                or self.truncated
            ):
                raise _validation_error(
                    "repeat termination requires complete repeat evidence"
                )
            if self.repeat.repeated_at_index != self.computed_steps:
                raise _validation_error(
                    "repeat evidence must bind the final orbit value"
                )
            if self.orbit[self.repeat.first_seen_index] != self.orbit[-1]:
                raise _validation_error("repeat evidence must bind equal orbit values")
        elif (
            self.repeat is not None
            or self.eventual_behavior_complete
            or not self.truncated
        ):
            raise _validation_error(
                "bounded termination cannot imply eventual behavior"
            )
        if self.termination == "STEP_BOUND_REACHED" and (
            self.computed_steps != self.requested_steps
        ):
            raise _validation_error(
                "step-bound termination must exhaust the requested prefix"
            )
        return self


class DynatomicPolynomialResult(StrictModel):
    source_coefficients: tuple[CanonicalRational, ...] = Field(
        min_length=1, max_length=MAX_DEGREE + 1
    )
    coefficients: tuple[CanonicalRational, ...] = Field(
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
        values = tuple(
            _bounded_fraction(
                value,
                max_digits=MAX_POLYNOMIAL_OUTPUT_DIGITS,
                label="dynatomic coefficient",
            )
            for value in self.coefficients
        )
        expected_degree = 0 if values == (Fraction(0),) else len(values) - 1
        if self.degree != expected_degree:
            raise _validation_error("degree must match the canonical coefficient tuple")
        return self


class CycleMultiplierResult(StrictModel):
    source_coefficients: tuple[CanonicalRational, ...] = Field(
        min_length=1, max_length=MAX_DEGREE + 1
    )
    multiplier: CanonicalRational
    cycle: tuple[CanonicalRational, ...] = Field(
        min_length=1, max_length=MAX_ORBIT_STEPS
    )
    period: int = Field(ge=1, le=MAX_ORBIT_STEPS)
    validated_cycle: Literal[True] = True
    complete: Literal[True] = True

    @model_validator(mode="after")
    def require_structural_consistency(self) -> Self:
        parse_polynomial_coefficients(self.source_coefficients)
        for value in self.cycle:
            _bounded_fraction(
                value, max_digits=MAX_COEFFICIENT_DIGITS, label="cycle point"
            )
        if self.period != len(self.cycle):
            raise _validation_error("period must match cycle length")
        require_bounded_rational(
            self.multiplier,
            max_digits=MAX_POLYNOMIAL_OUTPUT_DIGITS,
            label="multiplier",
        )
        return self

    @classmethod
    def _from_kernel(
        cls,
        request: CycleMultiplierRequest,
        *,
        multiplier: CanonicalRational,
    ) -> Self:
        """Build a result after the admitted cycle-multiplier kernel ran."""

        require_bounded_rational(
            multiplier,
            max_digits=MAX_POLYNOMIAL_OUTPUT_DIGITS,
            label="multiplier",
        )
        return cls(
            source_coefficients=request.coefficients,
            multiplier=multiplier,
            cycle=request.cycle,
            period=len(request.cycle),
        )


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
    def require_structural_consistency(self) -> Self:
        if not _is_prime(self.prime):
            raise _validation_error("prime must be a prime number")
        values = tuple(_parse_canonical_integer(value) for value in self.coefficients)
        if len(values) > 1 and values[-1] % self.prime == 0:
            raise _validation_error(
                "coefficients must omit trailing zeros modulo the prime"
            )
        self._require_complete_edges()
        cycle_set = self._require_canonical_cycles()
        self._require_tail_evidence(cycle_set)
        return self

    @classmethod
    def _from_kernel(
        cls,
        request: FiniteFieldMapRequest,
        *,
        edges: tuple[tuple[int, int], ...],
        cycles: tuple[tuple[int, ...], ...],
        tail_lengths: tuple[int, ...],
    ) -> Self:
        """Build a result after complete graph enumeration established it."""

        return cls(
            prime=request.prime,
            coefficients=request.coefficients,
            edges=edges,
            cycles=cycles,
            tail_lengths=tail_lengths,
        )

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
    source_coefficients: tuple[CanonicalRational, ...],
    start: CanonicalRational,
    orbit_values: tuple[CanonicalRational, ...],
) -> None:
    source = parse_polynomial_coefficients(source_coefficients)
    initial = _bounded_fraction(start, max_digits=MAX_COEFFICIENT_DIGITS, label="start")
    orbit = tuple(
        _bounded_fraction(
            value,
            max_digits=MAX_ORBIT_VALUE_DIGITS,
            label="orbit value",
        )
        for value in orbit_values
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
