"""Typed wire contracts for exact bounded arithmetic dynamics."""

from __future__ import annotations

from fractions import Fraction
from typing import Literal, Self

from pydantic import Field

from jacobian._exact import CanonicalRational, require_bounded_rational
from jacobian._models import StrictModel
from jacobian.math._rational_height import RationalHeight, sum_heights
from jacobian.math.finite_fields.values import FinitePolynomialMap
from jacobian.math.polynomials.values import RationalPolynomial

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
    f"iterate coefficient growth exceeds the {MAX_POLYNOMIAL_OUTPUT_DIGITS}-digit output bound": "iterate_coefficient_growth_exceeds_bound",
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
    "cycle must contain distinct points": "cycle_points_not_distinct",
    "cycle points do not follow the polynomial map": "cycle_map_mismatch",
    "polynomial coefficients must omit trailing zeros modulo p": "trailing_zero_coefficients",
}

_VALIDATION_FRAGMENTS = (
    ("coefficient exceeds the integer digit bound", "coefficient_digit_bound"),
    ("coefficient must be a canonical integer", "coefficient_not_canonical"),
    (f"exceeds the {MAX_COEFFICIENT_DIGITS}-digit bound", "rational_digit_bound"),
    (f"exceeds the {MAX_ORBIT_VALUE_DIGITS}-digit bound", "orbit_value_digit_bound"),
    (
        f"exceeds the {MAX_POLYNOMIAL_OUTPUT_DIGITS}-digit bound",
        "result_digit_bound",
    ),
    ("prime must be a prime number", "prime_not_prime"),
)


def _validation_code(message: str) -> str:
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
    return code


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
    """A source polynomial in the shared canonical QQ representation."""

    polynomial: RationalPolynomial


class MapIterateRequest(PolynomialCoefficientRequest):
    """Compute one exact polynomial iterate within an output-degree bound."""

    n: int = Field(ge=0, le=MAX_ITERATE)


class OrbitPrefixRequest(PolynomialCoefficientRequest):
    """Compute until a first repeat or an explicit finite/output bound."""

    start: CanonicalRational
    max_steps: int = Field(ge=0, le=MAX_ORBIT_STEPS)


class DynatomicPolynomialRequest(PolynomialCoefficientRequest):
    """Compute the n-th dynatomic polynomial of a degree-at-least-two map."""

    n: int = Field(ge=1, le=MAX_ITERATE)


class CycleMultiplierRequest(PolynomialCoefficientRequest):
    """Compute the multiplier of a supplied exact rational cycle."""

    cycle: tuple[CanonicalRational, ...] = Field(
        min_length=1, max_length=MAX_ORBIT_STEPS
    )


class FiniteFieldMapRequest(StrictModel):
    """A canonical polynomial map over the prime field GF(p)."""

    polynomial_map: FinitePolynomialMap


class MapIterateResult(StrictModel):
    source_polynomial: RationalPolynomial
    n: int = Field(ge=0, le=MAX_ITERATE)
    polynomial: RationalPolynomial
    degree: int = Field(ge=0, le=MAX_ITERATE_DEGREE)

    @classmethod
    def _from_kernel(
        cls,
        *,
        source_polynomial: RationalPolynomial,
        n: int,
        polynomial: RationalPolynomial,
        degree: int,
    ) -> Self:
        return cls.model_construct(
            source_polynomial=source_polynomial,
            n=n,
            polynomial=polynomial,
            degree=degree,
        )


class OrbitRepeatEvidence(StrictModel):
    first_seen_index: int = Field(ge=0)
    repeated_at_index: int = Field(ge=1)
    preperiod: int = Field(ge=0)
    period: int = Field(ge=1)


class OrbitPrefixResult(StrictModel):
    source_polynomial: RationalPolynomial
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

    @classmethod
    def _from_kernel(
        cls,
        *,
        source_polynomial: RationalPolynomial,
        start: CanonicalRational,
        requested_steps: int,
        orbit: tuple[CanonicalRational, ...],
        termination: Literal[
            "REPEAT_FOUND", "STEP_BOUND_REACHED", "OUTPUT_BOUND_REACHED"
        ],
        repeat: OrbitRepeatEvidence | None,
    ) -> Self:
        found_repeat = termination == "REPEAT_FOUND"
        return cls.model_construct(
            source_polynomial=source_polynomial,
            start=start,
            orbit=orbit,
            requested_steps=requested_steps,
            computed_steps=len(orbit) - 1,
            termination=termination,
            repeat=repeat,
            eventual_behavior_complete=found_repeat,
            truncated=not found_repeat,
        )


class DynatomicPolynomialResult(StrictModel):
    source_polynomial: RationalPolynomial
    polynomial: RationalPolynomial
    degree: int = Field(ge=0, le=MAX_DYNATOMIC_DEGREE)
    n: int = Field(ge=1, le=MAX_ITERATE)

    @classmethod
    def _from_kernel(
        cls,
        *,
        source_polynomial: RationalPolynomial,
        n: int,
        polynomial: RationalPolynomial,
        degree: int,
    ) -> Self:
        return cls.model_construct(
            source_polynomial=source_polynomial,
            polynomial=polynomial,
            degree=degree,
            n=n,
        )


class CycleMultiplierResult(StrictModel):
    source_polynomial: RationalPolynomial
    multiplier: CanonicalRational
    cycle: tuple[CanonicalRational, ...] = Field(
        min_length=1, max_length=MAX_ORBIT_STEPS
    )
    period: int = Field(ge=1, le=MAX_ORBIT_STEPS)

    @classmethod
    def _from_kernel(
        cls,
        *,
        source_polynomial: RationalPolynomial,
        cycle: tuple[CanonicalRational, ...],
        multiplier: CanonicalRational,
    ) -> Self:
        """Build a result after the admitted cycle-multiplier kernel ran."""

        require_bounded_rational(
            multiplier,
            max_digits=MAX_POLYNOMIAL_OUTPUT_DIGITS,
            label="multiplier",
        )
        return cls.model_construct(
            source_polynomial=source_polynomial,
            multiplier=multiplier,
            cycle=cycle,
            period=len(cycle),
        )


class FiniteFieldMapResult(StrictModel):
    polynomial_map: FinitePolynomialMap
    edges: tuple[tuple[int, int], ...]
    cycles: tuple[tuple[int, ...], ...]
    tail_lengths: tuple[int, ...]

    @classmethod
    def _from_kernel(
        cls,
        *,
        polynomial_map: FinitePolynomialMap,
        edges: tuple[tuple[int, int], ...],
        cycles: tuple[tuple[int, ...], ...],
        tail_lengths: tuple[int, ...],
    ) -> Self:
        """Build a result after complete graph enumeration established it."""

        return cls.model_construct(
            polynomial_map=polynomial_map,
            edges=edges,
            cycles=cycles,
            tail_lengths=tail_lengths,
        )


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
