"""Public contracts for complete rational-polynomial box enclosures."""

from __future__ import annotations

from dataclasses import dataclass
from math import gcd
from typing import Self

from pydantic import ConfigDict, model_validator
from pydantic_core import PydanticCustomError

from jacobian._digest import Sha256Digest
from jacobian._exact import MAX_CANONICAL_RATIONAL_DIGITS, require_bounded_rational
from jacobian._models import StrictModel
from jacobian.canonical import (
    CanonicalLimits,
    encode_strict_json,
    format_canonical_integer,
    sha256_digest,
)
from jacobian.math.intervals import ClosedRationalInterval, RationalBox
from jacobian.math.polynomials.intervals._kernel import term_is_zero_on_box
from jacobian.math.polynomials.values import (
    MAX_POLYNOMIAL_EXPONENT,
    MAX_POLYNOMIAL_TERMS,
    MAX_POLYNOMIAL_VARIABLES,
    RationalPolynomial,
    RationalPolynomialTerm,
    require_polynomial_budget,
)

MAX_BOX_ENCLOSURE_TERMS = MAX_POLYNOMIAL_TERMS
MAX_BOX_ENCLOSURE_PER_VARIABLE_DEGREE = MAX_POLYNOMIAL_EXPONENT
MAX_BOX_ENCLOSURE_TOTAL_DEGREE = (
    MAX_POLYNOMIAL_VARIABLES * MAX_BOX_ENCLOSURE_PER_VARIABLE_DEGREE
)
MAX_BOX_ENCLOSURE_TERM_AXIS_PAIRS = MAX_BOX_ENCLOSURE_TERMS * MAX_POLYNOMIAL_VARIABLES
MAX_BOX_ENCLOSURE_COEFFICIENT_DIGITS = MAX_CANONICAL_RATIONAL_DIGITS
MAX_BOX_ENCLOSURE_ENDPOINT_DIGITS = MAX_CANONICAL_RATIONAL_DIGITS
# Shared interval values validate before this operation's narrower source
# preflight. Ordering two maximum-size canonical rationals may therefore
# cross-multiply one numerator and the opposite denominator. The intermediate
# ceiling covers that value-level validation as well as Fraction addition and
# comparison in the kernel; source-derived growth is checked before any
# polynomial endpoint is exponentiated.
MAX_BOX_ENCLOSURE_INTERMEDIATE_DIGITS = 2 * MAX_CANONICAL_RATIONAL_DIGITS
MAX_BOX_ENCLOSURE_RESULT_DIGITS = MAX_CANONICAL_RATIONAL_DIGITS
MAX_BOX_ENCLOSURE_RESULT_BYTES = CanonicalLimits().max_output_bytes

BOX_ENCLOSURE_ADMISSION_SUMMARY = (
    f"Bounds: {MAX_BOX_ENCLOSURE_TERMS:,} terms; degree "
    f"{MAX_BOX_ENCLOSURE_PER_VARIABLE_DEGREE} per variable and "
    f"{MAX_BOX_ENCLOSURE_TOTAL_DEGREE} total; "
    f"{MAX_BOX_ENCLOSURE_COEFFICIENT_DIGITS}-digit coefficient and "
    f"{MAX_BOX_ENCLOSURE_ENDPOINT_DIGITS}-digit input-endpoint components; "
    f"{MAX_BOX_ENCLOSURE_TERM_AXIS_PAIRS:,} term-axis pairs; "
    f"{MAX_BOX_ENCLOSURE_INTERMEDIATE_DIGITS:,}-digit intermediate components; "
    f"{MAX_BOX_ENCLOSURE_RESULT_DIGITS:,}-digit result components; "
    f"{MAX_BOX_ENCLOSURE_RESULT_BYTES:,}-byte canonical retained-source result."
)

_RESULT_ENVELOPE_RESERVE_BYTES = 512


def _validation_error(message: str) -> PydanticCustomError:
    return PydanticCustomError("polynomial.box_invariant", message)


@dataclass(frozen=True, slots=True)
class _GrowthEstimate:
    result_numerator_digits: int
    result_denominator_digits: int
    intermediate_digits: int
    estimated_result_bytes: int


def _integer_digits(value: int) -> int:
    return len(format_canonical_integer(abs(value)))


def _component_contribution(value: str) -> int:
    magnitude = value.lstrip("-")
    return 0 if magnitude in {"0", "1"} else len(magnitude)


def _product_digit_bound(contributions: list[int]) -> int:
    return max(1, sum(contributions))


def _ceil_log10_count(count: int) -> int:
    return 0 if count <= 1 else len(str(count - 1))


def _coefficient_lcm_contribution(
    terms: tuple[RationalPolynomialTerm, ...],
) -> int:
    common_denominator = 1
    for term in terms:
        denominator = term.coefficient.as_integer_ratio()[1]
        factor = denominator // gcd(common_denominator, denominator)
        if factor == 1:
            continue
        predicted_product_digits = _integer_digits(
            common_denominator
        ) + _integer_digits(factor)
        if predicted_product_digits > MAX_BOX_ENCLOSURE_INTERMEDIATE_DIGITS:
            raise _validation_error(
                "polynomial-box coefficient denominator LCM exceeds the "
                f"{MAX_BOX_ENCLOSURE_INTERMEDIATE_DIGITS}-digit intermediate bound"
            )
        common_denominator *= factor
        if _integer_digits(common_denominator) > MAX_BOX_ENCLOSURE_RESULT_DIGITS:
            raise _validation_error(
                "polynomial-box common denominator exceeds the "
                f"{MAX_BOX_ENCLOSURE_RESULT_DIGITS}-digit result bound"
            )
    return 0 if common_denominator == 1 else _integer_digits(common_denominator)


def _endpoint_lcm_contribution(interval: ClosedRationalInterval) -> int:
    lower_denominator = interval.lower.as_integer_ratio()[1]
    upper_denominator = interval.upper.as_integer_ratio()[1]
    factor = lower_denominator // gcd(lower_denominator, upper_denominator)
    predicted_product_digits = _integer_digits(factor) + _integer_digits(
        upper_denominator
    )
    if predicted_product_digits > MAX_BOX_ENCLOSURE_INTERMEDIATE_DIGITS:
        raise _validation_error(
            "polynomial-box endpoint denominator LCM exceeds the "
            f"{MAX_BOX_ENCLOSURE_INTERMEDIATE_DIGITS}-digit intermediate bound"
        )
    common = factor * upper_denominator
    return 0 if common == 1 else _integer_digits(common)


def _source_payload(
    polynomial: RationalPolynomial,
    box: RationalBox,
) -> dict[str, object]:
    return {
        "polynomial": polynomial.model_dump(mode="json"),
        "box": box.model_dump(mode="json"),
    }


def polynomial_box_source_digest(
    polynomial: RationalPolynomial,
    box: RationalBox,
) -> Sha256Digest:
    """Return the strict-wire digest binding one polynomial and ordered box."""

    return sha256_digest(encode_strict_json(_source_payload(polynomial, box)))


def _term_component_bounds(
    term: RationalPolynomialTerm,
    box: RationalBox,
) -> tuple[int, int]:
    numerator_contributions = [_component_contribution(term.coefficient.num)]
    denominator_contributions = [_component_contribution(term.coefficient.den)]
    for exponent, interval in zip(term.exponents, box.intervals, strict=True):
        if exponent == 0:
            continue
        numerator_contributions.append(
            exponent
            * max(
                _component_contribution(interval.lower.num),
                _component_contribution(interval.upper.num),
            )
        )
        denominator_contributions.append(
            exponent
            * max(
                _component_contribution(interval.lower.den),
                _component_contribution(interval.upper.den),
            )
        )
    return (
        _product_digit_bound(numerator_contributions),
        _product_digit_bound(denominator_contributions),
    )


def _estimate_growth(
    polynomial: RationalPolynomial,
    box: RationalBox,
) -> _GrowthEstimate:
    source_bytes = len(encode_strict_json(_source_payload(polynomial, box)))
    effective_terms = tuple(
        term
        for term in polynomial.polynomial.terms
        if not term_is_zero_on_box(term, box)
    )
    if not effective_terms:
        return _GrowthEstimate(
            result_numerator_digits=1,
            result_denominator_digits=1,
            intermediate_digits=1,
            estimated_result_bytes=source_bytes + _RESULT_ENVELOPE_RESERVE_BYTES,
        )

    term_bounds = tuple(_term_component_bounds(term, box) for term in effective_terms)
    maximum_term_numerator_digits = max(bound[0] for bound in term_bounds)
    maximum_term_denominator_digits = max(bound[1] for bound in term_bounds)

    maximum_axis_degrees = tuple(
        max(term.exponents[index] for term in effective_terms)
        for index in range(len(box.variables))
    )
    common_denominator_contribution = _coefficient_lcm_contribution(effective_terms)
    common_denominator_contribution += sum(
        degree * _endpoint_lcm_contribution(interval)
        for degree, interval in zip(
            maximum_axis_degrees,
            box.intervals,
            strict=True,
        )
    )
    result_denominator_digits = max(1, common_denominator_contribution)

    if len(effective_terms) == 1:
        result_numerator_digits = maximum_term_numerator_digits
        result_denominator_digits = maximum_term_denominator_digits
    else:
        result_numerator_digits = (
            maximum_term_numerator_digits
            + result_denominator_digits
            + _ceil_log10_count(len(effective_terms))
        )

    comparison_digits = maximum_term_numerator_digits + maximum_term_denominator_digits
    output_comparison_digits = result_numerator_digits + result_denominator_digits
    if len(effective_terms) == 1:
        intermediate_digits = max(comparison_digits, output_comparison_digits)
    else:
        addition_digits = max(
            result_numerator_digits + maximum_term_denominator_digits + 1,
            maximum_term_numerator_digits + result_denominator_digits + 1,
            result_denominator_digits + maximum_term_denominator_digits,
        )
        intermediate_digits = max(
            comparison_digits,
            addition_digits,
            output_comparison_digits,
        )

    estimated_result_bytes = (
        source_bytes
        + 2 * (result_numerator_digits + result_denominator_digits + 64)
        + _RESULT_ENVELOPE_RESERVE_BYTES
    )
    return _GrowthEstimate(
        result_numerator_digits=result_numerator_digits,
        result_denominator_digits=result_denominator_digits,
        intermediate_digits=intermediate_digits,
        estimated_result_bytes=estimated_result_bytes,
    )


def _require_enclosure_preflight(
    polynomial: RationalPolynomial,
    box: RationalBox,
) -> None:
    if polynomial.domain != box.domain or polynomial.variables != box.variables:
        raise _validation_error(
            "polynomial box must use the polynomial's complete ordered axis and QQ parent"
        )
    require_polynomial_budget(
        polynomial,
        maximum_terms=MAX_BOX_ENCLOSURE_TERMS,
        maximum_exponent=MAX_BOX_ENCLOSURE_PER_VARIABLE_DEGREE,
        maximum_coefficient_digits=MAX_BOX_ENCLOSURE_COEFFICIENT_DIGITS,
        label="polynomial-box polynomial",
    )
    for variable, interval in zip(box.variables, box.intervals, strict=True):
        for endpoint in (interval.lower, interval.upper):
            require_bounded_rational(
                endpoint,
                max_digits=MAX_BOX_ENCLOSURE_ENDPOINT_DIGITS,
                label=f"polynomial-box {variable} endpoint",
            )

    growth = _estimate_growth(polynomial, box)
    if (
        growth.result_numerator_digits > MAX_BOX_ENCLOSURE_RESULT_DIGITS
        or growth.result_denominator_digits > MAX_BOX_ENCLOSURE_RESULT_DIGITS
    ):
        raise _validation_error(
            "polynomial-box enclosure exceeds the "
            f"{MAX_BOX_ENCLOSURE_RESULT_DIGITS}-digit exact-result bound"
        )
    if growth.intermediate_digits > MAX_BOX_ENCLOSURE_INTERMEDIATE_DIGITS:
        raise _validation_error(
            "polynomial-box enclosure exceeds the "
            f"{MAX_BOX_ENCLOSURE_INTERMEDIATE_DIGITS}-digit intermediate bound"
        )
    if growth.estimated_result_bytes > MAX_BOX_ENCLOSURE_RESULT_BYTES:
        raise _validation_error(
            "polynomial-box enclosure result would exceed the "
            f"{MAX_BOX_ENCLOSURE_RESULT_BYTES}-byte canonical output bound"
        )


class PolynomialBoxEnclosureRequest(StrictModel):
    """Enclose one scalar ``QQ`` polynomial on one complete closed rational box."""

    model_config = ConfigDict(
        json_schema_extra={
            "description": (
                "Enclose one canonical scalar QQ polynomial on a complete closed "
                "rational box. The box must use exactly the polynomial's ordered "
                "variables. The returned exact rational interval is deterministic "
                "and sound but need not be the exact range. "
                f"{BOX_ENCLOSURE_ADMISSION_SUMMARY}"
            )
        }
    )

    polynomial: RationalPolynomial
    box: RationalBox

    @model_validator(mode="after")
    def require_complete_bounded_source(self) -> Self:
        _require_enclosure_preflight(self.polynomial, self.box)
        return self


class PolynomialBoxEnclosureResult(StrictModel):
    """An exact enclosure bound to its source polynomial and box."""

    polynomial: RationalPolynomial
    box: RationalBox
    source_digest: Sha256Digest
    enclosure: ClosedRationalInterval

    @model_validator(mode="after")
    def require_source_binding(self) -> Self:
        if self.source_digest != polynomial_box_source_digest(
            self.polynomial,
            self.box,
        ):
            raise _validation_error(
                "source digest does not bind the polynomial and box"
            )
        return self

    @classmethod
    def _from_kernel(
        cls,
        request: PolynomialBoxEnclosureRequest,
        *,
        enclosure: ClosedRationalInterval,
    ) -> Self:
        """Build one result after the admitted kernel established its enclosure."""

        return cls.model_construct(
            polynomial=request.polynomial,
            box=request.box,
            source_digest=polynomial_box_source_digest(
                request.polynomial,
                request.box,
            ),
            enclosure=enclosure,
        )


__all__ = [
    "BOX_ENCLOSURE_ADMISSION_SUMMARY",
    "MAX_BOX_ENCLOSURE_COEFFICIENT_DIGITS",
    "MAX_BOX_ENCLOSURE_ENDPOINT_DIGITS",
    "MAX_BOX_ENCLOSURE_INTERMEDIATE_DIGITS",
    "MAX_BOX_ENCLOSURE_PER_VARIABLE_DEGREE",
    "MAX_BOX_ENCLOSURE_RESULT_BYTES",
    "MAX_BOX_ENCLOSURE_RESULT_DIGITS",
    "MAX_BOX_ENCLOSURE_TERMS",
    "MAX_BOX_ENCLOSURE_TERM_AXIS_PAIRS",
    "MAX_BOX_ENCLOSURE_TOTAL_DEGREE",
    "PolynomialBoxEnclosureRequest",
    "PolynomialBoxEnclosureResult",
    "polynomial_box_source_digest",
]
