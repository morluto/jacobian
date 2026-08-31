"""Typed wire contracts for polynomial root isolation and algebraic number comparison."""

from __future__ import annotations

from collections.abc import Mapping
from itertools import pairwise
from typing import Annotated, Any, Self

from pydantic import BeforeValidator, Field, WithJsonSchema, model_validator
from pydantic_core import PydanticCustomError

from jacobian._exact import CanonicalInteger, CanonicalRational
from jacobian._models import StrictModel, canonicalize_json_containers
from jacobian.canonical import format_canonical_integer, parse_canonical_integer
from jacobian.math.number_theory.algebraic_numbers.real import (
    MAX_REAL_ALGEBRAIC_COMPARISON_DEGREE,
    MAX_REAL_ALGEBRAIC_DEGREE,
    RealAlgebraicValue,
)

MAX_ROOT_ISOLATION_DEGREE = 8

# A degree-eight source with 996-digit coefficients has every primitive
# irreducible factor within its established 1,000-digit result envelope: the
# Landau--Mignotte bound contributes fewer than four decimal digits.
MAX_ROOT_ISOLATION_SOURCE_COEFFICIENT_DIGITS = 996
_MAX_ROOT_ISOLATION_SOURCE_COEFFICIENT_MAGNITUDE = (
    10**MAX_ROOT_ISOLATION_SOURCE_COEFFICIENT_DIGITS
)


def _validation_error(reason: str, message: str) -> PydanticCustomError:
    return PydanticCustomError(f"root_isolation.{reason}", message)


def _require_raw_comparison_degree(value: Any) -> Any:
    polynomial = (
        value.polynomial
        if isinstance(value, RealAlgebraicValue)
        else value.get("polynomial")
        if isinstance(value, Mapping)
        else None
    )
    if isinstance(polynomial, (list, tuple)) and len(polynomial) > (
        MAX_REAL_ALGEBRAIC_DEGREE + 1
    ):
        raise _validation_error(
            "comparison_degree_bound",
            "exact algebraic comparison admits degree at most "
            f"{MAX_REAL_ALGEBRAIC_DEGREE}",
        )
    if isinstance(value, Mapping):
        return canonicalize_json_containers(value)
    return value


def _comparison_value_schema() -> dict[str, Any]:
    schema = RealAlgebraicValue.model_json_schema()
    schema["properties"]["polynomial"]["maxItems"] = MAX_REAL_ALGEBRAIC_DEGREE + 1
    return schema


_ComparisonRealAlgebraicValue = Annotated[
    RealAlgebraicValue,
    BeforeValidator(_require_raw_comparison_degree),
    WithJsonSchema(_comparison_value_schema()),
]


class UnivariatePolynomialRequest(StrictModel):
    coefficients_descending: tuple[CanonicalRational, ...] = Field(
        min_length=2,
        max_length=MAX_ROOT_ISOLATION_DEGREE + 1,
        description=(
            "Canonical rational coefficients in descending degree. Their "
            "primitive integer normalization must use at most "
            f"{MAX_ROOT_ISOLATION_SOURCE_COEFFICIENT_DIGITS} decimal digits."
        ),
    )

    @model_validator(mode="before")
    @classmethod
    def require_raw_factorization_envelope(cls, value: Any) -> Any:
        """Reject oversized coefficients before nested rational parsing."""

        if not isinstance(value, Mapping):
            return value
        value = canonicalize_json_containers(value)
        coefficients = value.get("coefficients_descending")
        if not isinstance(coefficients, (list, tuple)):
            return value
        if len(coefficients) > MAX_ROOT_ISOLATION_DEGREE + 1:
            raise _validation_error(
                "source_degree_bound",
                f"root isolation admits degree at most {MAX_ROOT_ISOLATION_DEGREE}",
            )
        for coefficient in coefficients:
            if not isinstance(coefficient, Mapping):
                continue
            for component in ("num", "den"):
                raw_component = coefficient.get(component)
                if isinstance(raw_component, str) and len(raw_component.lstrip("-")) > (
                    MAX_ROOT_ISOLATION_SOURCE_COEFFICIENT_DIGITS
                ):
                    raise _validation_error(
                        "source_coefficient_bound",
                        "root isolation coefficients exceed the "
                        f"{MAX_ROOT_ISOLATION_SOURCE_COEFFICIENT_DIGITS}-digit "
                        "factorization envelope",
                    )
        return value

    @model_validator(mode="after")
    def require_nonzero_leading(self) -> Self:
        if self.coefficients_descending[0] == CanonicalRational(num="0", den="1"):
            raise _validation_error(
                "leading_coefficient_zero", "leading coefficient must be nonzero"
            )
        self.normalized_integer_coefficients()
        return self

    def normalized_integer_coefficients(self) -> tuple[int, ...]:
        """Return the canonical primitive positive-leading integer source."""

        from math import gcd, lcm

        numerators = tuple(
            parse_canonical_integer(value.num) for value in self.coefficients_descending
        )
        denominators = tuple(
            parse_canonical_integer(value.den) for value in self.coefficients_descending
        )
        common_denominator = lcm(*denominators)
        coefficients = tuple(
            numerator * (common_denominator // denominator)
            for numerator, denominator in zip(numerators, denominators, strict=True)
        )
        content = 0
        for coefficient in coefficients:
            content = gcd(content, abs(coefficient))
        normalized = tuple(coefficient // content for coefficient in coefficients)
        if normalized[0] < 0:
            normalized = tuple(-coefficient for coefficient in normalized)
        if any(
            abs(coefficient) >= _MAX_ROOT_ISOLATION_SOURCE_COEFFICIENT_MAGNITUDE
            for coefficient in normalized
        ):
            raise _validation_error(
                "source_coefficient_bound",
                "primitive integer source coefficients exceed the "
                f"{MAX_ROOT_ISOLATION_SOURCE_COEFFICIENT_DIGITS}-digit "
                "factorization envelope",
            )
        return normalized


class RootIsolationEntry(StrictModel):
    """One distinct source root with an exact composable algebraic identity."""

    isolating_interval: tuple[CanonicalRational, CanonicalRational]
    multiplicity: int = Field(ge=1, le=MAX_ROOT_ISOLATION_DEGREE)
    algebraic_value: RealAlgebraicValue

    @model_validator(mode="after")
    def require_ordered_interval(self) -> Self:
        lower, upper = self.isolating_interval
        if lower.as_fraction() > upper.as_fraction():
            raise _validation_error(
                "interval_bounds_invalid",
                "isolating interval must have lower <= upper",
            )
        if len(self.algebraic_value.polynomial) > MAX_ROOT_ISOLATION_DEGREE + 1:
            raise _validation_error(
                "algebraic_value_degree",
                "root-isolation algebraic values must have degree at most "
                f"{MAX_ROOT_ISOLATION_DEGREE}",
            )
        return self


class RootIsolationResult(StrictModel):
    """Source-bound, ordered real roots with canonical algebraic identities."""

    source_coefficients_descending: tuple[CanonicalInteger, ...] = Field(
        min_length=2, max_length=MAX_ROOT_ISOLATION_DEGREE + 1
    )
    roots: tuple[RootIsolationEntry, ...] = Field(max_length=MAX_ROOT_ISOLATION_DEGREE)

    @model_validator(mode="after")
    def require_structural_order(self) -> Self:
        if parse_canonical_integer(self.source_coefficients_descending[0]) <= 0:
            raise _validation_error(
                "source_leading_coefficient",
                "source polynomial must have positive leading coefficient",
            )
        intervals = tuple(root.isolating_interval for root in self.roots)
        if any(
            left[1].as_fraction() >= right[0].as_fraction()
            for left, right in pairwise(intervals)
        ):
            raise _validation_error(
                "intervals_not_disjoint",
                "isolating intervals must be strictly ordered and pairwise disjoint",
            )
        return self

    @classmethod
    def _from_kernel(
        cls,
        *,
        source_coefficients_descending: tuple[int, ...],
        roots: tuple[RootIsolationEntry, ...],
    ) -> Self:
        """Construct after the factor/isolation kernel established each identity."""

        return cls.model_construct(
            source_coefficients_descending=tuple(
                format_canonical_integer(coefficient)
                for coefficient in source_coefficients_descending
            ),
            roots=roots,
        )


class AlgebraicCompareRequest(StrictModel):
    left: _ComparisonRealAlgebraicValue
    right: _ComparisonRealAlgebraicValue

    @model_validator(mode="before")
    @classmethod
    def require_raw_pair_comparison_envelope(cls, data: Any) -> Any:
        if not isinstance(data, Mapping):
            return data
        left = data.get("left")
        right = data.get("right")
        left_polynomial = left.get("polynomial") if isinstance(left, Mapping) else None
        right_polynomial = (
            right.get("polynomial") if isinstance(right, Mapping) else None
        )
        if (
            isinstance(left_polynomial, (list, tuple))
            and isinstance(right_polynomial, (list, tuple))
            and left_polynomial != right_polynomial
            and (
                len(left_polynomial) - 1 > MAX_REAL_ALGEBRAIC_COMPARISON_DEGREE
                or len(right_polynomial) - 1 > MAX_REAL_ALGEBRAIC_COMPARISON_DEGREE
            )
        ):
            raise _validation_error(
                "comparison_degree_bound",
                "distinct-polynomial comparison admits degree at most "
                f"{MAX_REAL_ALGEBRAIC_COMPARISON_DEGREE}",
            )
        return data

    @model_validator(mode="after")
    def require_pair_comparison_envelope(self) -> Self:
        if self.left.polynomial != self.right.polynomial and (
            len(self.left.polynomial) - 1 > MAX_REAL_ALGEBRAIC_COMPARISON_DEGREE
            or len(self.right.polynomial) - 1 > MAX_REAL_ALGEBRAIC_COMPARISON_DEGREE
        ):
            raise _validation_error(
                "comparison_degree_bound",
                "distinct-polynomial comparison admits degree at most "
                f"{MAX_REAL_ALGEBRAIC_COMPARISON_DEGREE}",
            )
        return self


__all__ = [
    "MAX_ROOT_ISOLATION_DEGREE",
    "MAX_ROOT_ISOLATION_SOURCE_COEFFICIENT_DIGITS",
    "AlgebraicCompareRequest",
    "RootIsolationEntry",
    "RootIsolationResult",
    "UnivariatePolynomialRequest",
]
