"""Typed wire contracts for polynomial root isolation and algebraic number comparison."""

from __future__ import annotations

from collections.abc import Mapping
from itertools import pairwise
from typing import Any, Literal, Self

from pydantic import Field, model_validator
from pydantic_core import PydanticCustomError

from jacobian._exact import CanonicalInteger, CanonicalRational
from jacobian._models import StrictModel, canonicalize_json_containers
from jacobian.canonical import format_canonical_integer, parse_canonical_integer
from jacobian.math.real_algebraic import (
    MAX_REAL_ALGEBRAIC_COEFFICIENT_DIGITS,
    MAX_REAL_ALGEBRAIC_DEGREE,
    RealAlgebraicValue,
)

# A degree-eight source with 996-digit coefficients has every primitive
# irreducible factor within the shared 1,000-digit algebraic-value envelope:
# the Landau--Mignotte bound contributes fewer than four decimal digits.
MAX_ROOT_ISOLATION_SOURCE_COEFFICIENT_DIGITS = MAX_REAL_ALGEBRAIC_COEFFICIENT_DIGITS - 4
_MAX_ROOT_ISOLATION_SOURCE_COEFFICIENT_MAGNITUDE = (
    10**MAX_ROOT_ISOLATION_SOURCE_COEFFICIENT_DIGITS
)


def _validation_error(reason: str, message: str) -> PydanticCustomError:
    return PydanticCustomError(f"root_isolation.{reason}", message)


class UnivariatePolynomialRequest(StrictModel):
    coefficients_descending: tuple[CanonicalRational, ...] = Field(
        min_length=2,
        max_length=MAX_REAL_ALGEBRAIC_DEGREE + 1,
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
        if len(coefficients) > MAX_REAL_ALGEBRAIC_DEGREE + 1:
            raise _validation_error(
                "source_degree_bound",
                f"root isolation admits degree at most {MAX_REAL_ALGEBRAIC_DEGREE}",
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
    multiplicity: int = Field(ge=1, le=MAX_REAL_ALGEBRAIC_DEGREE)
    algebraic_value: RealAlgebraicValue

    @model_validator(mode="after")
    def require_ordered_interval(self) -> Self:
        lower, upper = self.isolating_interval
        if lower.as_fraction() > upper.as_fraction():
            raise _validation_error(
                "interval_bounds_invalid",
                "isolating interval must have lower <= upper",
            )
        return self


class RootIsolationResult(StrictModel):
    """Source-bound, ordered real roots with canonical algebraic identities."""

    source_coefficients_descending: tuple[CanonicalInteger, ...] = Field(
        min_length=2, max_length=MAX_REAL_ALGEBRAIC_DEGREE + 1
    )
    roots: tuple[RootIsolationEntry, ...] = Field(max_length=MAX_REAL_ALGEBRAIC_DEGREE)
    convention: Literal["SYMPY_REAL_ROOTS"] = "SYMPY_REAL_ROOTS"

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
    left: RealAlgebraicValue
    right: RealAlgebraicValue


__all__ = [
    "MAX_ROOT_ISOLATION_SOURCE_COEFFICIENT_DIGITS",
    "AlgebraicCompareRequest",
    "RootIsolationEntry",
    "RootIsolationResult",
    "UnivariatePolynomialRequest",
]
