"""Typed wire contracts for polynomial root isolation and algebraic number comparison."""

from __future__ import annotations

from collections.abc import Mapping
from itertools import pairwise
from typing import Annotated, Any, Self

from pydantic import BeforeValidator, Field, WithJsonSchema, model_validator
from pydantic_core import PydanticCustomError

from jacobian._exact import CanonicalRational
from jacobian._models import StrictModel, canonicalize_json_containers
from jacobian.canonical import format_canonical_integer
from jacobian.math.number_theory.algebraic_numbers.real import (
    MAX_REAL_ALGEBRAIC_COMPARISON_DEGREE,
    MAX_REAL_ALGEBRAIC_DEGREE,
    RealAlgebraicValue,
)
from jacobian.math.polynomials.values import RationalPolynomial

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


def _root_isolation_value_schema() -> dict[str, Any]:
    schema = RealAlgebraicValue.model_json_schema()
    schema["properties"]["polynomial"]["maxItems"] = MAX_ROOT_ISOLATION_DEGREE + 1
    return schema


_ComparisonRealAlgebraicValue = Annotated[
    RealAlgebraicValue,
    BeforeValidator(_require_raw_comparison_degree),
    WithJsonSchema(_comparison_value_schema()),
]

_RootIsolationRealAlgebraicValue = Annotated[
    RealAlgebraicValue,
    WithJsonSchema(_root_isolation_value_schema()),
]


class UnivariatePolynomialRequest(StrictModel):
    polynomial: RationalPolynomial = Field(
        description="One nonconstant univariate QQ polynomial of degree at most 8; coefficient numerators and denominators have at most 996 digits."
    )

    @model_validator(mode="before")
    @classmethod
    def require_raw_factorization_envelope(cls, value: Any) -> Any:
        value = canonicalize_json_containers(value)
        if not isinstance(value, Mapping):
            return value
        polynomial = value.get("polynomial")
        body = polynomial.get("polynomial") if isinstance(polynomial, Mapping) else None
        terms = body.get("terms") if isinstance(body, Mapping) else None
        if not isinstance(terms, (list, tuple)):
            return value
        if len(terms) > MAX_ROOT_ISOLATION_DEGREE + 1:
            raise _validation_error(
                "source_degree_bound", "root isolation admits degree at most 8"
            )
        for term in terms:
            coefficient = term.get("coefficient") if isinstance(term, Mapping) else None
            if isinstance(coefficient, Mapping) and any(
                isinstance(coefficient.get(component), str)
                and len(coefficient[component].lstrip("-"))
                > MAX_ROOT_ISOLATION_SOURCE_COEFFICIENT_DIGITS
                for component in ("num", "den")
            ):
                raise _validation_error(
                    "source_coefficient_bound",
                    "root isolation coefficients exceed the 996-digit factorization envelope",
                )
        return value

    @model_validator(mode="after")
    def require_isolation_shape(self) -> Self:
        terms = self.polynomial.polynomial.terms
        if (
            len(self.polynomial.variables) != 1
            or not terms
            or not 1 <= terms[0].exponents[0] <= MAX_ROOT_ISOLATION_DEGREE
        ):
            raise _validation_error(
                "source_degree_bound",
                "root isolation requires one variable and degree at most 8 (positive)",
            )
        if any(
            len(format_canonical_integer(abs(component)))
            > MAX_ROOT_ISOLATION_SOURCE_COEFFICIENT_DIGITS
            for term in terms
            for component in (term.coefficient.num, term.coefficient.den)
        ):
            raise _validation_error(
                "source_coefficient_bound",
                "root isolation coefficients exceed the 996-digit factorization envelope",
            )
        return self

    @property
    def coefficients_descending(self) -> tuple[CanonicalRational, ...]:
        terms = self.polynomial.polynomial.terms
        coefficients = [CanonicalRational(num=0, den=1)] * (terms[0].exponents[0] + 1)
        for term in terms:
            coefficients[-1 - term.exponents[0]] = term.coefficient
        return tuple(coefficients)

    def normalized_integer_coefficients(self) -> tuple[int, ...]:
        """Return the canonical primitive positive-leading integer source."""

        from math import gcd, lcm

        numerators = tuple(value.num for value in self.coefficients_descending)
        denominators = tuple(value.den for value in self.coefficients_descending)
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
    algebraic_value: _RootIsolationRealAlgebraicValue

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

    source_polynomial: RationalPolynomial
    roots: tuple[RootIsolationEntry, ...] = Field(max_length=MAX_ROOT_ISOLATION_DEGREE)

    @model_validator(mode="before")
    @classmethod
    def require_raw_root_degree_envelope(cls, data: Any) -> Any:
        """Reject oversized worker roots before nested algebraic recognition."""

        if not isinstance(data, Mapping):
            return data
        roots = data.get("roots")
        if not isinstance(roots, (list, tuple)):
            return data
        if len(roots) > MAX_ROOT_ISOLATION_DEGREE:
            raise _validation_error(
                "root_count_bound",
                f"root isolation admits at most {MAX_ROOT_ISOLATION_DEGREE} roots",
            )
        for root in roots:
            if not isinstance(root, Mapping):
                continue
            algebraic_value = root.get("algebraic_value")
            if not isinstance(algebraic_value, Mapping):
                continue
            polynomial = algebraic_value.get("polynomial")
            if isinstance(polynomial, (list, tuple)) and len(polynomial) > (
                MAX_ROOT_ISOLATION_DEGREE + 1
            ):
                raise _validation_error(
                    "algebraic_value_degree",
                    "root-isolation algebraic values must have degree at most "
                    f"{MAX_ROOT_ISOLATION_DEGREE}",
                )
        return canonicalize_json_containers(data)

    @model_validator(mode="after")
    def require_structural_order(self) -> Self:
        terms = self.source_polynomial.polynomial.terms
        if (
            len(self.source_polynomial.variables) != 1
            or not terms
            or not 1 <= terms[0].exponents[0] <= MAX_ROOT_ISOLATION_DEGREE
        ):
            raise _validation_error(
                "source_degree_bound",
                "root isolation requires one variable and degree at most 8 (positive)",
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
        source_polynomial: RationalPolynomial,
        roots: tuple[RootIsolationEntry, ...],
    ) -> Self:
        """Construct after the factor/isolation kernel established each identity."""

        return cls.model_construct(
            source_polynomial=source_polynomial,
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
        return canonicalize_json_containers(data)

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
