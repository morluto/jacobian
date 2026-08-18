"""Provider-independent values for exact bounded finite graphical models."""

from __future__ import annotations

from fractions import Fraction
from typing import Annotated, Self

from pydantic import Field, model_validator

from jacobian._models import StrictModel

MAX_MODEL_VARS = 16
MAX_VAR_DOMAIN = 32
MAX_FACTOR_TABLE_SIZE = 4_096
MAX_FACTOR_COUNT = 64
MAX_RATIONAL_DIGITS = 256

DomainSize = Annotated[int, Field(ge=1, le=MAX_VAR_DOMAIN)]
Variable = Annotated[int, Field(ge=0, lt=MAX_MODEL_VARS)]


def parse_canonical_rational(value: str) -> Fraction:
    """Parse one bounded reduced rational string."""

    if len(value) > 2 * MAX_RATIONAL_DIGITS + 2:
        raise ValueError("factor entry exceeds the rational digit bound")
    try:
        parsed = Fraction(value)
    except (ValueError, ZeroDivisionError):
        raise ValueError("factor entries must be canonical rationals") from None
    if str(parsed) != value:
        raise ValueError("factor entries must be reduced canonical rationals")
    if (
        len(str(abs(parsed.numerator))) > MAX_RATIONAL_DIGITS
        or len(str(parsed.denominator)) > MAX_RATIONAL_DIGITS
    ):
        raise ValueError("factor entry exceeds the rational digit bound")
    return parsed


def scope_size(variables: tuple[int, ...], domain_sizes: tuple[int, ...]) -> int:
    """Return a bounded factor-table size for a validated variable scope."""

    size = 1
    for variable in variables:
        if variable >= len(domain_sizes):
            raise ValueError("variable index exceeds domain_sizes")
        size *= domain_sizes[variable]
        if size > MAX_FACTOR_TABLE_SIZE:
            raise ValueError("factor table exceeds the supported size bound")
    return size


class Factor(StrictModel):
    """An exact factor indexed lexicographically by its ordered variable scope.

    The empty scope represents a scalar and therefore has exactly one table entry.
    ``domain_sizes`` describes the complete shared model domain.
    """

    variables: tuple[Variable, ...] = Field(max_length=MAX_MODEL_VARS)
    domain_sizes: tuple[DomainSize, ...] = Field(
        min_length=1, max_length=MAX_MODEL_VARS
    )
    table: tuple[str, ...] = Field(min_length=1, max_length=MAX_FACTOR_TABLE_SIZE)

    @model_validator(mode="after")
    def require_valid_factor(self) -> Self:
        if len(set(self.variables)) != len(self.variables):
            raise ValueError("factor variables must be distinct")
        expected_size = scope_size(self.variables, self.domain_sizes)
        if len(self.table) != expected_size:
            raise ValueError(
                f"table size {len(self.table)} does not match expected {expected_size}"
            )
        for value in self.table:
            parse_canonical_rational(value)
        return self


__all__ = [
    "MAX_FACTOR_COUNT",
    "MAX_FACTOR_TABLE_SIZE",
    "MAX_MODEL_VARS",
    "MAX_RATIONAL_DIGITS",
    "MAX_VAR_DOMAIN",
    "Factor",
    "parse_canonical_rational",
    "scope_size",
]
