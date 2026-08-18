"""Typed wire contracts for arithmetic dynamics operations."""

from __future__ import annotations

from typing import Self

from pydantic import Field, model_validator

from jacobian._models import StrictModel

MAX_COEFF = 10_000
MAX_DEGREE = 30
MAX_ITERATE = 20
MAX_ORBIT = 1_000
MAX_FIELD_PRIME = 10_000


class PolynomialMapRequest(StrictModel):
    """A polynomial dynamical system f: A^1 -> A^1 over QQ."""

    coefficients: tuple[str, ...] = Field(min_length=1, max_length=MAX_DEGREE + 1)

    @model_validator(mode="after")
    def validate_coefficients(self) -> Self:
        for coeff in self.coefficients:
            value = int(coeff)
            if abs(value) > MAX_COEFF:
                raise ValueError(f"coefficients must be at most {MAX_COEFF} in absolute value")
        return self


class MapIterateRequest(StrictModel):
    """Compute the n-th iterate of a polynomial map by composition."""

    coefficients: tuple[str, ...] = Field(min_length=1, max_length=MAX_DEGREE + 1)
    n: int = Field(ge=0, le=MAX_ITERATE)


class OrbitPrefixRequest(StrictModel):
    """Compute the orbit prefix of a point under a polynomial map."""

    coefficients: tuple[str, ...] = Field(min_length=1, max_length=MAX_DEGREE + 1)
    start: str
    length: int = Field(ge=0, le=MAX_ORBIT)


class FixedPointEquationRequest(StrictModel):
    """Compute the fixed-point equation of the n-th iterate."""

    coefficients: tuple[str, ...] = Field(min_length=1, max_length=MAX_DEGREE + 1)
    n: int = Field(ge=1, le=MAX_ITERATE)


class DynatomicPolynomialRequest(StrictModel):
    """Compute the n-th dynatomic polynomial of a polynomial map."""

    coefficients: tuple[str, ...] = Field(min_length=1, max_length=MAX_DEGREE + 1)
    n: int = Field(ge=1, le=MAX_ITERATE)


class CycleMultiplierRequest(StrictModel):
    """Compute the multiplier of a periodic cycle.

    The cycle is given as a list of exact rational points (as strings)
    forming one orbit under the map. The multiplier is the product of
    the derivative at each cycle point.
    """

    coefficients: tuple[str, ...] = Field(min_length=1, max_length=MAX_DEGREE + 1)
    cycle: tuple[str, ...] = Field(min_length=1, max_length=MAX_ORBIT)


class FiniteFieldMapRequest(StrictModel):
    """A polynomial map over a finite field GF(p)."""

    prime: int = Field(ge=2, le=MAX_FIELD_PRIME)
    coefficients: tuple[str, ...] = Field(min_length=1, max_length=MAX_DEGREE + 1)

    @model_validator(mode="after")
    def validate(self) -> Self:
        if not _is_prime(self.prime):
            raise ValueError("prime must be a prime number")
        for coeff in self.coefficients:
            int(coeff)
        return self


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


# -- Results -----------------------------------------------------------------


class MapIterateResult(StrictModel):
    """Result of computing the n-th iterate."""

    n: int = Field(ge=0)
    coefficients: tuple[str, ...]
    degree: int = Field(ge=0)


class OrbitPrefixResult(StrictModel):
    """Orbit prefix of a point."""

    orbit: tuple[str, ...]
    length: int = Field(ge=0)
    first_repeat_index: int | None = None
    first_repeat_match: int | None = None


class FixedPointEquationResult(StrictModel):
    """Fixed-point equation f^n(x) - x as a polynomial."""

    coefficients: tuple[str, ...]
    degree: int = Field(ge=0)


class DynatomicPolynomialResult(StrictModel):
    """The n-th dynatomic polynomial."""

    coefficients: tuple[str, ...]
    degree: int = Field(ge=0)
    n: int = Field(ge=1)


class CycleMultiplierResult(StrictModel):
    """The multiplier of a periodic cycle."""

    multiplier: str
    cycle: tuple[str, ...]


class FiniteFieldMapResult(StrictModel):
    """Functional graph of a polynomial map over a finite field."""

    prime: int = Field(ge=2)
    edges: tuple[tuple[int, int], ...]
    cycles: tuple[tuple[int, ...], ...]
    tail_lengths: tuple[int, ...]


__all__ = [
    "CycleMultiplierRequest",
    "CycleMultiplierResult",
    "DynatomicPolynomialRequest",
    "DynatomicPolynomialResult",
    "FiniteFieldMapRequest",
    "FiniteFieldMapResult",
    "FixedPointEquationRequest",
    "FixedPointEquationResult",
    "MapIterateRequest",
    "MapIterateResult",
    "OrbitPrefixRequest",
    "OrbitPrefixResult",
    "PolynomialMapRequest",
]
