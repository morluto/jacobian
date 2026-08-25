"""Typed wire contracts for arithmetic function operations."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import model_validator
from pydantic_core import PydanticCustomError

from jacobian._exact import MAX_CANONICAL_RATIONAL_DIGITS, CanonicalRational
from jacobian._models import StrictModel
from jacobian.math._rational_height import RationalHeight, sum_heights

# Bounds shared by every arithmetic-function operation.
_MIN_LENGTH = 1
_MAX_LENGTH = 10_000


def _validation_error(reason: str, message: str) -> PydanticCustomError:
    return PydanticCustomError(f"arithmetic_functions.{reason}", message)


def _heights(values: tuple[CanonicalRational, ...]) -> tuple[RationalHeight, ...]:
    return tuple(RationalHeight.from_canonical(value) for value in values)


def _divisors(value: int) -> tuple[int, ...]:
    small: list[int] = []
    large: list[int] = []
    candidate = 1
    while candidate * candidate <= value:
        if value % candidate == 0:
            small.append(candidate)
            if candidate != value // candidate:
                large.append(value // candidate)
        candidate += 1
    return (*small, *reversed(large))


def _require_result_height(height: RationalHeight, operation: str) -> None:
    if height.exceeds(MAX_CANONICAL_RATIONAL_DIGITS):
        raise _validation_error(
            "result_height_exceeded",
            f"{operation} rational height exceeds the "
            f"{MAX_CANONICAL_RATIONAL_DIGITS}-digit result bound",
        )


class DirichletConvolutionRequest(StrictModel):
    """Request: Dirichlet convolution of two arithmetic functions.

    The two functions ``f`` and ``g`` must be given at the same indices
    1, 2, ..., n.  The result ``h = f * g`` is defined for K = 1..n by
    ``h(K) = sum_{d | K} f(d) * g(K // d)``.
    """

    f: tuple[CanonicalRational, ...]
    g: tuple[CanonicalRational, ...]

    @model_validator(mode="after")
    def require_matching_lengths(self) -> Self:
        if not (_MIN_LENGTH <= len(self.f) <= _MAX_LENGTH):
            raise _validation_error(
                "invalid_length",
                f"f must have between {_MIN_LENGTH} and {_MAX_LENGTH} values",
            )
        if len(self.f) != len(self.g):
            raise _validation_error(
                "length_mismatch", "f and g must have the same length"
            )
        return self

    @model_validator(mode="after")
    def require_bounded_result_height(self) -> Self:
        left = _heights(self.f)
        right = _heights(self.g)
        for index in range(1, len(left) + 1):
            terms = (
                left[divisor - 1].product(right[index // divisor - 1])
                for divisor in _divisors(index)
            )
            _require_result_height(sum_heights(terms), "Dirichlet convolution")
        return self


class DirichletConvolutionResult(StrictModel):
    """Result: the Dirichlet convolution ``(f*g)(1)..(f*g)(n)``."""

    values: tuple[CanonicalRational, ...]
    length: int
    convention: Literal["JACOBIAN_DIRICHLET_CONVOLUTION"] = (
        "JACOBIAN_DIRICHLET_CONVOLUTION"
    )

    @model_validator(mode="after")
    def bind_length(self) -> Self:
        if self.length != len(self.values):
            raise _validation_error(
                "length_mismatch", "length must match the number of values"
            )
        return self


class MobiusTransformRequest(StrictModel):
    """Request: Möbius (inverse) transform of an arithmetic function.

    Given ``F`` at indices 1..n the forward Möbius transform returns
    ``f(K) = sum_{d | K} mu(d) * F(K // d)``.  When ``inverse`` is true the
    inverse transform is Dirichlet convolution with the constant-one function:
    ``F(K) = sum_{d | K} f(K // d)``.
    """

    values: tuple[CanonicalRational, ...]
    inverse: bool = False

    @model_validator(mode="after")
    def require_valid_length(self) -> Self:
        if not (_MIN_LENGTH <= len(self.values) <= _MAX_LENGTH):
            raise _validation_error(
                "invalid_length",
                f"values must have between {_MIN_LENGTH} and {_MAX_LENGTH} entries",
            )
        return self

    @model_validator(mode="after")
    def require_bounded_result_height(self) -> Self:
        heights = _heights(self.values)
        for index in range(1, len(heights) + 1):
            terms = (heights[index // divisor - 1] for divisor in _divisors(index))
            _require_result_height(sum_heights(terms), "Möbius transform")
        return self


class MobiusTransformResult(StrictModel):
    """Result: the (inverse) Möbius transform at indices 1..n."""

    values: tuple[CanonicalRational, ...]
    length: int
    inverse: bool
    convention: Literal["JACOBIAN_MOBIUS_TRANSFORM"] = "JACOBIAN_MOBIUS_TRANSFORM"

    @model_validator(mode="after")
    def bind_length(self) -> Self:
        if self.length != len(self.values):
            raise _validation_error(
                "length_mismatch", "length must match the number of values"
            )
        return self


class SummatoryFunctionRequest(StrictModel):
    """Request: partial sums ``S(K) = sum_{i=1}^{K} f(i)`` for K = 1..n."""

    values: tuple[CanonicalRational, ...]

    @model_validator(mode="after")
    def require_valid_length(self) -> Self:
        if not (_MIN_LENGTH <= len(self.values) <= _MAX_LENGTH):
            raise _validation_error(
                "invalid_length",
                f"values must have between {_MIN_LENGTH} and {_MAX_LENGTH} entries",
            )
        return self

    @model_validator(mode="after")
    def require_bounded_result_height(self) -> Self:
        _require_result_height(sum_heights(_heights(self.values)), "summatory function")
        return self


class SummatoryFunctionResult(StrictModel):
    """Result: the partial sums ``S(1)..S(n)``."""

    values: tuple[CanonicalRational, ...]
    length: int
    convention: Literal["JACOBIAN_SUMMATORY_FUNCTION"] = "JACOBIAN_SUMMATORY_FUNCTION"

    @model_validator(mode="after")
    def bind_length(self) -> Self:
        if self.length != len(self.values):
            raise _validation_error(
                "length_mismatch", "length must match the number of values"
            )
        return self


class DirichletInverseRequest(StrictModel):
    """Request: Dirichlet inverse ``g`` such that ``f * g = epsilon``.

    The arithmetic function ``f`` must satisfy ``f(1) != 0``.
    """

    values: tuple[CanonicalRational, ...]

    @model_validator(mode="after")
    def require_valid_length_and_nonzero_unit(self) -> Self:
        if not (_MIN_LENGTH <= len(self.values) <= _MAX_LENGTH):
            raise _validation_error(
                "invalid_length",
                f"values must have between {_MIN_LENGTH} and {_MAX_LENGTH} entries",
            )
        if self.values[0].as_fraction() == 0:
            raise _validation_error("zero_unit", "f(1) must be nonzero")
        return self

    @model_validator(mode="after")
    def require_bounded_result_height(self) -> Self:
        source = _heights(self.values)
        inverse = [RationalHeight(1, 1).quotient(source[0])]
        _require_result_height(inverse[0], "Dirichlet inverse")
        for index in range(2, len(source) + 1):
            terms = (
                source[divisor - 1].product(inverse[index // divisor - 1])
                for divisor in _divisors(index)
                if divisor > 1
            )
            height = sum_heights(terms).quotient(source[0])
            _require_result_height(height, "Dirichlet inverse")
            inverse.append(height)
        return self


class DirichletInverseResult(StrictModel):
    """Result: the Dirichlet inverse at indices 1..n."""

    values: tuple[CanonicalRational, ...]
    length: int
    convention: Literal["JACOBIAN_DIRICHLET_INVERSE"] = "JACOBIAN_DIRICHLET_INVERSE"

    @model_validator(mode="after")
    def bind_length(self) -> Self:
        if self.length != len(self.values):
            raise _validation_error(
                "length_mismatch", "length must match the number of values"
            )
        return self


__all__ = [
    "DirichletConvolutionRequest",
    "DirichletConvolutionResult",
    "DirichletInverseRequest",
    "DirichletInverseResult",
    "MobiusTransformRequest",
    "MobiusTransformResult",
    "SummatoryFunctionRequest",
    "SummatoryFunctionResult",
]
