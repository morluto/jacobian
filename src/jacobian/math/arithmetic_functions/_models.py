"""Typed wire contracts for arithmetic function operations."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import model_validator

from jacobian._exact import CanonicalRational
from jacobian._models import StrictModel

# Bounds shared by every arithmetic-function operation.
_MIN_LENGTH = 1
_MAX_LENGTH = 10_000


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
            raise ValueError(
                f"f must have between {_MIN_LENGTH} and {_MAX_LENGTH} values",
            )
        if len(self.f) != len(self.g):
            raise ValueError("f and g must have the same length")
        return self


class DirichletConvolutionResult(StrictModel):
    """Result: the Dirichlet convolution ``(f*g)(1)..(f*g)(n)``."""

    values: tuple[CanonicalRational, ...]
    length: int
    convention: Literal["JACOBIAN_DIRICHLET_CONVOLUTION"] = (
        "JACOBIAN_DIRICHLET_CONVOLUTION"
    )


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
            raise ValueError(
                f"values must have between {_MIN_LENGTH} and {_MAX_LENGTH} entries",
            )
        return self


class MobiusTransformResult(StrictModel):
    """Result: the (inverse) Möbius transform at indices 1..n."""

    values: tuple[CanonicalRational, ...]
    length: int
    inverse: bool
    convention: Literal["JACOBIAN_MOBIUS_TRANSFORM"] = "JACOBIAN_MOBIUS_TRANSFORM"


class SummatoryFunctionRequest(StrictModel):
    """Request: partial sums ``S(K) = sum_{i=1}^{K} f(i)`` for K = 1..n."""

    values: tuple[CanonicalRational, ...]

    @model_validator(mode="after")
    def require_valid_length(self) -> Self:
        if not (_MIN_LENGTH <= len(self.values) <= _MAX_LENGTH):
            raise ValueError(
                f"values must have between {_MIN_LENGTH} and {_MAX_LENGTH} entries",
            )
        return self


class SummatoryFunctionResult(StrictModel):
    """Result: the partial sums ``S(1)..S(n)``."""

    values: tuple[CanonicalRational, ...]
    length: int
    convention: Literal["JACOBIAN_SUMMATORY_FUNCTION"] = "JACOBIAN_SUMMATORY_FUNCTION"


class DirichletInverseRequest(StrictModel):
    """Request: Dirichlet inverse ``g`` such that ``f * g = epsilon``.

    The arithmetic function ``f`` must satisfy ``f(1) != 0``.
    """

    values: tuple[CanonicalRational, ...]

    @model_validator(mode="after")
    def require_valid_length_and_nonzero_unit(self) -> Self:
        if not (_MIN_LENGTH <= len(self.values) <= _MAX_LENGTH):
            raise ValueError(
                f"values must have between {_MIN_LENGTH} and {_MAX_LENGTH} entries",
            )
        if self.values[0].as_fraction() == 0:
            raise ValueError("f(1) must be nonzero")
        return self


class DirichletInverseResult(StrictModel):
    """Result: the Dirichlet inverse at indices 1..n."""

    values: tuple[CanonicalRational, ...]
    length: int
    convention: Literal["JACOBIAN_DIRICHLET_INVERSE"] = "JACOBIAN_DIRICHLET_INVERSE"


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
