"""Typed wire contracts for symbolic dynamics operations."""

from __future__ import annotations

from typing import Self

from pydantic import Field, model_validator

from jacobian._models import StrictModel

MAX_ALPHABET = 50
MAX_FORBIDDEN = 100
MAX_BLOCK_LEN = 20
MAX_STATES = 100
MAX_PERIOD_BOUND = 50


class FiniteTypeShiftRequest(StrictModel):
    """Construct a shift of finite type from a forbidden-block family."""

    alphabet: tuple[str, ...] = Field(min_length=1, max_length=MAX_ALPHABET)
    forbidden_blocks: tuple[tuple[str, ...], ...] = Field(max_length=MAX_FORBIDDEN)
    two_sided: bool = True

    @model_validator(mode="after")
    def validate_blocks(self) -> Self:
        valid = set(self.alphabet)
        for block in self.forbidden_blocks:
            for letter in block:
                if letter not in valid:
                    raise ValueError(f"forbidden block letter {letter!r} is not in the alphabet")
        return self


class BlockLanguageRequest(StrictModel):
    """Compute the allowed block language of a shift at a given length."""

    alphabet: tuple[str, ...] = Field(min_length=1, max_length=MAX_ALPHABET)
    forbidden_blocks: tuple[tuple[str, ...], ...] = Field(max_length=MAX_FORBIDDEN)
    block_length: int = Field(ge=1, le=MAX_BLOCK_LEN)

    @model_validator(mode="after")
    def validate_blocks(self) -> Self:
        valid = set(self.alphabet)
        for block in self.forbidden_blocks:
            for letter in block:
                if letter not in valid:
                    raise ValueError(f"forbidden block letter {letter!r} is not in the alphabet")
        return self


class AdjacencyShiftRequest(StrictModel):
    """Construct a shift from a nonnegative integer adjacency matrix."""

    matrix: tuple[tuple[int, ...], ...]
    two_sided: bool = True

    @model_validator(mode="after")
    def validate_matrix(self) -> Self:
        n = len(self.matrix)
        if n == 0:
            raise ValueError("matrix must be non-empty")
        for row in self.matrix:
            if len(row) != n:
                raise ValueError("matrix must be square")
            for val in row:
                if val < 0:
                    raise ValueError("matrix entries must be non-negative")
        return self


class PeriodicPointProfileRequest(StrictModel):
    """Compute the periodic point counts of a shift from its adjacency matrix."""

    matrix: tuple[tuple[int, ...], ...]
    max_period: int = Field(ge=1, le=MAX_PERIOD_BOUND)

    @model_validator(mode="after")
    def validate_matrix(self) -> Self:
        n = len(self.matrix)
        if n == 0:
            raise ValueError("matrix must be non-empty")
        for row in self.matrix:
            if len(row) != n:
                raise ValueError("matrix must be square")
            for val in row:
                if val < 0:
                    raise ValueError("matrix entries must be non-negative")
        return self


class HigherBlockRequest(StrictModel):
    """Compute the higher-block presentation (n-th higher block)."""

    alphabet: tuple[str, ...] = Field(min_length=1, max_length=MAX_ALPHABET)
    forbidden_blocks: tuple[tuple[str, ...], ...] = Field(max_length=MAX_FORBIDDEN)
    n: int = Field(ge=2, le=MAX_BLOCK_LEN)

    @model_validator(mode="after")
    def validate_blocks(self) -> Self:
        valid = set(self.alphabet)
        for block in self.forbidden_blocks:
            for letter in block:
                if letter not in valid:
                    raise ValueError(f"forbidden block letter {letter!r} is not in the alphabet")
        return self


class FiniteTypeShiftResult(StrictModel):
    """A shift of finite type presentation."""

    alphabet: tuple[str, ...]
    forbidden_blocks: tuple[tuple[str, ...], ...]
    max_forbidden_length: int = Field(ge=0)
    is_empty: bool
    adjacency_matrix: tuple[tuple[int, ...], ...]
    num_states: int = Field(ge=0)


class BlockLanguageResult(StrictModel):
    """Allowed block language of a shift at a given length."""

    block_length: int = Field(ge=1)
    allowed_blocks: tuple[tuple[str, ...], ...]
    count: int = Field(ge=0)


class AdjacencyShiftResult(StrictModel):
    """A shift presentation from an adjacency matrix."""

    matrix: tuple[tuple[int, ...], ...]
    is_essential: bool
    is_irreducible: bool
    period: int = Field(ge=0)
    is_mixing: bool


class PeriodicPointProfileResult(StrictModel):
    """Periodic point profile of a shift."""

    fix_counts: tuple[int, ...]
    exact_counts: tuple[int, ...]
    orbit_counts: tuple[int, ...]
    zeta_numerator: tuple[int, ...]
    zeta_denominator: tuple[int, ...]


class HigherBlockResult(StrictModel):
    """Higher-block presentation of a shift."""

    new_alphabet: tuple[str, ...]
    new_forbidden_blocks: tuple[tuple[str, ...], ...]
    n: int = Field(ge=2)


__all__ = [
    "AdjacencyShiftRequest",
    "AdjacencyShiftResult",
    "BlockLanguageRequest",
    "BlockLanguageResult",
    "FiniteTypeShiftRequest",
    "FiniteTypeShiftResult",
    "HigherBlockRequest",
    "HigherBlockResult",
    "PeriodicPointProfileRequest",
    "PeriodicPointProfileResult",
]
