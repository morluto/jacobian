"""Typed wire contracts for linear code structural operations."""

from __future__ import annotations

from typing import Self

from pydantic import Field, model_validator

from jacobian._models import StrictModel

MAX_CODEWORDS = 16384  # binary k=14 (2^14), ternary k=8 (3^8=6561); mainly for equal.decide witness enumeration
MAX_LENGTH = 64  # rowspace operations are O(k^2 n) and remain cheap; raised from 32


def _validate_prime_matrix(
    field_order: int,
    generator_matrix: tuple[tuple[int, ...], ...],
) -> int:
    from sympy import isprime

    if not isprime(field_order):
        raise ValueError("field_order must be prime")
    width = len(generator_matrix[0])
    if width == 0 or width > MAX_LENGTH:
        raise ValueError(f"generator rows must have between 1 and {MAX_LENGTH} entries")
    if any(len(row) != width for row in generator_matrix):
        raise ValueError("generator matrix rows must have equal length")
    if any(not 0 <= entry < field_order for row in generator_matrix for entry in row):
        raise ValueError("generator entries must be canonical field residues")
    return width


class GeneratorMatrixRequest(StrictModel):
    """A linear code given by a generator matrix over a bounded prime field."""

    field_order: int = Field(ge=2, le=251)
    generator_matrix: tuple[tuple[int, ...], ...] = Field(min_length=1, max_length=16)

    @model_validator(mode="after")
    def require_bounded_prime_matrix(self) -> Self:
        width = _validate_prime_matrix(self.field_order, self.generator_matrix)
        if self.field_order ** len(self.generator_matrix) > MAX_CODEWORDS:
            raise ValueError("generator matrix exceeds exact enumeration bound")
        if width > MAX_LENGTH:
            raise ValueError("code length exceeds bound")
        return self


class ParityCheckRequest(StrictModel):
    """A linear code given by a generator matrix for computing a parity-check."""

    field_order: int = Field(ge=2, le=251)
    generator_matrix: tuple[tuple[int, ...], ...] = Field(min_length=1, max_length=16)

    @model_validator(mode="after")
    def require_valid(self) -> Self:
        _validate_prime_matrix(self.field_order, self.generator_matrix)
        return self


class CodewordCheckRequest(StrictModel):
    """Check whether a word is a codeword of the code."""

    field_order: int = Field(ge=2, le=251)
    generator_matrix: tuple[tuple[int, ...], ...] = Field(min_length=1, max_length=16)
    word: tuple[int, ...] = Field(min_length=1, max_length=MAX_LENGTH)

    @model_validator(mode="after")
    def require_valid(self) -> Self:
        width = _validate_prime_matrix(self.field_order, self.generator_matrix)
        if len(self.word) != width:
            raise ValueError("word length must match code length")
        if any(not 0 <= v < self.field_order for v in self.word):
            raise ValueError("word entries must be canonical field residues")
        return self


class ParityCheckMatrix(StrictModel):
    """A prime-field matrix retaining its column count when it has no rows."""

    field_order: int = Field(ge=2, le=251)
    column_count: int = Field(ge=1, le=MAX_LENGTH)
    rows: tuple[tuple[int, ...], ...] = Field(max_length=MAX_LENGTH)

    @model_validator(mode="after")
    def require_valid(self) -> Self:
        from sympy import isprime

        if not isprime(self.field_order):
            raise ValueError("field_order must be prime")
        if any(len(row) != self.column_count for row in self.rows):
            raise ValueError("parity-check rows must match the declared column count")
        if any(not 0 <= value < self.field_order for row in self.rows for value in row):
            raise ValueError("parity-check entries must be canonical field residues")
        return self


class SyndromeRequest(StrictModel):
    """Compute the syndrome of a word under a parity-check matrix."""

    parity_check: ParityCheckMatrix
    word: tuple[int, ...] = Field(min_length=1, max_length=MAX_LENGTH)

    @model_validator(mode="after")
    def require_valid(self) -> Self:
        if len(self.word) != self.parity_check.column_count:
            raise ValueError("word length must match code length")
        if any(not 0 <= v < self.parity_check.field_order for v in self.word):
            raise ValueError("word entries must be canonical field residues")
        return self


class CodeEqualRequest(StrictModel):
    """Check whether two generator matrices define the same code."""

    field_order: int = Field(ge=2, le=251)
    generator_matrix_a: tuple[tuple[int, ...], ...] = Field(min_length=1, max_length=16)
    generator_matrix_b: tuple[tuple[int, ...], ...] = Field(min_length=1, max_length=16)

    @model_validator(mode="after")
    def require_valid(self) -> Self:
        _validate_prime_matrix(self.field_order, self.generator_matrix_a)
        _validate_prime_matrix(self.field_order, self.generator_matrix_b)
        width_a = len(self.generator_matrix_a[0])
        width_b = len(self.generator_matrix_b[0])
        if width_a != width_b:
            raise ValueError("generator matrices must have the same code length")
        if self.field_order ** len(self.generator_matrix_a) > MAX_CODEWORDS:
            raise ValueError("code cardinality exceeds enumeration bound")
        if self.field_order ** len(self.generator_matrix_b) > MAX_CODEWORDS:
            raise ValueError("code cardinality exceeds enumeration bound")
        return self


class MacWilliamsRequest(StrictModel):
    """Primal weight distribution for MacWilliams transform."""

    field_order: int = Field(ge=2, le=251)
    code_cardinality: int = Field(ge=1)
    length: int = Field(ge=1, le=MAX_LENGTH)
    weights: tuple[int, ...] = Field(min_length=1, max_length=MAX_LENGTH + 1)

    @model_validator(mode="after")
    def require_valid_distribution(self) -> Self:
        if len(self.weights) != self.length + 1:
            raise ValueError("weights must have length + 1 entries")
        if any(w < 0 for w in self.weights):
            raise ValueError("weight counts must be non-negative")
        if self.weights[0] != 1:
            raise ValueError("first weight count must be 1 (zero codeword)")
        if sum(self.weights) != self.code_cardinality:
            raise ValueError("weight counts must sum to code cardinality")
        return self


class PunctureRequest(StrictModel):
    """Puncture a linear code by deleting one coordinate."""

    field_order: int = Field(ge=2, le=251)
    generator_matrix: tuple[tuple[int, ...], ...] = Field(min_length=1, max_length=16)
    coordinate: int = Field(ge=0)

    @model_validator(mode="after")
    def require_valid_request(self) -> Self:
        _validate_prime_matrix(self.field_order, self.generator_matrix)
        width = len(self.generator_matrix[0])
        if self.coordinate >= width:
            raise ValueError("coordinate index out of range")
        return self


class ShortenRequest(StrictModel):
    """Shorten a linear code by fixing one coordinate to zero and puncturing it."""

    field_order: int = Field(ge=2, le=251)
    generator_matrix: tuple[tuple[int, ...], ...] = Field(min_length=1, max_length=16)
    coordinate: int = Field(ge=0)

    @model_validator(mode="after")
    def require_valid_request(self) -> Self:
        _validate_prime_matrix(self.field_order, self.generator_matrix)
        width = len(self.generator_matrix[0])
        if self.coordinate >= width:
            raise ValueError("coordinate index out of range")
        return self


# Results


class FromGeneratorResult(StrictModel):
    canonical_generator: tuple[tuple[int, ...], ...]
    dimension: int = Field(ge=0)
    length: int = Field(ge=0)
    cardinality: int = Field(ge=1)
    method: str = "RREF"


class DualCodeResult(StrictModel):
    dual_generator: tuple[tuple[int, ...], ...]
    parity_check: ParityCheckMatrix
    dimension: int = Field(ge=0)
    dual_dimension: int = Field(ge=0)
    length: int = Field(ge=0)
    method: str = "NULLSPACE"


class ParityCheckResult(StrictModel):
    parity_check: ParityCheckMatrix
    dimension: int = Field(ge=0)
    rank_h: int = Field(ge=0)
    length: int = Field(ge=0)
    method: str = "NULLSPACE"


class CodewordCheckResult(StrictModel):
    is_member: bool
    hamming_weight: int = Field(ge=0)
    coefficients: tuple[int, ...] = ()
    syndrome: tuple[int, ...] = ()
    method: str = "RREF_MEMBERSHIP"


class SyndromeResult(StrictModel):
    syndrome: tuple[int, ...]
    is_member: bool
    method: str = "MATRIX_VECTOR_PRODUCT"


class CodeEqualResult(StrictModel):
    equal: bool
    dimension_a: int = Field(ge=0)
    dimension_b: int = Field(ge=0)
    witness_word: tuple[int, ...] | None = None
    method: str = "MUTUAL_ROW_SPACE_CONTAINMENT"


class MacWilliamsResult(StrictModel):
    dual_weights: tuple[int, ...]
    method: str = "MACWILLIAMS_IDENTITY"


class PunctureResult(StrictModel):
    generator: tuple[tuple[int, ...], ...]
    dimension: int = Field(ge=0)
    length: int = Field(ge=0)
    method: str = "PUNCTURE"


class ShortenResult(StrictModel):
    generator: tuple[tuple[int, ...], ...]
    dimension: int = Field(ge=0)
    length: int = Field(ge=0)
    method: str = "SHORTEN"
