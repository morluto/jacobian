"""Typed wire contracts for linear code structural operations."""

from __future__ import annotations

from typing import Annotated, Literal, Self

from pydantic import Field, StrictInt, model_validator

from jacobian._models import StrictModel
from jacobian.math.code_linear.values import (
    MAX_LINEAR_CODE_DIMENSION,
    MAX_LINEAR_CODE_LENGTH,
    PrimeFieldLinearEncoder,
)

MAX_CODEWORDS = 4096
MAX_LENGTH = MAX_LINEAR_CODE_LENGTH
MAX_RECEIVED_PROFILE_CODEWORDS = 4_096
MAX_RECEIVED_PROFILE_REPLAY_WORK = 3_000_000
MAX_RECEIVED_PROFILE_WITNESS_CELLS = 65_536

_FieldElement = Annotated[StrictInt, Field(ge=0, le=250)]
_HistogramCount = Annotated[
    StrictInt,
    Field(ge=0, le=MAX_RECEIVED_PROFILE_CODEWORDS),
]


class ReceivedWordThreshold(StrictModel):
    """One exact integer threshold on distance or coordinate agreement."""

    metric: Literal["DISTANCE", "AGREEMENT"]
    comparison: Literal["LT", "LE", "GT", "GE"]
    value: StrictInt = Field(ge=0, le=MAX_LINEAR_CODE_LENGTH)


class ReceivedWordProfileRequest(StrictModel):
    """Profile one received word against every word of a linear encoder."""

    encoder: PrimeFieldLinearEncoder
    received_word: tuple[_FieldElement, ...] = Field(
        min_length=1,
        max_length=MAX_LINEAR_CODE_LENGTH,
        description=(
            "Canonical GF(p) residues on the encoder's ordered coordinate axis."
        ),
    )
    threshold: ReceivedWordThreshold | None = Field(
        default=None,
        description=(
            "Optional exact distance/agreement threshold. Omit it only with "
            "witness_mode NONE; COUNT, FIRST, and ALL require it."
        ),
    )
    witness_mode: Literal["NONE", "COUNT", "FIRST", "ALL"] = Field(
        default="NONE",
        description=(
            "Threshold evidence to return: NONE without a threshold, COUNT "
            "only, FIRST in lexicographic message order, or every match."
        ),
    )

    @model_validator(mode="after")
    def require_aligned_bounded_profile(self) -> Self:
        if len(self.received_word) != len(self.encoder.coordinate_axis):
            raise ValueError("received word must match the encoder coordinate axis")
        if any(value >= self.encoder.field_order for value in self.received_word):
            raise ValueError("received-word entries must be canonical field residues")
        if self.encoder.codeword_count > MAX_RECEIVED_PROFILE_CODEWORDS:
            raise ValueError(
                "encoder codeword count exceeds the exact profile bound of "
                f"{MAX_RECEIVED_PROFILE_CODEWORDS}"
            )
        if self.profile_replay_work > MAX_RECEIVED_PROFILE_REPLAY_WORK:
            raise ValueError(
                "received-word profile replay work exceeds the bound of "
                f"{MAX_RECEIVED_PROFILE_REPLAY_WORK}"
            )

        if self.threshold is None and self.witness_mode != "NONE":
            raise ValueError("a witness mode requires an exact threshold")
        if self.threshold is not None:
            if self.witness_mode == "NONE":
                raise ValueError("a threshold requires COUNT, FIRST, or ALL mode")
            if self.threshold.value > len(self.encoder.coordinate_axis):
                raise ValueError("threshold value cannot exceed the code length")

        if (
            self.witness_mode == "ALL"
            and self.maximum_witness_cells > MAX_RECEIVED_PROFILE_WITNESS_CELLS
        ):
            raise ValueError(
                "all-witness result exceeds the aggregate witness-cell bound of "
                f"{MAX_RECEIVED_PROFILE_WITNESS_CELLS}"
            )
        return self

    @property
    def maximum_witness_cells(self) -> int:
        """Return the worst-case integer cells in a complete witness result."""

        row_width = (
            len(self.encoder.message_axis) + len(self.encoder.coordinate_axis) + 2
        )
        return self.encoder.codeword_count * row_width

    @property
    def profile_replay_work(self) -> int:
        """Bound kernel construction, comparison, and result-validation replay."""

        dimension = len(self.encoder.message_axis)
        length = len(self.encoder.coordinate_axis)
        return 2 * self.encoder.codeword_count * length * (dimension + 1)


class ReceivedWordWitness(StrictModel):
    """One source-replayable message, codeword, and Hamming relation."""

    message: tuple[_FieldElement, ...] = Field(max_length=MAX_LINEAR_CODE_DIMENSION)
    codeword: tuple[_FieldElement, ...] = Field(
        min_length=1,
        max_length=MAX_LINEAR_CODE_LENGTH,
    )
    distance: StrictInt = Field(ge=0, le=MAX_LINEAR_CODE_LENGTH)
    agreement: StrictInt = Field(ge=0, le=MAX_LINEAR_CODE_LENGTH)


class ReceivedWordProfileResult(StrictModel):
    """Complete coset weight distribution, bound to its encoder and word."""

    source: ReceivedWordProfileRequest
    distance_histogram: tuple[_HistogramCount, ...] = Field(
        min_length=2,
        max_length=MAX_LINEAR_CODE_LENGTH + 1,
        description=(
            "Dense histogram indexed by Hamming distance; agreement a is the "
            "entry at distance code_length - a."
        ),
    )
    codeword_count: StrictInt = Field(ge=1, le=MAX_RECEIVED_PROFILE_CODEWORDS)
    minimum_distance: StrictInt = Field(
        ge=0,
        le=MAX_LINEAR_CODE_LENGTH,
        description="Minimum Hamming distance from the received word to the code.",
    )
    maximum_agreement: StrictInt = Field(
        ge=0,
        le=MAX_LINEAR_CODE_LENGTH,
        description="Maximum coordinate agreement with the received word.",
    )
    threshold_match_count: StrictInt | None = Field(
        default=None,
        ge=0,
        le=MAX_RECEIVED_PROFILE_CODEWORDS,
    )
    witnesses: tuple[ReceivedWordWitness, ...] = Field(
        default=(),
        max_length=MAX_RECEIVED_PROFILE_CODEWORDS,
    )

    @model_validator(mode="after")
    def bind_exact_profile_to_source(self) -> Self:
        from jacobian.math.code_linear._operations import _received_word_profile_data

        expected = _received_word_profile_data(self.source)
        if self.distance_histogram != expected.distance_histogram:
            raise ValueError("distance histogram does not match the source relation")
        if self.codeword_count != expected.codeword_count:
            raise ValueError("codeword count does not match the source encoder")
        if self.minimum_distance != expected.minimum_distance:
            raise ValueError("minimum distance does not match the source relation")
        if self.maximum_agreement != expected.maximum_agreement:
            raise ValueError("maximum agreement does not match the source relation")
        if self.threshold_match_count != expected.threshold_match_count:
            raise ValueError("threshold count does not match the source threshold")
        if self.witnesses != expected.witnesses:
            raise ValueError("threshold witnesses do not replay against the source")
        return self


def _validate_prime_matrix(
    field_order: int,
    generator_matrix: tuple[tuple[int, ...], ...],
) -> int:
    from sympy import isprime

    if not isprime(field_order):
        raise ValueError("field_order must be prime")
    width = len(generator_matrix[0])
    if width == 0 or width > MAX_LENGTH:
        raise ValueError("generator rows must have between 1 and 32 entries")
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
