"""Typed wire contracts for linear code structural operations."""

from __future__ import annotations

from math import comb
from typing import Annotated, Literal, Self

from pydantic import Field, StrictInt, model_validator

from jacobian._models import StrictModel
from jacobian.math._labels import OpaqueLabel
from jacobian.math.code_linear.values import (
    MAX_LINEAR_CODE_DIMENSION,
    MAX_LINEAR_CODE_LENGTH,
    PrimeFieldLinearEncoder,
)

MAX_CODEWORDS = 16384  # binary k=14 (2^14), ternary k=8 (3^8=6561); mainly for equal.decide witness enumeration
MAX_LENGTH = MAX_LINEAR_CODE_LENGTH  # rowspace operations are O(k^2 n) and remain cheap; raised from 32
MAX_RECEIVED_PROFILE_REPLAY_WORK = 3_000_000
MAX_RECEIVED_PROFILE_WITNESS_CELLS = 65_536


def _max_codewords_within_replay_work(replay_work_bound: int) -> int:
    """Largest encoder image admitted by the exact replay-work bound."""

    best = 1
    for order in range(2, 252):
        if any(order % divisor == 0 for divisor in range(2, int(order**0.5) + 1)):
            continue
        for dimension in range(1, MAX_LINEAR_CODE_DIMENSION + 1):
            count = order**dimension
            # Full rank forces length >= dimension, so the work product is
            # minimized at length == dimension; larger lengths reject sooner.
            if 2 * count * dimension * (dimension + 1) > replay_work_bound:
                break
            if count > best:
                best = count
    return best


# Derived from the replay-work bound over primes <= 251: every admitted
# encoder image satisfies codeword_count <= this constant.
MAX_RECEIVED_PROFILE_CODEWORDS = _max_codewords_within_replay_work(
    MAX_RECEIVED_PROFILE_REPLAY_WORK
)

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


def _threshold_matches_distance(
    threshold: ReceivedWordThreshold,
    *,
    distance: int,
    length: int,
) -> bool:
    observed = distance if threshold.metric == "DISTANCE" else length - distance
    if threshold.comparison == "LT":
        return observed < threshold.value
    if threshold.comparison == "LE":
        return observed <= threshold.value
    if threshold.comparison == "GT":
        return observed > threshold.value
    return observed >= threshold.value


class ReceivedWordProfileRequest(StrictModel):
    """Profile one received word against every word of a linear encoder."""

    encoder: PrimeFieldLinearEncoder
    received_word: tuple[_FieldElement, ...] = Field(
        max_length=MAX_LINEAR_CODE_LENGTH,
        description=(
            "Canonical GF(p) residues on the encoder's ordered coordinate axis; "
            "the empty word is valid exactly for a length-zero encoder."
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

        threshold = self.threshold
        if threshold is None:
            return 0
        row_width = (
            len(self.encoder.message_axis) + len(self.encoder.coordinate_axis) + 2
        )
        length = len(self.encoder.coordinate_axis)
        field_order = int(self.encoder.field_order)
        ambient_match_count: int = sum(
            comb(length, distance) * (field_order - 1) ** distance
            for distance in range(length + 1)
            if _threshold_matches_distance(
                threshold,
                distance=distance,
                length=length,
            )
        )
        return min(self.encoder.codeword_count, ambient_match_count) * row_width

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
        max_length=MAX_LINEAR_CODE_LENGTH,
    )
    distance: StrictInt = Field(ge=0, le=MAX_LINEAR_CODE_LENGTH)
    agreement: StrictInt = Field(ge=0, le=MAX_LINEAR_CODE_LENGTH)


class ReceivedWordProfileResult(StrictModel):
    """Complete coset weight distribution, bound to its encoder and word."""

    source: ReceivedWordProfileRequest
    distance_histogram: tuple[_HistogramCount, ...] = Field(
        min_length=1,
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
        raise ValueError(f"generator rows must have between 1 and {MAX_LENGTH} entries")
    if any(len(row) != width for row in generator_matrix):
        raise ValueError("generator matrix rows must have equal length")
    if any(not 0 <= entry < field_order for row in generator_matrix for entry in row):
        raise ValueError("generator entries must be canonical field residues")
    return width


def _validate_coordinate_axis(
    coordinate_axis: tuple[OpaqueLabel, ...],
    *,
    width: int,
) -> None:
    if len(coordinate_axis) != width:
        raise ValueError("coordinate axis must match the generator-matrix columns")
    if len(set(coordinate_axis)) != len(coordinate_axis):
        raise ValueError("coordinate-axis labels must be unique")


class GeneratorMatrixRequest(StrictModel):
    """A linear code given by a generator matrix over a bounded prime field."""

    field_order: int = Field(ge=2, le=251)
    generator_matrix: tuple[tuple[int, ...], ...] = Field(min_length=1, max_length=16)
    coordinate_axis: tuple[OpaqueLabel, ...] = Field(
        min_length=1,
        max_length=MAX_LENGTH,
        description=(
            "Ordered unique labels for generator-matrix columns; code-producing "
            "results preserve these labels."
        ),
    )

    @model_validator(mode="after")
    def require_bounded_prime_matrix(self) -> Self:
        width = _validate_prime_matrix(self.field_order, self.generator_matrix)
        _validate_coordinate_axis(self.coordinate_axis, width=width)
        if self.field_order ** len(self.generator_matrix) > MAX_CODEWORDS:
            raise ValueError("generator matrix exceeds exact enumeration bound")
        if width > MAX_LENGTH:
            raise ValueError("code length exceeds bound")
        return self


class DualCodeRequest(StrictModel):
    """Compute the dual code of a canonical linear encoder."""

    encoder: PrimeFieldLinearEncoder


class ParityCheckRequest(StrictModel):
    """A canonical linear encoder for computing a parity-check."""

    encoder: PrimeFieldLinearEncoder


class CodewordCheckRequest(StrictModel):
    """Check whether a word is a codeword of a canonical linear encoder."""

    encoder: PrimeFieldLinearEncoder
    word: tuple[_FieldElement, ...] = Field(
        max_length=MAX_LINEAR_CODE_LENGTH,
        description=(
            "Canonical GF(p) residues on the encoder's ordered coordinate axis; "
            "the empty word is valid exactly for a length-zero encoder."
        ),
    )

    @model_validator(mode="after")
    def require_aligned_canonical_word(self) -> Self:
        if len(self.word) != len(self.encoder.coordinate_axis):
            raise ValueError("word length must match the encoder coordinate axis")
        if any(value >= self.encoder.field_order for value in self.word):
            raise ValueError("word entries must be canonical field residues")
        return self


class ParityCheckMatrix(StrictModel):
    """A prime-field matrix retaining its ordered column axis when it has no rows."""

    field_order: int = Field(ge=2, le=251)
    coordinate_axis: tuple[OpaqueLabel, ...] = Field(
        max_length=MAX_LENGTH,
        description=(
            "Ordered unique labels for matrix columns; syndrome words must "
            "present this same ordered axis."
        ),
    )
    rows: tuple[tuple[int, ...], ...] = Field(max_length=MAX_LENGTH)

    @model_validator(mode="after")
    def require_valid(self) -> Self:
        from sympy import isprime

        if not isprime(self.field_order):
            raise ValueError("field_order must be prime")
        if len(set(self.coordinate_axis)) != len(self.coordinate_axis):
            raise ValueError("coordinate-axis labels must be unique")
        if any(len(row) != len(self.coordinate_axis) for row in self.rows):
            raise ValueError("parity-check rows must match the column axis")
        if any(not 0 <= value < self.field_order for row in self.rows for value in row):
            raise ValueError("parity-check entries must be canonical field residues")
        return self


class SyndromeRequest(StrictModel):
    """Compute the syndrome of a word under a parity-check matrix."""

    parity_check: ParityCheckMatrix
    coordinate_axis: tuple[OpaqueLabel, ...] = Field(
        max_length=MAX_LENGTH,
        description=(
            "Ordered labels of the word's coordinates; must equal the "
            "parity-check column axis so word entries cannot align to the "
            "wrong columns."
        ),
    )
    word: tuple[int, ...] = Field(max_length=MAX_LENGTH)

    @model_validator(mode="after")
    def require_valid(self) -> Self:
        if self.coordinate_axis != self.parity_check.coordinate_axis:
            raise ValueError("word axis must match the parity-check column axis")
        if len(self.word) != len(self.coordinate_axis):
            raise ValueError("word length must match code length")
        if any(not 0 <= v < self.parity_check.field_order for v in self.word):
            raise ValueError("word entries must be canonical field residues")
        return self


class CodeEqualRequest(StrictModel):
    """Check whether two canonical encoders define the same code."""

    encoder_a: PrimeFieldLinearEncoder
    encoder_b: PrimeFieldLinearEncoder

    @model_validator(mode="after")
    def require_comparable_encoders(self) -> Self:
        if self.encoder_a.field_order != self.encoder_b.field_order:
            raise ValueError("encoders must share one prime field order")
        if self.encoder_a.coordinate_axis != self.encoder_b.coordinate_axis:
            raise ValueError("encoders must share one ordered coordinate axis")
        if (
            self.encoder_a.codeword_count > MAX_CODEWORDS
            or self.encoder_b.codeword_count > MAX_CODEWORDS
        ):
            raise ValueError("code cardinality exceeds exact enumeration bound")
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


def _require_selected_coordinate(
    encoder: PrimeFieldLinearEncoder,
    coordinate: int,
) -> None:
    if not encoder.coordinate_axis:
        raise ValueError("encoder must have at least one coordinate to select")
    if coordinate >= len(encoder.coordinate_axis):
        raise ValueError("coordinate index out of range")


class PunctureRequest(StrictModel):
    """Puncture a linear code by deleting one coordinate."""

    encoder: PrimeFieldLinearEncoder
    coordinate: int = Field(
        ge=0,
        description=(
            "Zero-based index into encoder.coordinate_axis identifying the "
            "single coordinate to delete."
        ),
    )

    @model_validator(mode="after")
    def require_valid_request(self) -> Self:
        _require_selected_coordinate(self.encoder, self.coordinate)
        return self


class ShortenRequest(StrictModel):
    """Shorten a linear code by fixing one coordinate to zero and puncturing it."""

    encoder: PrimeFieldLinearEncoder
    coordinate: int = Field(
        ge=0,
        description=(
            "Zero-based index into encoder.coordinate_axis identifying the "
            "single coordinate to fix and delete."
        ),
    )

    @model_validator(mode="after")
    def require_valid_request(self) -> Self:
        _require_selected_coordinate(self.encoder, self.coordinate)
        return self


# Results


class FromGeneratorResult(StrictModel):
    """A canonical encoder with summaries derived from that encoder."""

    encoder: PrimeFieldLinearEncoder
    dimension: int = Field(ge=0)
    length: int = Field(ge=0)
    cardinality: int = Field(ge=1)
    method: str = "RREF"

    @model_validator(mode="after")
    def require_consistent_summaries(self) -> Self:
        if self.dimension != len(self.encoder.message_axis):
            raise ValueError("dimension must match the encoder message axis")
        if self.length != len(self.encoder.coordinate_axis):
            raise ValueError("length must match the encoder coordinate axis")
        if self.cardinality != self.encoder.codeword_count:
            raise ValueError("cardinality must match the encoder image")
        return self


class DualCodeResult(StrictModel):
    """A canonical dual encoder and its matching parity-check matrix."""

    encoder: PrimeFieldLinearEncoder
    parity_check: ParityCheckMatrix
    dimension: int = Field(ge=0)
    dual_dimension: int = Field(ge=0)
    length: int = Field(ge=0)
    method: str = "NULLSPACE"

    @model_validator(mode="after")
    def require_consistent_dual(self) -> Self:
        if self.dual_dimension != len(self.encoder.message_axis):
            raise ValueError("dual dimension must match the encoder message axis")
        if self.length != len(self.encoder.coordinate_axis):
            raise ValueError("length must match the encoder coordinate axis")
        if self.dimension + self.dual_dimension != self.length:
            raise ValueError("primal and dual dimensions must sum to the code length")
        if self.parity_check.field_order != self.encoder.field_order:
            raise ValueError("parity-check field must match the dual encoder")
        if self.parity_check.coordinate_axis != self.encoder.coordinate_axis:
            raise ValueError("parity-check must preserve the dual coordinate axis")
        if self.parity_check.rows != self.encoder.generator_matrix:
            raise ValueError("parity-check rows must match the dual encoder")
        return self


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
    """A canonical punctured encoder with derived dimension and length."""

    encoder: PrimeFieldLinearEncoder
    dimension: int = Field(ge=0)
    length: int = Field(ge=0)
    method: str = "PUNCTURE"

    @model_validator(mode="after")
    def require_consistent_summaries(self) -> Self:
        if self.dimension != len(self.encoder.message_axis):
            raise ValueError("dimension must match the encoder message axis")
        if self.length != len(self.encoder.coordinate_axis):
            raise ValueError("length must match the encoder coordinate axis")
        return self


class ShortenResult(StrictModel):
    """A canonical shortened encoder with derived dimension and length."""

    encoder: PrimeFieldLinearEncoder
    dimension: int = Field(ge=0)
    length: int = Field(ge=0)
    method: str = "SHORTEN"

    @model_validator(mode="after")
    def require_consistent_summaries(self) -> Self:
        if self.dimension != len(self.encoder.message_axis):
            raise ValueError("dimension must match the encoder message axis")
        if self.length != len(self.encoder.coordinate_axis):
            raise ValueError("length must match the encoder coordinate axis")
        return self
