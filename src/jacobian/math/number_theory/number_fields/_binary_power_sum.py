"""Complete exact binary power-sum gap profiles in one real embedding."""

from __future__ import annotations

import time
from collections.abc import Mapping
from dataclasses import dataclass
from fractions import Fraction
from functools import cmp_to_key
from itertools import pairwise
from math import lcm
from typing import Annotated, Any, Literal, Self

from pydantic import Field, StrictInt, model_validator
from pydantic_core import PydanticCustomError

from jacobian._exact import CanonicalRational
from jacobian._execution import (
    OperationExecutionCancelledError,
    OperationExecutionTimeoutError,
    bind_request_deadline,
    current_request_execution,
)
from jacobian._models import StrictModel, canonicalize_json_containers
from jacobian.canonical import encode_strict_json, parse_canonical_integer
from jacobian.math.number_theory.number_fields._embeddings_process import (
    embeddings_worker_cancelled,
)
from jacobian.math.number_theory.number_fields._real_embedding_order import (
    NumberFieldRealEmbeddingOrderError,
    RealEmbeddingDifferenceAdmission,
    RecognizedRealEmbeddingContext,
    admit_real_embedding_difference,
    admit_real_embedding_difference_envelope,
    isolate_backend_real_value,
    recognize_real_embedding_binding,
)
from jacobian.math.number_theory.number_fields.values import (
    MAX_NUMBER_FIELD_ISOLATOR_COMPONENT_DIGITS,
    MAX_SIMPLE_NUMBER_FIELD_ELEMENT_DIGITS,
    NumberFieldRealValueEnclosure,
    SimpleNumberFieldElement,
    SimpleNumberFieldRealEmbeddingBinding,
    SimpleNumberFieldRealOrder,
)

MAX_BINARY_POWER_SUM_SOURCE_REPRESENTATIONS = 4_096
MAX_BINARY_POWER_SUM_EXPONENT_COUNT = (
    MAX_BINARY_POWER_SUM_SOURCE_REPRESENTATIONS.bit_length() - 1
)
MAX_BINARY_POWER_SUM_REPRESENTATION_BIT_SLOTS = 49_152
MAX_BINARY_POWER_SUM_FIELD_OPERATIONS = 16_384
MAX_BINARY_POWER_SUM_COMPARISONS = 100_000
MAX_BINARY_POWER_SUM_RESULT_BYTES = 10 * 1024 * 1024
MAX_BINARY_POWER_SUM_SHARED_ROOT_REFINEMENT_BITS = 4_096
BINARY_POWER_SUM_WALL_SECONDS = 600.0

BinaryDigit = Annotated[int, Field(ge=0, le=1, strict=True)]
BinaryPowerSumBitVector = tuple[BinaryDigit, ...]


def _validation_error(reason: str, message: str) -> PydanticCustomError:
    return PydanticCustomError(f"binary_power_sum.{reason}", message)


class NumberFieldBinaryPowerSumGapProfileRequest(StrictModel):
    """Request a complete finite binary power-sum profile."""

    base: SimpleNumberFieldRealEmbeddingBinding = Field(
        description=(
            "Exact field element q structurally bound to one real embedding "
            "record. The operation recognizes that record and requires "
            "1 < sigma(q) < 2 before enumeration."
        )
    )
    exponent_count: StrictInt = Field(
        ge=0,
        description=(
            "Number m of powers q^0 through q^(m-1). The complete result "
            "retains all 2^m indexed bit-vector representations; owner-local "
            "admission currently permits at most 4,096 representations."
        ),
    )


class BinaryPowerSumValueBucket(StrictModel):
    """One exact quotient-field value and every source representation."""

    value: SimpleNumberFieldElement
    representations: tuple[BinaryPowerSumBitVector, ...] = Field(
        min_length=1,
        max_length=MAX_BINARY_POWER_SUM_SOURCE_REPRESENTATIONS,
    )
    multiplicity: StrictInt = Field(
        ge=1, le=MAX_BINARY_POWER_SUM_SOURCE_REPRESENTATIONS
    )

    @model_validator(mode="before")
    @classmethod
    def require_raw_representation_shape(cls, data: Any) -> Any:
        if not isinstance(data, Mapping):
            return data
        representations = data.get("representations")
        if isinstance(representations, (list, tuple)):
            if len(representations) > MAX_BINARY_POWER_SUM_SOURCE_REPRESENTATIONS:
                raise _validation_error(
                    "bucket_representation_bound",
                    "one collision bucket exceeds the retained representation bound",
                )
            if any(
                isinstance(bits, (list, tuple))
                and len(bits) > MAX_BINARY_POWER_SUM_EXPONENT_COUNT
                for bits in representations
            ):
                raise _validation_error(
                    "bit_vector_bound",
                    "binary power-sum bit vector exceeds the exponent bound",
                )
        return canonicalize_json_containers(data)

    @model_validator(mode="after")
    def bind_multiplicity_and_order(self) -> Self:
        if self.multiplicity != len(self.representations):
            raise _validation_error(
                "bucket_multiplicity",
                "bucket multiplicity must equal its representation count",
            )
        if self.representations != tuple(sorted(self.representations)):
            raise _validation_error(
                "bucket_representation_order",
                "bucket representations must be strictly lexicographically ordered",
            )
        if len(set(self.representations)) != len(self.representations):
            raise _validation_error(
                "bucket_duplicate_representation",
                "a collision bucket cannot repeat a source bit vector",
            )
        return self


class BinaryPowerSumGap(StrictModel):
    """One adjacent positive difference in the selected embedded order."""

    lower_value_index: StrictInt = Field(
        ge=0, le=MAX_BINARY_POWER_SUM_SOURCE_REPRESENTATIONS - 2
    )
    upper_value_index: StrictInt = Field(
        ge=1, le=MAX_BINARY_POWER_SUM_SOURCE_REPRESENTATIONS - 1
    )
    difference: SimpleNumberFieldElement
    positive_enclosure: NumberFieldRealValueEnclosure

    @model_validator(mode="after")
    def require_adjacent_positive_shape(self) -> Self:
        if self.upper_value_index != self.lower_value_index + 1:
            raise _validation_error(
                "gap_indices",
                "a binary power-sum gap must join adjacent value indices",
            )
        if self.positive_enclosure.lower.as_fraction() <= 0:
            raise _validation_error(
                "gap_not_positive",
                "binary power-sum gap evidence must be strictly positive",
            )
        return self


class BinaryPowerSumGapProfile(StrictModel):
    """A source-complete exact finite binary power-sum gap profile.

    Parsing checks bounded source binding, exact quotient-coordinate gap
    reconstruction, representation partition shape, summaries, and positive
    evidence shape. The producer establishes selected-embedding order and the
    value of every retained representation inside one recognized invocation.
    """

    base: SimpleNumberFieldRealEmbeddingBinding
    exponent_count: StrictInt = Field(ge=0, le=MAX_BINARY_POWER_SUM_EXPONENT_COUNT)
    value_buckets: tuple[BinaryPowerSumValueBucket, ...] = Field(
        min_length=1,
        max_length=MAX_BINARY_POWER_SUM_SOURCE_REPRESENTATIONS,
    )
    gaps: tuple[BinaryPowerSumGap, ...] = Field(
        max_length=MAX_BINARY_POWER_SUM_SOURCE_REPRESENTATIONS - 1
    )
    source_representation_count: StrictInt = Field(
        ge=1, le=MAX_BINARY_POWER_SUM_SOURCE_REPRESENTATIONS
    )
    distinct_value_count: StrictInt = Field(
        ge=1, le=MAX_BINARY_POWER_SUM_SOURCE_REPRESENTATIONS
    )
    largest_multiplicity: StrictInt = Field(
        ge=1, le=MAX_BINARY_POWER_SUM_SOURCE_REPRESENTATIONS
    )
    least_gap: SimpleNumberFieldElement | None
    largest_gap: SimpleNumberFieldElement | None
    least_gap_index: StrictInt | None = Field(
        default=None, ge=0, le=MAX_BINARY_POWER_SUM_SOURCE_REPRESENTATIONS - 2
    )
    largest_gap_index: StrictInt | None = Field(
        default=None, ge=0, le=MAX_BINARY_POWER_SUM_SOURCE_REPRESENTATIONS - 2
    )
    ordering: Literal["SELECTED_REAL_EMBEDDING_INCREASING_V1"] = (
        "SELECTED_REAL_EMBEDDING_INCREASING_V1"
    )

    @model_validator(mode="before")
    @classmethod
    def require_raw_profile_cardinality(cls, data: Any) -> Any:
        if not isinstance(data, Mapping):
            return data
        buckets = data.get("value_buckets")
        gaps = data.get("gaps")
        if (
            isinstance(buckets, (list, tuple))
            and len(buckets) > MAX_BINARY_POWER_SUM_SOURCE_REPRESENTATIONS
        ):
            raise _validation_error(
                "value_bucket_bound",
                "binary power-sum result exceeds the value-bucket bound",
            )
        if isinstance(buckets, (list, tuple)):
            raw_representation_count = 0
            raw_bit_slots = 0
            for bucket in buckets:
                if not isinstance(bucket, Mapping):
                    continue
                representations = bucket.get("representations")
                if not isinstance(representations, (list, tuple)):
                    continue
                raw_representation_count += len(representations)
                raw_bit_slots += sum(
                    len(bits)
                    for bits in representations
                    if isinstance(bits, (list, tuple))
                )
            if (
                raw_representation_count > MAX_BINARY_POWER_SUM_SOURCE_REPRESENTATIONS
                or raw_bit_slots > MAX_BINARY_POWER_SUM_REPRESENTATION_BIT_SLOTS
            ):
                raise _validation_error(
                    "representation_partition_bound",
                    "binary power-sum result exceeds the retained source partition bound",
                )
        if isinstance(gaps, (list, tuple)) and len(gaps) >= (
            MAX_BINARY_POWER_SUM_SOURCE_REPRESENTATIONS
        ):
            raise _validation_error(
                "gap_bound",
                "binary power-sum result exceeds the adjacent-gap bound",
            )
        return canonicalize_json_containers(data)

    @model_validator(mode="after")
    def bind_complete_profile_shape(self) -> Self:
        self._require_counts_and_partition()
        self._require_gap_reconstruction()
        self._require_gap_summaries()
        return self

    def _require_counts_and_partition(self) -> None:
        field = self.base.element.presentation
        expected_source_count = 1 << self.exponent_count
        if self.source_representation_count != expected_source_count:
            raise _validation_error(
                "source_representation_count",
                "source representation count must equal 2^exponent_count",
            )
        if self.distinct_value_count != len(self.value_buckets):
            raise _validation_error(
                "distinct_value_count",
                "distinct value count must equal the number of value buckets",
            )
        if len(self.gaps) != len(self.value_buckets) - 1:
            raise _validation_error(
                "gap_count",
                "the gap family must contain one entry per adjacent value pair",
            )
        if any(bucket.value.presentation != field for bucket in self.value_buckets):
            raise _validation_error(
                "bucket_field",
                "every binary power-sum value must belong to the source field",
            )

        value_keys = tuple(
            tuple(
                coordinate.as_fraction()
                for coordinate in bucket.value.coefficients_ascending
            )
            for bucket in self.value_buckets
        )
        if len(set(value_keys)) != len(value_keys):
            raise _validation_error(
                "duplicate_value",
                "binary power-sum value buckets must have distinct field values",
            )

        representations: list[BinaryPowerSumBitVector] = []
        for bucket in self.value_buckets:
            if any(
                len(representation) != self.exponent_count
                for representation in bucket.representations
            ):
                raise _validation_error(
                    "bit_vector_length",
                    "every source bit vector must have exponent_count entries",
                )
            representations.extend(bucket.representations)
        if (
            len(representations) != expected_source_count
            or len(set(representations)) != expected_source_count
        ):
            raise _validation_error(
                "representation_partition",
                "value buckets must partition every binary source bit vector exactly once",
            )
        if self.largest_multiplicity != max(
            bucket.multiplicity for bucket in self.value_buckets
        ):
            raise _validation_error(
                "largest_multiplicity",
                "largest multiplicity must equal the maximum bucket multiplicity",
            )

    def _require_gap_reconstruction(self) -> None:
        field = self.base.element.presentation
        for index, gap in enumerate(self.gaps):
            if (
                gap.lower_value_index != index
                or gap.upper_value_index != index + 1
                or gap.difference.presentation != field
            ):
                raise _validation_error(
                    "gap_source",
                    "every gap must join its corresponding adjacent source-field values",
                )
            lower = self.value_buckets[index].value
            upper = self.value_buckets[index + 1].value
            expected_difference = tuple(
                upper_value.as_fraction() - lower_value.as_fraction()
                for lower_value, upper_value in zip(
                    lower.coefficients_ascending,
                    upper.coefficients_ascending,
                    strict=True,
                )
            )
            actual_difference = tuple(
                coordinate.as_fraction()
                for coordinate in gap.difference.coefficients_ascending
            )
            if actual_difference != expected_difference:
                raise _validation_error(
                    "gap_difference",
                    "every gap must reconstruct as its upper value minus lower value",
                )

    def _require_gap_summaries(self) -> None:
        if not self.gaps:
            if any(
                value is not None
                for value in (
                    self.least_gap,
                    self.largest_gap,
                    self.least_gap_index,
                    self.largest_gap_index,
                )
            ):
                raise _validation_error(
                    "empty_gap_summary",
                    "a gapless profile must omit least and largest gap summaries",
                )
        else:
            if (
                self.least_gap is None
                or self.largest_gap is None
                or self.least_gap_index is None
                or self.largest_gap_index is None
            ):
                raise _validation_error(
                    "nonempty_gap_summary",
                    "a nonempty gap family requires least and largest gap summaries",
                )
            if self.least_gap_index >= len(self.gaps) or self.largest_gap_index >= len(
                self.gaps
            ):
                raise _validation_error(
                    "gap_summary_index",
                    "least and largest gap indices must reference the gap family",
                )
            if (
                self.least_gap != self.gaps[self.least_gap_index].difference
                or self.largest_gap != self.gaps[self.largest_gap_index].difference
            ):
                raise _validation_error(
                    "gap_summary_source",
                    "least and largest gap values must reference their indexed gap entries",
                )
            for label, gap_value, gap_index in (
                ("least", self.least_gap, self.least_gap_index),
                ("largest", self.largest_gap, self.largest_gap_index),
            ):
                first_matching_index = next(
                    index
                    for index, gap in enumerate(self.gaps)
                    if gap.difference == gap_value
                )
                if gap_index != first_matching_index:
                    raise _validation_error(
                        "gap_summary_first_match",
                        f"{label} gap index must be the first exactly matching gap",
                    )

    @classmethod
    def _from_kernel(
        cls,
        *,
        base: SimpleNumberFieldRealEmbeddingBinding,
        exponent_count: int,
        value_buckets: tuple[BinaryPowerSumValueBucket, ...],
        gaps: tuple[BinaryPowerSumGap, ...],
        least_gap_index: int | None,
        largest_gap_index: int | None,
    ) -> Self:
        return cls.model_construct(
            base=base,
            exponent_count=exponent_count,
            value_buckets=value_buckets,
            gaps=gaps,
            source_representation_count=1 << exponent_count,
            distinct_value_count=len(value_buckets),
            largest_multiplicity=max(bucket.multiplicity for bucket in value_buckets),
            least_gap=(
                None if least_gap_index is None else gaps[least_gap_index].difference
            ),
            largest_gap=(
                None
                if largest_gap_index is None
                else gaps[largest_gap_index].difference
            ),
            least_gap_index=least_gap_index,
            largest_gap_index=largest_gap_index,
            ordering="SELECTED_REAL_EMBEDDING_INCREASING_V1",
        )


class BinaryPowerSumAdmissionError(ValueError):
    """A proved owner-local rejection for a complete power-sum profile."""

    def __init__(self, reason: str, message: str) -> None:
        self.reason = reason
        super().__init__(message)


@dataclass(frozen=True, slots=True)
class BinaryPowerSumAdmission:
    source_representation_count: int
    representation_bit_slots: int
    base_slice_comparison_bound: int
    frontier_addition_bound: int
    power_multiplication_bound: int
    gap_subtraction_bound: int
    gap_certification_bound: int
    sort_comparison_bound: int
    summary_comparison_bound: int
    field_operation_count: int
    comparison_count: int
    coordinate_numerator_bound: int
    coordinate_denominator_bound: int
    comparison_admission: RealEmbeddingDifferenceAdmission
    base_lower_admission: RealEmbeddingDifferenceAdmission
    base_upper_admission: RealEmbeddingDifferenceAdmission
    predicted_result_bytes: int

    def work_bounds(self) -> dict[str, int]:
        """Return the preflight charge for every executed mathematical phase."""

        return {
            "base_slice_comparison": self.base_slice_comparison_bound,
            "frontier_addition": self.frontier_addition_bound,
            "power_multiplication": self.power_multiplication_bound,
            "gap_subtraction": self.gap_subtraction_bound,
            "gap_certification": self.gap_certification_bound,
            "sort_comparison": self.sort_comparison_bound,
            "summary_comparison": self.summary_comparison_bound,
        }


@dataclass(slots=True)
class _BinaryPowerSumWorkLedger:
    """Actual mathematical primitives executed by one admitted profile."""

    base_slice_comparisons: int = 0
    frontier_additions: int = 0
    power_multiplications: int = 0
    gap_subtractions: int = 0
    gap_certifications: int = 0
    sort_comparisons: int = 0
    summary_comparisons: int = 0

    @property
    def field_operation_count(self) -> int:
        return (
            self.frontier_additions + self.power_multiplications + self.gap_subtractions
        )

    @property
    def comparison_count(self) -> int:
        return (
            self.base_slice_comparisons
            + self.gap_certifications
            + self.sort_comparisons
            + self.summary_comparisons
        )

    def phase_counts(self) -> dict[str, int]:
        return {
            "base_slice_comparison": self.base_slice_comparisons,
            "frontier_addition": self.frontier_additions,
            "power_multiplication": self.power_multiplications,
            "gap_subtraction": self.gap_subtractions,
            "gap_certification": self.gap_certifications,
            "sort_comparison": self.sort_comparisons,
            "summary_comparison": self.summary_comparisons,
        }


@dataclass(frozen=True, slots=True)
class _BinaryPowerSumDeadlineTrace:
    profile_deadline: float
    embedding_recognition_deadline: float | None
    resumed_profile_deadline: float | None


@dataclass(frozen=True, slots=True)
class _BinaryPowerSumExecution:
    result: BinaryPowerSumGapProfile
    admission: BinaryPowerSumAdmission
    work: _BinaryPowerSumWorkLedger
    deadlines: _BinaryPowerSumDeadlineTrace


def _require_work_within_admission(
    admission: BinaryPowerSumAdmission,
    work: _BinaryPowerSumWorkLedger,
) -> None:
    bounds = admission.work_bounds()
    for phase, actual in work.phase_counts().items():
        if actual > bounds[phase]:
            raise RuntimeError(
                f"binary power-sum {phase} work exceeded its admitted bound"
            )
    if work.field_operation_count > admission.field_operation_count:
        raise RuntimeError("binary power-sum field work exceeded its admitted bound")
    if work.comparison_count > admission.comparison_count:
        raise RuntimeError(
            "binary power-sum comparison work exceeded its admitted bound"
        )


def _decimal_digits_from_bits(bits: int) -> int:
    return (max(bits, 1) * 30_103) // 100_000 + 1


def _element_coordinates(element: SimpleNumberFieldElement) -> tuple[Fraction, ...]:
    return tuple(
        coordinate.as_fraction() for coordinate in element.coefficients_ascending
    )


def _coordinates_with_constant_shift(
    element: SimpleNumberFieldElement,
    shift: int,
) -> tuple[Fraction, ...]:
    coordinates = list(_element_coordinates(element))
    coordinates[0] -= shift
    return tuple(coordinates)


def _power_sum_coordinate_bounds(
    base: SimpleNumberFieldElement,
    exponent_count: int,
) -> tuple[int, int]:
    """Bound reduced numerator/denominator components of every subset sum.

    Write ``q=H(alpha)/D`` with integer ``H``. A power ``H^i`` has L1 norm at
    most ``||H||_1^i``. Reducing a monomial of degree ``k`` through a degree-n
    relation needs at most ``k-n+1`` recursive substitutions; after clearing
    the leading coefficient, ``(a_n+n*height(f))`` bounds every branch-and-scale
    step. All power denominators divide the one for the largest exponent, so
    summing their scaled numerator bounds covers every binary subset.
    """

    if exponent_count <= 1:
        return 1, 1

    coordinates = _element_coordinates(base)
    if exponent_count == 2:
        numerator_bound = 1
        denominator_bound = 1
        for index, coordinate in enumerate(coordinates):
            numerator_bound = max(numerator_bound, abs(coordinate.numerator))
            denominator_bound = max(denominator_bound, coordinate.denominator)
            if index == 0:
                numerator_bound = max(
                    numerator_bound,
                    abs(coordinate.numerator + coordinate.denominator),
                )
        return numerator_bound, denominator_bound

    denominator = 1
    for coordinate in coordinates:
        denominator = lcm(denominator, coordinate.denominator)
    integer_coordinates = tuple(
        coordinate.numerator * (denominator // coordinate.denominator)
        for coordinate in coordinates
    )
    numerator_l1 = sum(abs(value) for value in integer_coordinates)
    field = base.presentation
    defining = tuple(
        parse_canonical_integer(value) for value in field.coefficients_descending
    )
    degree = field.degree
    base_polynomial_degree = max(
        (index for index, coefficient in enumerate(integer_coordinates) if coefficient),
        default=0,
    )
    leading = defining[0]
    height = max(abs(value) for value in defining)
    reduction_step_bound = leading + degree * height

    largest_exponent = exponent_count - 1
    largest_degree = largest_exponent * base_polynomial_degree
    largest_reduction_depth = max(largest_degree - degree + 1, 0)
    common_denominator = (
        denominator**largest_exponent * leading**largest_reduction_depth
    )
    numerator_bound = 0
    for exponent in range(exponent_count):
        polynomial_degree = exponent * base_polynomial_degree
        reduction_depth = max(polynomial_degree - degree + 1, 0)
        reduced_numerator_bound = (
            numerator_l1**exponent * reduction_step_bound**reduction_depth
        )
        numerator_bound += (
            reduced_numerator_bound
            * denominator ** (largest_exponent - exponent)
            * leading ** (largest_reduction_depth - reduction_depth)
        )
    return max(numerator_bound, 1), max(common_denominator, 1)


def _predicted_result_bytes(
    base: SimpleNumberFieldRealEmbeddingBinding,
    *,
    exponent_count: int,
    representation_count: int,
    coordinate_digits: int,
    isolator_digits: int,
) -> int:
    field = base.element.presentation
    source_bytes = len(encode_strict_json(base.model_dump(mode="json")))
    field_bytes = len(encode_strict_json(field.model_dump(mode="json")))
    rational_bytes = 2 * coordinate_digits + 32
    element_bytes = field_bytes + field.degree * rational_bytes + 256
    enclosure_bytes = 4 * isolator_digits + 192
    representation_bytes = representation_count * (2 * exponent_count + 8)
    value_bytes = representation_count * (element_bytes + 192)
    gap_bytes = max(representation_count - 1, 0) * (
        element_bytes + enclosure_bytes + 256
    )
    # Two summary elements are retained in the largest possible result. The
    # fixed allowance covers field names, counters, indices, enums, and JSON
    # punctuation in both the result and outer operation projection.
    return (
        8_192
        + source_bytes
        + representation_bytes
        + value_bytes
        + gap_bytes
        + 2 * element_bytes
    )


def admit_binary_power_sum_gap_profile(
    base: SimpleNumberFieldRealEmbeddingBinding,
    exponent_count: int,
) -> BinaryPowerSumAdmission:
    """Preflight exhaustive source, arithmetic, comparison, and exact output."""

    if type(exponent_count) is not int:
        raise BinaryPowerSumAdmissionError(
            "invalid_exponent_count", "exponent_count must be a strict integer"
        )
    if exponent_count < 0:
        raise BinaryPowerSumAdmissionError(
            "negative_exponent_count", "exponent_count must be nonnegative"
        )
    if exponent_count > MAX_BINARY_POWER_SUM_EXPONENT_COUNT:
        raise BinaryPowerSumAdmissionError(
            "source_representation_bound",
            "the complete binary source exceeds the "
            f"{MAX_BINARY_POWER_SUM_SOURCE_REPRESENTATIONS:,}-representation bound",
        )
    representation_count = 1 << exponent_count
    representation_bit_slots = exponent_count * representation_count
    if representation_bit_slots > MAX_BINARY_POWER_SUM_REPRESENTATION_BIT_SLOTS:
        raise BinaryPowerSumAdmissionError(
            "representation_bit_bound",
            "retained bit vectors exceed the "
            f"{MAX_BINARY_POWER_SUM_REPRESENTATION_BIT_SLOTS:,}-bit-slot bound",
        )

    base_slice_comparison_bound = 2
    frontier_addition_bound = representation_count - 1
    power_multiplication_bound = max(exponent_count - 1, 0)
    gap_subtraction_bound = representation_count - 1
    gap_certification_bound = representation_count - 1
    # CPython's stable comparison sort is charged conservatively above the
    # binary information bound. Every actual comparator call is counted below.
    sort_comparison_bound = (
        0 if representation_count == 1 else representation_count * (exponent_count + 2)
    )
    summary_comparison_bound = 2 * max(representation_count - 2, 0)
    field_operation_count = (
        frontier_addition_bound + power_multiplication_bound + gap_subtraction_bound
    )
    if field_operation_count > MAX_BINARY_POWER_SUM_FIELD_OPERATIONS:
        raise BinaryPowerSumAdmissionError(
            "field_operation_bound",
            "binary power-sum recurrence exceeds the exact field-operation bound",
        )
    comparison_count = (
        base_slice_comparison_bound
        + gap_certification_bound
        + sort_comparison_bound
        + summary_comparison_bound
    )
    if comparison_count > MAX_BINARY_POWER_SUM_COMPARISONS:
        raise BinaryPowerSumAdmissionError(
            "comparison_bound",
            "complete selected-embedding order exceeds the exact comparison bound",
        )

    numerator_bound, denominator_bound = _power_sum_coordinate_bounds(
        base.element, exponent_count
    )
    component_limit = 10**MAX_SIMPLE_NUMBER_FIELD_ELEMENT_DIGITS - 1
    difference_numerator_bound = 2 * numerator_bound
    if (
        numerator_bound > component_limit
        or difference_numerator_bound > component_limit
        or denominator_bound > component_limit
    ):
        raise BinaryPowerSumAdmissionError(
            "element_coordinate_bound",
            "the reduced power-sum coordinate envelope exceeds the canonical "
            f"{MAX_SIMPLE_NUMBER_FIELD_ELEMENT_DIGITS}-digit field-element bound",
        )

    try:
        if exponent_count <= 1:
            comparison_admission = admit_real_embedding_difference(
                base.element.presentation,
                (
                    Fraction(1),
                    *(Fraction(0) for _ in range(base.element.presentation.degree - 1)),
                ),
            )
        elif all(
            coordinate == 0 for coordinate in _element_coordinates(base.element)[1:]
        ):
            comparison_admission = admit_real_embedding_difference(
                base.element.presentation,
                (
                    Fraction(difference_numerator_bound, denominator_bound),
                    *(Fraction(0) for _ in range(base.element.presentation.degree - 1)),
                ),
            )
        else:
            comparison_admission = admit_real_embedding_difference_envelope(
                base.element.presentation,
                coordinate_numerator_bound=difference_numerator_bound,
                coordinate_denominator_bound=denominator_bound,
            )
        base_lower_admission = admit_real_embedding_difference(
            base.element.presentation,
            _coordinates_with_constant_shift(base.element, 1),
        )
        base_upper_admission = admit_real_embedding_difference(
            base.element.presentation,
            _coordinates_with_constant_shift(base.element, 2),
        )
    except NumberFieldRealEmbeddingOrderError as exc:
        raise BinaryPowerSumAdmissionError(exc.reason, str(exc)) from exc

    coordinate_digits = max(
        _decimal_digits_from_bits(difference_numerator_bound.bit_length()),
        _decimal_digits_from_bits(denominator_bound.bit_length()),
    )
    isolator_digits = comparison_admission.predicted_isolator_component_digits
    if all(coordinate == 0 for coordinate in _element_coordinates(base.element)[1:]):
        isolator_digits = max(isolator_digits, coordinate_digits)
    predicted_result_bytes = _predicted_result_bytes(
        base,
        exponent_count=exponent_count,
        representation_count=representation_count,
        coordinate_digits=coordinate_digits,
        isolator_digits=isolator_digits,
    )
    if predicted_result_bytes > MAX_BINARY_POWER_SUM_RESULT_BYTES:
        raise BinaryPowerSumAdmissionError(
            "result_byte_bound",
            "the complete retained power-sum profile exceeds the "
            f"{MAX_BINARY_POWER_SUM_RESULT_BYTES:,}-byte result bound",
        )
    return BinaryPowerSumAdmission(
        source_representation_count=representation_count,
        representation_bit_slots=representation_bit_slots,
        base_slice_comparison_bound=base_slice_comparison_bound,
        frontier_addition_bound=frontier_addition_bound,
        power_multiplication_bound=power_multiplication_bound,
        gap_subtraction_bound=gap_subtraction_bound,
        gap_certification_bound=gap_certification_bound,
        sort_comparison_bound=sort_comparison_bound,
        summary_comparison_bound=summary_comparison_bound,
        field_operation_count=field_operation_count,
        comparison_count=comparison_count,
        coordinate_numerator_bound=numerator_bound,
        coordinate_denominator_bound=denominator_bound,
        comparison_admission=comparison_admission,
        base_lower_admission=base_lower_admission,
        base_upper_admission=base_upper_admission,
        predicted_result_bytes=predicted_result_bytes,
    )


def _require_execution_active(deadline: float, phase: str) -> None:
    if embeddings_worker_cancelled():
        raise OperationExecutionCancelledError(f"request cancelled {phase}")
    if deadline <= time.monotonic():
        raise OperationExecutionTimeoutError(f"request deadline expired {phase}")


def _canonical_rational(value: Fraction) -> CanonicalRational:
    return CanonicalRational.from_integer_ratio(value.numerator, value.denominator)


def _enclosure(
    lower: Fraction,
    upper: Fraction,
) -> NumberFieldRealValueEnclosure:
    return NumberFieldRealValueEnclosure(
        lower=_canonical_rational(lower),
        upper=_canonical_rational(upper),
        interval_type="SINGLETON" if lower == upper else "CLOSED",
    )


def _rational_component_digits(value: Fraction) -> int:
    return max(
        _decimal_digits_from_bits(abs(value.numerator).bit_length()),
        _decimal_digits_from_bits(value.denominator.bit_length()),
    )


class _SelectedRealEmbeddingEvaluator:
    """One recognized, monotonically refined exact real-axis evaluator."""

    def __init__(
        self,
        context: RecognizedRealEmbeddingContext,
        *,
        comparison_admission: RealEmbeddingDifferenceAdmission,
        deadline: float,
    ) -> None:
        self.context = context
        self.comparison_admission = comparison_admission
        self.deadline = deadline
        interval = context.record.isolating_interval
        self.lower = interval.lower.as_fraction()
        self.upper = interval.upper.as_fraction()
        self._cache: dict[
            tuple[Fraction, ...],
            tuple[
                SimpleNumberFieldRealOrder,
                NumberFieldRealValueEnclosure,
            ],
        ] = {}
        self._certified_cache: dict[
            tuple[Fraction, ...],
            tuple[
                SimpleNumberFieldRealOrder,
                NumberFieldRealValueEnclosure,
            ],
        ] = {}

    @staticmethod
    def _coordinate_key(element: SimpleNumberFieldElement) -> tuple[Fraction, ...]:
        return _element_coordinates(element)

    def _power_basis_enclosure(
        self,
        coordinates: tuple[Fraction, ...],
    ) -> tuple[Fraction, Fraction]:
        midpoint = (self.lower + self.upper) / 2
        value = Fraction(0)
        for coefficient in reversed(coordinates):
            value = value * midpoint + coefficient
        magnitude = max(abs(self.lower), abs(self.upper))
        derivative_bound = sum(
            index * abs(coefficient) * magnitude ** (index - 1)
            for index, coefficient in enumerate(coordinates)
            if index
        )
        error = (self.upper - self.lower) * derivative_bound / 2
        return value - error, value + error

    def _refine_selected_root(self, bits: int) -> None:
        if self.lower == self.upper:
            return
        _require_execution_active(self.deadline, "before selected-root refinement")
        import sympy

        lower, upper = self.context.polynomial.refine_root(
            sympy.Rational(self.lower.numerator, self.lower.denominator),
            sympy.Rational(self.upper.numerator, self.upper.denominator),
            eps=sympy.Rational(1, 2**bits),
            fast=True,
        )
        self.lower = Fraction(int(lower.p), int(lower.q))
        self.upper = Fraction(int(upper.p), int(upper.q))
        _require_execution_active(self.deadline, "after selected-root refinement")

    @staticmethod
    def _invert(
        result: tuple[
            SimpleNumberFieldRealOrder,
            NumberFieldRealValueEnclosure,
        ],
    ) -> tuple[
        SimpleNumberFieldRealOrder,
        NumberFieldRealValueEnclosure,
    ]:
        order, enclosure = result
        inverse_order: SimpleNumberFieldRealOrder = (
            "LT" if order == "GT" else "GT" if order == "LT" else "EQ"
        )
        return (
            inverse_order,
            _enclosure(
                -enclosure.upper.as_fraction(),
                -enclosure.lower.as_fraction(),
            ),
        )

    def sign(
        self,
        backend_value: Any,
        public_value: SimpleNumberFieldElement,
        admission: RealEmbeddingDifferenceAdmission,
    ) -> tuple[
        SimpleNumberFieldRealOrder,
        NumberFieldRealValueEnclosure,
    ]:
        _require_execution_active(self.deadline, "during selected-embedding order")
        coordinates = self._coordinate_key(public_value)
        cached = self._cache.get(coordinates)
        if cached is not None:
            return cached
        inverse = self._cache.get(tuple(-coordinate for coordinate in coordinates))
        if inverse is not None:
            result = self._invert(inverse)
            self._cache[coordinates] = result
            return result
        if all(coordinate == 0 for coordinate in coordinates):
            result = ("EQ", _enclosure(Fraction(0), Fraction(0)))
            self._cache[coordinates] = result
            return result

        lower, upper = self._power_basis_enclosure(coordinates)
        if lower <= 0 <= upper:
            self._refine_selected_root(
                min(
                    max(admission.root_refinement_bits, 64),
                    MAX_BINARY_POWER_SUM_SHARED_ROOT_REFINEMENT_BITS,
                )
            )
            lower, upper = self._power_basis_enclosure(coordinates)

        if (
            (upper < 0 or lower > 0)
            and _rational_component_digits(lower)
            <= MAX_NUMBER_FIELD_ISOLATOR_COMPONENT_DIGITS
            and _rational_component_digits(upper)
            <= MAX_NUMBER_FIELD_ISOLATOR_COMPONENT_DIGITS
        ):
            order: SimpleNumberFieldRealOrder = "LT" if upper < 0 else "GT"
            result = (
                order,
                _enclosure(lower, upper),
            )
        else:
            order, isolating_interval = isolate_backend_real_value(
                self.context,
                backend_value,
                admission,
            )
            result = (
                order,
                _enclosure(
                    isolating_interval.lower.as_fraction(),
                    isolating_interval.upper.as_fraction(),
                ),
            )
        self._cache[coordinates] = result
        return result

    def compare(
        self,
        left_backend: Any,
        right_backend: Any,
    ) -> int:
        difference_backend = left_backend - right_backend
        difference_public = self.context.from_backend(difference_backend)
        order, _enclosure_value = self.sign(
            difference_backend,
            difference_public,
            self.comparison_admission,
        )
        if order == "EQ":
            return 0
        return -1 if order == "LT" else 1

    def certify(
        self,
        backend_value: Any,
        public_value: SimpleNumberFieldElement,
    ) -> tuple[
        SimpleNumberFieldRealOrder,
        NumberFieldRealValueEnclosure,
    ]:
        """Return bounded public evidence from the admitted image polynomial."""

        coordinates = self._coordinate_key(public_value)
        cached = self._certified_cache.get(coordinates)
        if cached is not None:
            return cached
        _require_execution_active(
            self.deadline, "before selected-image evidence isolation"
        )
        order, isolating_interval = isolate_backend_real_value(
            self.context,
            backend_value,
            self.comparison_admission,
        )
        _require_execution_active(
            self.deadline, "after selected-image evidence isolation"
        )
        result: tuple[
            SimpleNumberFieldRealOrder,
            NumberFieldRealValueEnclosure,
        ] = (
            order,
            _enclosure(
                isolating_interval.lower.as_fraction(),
                isolating_interval.upper.as_fraction(),
            ),
        )
        self._certified_cache[coordinates] = result
        return result


@dataclass(frozen=True, slots=True)
class _PowerSumBucket:
    backend_value: Any
    public_value: SimpleNumberFieldElement
    representations: tuple[BinaryPowerSumBitVector, ...]


def _constant_element(
    context: RecognizedRealEmbeddingContext,
    value: int,
) -> SimpleNumberFieldElement:
    coordinates = (Fraction(value),) + (Fraction(0),) * (
        context.presentation.degree - 1
    )
    return SimpleNumberFieldElement(
        presentation=context.presentation,
        coefficients_ascending=tuple(
            _canonical_rational(coordinate) for coordinate in coordinates
        ),
    )


def _recognize_profile_base(
    base: SimpleNumberFieldRealEmbeddingBinding,
    *,
    profile_deadline: float,
) -> tuple[RecognizedRealEmbeddingContext, _BinaryPowerSumDeadlineTrace]:
    """Recognize the selected root without widening the caller's envelope."""

    try:
        context = recognize_real_embedding_binding(base)
    except NumberFieldRealEmbeddingOrderError as exc:
        raise BinaryPowerSumAdmissionError(exc.reason, str(exc)) from exc
    recognition_execution = current_request_execution()
    recognition_deadline = (
        recognition_execution.deadline if recognition_execution is not None else None
    )
    if recognition_deadline is not None and recognition_deadline > profile_deadline:
        raise RuntimeError("embedding recognition extended the profile deadline")

    # The embedding producer applies its stricter one-worker subdeadline. The
    # remaining admitted profile work resumes under this operation's envelope.
    bind_request_deadline(profile_deadline)
    resumed_execution = current_request_execution()
    resumed_deadline = (
        resumed_execution.deadline if resumed_execution is not None else None
    )
    if resumed_deadline is not None and resumed_deadline != profile_deadline:
        raise RuntimeError("binary power-sum profile deadline was not restored")
    return context, _BinaryPowerSumDeadlineTrace(
        profile_deadline=profile_deadline,
        embedding_recognition_deadline=recognition_deadline,
        resumed_profile_deadline=resumed_deadline,
    )


def _execute_binary_power_sum_gap_profile(
    base: SimpleNumberFieldRealEmbeddingBinding,
    exponent_count: int,
) -> _BinaryPowerSumExecution:
    """Execute one profile and retain its owner-local accounting evidence."""

    execution = current_request_execution()
    started = execution.started_at if execution is not None else time.monotonic()
    owner_deadline = started + BINARY_POWER_SUM_WALL_SECONDS
    deadline = (
        min(execution.deadline, owner_deadline)
        if execution is not None and execution.deadline is not None
        else owner_deadline
    )
    bind_request_deadline(deadline)
    _require_execution_active(deadline, "before binary power-sum admission")
    admission = admit_binary_power_sum_gap_profile(base, exponent_count)
    _require_execution_active(deadline, "after binary power-sum admission")
    work = _BinaryPowerSumWorkLedger()

    context, deadlines = _recognize_profile_base(
        base,
        profile_deadline=deadline,
    )
    _require_execution_active(deadline, "after selected-embedding recognition")
    evaluator = _SelectedRealEmbeddingEvaluator(
        context,
        comparison_admission=admission.comparison_admission,
        deadline=deadline,
    )
    recognized_base = SimpleNumberFieldRealEmbeddingBinding(
        element=base.element,
        embedding_record=context.record,
    )
    backend_base = context.to_backend(base.element)
    one = _constant_element(context, 1)
    two = _constant_element(context, 2)
    lower_difference = backend_base - context.to_backend(one)
    work.base_slice_comparisons += 1
    lower_order, _lower_enclosure = evaluator.sign(
        lower_difference,
        context.from_backend(lower_difference),
        admission.base_lower_admission,
    )
    upper_difference = backend_base - context.to_backend(two)
    work.base_slice_comparisons += 1
    upper_order, _upper_enclosure = evaluator.sign(
        upper_difference,
        context.from_backend(upper_difference),
        admission.base_upper_admission,
    )
    if lower_order != "GT" or upper_order != "LT":
        raise BinaryPowerSumAdmissionError(
            "base_interval",
            "the selected real image of q must satisfy the strict slice 1 < sigma(q) < 2",
        )

    backend_buckets: dict[Any, list[BinaryPowerSumBitVector]] = {
        context.algebraic_field.zero: [()]
    }
    power = context.algebraic_field.one
    for exponent in range(exponent_count):
        _require_execution_active(deadline, f"before power-sum frontier {exponent}")
        next_buckets: dict[Any, list[BinaryPowerSumBitVector]] = {}
        for value, representations in backend_buckets.items():
            unshifted = next_buckets.setdefault(value, [])
            work.frontier_additions += 1
            shifted = next_buckets.setdefault(value + power, [])
            unshifted.extend((*representation, 0) for representation in representations)
            shifted.extend((*representation, 1) for representation in representations)
        backend_buckets = next_buckets
        if exponent + 1 < exponent_count:
            work.power_multiplications += 1
            power *= backend_base
        _require_execution_active(deadline, f"after power-sum frontier {exponent}")

    buckets = [
        _PowerSumBucket(
            backend_value=backend_value,
            public_value=context.from_backend(backend_value),
            representations=tuple(sorted(representations)),
        )
        for backend_value, representations in backend_buckets.items()
    ]

    def compare_buckets(left: _PowerSumBucket, right: _PowerSumBucket) -> int:
        work.sort_comparisons += 1
        if work.sort_comparisons > admission.sort_comparison_bound:
            raise RuntimeError(
                "binary power-sum sort exceeded its admitted comparison bound"
            )
        return evaluator.compare(
            left.backend_value,
            right.backend_value,
        )

    buckets.sort(key=cmp_to_key(compare_buckets))
    value_buckets = tuple(
        BinaryPowerSumValueBucket(
            value=bucket.public_value,
            representations=bucket.representations,
            multiplicity=len(bucket.representations),
        )
        for bucket in buckets
    )

    gaps: list[BinaryPowerSumGap] = []
    backend_gaps: list[Any] = []
    for index, (lower_bucket, upper_bucket) in enumerate(pairwise(buckets)):
        work.gap_subtractions += 1
        backend_difference = upper_bucket.backend_value - lower_bucket.backend_value
        public_difference = context.from_backend(backend_difference)
        work.gap_certifications += 1
        order, positive_enclosure = evaluator.certify(
            backend_difference,
            public_difference,
        )
        if order != "GT":
            raise RuntimeError(
                "exact embedded sort produced a nonpositive adjacent gap"
            )
        gaps.append(
            BinaryPowerSumGap(
                lower_value_index=index,
                upper_value_index=index + 1,
                difference=public_difference,
                positive_enclosure=positive_enclosure,
            )
        )
        backend_gaps.append(backend_difference)

    least_gap_index: int | None = None
    largest_gap_index: int | None = None
    if gaps:
        least_gap_index = largest_gap_index = 0
        for index in range(1, len(gaps)):
            work.summary_comparisons += 1
            if (
                evaluator.compare(
                    backend_gaps[index],
                    backend_gaps[least_gap_index],
                )
                < 0
            ):
                least_gap_index = index
            work.summary_comparisons += 1
            if (
                evaluator.compare(
                    backend_gaps[index],
                    backend_gaps[largest_gap_index],
                )
                > 0
            ):
                largest_gap_index = index

    result = BinaryPowerSumGapProfile._from_kernel(
        base=recognized_base,
        exponent_count=exponent_count,
        value_buckets=value_buckets,
        gaps=tuple(gaps),
        least_gap_index=least_gap_index,
        largest_gap_index=largest_gap_index,
    )
    _require_work_within_admission(admission, work)
    _require_execution_active(deadline, "after binary power-sum result construction")
    actual_result_bytes = len(encode_strict_json(result.model_dump(mode="json")))
    if actual_result_bytes > admission.predicted_result_bytes:
        raise RuntimeError("binary power-sum result exceeded its admitted byte bound")
    _require_execution_active(deadline, "after binary power-sum serialization check")
    return _BinaryPowerSumExecution(
        result=result,
        admission=admission,
        work=work,
        deadlines=deadlines,
    )


def binary_power_sum_gap_profile(
    base: SimpleNumberFieldRealEmbeddingBinding,
    exponent_count: int,
) -> BinaryPowerSumGapProfile:
    """Return every exact binary power sum, collision, and consecutive gap."""

    return _execute_binary_power_sum_gap_profile(base, exponent_count).result


__all__ = [
    "BINARY_POWER_SUM_WALL_SECONDS",
    "MAX_BINARY_POWER_SUM_COMPARISONS",
    "MAX_BINARY_POWER_SUM_EXPONENT_COUNT",
    "MAX_BINARY_POWER_SUM_FIELD_OPERATIONS",
    "MAX_BINARY_POWER_SUM_REPRESENTATION_BIT_SLOTS",
    "MAX_BINARY_POWER_SUM_RESULT_BYTES",
    "MAX_BINARY_POWER_SUM_SHARED_ROOT_REFINEMENT_BITS",
    "MAX_BINARY_POWER_SUM_SOURCE_REPRESENTATIONS",
    "BinaryPowerSumAdmission",
    "BinaryPowerSumAdmissionError",
    "BinaryPowerSumBitVector",
    "BinaryPowerSumGap",
    "BinaryPowerSumGapProfile",
    "BinaryPowerSumValueBucket",
    "NumberFieldBinaryPowerSumGapProfileRequest",
    "admit_binary_power_sum_gap_profile",
    "binary_power_sum_gap_profile",
]
