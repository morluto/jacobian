"""Complete exact binary power-sum collision and gap profiles."""

from __future__ import annotations

import json
import time
from collections import defaultdict
from fractions import Fraction
from functools import cmp_to_key

import pytest
from pydantic import ValidationError
from tests.fixtures.accounting import assert_charged_work_parity

from jacobian._exact import CanonicalRational
from jacobian._execution import (
    OperationExecutionTimeoutError,
    bind_request_deadline,
    current_request_execution,
    request_execution,
)
from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.number_theory.number_fields import (
    BinaryPowerSumGap,
    BinaryPowerSumGapProfile,
    SimpleNumberFieldElement,
    SimpleNumberFieldPresentation,
    SimpleNumberFieldRealEmbeddingBinding,
    binary_power_sum_gap_profile,
    embeddings,
    verify_binary_power_sum_gap_profile,
)
from jacobian.math.number_theory.number_fields._binary_power_sum import (
    BINARY_POWER_SUM_WALL_SECONDS,
    MAX_BINARY_POWER_SUM_EXPONENT_COUNT,
    BinaryPowerSumAdmissionError,
    NumberFieldBinaryPowerSumGapProfileRequest,
    _execute_binary_power_sum_gap_profile,
    admit_binary_power_sum_gap_profile,
)
from jacobian.math.number_theory.number_fields._embeddings_process import (
    EMBEDDINGS_WORKER_WALL_SECONDS,
)
from jacobian.math.number_theory.number_fields._tools import TOOLS
from jacobian.math.number_theory.number_fields.values import (
    RealNumberFieldEmbeddingRecord,
)

type QuadraticPair = tuple[Fraction, Fraction]


def _field(*coefficients: str) -> SimpleNumberFieldPresentation:
    return SimpleNumberFieldPresentation(
        coefficients_descending=tuple(int(coefficient) for coefficient in coefficients)
    )


def _element(
    field: SimpleNumberFieldPresentation,
    *coordinates: int | Fraction,
) -> SimpleNumberFieldElement:
    return SimpleNumberFieldElement(
        presentation=field,
        coefficients_ascending=tuple(
            CanonicalRational.from_fraction(Fraction(coordinate))
            for coordinate in coordinates
        ),
    )


def _binding(
    element: SimpleNumberFieldElement,
    record: RealNumberFieldEmbeddingRecord,
) -> SimpleNumberFieldRealEmbeddingBinding:
    return SimpleNumberFieldRealEmbeddingBinding(
        element=element,
        embedding_record=record,
    )


def _real_records(
    field: SimpleNumberFieldPresentation,
) -> tuple[RealNumberFieldEmbeddingRecord, ...]:
    records = embeddings(field).records
    assert all(isinstance(record, RealNumberFieldEmbeddingRecord) for record in records)
    return records  # type: ignore[return-value]


@pytest.fixture(scope="module")
def sqrt_two_records() -> tuple[
    RealNumberFieldEmbeddingRecord, RealNumberFieldEmbeddingRecord
]:
    negative, positive = _real_records(_field("1", "0", "-2"))
    return negative, positive


@pytest.fixture(scope="module")
def golden_ratio_record() -> RealNumberFieldEmbeddingRecord:
    _negative_inverse, golden_ratio = _real_records(_field("1", "-1", "-1"))
    return golden_ratio


def _sqrt_two_add(left: QuadraticPair, right: QuadraticPair) -> QuadraticPair:
    return left[0] + right[0], left[1] + right[1]


def _sqrt_two_multiply(left: QuadraticPair, right: QuadraticPair) -> QuadraticPair:
    return (
        left[0] * right[0] + 2 * left[1] * right[1],
        left[0] * right[1] + left[1] * right[0],
    )


def _sqrt_two_sign(value: QuadraticPair) -> int:
    rational, radical = value
    if radical == 0:
        return (rational > 0) - (rational < 0)
    if rational == 0:
        return (radical > 0) - (radical < 0)
    if rational > 0 and radical > 0:
        return 1
    if rational < 0 and radical < 0:
        return -1
    comparison = rational * rational - 2 * radical * radical
    if rational > 0:
        return (comparison > 0) - (comparison < 0)
    return -((comparison > 0) - (comparison < 0))


def _sqrt_two_compare(left: QuadraticPair, right: QuadraticPair) -> int:
    return _sqrt_two_sign((left[0] - right[0], left[1] - right[1]))


def _sqrt_two_oracle(
    exponent_count: int,
) -> tuple[tuple[QuadraticPair, tuple[tuple[int, ...], ...]], ...]:
    powers: list[QuadraticPair] = []
    power: QuadraticPair = (Fraction(1), Fraction(0))
    for _ in range(exponent_count):
        powers.append(power)
        power = _sqrt_two_multiply(power, (Fraction(0), Fraction(1)))
    buckets: dict[QuadraticPair, list[tuple[int, ...]]] = defaultdict(list)
    for source_index in range(1 << exponent_count):
        bits = tuple((source_index >> index) & 1 for index in range(exponent_count))
        value: QuadraticPair = (Fraction(0), Fraction(0))
        for bit, source_power in zip(bits, powers, strict=True):
            if bit:
                value = _sqrt_two_add(value, source_power)
        buckets[value].append(bits)
    ordered = sorted(buckets, key=cmp_to_key(_sqrt_two_compare))
    return tuple((value, tuple(sorted(buckets[value]))) for value in ordered)


def _pair(value: SimpleNumberFieldElement) -> QuadraticPair:
    first, second = value.coefficients_ascending
    return first.as_fraction(), second.as_fraction()


def test_sqrt_two_profile_matches_an_independent_exact_quadratic_oracle(
    sqrt_two_records: tuple[
        RealNumberFieldEmbeddingRecord, RealNumberFieldEmbeddingRecord
    ],
) -> None:
    record = sqrt_two_records[1]
    result = binary_power_sum_gap_profile(
        _binding(_element(record.embedding.presentation, 0, 1), record),
        3,
    )
    oracle = _sqrt_two_oracle(3)

    assert (
        tuple(
            (_pair(bucket.value), bucket.representations)
            for bucket in result.value_buckets
        )
        == oracle
    )
    assert result.source_representation_count == 8
    assert result.distinct_value_count == 8
    assert result.largest_multiplicity == 1
    for index, gap in enumerate(result.gaps):
        expected = (
            oracle[index + 1][0][0] - oracle[index][0][0],
            oracle[index + 1][0][1] - oracle[index][0][1],
        )
        assert _pair(gap.difference) == expected
        assert _sqrt_two_sign(expected) == 1
        enclosure_lower = gap.positive_enclosure.lower.as_fraction()
        enclosure_upper = gap.positive_enclosure.upper.as_fraction()
        assert enclosure_lower > 0
        assert _sqrt_two_sign((expected[0] - enclosure_lower, expected[1])) >= 0
        assert _sqrt_two_sign((enclosure_upper - expected[0], -expected[1])) >= 0

    oracle_gaps = tuple(_pair(gap.difference) for gap in result.gaps)
    expected_least = min(oracle_gaps, key=cmp_to_key(_sqrt_two_compare))
    expected_largest = max(oracle_gaps, key=cmp_to_key(_sqrt_two_compare))
    assert result.least_gap is not None
    assert result.largest_gap is not None
    assert _pair(result.least_gap) == expected_least
    assert _pair(result.largest_gap) == expected_largest


def test_mixed_denominator_base_admits_every_comparison_operand(
    sqrt_two_records: tuple[
        RealNumberFieldEmbeddingRecord, RealNumberFieldEmbeddingRecord
    ],
) -> None:
    record = sqrt_two_records[1]
    base = _binding(
        _element(
            record.embedding.presentation,
            Fraction(-50, 101),
            Fraction(3, 2),
        ),
        record,
    )

    execution = _execute_binary_power_sum_gap_profile(base, 2)
    admission = execution.admission
    value_coordinates = tuple(
        _pair(bucket.value) for bucket in execution.result.value_buckets
    )
    gap_coordinates = tuple(_pair(gap.difference) for gap in execution.result.gaps)

    assert gap_coordinates == (
        (Fraction(1), Fraction(0)),
        (Fraction(-151, 101), Fraction(3, 2)),
        (Fraction(1), Fraction(0)),
    )

    def difference(left: QuadraticPair, right: QuadraticPair) -> QuadraticPair:
        return left[0] - right[0], left[1] - right[1]

    def fits(coordinates: QuadraticPair, numerator_bound: int) -> bool:
        return all(
            abs(coordinate.numerator) <= numerator_bound
            and coordinate.denominator <= admission.coordinate_denominator_bound
            for coordinate in coordinates
        )

    assert difference(gap_coordinates[1], gap_coordinates[0]) == (
        Fraction(-252, 101),
        Fraction(3, 2),
    )
    # Sorting may compare any two retained values; gap certification receives
    # an adjacent value difference.  Summary selection may compare any two
    # gaps, so its downstream admission must cover a difference of differences.
    assert all(
        fits(
            difference(left, right),
            admission.value_difference_numerator_bound,
        )
        for left in value_coordinates
        for right in value_coordinates
    )
    assert all(
        fits(gap, admission.value_difference_numerator_bound) for gap in gap_coordinates
    )
    assert all(
        fits(
            difference(left, right),
            admission.gap_difference_numerator_bound,
        )
        for left in gap_coordinates
        for right in gap_coordinates
    )


def test_golden_ratio_collision_retains_both_indexed_sources(
    golden_ratio_record: RealNumberFieldEmbeddingRecord,
) -> None:
    field = golden_ratio_record.embedding.presentation
    result = binary_power_sum_gap_profile(
        _binding(_element(field, 0, 1), golden_ratio_record),
        3,
    )
    collision = next(
        bucket
        for bucket in result.value_buckets
        if _pair(bucket.value) == (Fraction(1), Fraction(1))
    )

    assert collision.representations == ((0, 0, 1), (1, 1, 0))
    assert collision.multiplicity == 2
    assert result.source_representation_count == 8
    assert result.distinct_value_count == 7
    assert result.largest_multiplicity == 2


def test_rational_base_agrees_with_an_independent_fraction_profile() -> None:
    field = _field("1", "0")
    (record,) = _real_records(field)
    base = Fraction(3, 2)
    exponent_count = 4
    result = binary_power_sum_gap_profile(
        _binding(_element(field, base), record), exponent_count
    )
    expected: dict[Fraction, list[tuple[int, ...]]] = defaultdict(list)
    for source_index in range(1 << exponent_count):
        bits = tuple((source_index >> index) & 1 for index in range(exponent_count))
        value = sum(
            (base**index for index, bit in enumerate(bits) if bit),
            start=Fraction(0),
        )
        expected[value].append(bits)

    assert tuple(
        (
            bucket.value.coefficients_ascending[0].as_fraction(),
            bucket.representations,
        )
        for bucket in result.value_buckets
    ) == tuple((value, tuple(sorted(expected[value]))) for value in sorted(expected))


@pytest.mark.parametrize("exponent_count", [0, 1])
def test_empty_and_single_power_boundaries_have_exact_gap_semantics(
    exponent_count: int,
) -> None:
    field = _field("1", "0")
    (record,) = _real_records(field)
    result = binary_power_sum_gap_profile(
        _binding(_element(field, Fraction(3, 2)), record), exponent_count
    )

    assert result.source_representation_count == 1 << exponent_count
    if exponent_count == 0:
        assert len(result.value_buckets) == 1
        assert result.value_buckets[0].representations == ((),)
        assert result.gaps == ()
        assert result.least_gap is None
        assert result.largest_gap is None
    else:
        assert tuple(
            bucket.value.coefficients_ascending[0].as_fraction()
            for bucket in result.value_buckets
        ) == (Fraction(0), Fraction(1))
        assert len(result.gaps) == 1
        assert result.least_gap == result.gaps[0].difference
        assert result.largest_gap == result.gaps[0].difference


def test_one_abstract_base_is_sorted_by_each_selected_real_embedding(
    sqrt_two_records: tuple[
        RealNumberFieldEmbeddingRecord, RealNumberFieldEmbeddingRecord
    ],
) -> None:
    negative, positive = sqrt_two_records
    base = _element(negative.embedding.presentation, Fraction(3, 2), Fraction(1, 10))

    negative_profile = binary_power_sum_gap_profile(_binding(base, negative), 3)
    positive_profile = binary_power_sum_gap_profile(_binding(base, positive), 3)

    negative_values = tuple(
        _pair(bucket.value) for bucket in negative_profile.value_buckets
    )
    positive_values = tuple(
        _pair(bucket.value) for bucket in positive_profile.value_buckets
    )
    assert set(negative_values) == set(positive_values)
    assert negative_values != positive_values
    assert negative_values[3] == (Fraction(227, 100), Fraction(3, 10))
    assert positive_values[3] == (Fraction(5, 2), Fraction(1, 10))
    assert negative_profile.base.embedding_record == negative
    assert positive_profile.base.embedding_record == positive


@pytest.mark.parametrize("base_value", [Fraction(1), Fraction(2), Fraction(1, 2)])
def test_first_release_rejects_bases_outside_the_strict_real_slice(
    base_value: Fraction,
) -> None:
    field = _field("1", "0")
    (record,) = _real_records(field)

    with pytest.raises(BinaryPowerSumAdmissionError) as caught:
        binary_power_sum_gap_profile(
            _binding(_element(field, base_value), record),
            3,
        )

    assert caught.value.reason == "base_interval"


def test_consumer_rejects_a_forged_real_embedding_record(
    sqrt_two_records: tuple[
        RealNumberFieldEmbeddingRecord, RealNumberFieldEmbeddingRecord
    ],
) -> None:
    record = sqrt_two_records[1]
    forged_data = record.model_dump(mode="json")
    forged_data["isolating_interval"] = {
        "lower": {"num": "2", "den": "1"},
        "upper": {"num": "3", "den": "1"},
        "interval_type": "OPEN",
    }
    forged = RealNumberFieldEmbeddingRecord.model_validate_json(json.dumps(forged_data))

    with pytest.raises(BinaryPowerSumAdmissionError) as caught:
        binary_power_sum_gap_profile(
            _binding(
                _element(record.embedding.presentation, Fraction(3, 2), 0), forged
            ),
            2,
        )

    assert caught.value.reason == "embedding_record_not_recognized"


def test_nonreal_embedding_cannot_enter_the_real_binding() -> None:
    nonreal = embeddings(_field("1", "0", "1")).records[0]

    with pytest.raises(ValidationError):
        SimpleNumberFieldRealEmbeddingBinding.model_validate(
            {
                "element": _element(
                    nonreal.embedding.presentation, Fraction(3, 2), 0
                ).model_dump(mode="json"),
                "embedding_record": nonreal.model_dump(mode="json"),
            }
        )


def test_exponent_boundary_is_complete_and_next_value_is_rejected() -> None:
    field = _field("1", "0")
    (record,) = _real_records(field)
    base = _binding(_element(field, Fraction(3, 2)), record)

    result = binary_power_sum_gap_profile(base, MAX_BINARY_POWER_SUM_EXPONENT_COUNT)

    assert result.source_representation_count == 4_096
    assert sum(bucket.multiplicity for bucket in result.value_buckets) == 4_096
    with pytest.raises(BinaryPowerSumAdmissionError) as caught:
        binary_power_sum_gap_profile(base, MAX_BINARY_POWER_SUM_EXPONENT_COUNT + 1)
    assert caught.value.reason == "source_representation_bound"

    with pytest.raises(BinaryPowerSumAdmissionError) as strict_caught:
        binary_power_sum_gap_profile(base, True)
    assert strict_caught.value.reason == "invalid_exponent_count"


def test_coordinate_growth_is_rejected_before_embedding_recognition(
    sqrt_two_records: tuple[
        RealNumberFieldEmbeddingRecord, RealNumberFieldEmbeddingRecord
    ],
) -> None:
    record = sqrt_two_records[1]
    field = record.embedding.presentation
    large_denominator = 10**255
    base = _binding(
        _element(field, Fraction(3, 2), Fraction(1, large_denominator)),
        record,
    )

    with pytest.raises(BinaryPowerSumAdmissionError) as caught:
        admit_binary_power_sum_gap_profile(base, 3)

    assert caught.value.reason == "element_coordinate_bound"


def test_structurally_admitted_profile_is_independent_of_serialized_size() -> None:
    coefficients = ("9" * 256, *("8" * 256 for _ in range(7)), "1")
    field = _field(*coefficients)
    record = RealNumberFieldEmbeddingRecord.model_validate_json(
        json.dumps(
            {
                "kind": "REAL",
                "embedding": {
                    "kind": "REAL",
                    "presentation": field.model_dump(mode="json"),
                    "root": {
                        "polynomial": list(coefficients),
                        "real_root_index": 0,
                    },
                },
                "isolating_interval": {
                    "lower": {"num": "0", "den": "1"},
                    "upper": {"num": "1", "den": "1"},
                    "interval_type": "OPEN",
                },
            }
        )
    )
    base = _binding(_element(field, Fraction(3, 2), *(0 for _ in range(7))), record)

    admission = admit_binary_power_sum_gap_profile(base, 12)
    assert admission.source_representation_count == 4_096


def test_result_round_trip_preserves_source_partition_and_gap_reconstruction(
    golden_ratio_record: RealNumberFieldEmbeddingRecord,
) -> None:
    field = golden_ratio_record.embedding.presentation
    result = binary_power_sum_gap_profile(
        _binding(_element(field, 0, 1), golden_ratio_record), 3
    )

    assert (
        BinaryPowerSumGapProfile.model_validate_json(
            result.model_dump_json(), strict=True
        )
        == result
    )
    assert "evidence_basis" not in result.gaps[0].model_dump()
    assert "evidence_basis" not in BinaryPowerSumGap.model_json_schema()["properties"]


def test_result_validation_rejects_duplicate_sources_and_wrong_gap_difference(
    golden_ratio_record: RealNumberFieldEmbeddingRecord,
) -> None:
    field = golden_ratio_record.embedding.presentation
    result = binary_power_sum_gap_profile(
        _binding(_element(field, 0, 1), golden_ratio_record), 3
    )
    duplicate = result.model_dump(mode="json")
    duplicate["value_buckets"][1]["representations"] = [[0, 0, 0]]
    duplicate_claim = BinaryPowerSumGapProfile.model_validate_json(json.dumps(duplicate))
    assert not verify_binary_power_sum_gap_profile(duplicate_claim)

    wrong_gap = result.model_dump(mode="json")
    wrong_gap["gaps"][0]["difference"] = result.value_buckets[0].value.model_dump(
        mode="json"
    )
    wrong_gap_claim = BinaryPowerSumGapProfile.model_validate_json(json.dumps(wrong_gap))
    assert not verify_binary_power_sum_gap_profile(wrong_gap_claim)


@pytest.mark.parametrize("index_field", ["least_gap_index", "largest_gap_index"])
def test_result_validation_rejects_out_of_range_summary_indices_as_typed_errors(
    index_field: str,
) -> None:
    field = _field("1", "0")
    (record,) = _real_records(field)
    result = binary_power_sum_gap_profile(
        _binding(_element(field, Fraction(3, 2)), record), 2
    )
    invalid = result.model_dump(mode="json")
    invalid[index_field] = 100

    claim = BinaryPowerSumGapProfile.model_validate_json(json.dumps(invalid))
    assert not verify_binary_power_sum_gap_profile(claim)


def test_result_validation_requires_first_exactly_matching_gap_summary() -> None:
    field = _field("1", "0")
    (record,) = _real_records(field)
    result = binary_power_sum_gap_profile(
        _binding(_element(field, Fraction(3, 2)), record), 2
    )
    assert result.largest_gap_index == 0
    assert result.gaps[0].difference == result.gaps[2].difference

    noncanonical = result.model_dump(mode="json")
    noncanonical["largest_gap_index"] = 2
    claim = BinaryPowerSumGapProfile.model_validate_json(json.dumps(noncanonical))
    assert not verify_binary_power_sum_gap_profile(claim)


def test_representative_profile_charges_every_executed_math_phase() -> None:
    field = _field("1", "0")
    (record,) = _real_records(field)
    execution = _execute_binary_power_sum_gap_profile(
        _binding(_element(field, Fraction(3, 2)), record), 3
    )

    assert execution.work.phase_counts() == {
        "base_slice_comparison": 2,
        "frontier_addition": 7,
        "power_multiplication": 2,
        "gap_subtraction": 7,
        "gap_certification": 7,
        "sort_comparison": 18,
        "summary_comparison": 12,
    }
    assert_charged_work_parity(
        charged=execution.admission.work_bounds(),
        executed=execution.work.phase_counts(),
    )
    assert execution.work.field_operation_count == (
        execution.admission.field_operation_count
    )
    assert execution.work.comparison_count <= execution.admission.comparison_count


@pytest.mark.parametrize(
    ("caller_seconds", "recognition_seconds"),
    [
        (60.0, 60.0),
        (300.0, EMBEDDINGS_WORKER_WALL_SECONDS),
    ],
)
def test_embedding_recognition_subdeadline_preserves_the_caller_envelope(
    caller_seconds: float,
    recognition_seconds: float,
) -> None:
    field = _field("1", "0")
    (record,) = _real_records(field)
    base = _binding(_element(field, Fraction(3, 2)), record)
    started = time.monotonic()
    caller_deadline = started + caller_seconds

    with request_execution(started):
        bind_request_deadline(caller_deadline)
        execution = _execute_binary_power_sum_gap_profile(base, 0)
        active = current_request_execution()

    assert execution.deadlines.profile_deadline == caller_deadline
    assert execution.deadlines.embedding_recognition_deadline == (
        started + recognition_seconds
    )
    assert execution.deadlines.resumed_profile_deadline == caller_deadline
    assert active is not None
    assert active.deadline == caller_deadline


def test_catalog_operation_runs_example_and_projects_domain_errors() -> None:
    operation = next(
        tool
        for tool in TOOLS
        if tool.operation_id
        == "number_field.real_embedding.binary_power_sum_gap_profile.compute"
    )
    request = NumberFieldBinaryPowerSumGapProfileRequest.model_validate_json(
        json.dumps(operation.examples[0].input)
    )

    result = operation.run(request)

    assert result.source_representation_count == 8
    assert result.distinct_value_count == 8

    invalid = request.model_copy(update={"exponent_count": 13})
    with pytest.raises(OperationDomainValidationError) as caught:
        operation.run(invalid)
    error = caught.value.errors()[0]
    assert error["loc"] == ("exponent_count",)
    assert error["type"] == (
        "number_field.binary_power_sum.source_representation_bound"
    )


def test_expired_request_is_rejected_before_admission() -> None:
    operation = next(
        tool
        for tool in TOOLS
        if tool.operation_id
        == "number_field.real_embedding.binary_power_sum_gap_profile.compute"
    )
    request = NumberFieldBinaryPowerSumGapProfileRequest.model_validate_json(
        json.dumps(operation.examples[0].input)
    )

    with (
        request_execution(
            started_at=time.monotonic() - BINARY_POWER_SUM_WALL_SECONDS - 1
        ),
        pytest.raises(OperationExecutionTimeoutError, match="before"),
    ):
        operation.run(request)
