"""Exact received-word profiles for bounded prime-field linear codes."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from itertools import product
from typing import Literal

import pytest
from pydantic import ValidationError

from jacobian.math.combinatorics.codes import linear as code_linear
from jacobian.math.combinatorics.codes.linear._models import (
    MAX_RECEIVED_PROFILE_CODEWORDS,
    DualCodeRequest,
    GeneratorMatrixRequest,
    PunctureRequest,
    ReceivedWordProfileRequest,
    ReceivedWordProfileResult,
    ReceivedWordThreshold,
    ShortenRequest,
)
from jacobian.math.combinatorics.codes.linear._tools import (
    compute_dual_code,
    compute_from_generator,
    compute_puncture,
    compute_received_word_profile,
    compute_shorten,
)
from jacobian.math.combinatorics.codes.linear.values import PrimeFieldLinearEncoder

Metric = Literal["DISTANCE", "AGREEMENT"]
Comparison = Literal["LT", "LE", "GT", "GE"]
WitnessMode = Literal["NONE", "COUNT", "FIRST", "ALL"]


@contextmanager
def _validation_error(code: str) -> Iterator[None]:
    with pytest.raises(ValidationError) as exc_info:
        yield
    assert code in exc_info.value.errors()[0]["type"]


@contextmanager
def _operation_error(code: str) -> Iterator[None]:
    with pytest.raises(ValueError) as exc_info:
        yield
    assert exc_info.value.errors()[0]["type"] == code


def _encoder(
    generator: tuple[tuple[int, ...], ...],
    *,
    field_order: int = 2,
) -> PrimeFieldLinearEncoder:
    dimension = len(generator)
    length = len(generator[0]) if generator else 2
    return PrimeFieldLinearEncoder(
        field_order=field_order,
        message_axis=tuple(f"m{index}" for index in range(dimension)),
        coordinate_axis=tuple(f"x{index}" for index in range(length)),
        generator_matrix=generator,
    )


def _profile(
    generator: tuple[tuple[int, ...], ...],
    received_word: tuple[int, ...],
    *,
    field_order: int = 2,
) -> ReceivedWordProfileResult:
    return compute_received_word_profile(
        ReceivedWordProfileRequest(
            encoder=_encoder(generator, field_order=field_order),
            received_word=received_word,
        )
    )


def _threshold(
    metric: Metric,
    comparison: Comparison,
    value: int,
) -> ReceivedWordThreshold:
    return ReceivedWordThreshold.model_validate(
        {"metric": metric, "comparison": comparison, "value": value}
    )


def test_outside_code_profile_is_not_the_intrinsic_weight_distribution() -> None:
    result = _profile(((1, 1),), (1, 0))

    assert result.distance_histogram == (0, 2, 0)
    assert result.distance_histogram != (1, 0, 1)
    assert result.minimum_distance == 1
    assert result.maximum_agreement == 1
    assert result.codeword_count == 2


def test_zero_code_and_full_space_have_complete_histograms() -> None:
    zero_code = _profile((), (1, 1))
    full_space = _profile(((1, 0), (0, 1)), (1, 1))

    assert zero_code.distance_histogram == (0, 0, 1)
    assert zero_code.codeword_count == 1
    assert full_space.distance_histogram == (1, 2, 1)
    assert full_space.codeword_count == 4


def test_row_equivalent_generators_have_the_same_profile() -> None:
    received = (1, 0, 0)
    first = _profile(((1, 0, 1), (0, 1, 1)), received)
    second = _profile(((1, 1, 0), (0, 1, 1)), received)

    assert first.distance_histogram == second.distance_histogram


def test_canonicalized_encoder_serializes_directly_into_profile() -> None:
    canonicalized = compute_from_generator(
        GeneratorMatrixRequest(
            field_order=2,
            generator_matrix=((1, 1), (1, 1)),
            coordinate_axis=("left", "right"),
        )
    )

    request = ReceivedWordProfileRequest.model_validate(
        {
            "encoder": canonicalized.model_dump(mode="json")["encoder"],
            "received_word": [1, 0],
        }
    )
    result = compute_received_word_profile(request)

    assert request.encoder == canonicalized.encoder
    assert request.encoder.message_axis == ("m0",)
    assert request.encoder.coordinate_axis == ("left", "right")
    assert result.distance_histogram == (0, 2, 0)


def test_dual_encoder_serializes_directly_into_profile() -> None:
    dual = compute_dual_code(
        DualCodeRequest(
            encoder=PrimeFieldLinearEncoder(
                field_order=2,
                message_axis=("m0",),
                coordinate_axis=("left", "middle", "right"),
                generator_matrix=((1, 1, 1),),
            )
        )
    )

    request = ReceivedWordProfileRequest.model_validate(
        {
            "encoder": dual.model_dump(mode="json")["encoder"],
            "received_word": [0, 0, 0],
        }
    )
    result = compute_received_word_profile(request)

    assert request.encoder == dual.encoder
    assert request.encoder.coordinate_axis == ("left", "middle", "right")
    assert request.encoder.message_axis == ("m0", "m1")
    assert result.distance_histogram == (1, 0, 3, 0)


def test_punctured_encoder_serializes_directly_into_profile() -> None:
    punctured = compute_puncture(
        PunctureRequest(encoder=_encoder(((1, 0, 1), (0, 1, 1))), coordinate=1)
    )

    request = ReceivedWordProfileRequest.model_validate(
        {
            "encoder": punctured.model_dump(mode="json")["encoder"],
            "received_word": [0, 0],
        }
    )
    result = compute_received_word_profile(request)

    assert request.encoder == punctured.encoder
    assert request.encoder.coordinate_axis == ("x0", "x2")
    assert result.distance_histogram == (1, 2, 1)


def test_shortened_encoder_serializes_directly_into_profile() -> None:
    shortened = compute_shorten(
        ShortenRequest(encoder=_encoder(((1, 1, 0), (1, 0, 1))), coordinate=0)
    )

    request = ReceivedWordProfileRequest.model_validate(
        {
            "encoder": shortened.model_dump(mode="json")["encoder"],
            "received_word": [1, 0],
        }
    )
    result = compute_received_word_profile(request)

    assert request.encoder == shortened.encoder
    assert request.encoder.coordinate_axis == ("x1", "x2")
    assert result.distance_histogram == (0, 2, 0)


def test_length_one_puncture_composes_into_length_zero_profile() -> None:
    punctured = compute_puncture(
        PunctureRequest(encoder=_encoder(((1,),)), coordinate=0)
    )
    request = ReceivedWordProfileRequest.model_validate(
        {
            "encoder": punctured.model_dump(mode="json")["encoder"],
            "received_word": [],
            "threshold": {
                "metric": "DISTANCE",
                "comparison": "LE",
                "value": 0,
            },
            "witness_mode": "ALL",
        }
    )

    result = compute_received_word_profile(request)

    assert punctured.encoder.coordinate_axis == ()
    assert punctured.encoder.message_axis == ()
    assert punctured.encoder.generator_matrix == ()
    assert result.distance_histogram == (1,)
    assert result.codeword_count == 1
    assert result.minimum_distance == 0
    assert result.maximum_agreement == 0
    assert result.threshold_match_count == 1
    assert tuple(
        (witness.message, witness.codeword) for witness in result.witnesses
    ) == (((), ()),)


def test_ternary_repetition_profile_counts_every_distinct_codeword() -> None:
    result = _profile(((1, 1),), (1, 0), field_order=3)

    assert result.distance_histogram == (0, 2, 1)
    assert sum(result.distance_histogram) == 3


@pytest.mark.parametrize(
    ("metric", "comparison", "mode", "expected_count", "expected_witnesses"),
    [
        ("AGREEMENT", "GT", "COUNT", 0, 0),
        ("AGREEMENT", "GE", "COUNT", 2, 0),
        ("AGREEMENT", "GE", "FIRST", 2, 1),
        ("AGREEMENT", "GE", "ALL", 2, 2),
        ("DISTANCE", "LT", "COUNT", 0, 0),
        ("DISTANCE", "LE", "COUNT", 2, 0),
    ],
)
def test_exact_threshold_modes_preserve_strict_boundary_semantics(
    metric: Metric,
    comparison: Comparison,
    mode: WitnessMode,
    expected_count: int,
    expected_witnesses: int,
) -> None:
    request = ReceivedWordProfileRequest(
        encoder=_encoder(((1, 1),)),
        received_word=(1, 0),
        threshold=_threshold(metric, comparison, 1),
        witness_mode=mode,
    )

    result = compute_received_word_profile(request)

    assert result.threshold_match_count == expected_count
    assert len(result.witnesses) == expected_witnesses
    assert all(witness.agreement == 1 for witness in result.witnesses)
    if mode == "FIRST":
        assert result.witnesses[0].message == (0,)


def test_small_binary_profiles_match_independent_set_enumeration() -> None:
    for length in range(1, 4):
        for generator_row in product((0, 1), repeat=length):
            if not any(generator_row):
                continue
            for received_word in product((0, 1), repeat=length):
                result = _profile((generator_row,), received_word)
                expected = [0] * (length + 1)
                for scalar in (0, 1):
                    word = tuple(scalar * value for value in generator_row)
                    distance = sum(
                        left != right
                        for left, right in zip(word, received_word, strict=True)
                    )
                    expected[distance] += 1
                assert result.distance_histogram == tuple(expected)


def test_encoder_rejects_ambiguous_or_invalid_presentations() -> None:
    with _validation_error("full_row_rank"):
        _encoder(((1, 1), (1, 1)))
    with _validation_error("prime"):
        _encoder(((1,),), field_order=4)
    with _validation_error("canonical"):
        _encoder(((2,),))
    with _validation_error("message_axis"):
        PrimeFieldLinearEncoder(
            field_order=2,
            message_axis=(),
            coordinate_axis=("x",),
            generator_matrix=((1,),),
        )
    with _validation_error("unique"):
        PrimeFieldLinearEncoder(
            field_order=2,
            message_axis=("m",),
            coordinate_axis=("x", "x"),
            generator_matrix=((1, 0),),
        )


def test_profile_request_rejects_misalignment_and_mode_holes() -> None:
    encoder = _encoder(((1, 1),))
    with _validation_error("coordinate_axis"):
        ReceivedWordProfileRequest(encoder=encoder, received_word=(1,))
    with _validation_error("canonical"):
        ReceivedWordProfileRequest(encoder=encoder, received_word=(1, 2))
    with _validation_error("requires_an_exact_threshold"):
        ReceivedWordProfileRequest(
            encoder=encoder,
            received_word=(1, 0),
            witness_mode="FIRST",
        )
    with _validation_error("requires_count"):
        ReceivedWordProfileRequest.model_validate(
            {
                "encoder": encoder,
                "received_word": (1, 0),
                "threshold": {
                    "metric": "DISTANCE",
                    "comparison": "LE",
                    "value": 1,
                },
            }
        )


def test_threshold_mode_relation_is_schema_visible() -> None:
    schema = ReceivedWordProfileRequest.model_json_schema()
    threshold_description = schema["properties"]["threshold"]["description"]
    mode_description = schema["properties"]["witness_mode"]["description"]

    assert "COUNT, FIRST, and ALL require it" in threshold_description
    assert "NONE without a threshold" in mode_description


def test_execution_work_admits_larger_profiles_without_a_codeword_cap() -> None:
    # A binary 13x13 identity encoder has 8,192 codewords and execution work
    # 2,981,888, so admission comes from the work bound alone.
    identity_13 = tuple(
        tuple(int(row == column) for column in range(13)) for row in range(13)
    )
    request = ReceivedWordProfileRequest(
        encoder=_encoder(identity_13),
        received_word=(0,) * 13,
    )
    assert request.encoder.codeword_count == 8_192
    assert request.profile_execution_work == 2_981_888

    result = compute_received_word_profile(request)
    assert result.codeword_count == 8_192
    assert len(result.distance_histogram) == 14
    assert sum(result.distance_histogram) == 8_192


def test_work_bound_still_rejects_before_enumeration() -> None:
    def rectangular_identity(length: int) -> ReceivedWordProfileRequest:
        generator = tuple(
            tuple(int(row == column) for column in range(length)) for row in range(12)
        )
        return ReceivedWordProfileRequest(
            encoder=_encoder(generator),
            received_word=(0,) * length,
        )

    assert rectangular_identity(28).profile_execution_work == 2_981_888
    request = rectangular_identity(29)
    with _operation_error("code_linear.profile_execution_work_exceeded"):
        compute_received_word_profile(request)


def test_codeword_budget_is_derived_from_the_execution_work_bound() -> None:
    # GF(47)^3 is the largest image admitted by the 3,000,000 execution-work
    # budget: 2 * 47^3 * 3 * 4 = 2,491,752; GF(53)^3 needs 3,573,048.
    assert MAX_RECEIVED_PROFILE_CODEWORDS == 47**3 == 103_823

    identity = ((1, 0, 0), (0, 1, 0), (0, 0, 1))
    admitted = ReceivedWordProfileRequest(
        encoder=_encoder(identity, field_order=47),
        received_word=(0, 0, 0),
    )
    assert admitted.encoder.codeword_count == MAX_RECEIVED_PROFILE_CODEWORDS
    assert admitted.profile_execution_work == 2_491_752

    over_budget = ReceivedWordProfileRequest(
        encoder=_encoder(identity, field_order=53),
        received_word=(0, 0, 0),
    )
    with _operation_error("code_linear.profile_execution_work_exceeded"):
        compute_received_word_profile(over_budget)


def test_all_witness_output_has_a_separate_preflight_bound() -> None:
    generator = tuple(
        tuple(int(row == column) for column in range(32)) for row in range(11)
    )
    encoder = _encoder(generator)

    ReceivedWordProfileRequest(
        encoder=encoder,
        received_word=(0,) * 32,
        threshold=_threshold("DISTANCE", "GE", 0),
        witness_mode="FIRST",
    )
    all_witness = ReceivedWordProfileRequest.model_validate(
        {
            "encoder": encoder,
            "received_word": (0,) * 32,
            "threshold": {
                "metric": "DISTANCE",
                "comparison": "GE",
                "value": 0,
            },
            "witness_mode": "ALL",
        }
    )
    with _operation_error("code_linear.witness_cells_exceeded"):
        compute_received_word_profile(all_witness)


def test_all_witness_bound_uses_the_threshold_hamming_ball() -> None:
    dimension = 12
    length = 20
    generator = tuple(
        tuple(int(row == column) for column in range(length))
        for row in range(dimension)
    )
    request = ReceivedWordProfileRequest(
        encoder=_encoder(generator),
        received_word=(0,) * length,
        threshold=_threshold("DISTANCE", "LE", 0),
        witness_mode="ALL",
    )

    assert request.maximum_witness_cells == dimension + length + 2
    result = compute_received_word_profile(request)
    assert result.threshold_match_count == 1
    assert tuple(witness.message for witness in result.witnesses) == ((0,) * dimension,)

    impossible = ReceivedWordProfileRequest(
        encoder=request.encoder,
        received_word=request.received_word,
        threshold=_threshold("DISTANCE", "LT", 0),
        witness_mode="ALL",
    )
    assert impossible.maximum_witness_cells == 0


def test_public_value_api_is_explicit() -> None:
    assert tuple(code_linear.__all__) == ("PrimeFieldLinearEncoder",)
    assert code_linear.PrimeFieldLinearEncoder is PrimeFieldLinearEncoder
