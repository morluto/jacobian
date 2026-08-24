"""Exact contracts for canonical explicit nonlinear binary codes."""

from __future__ import annotations

from copy import deepcopy
from math import comb
from pathlib import Path

import pytest
from pydantic import ValidationError

from jacobian.canonical import encode_strict_json
from jacobian.math.code_nonlinear import ExplicitBinaryCode, to_set_system
from jacobian.math.code_nonlinear._budget import (
    MAX_CODE_RESULT_BYTES,
    MAX_GENERATED_CODE_ENTRIES,
    MAX_PROFILE_BITSET_CHUNK_WORK,
    MAX_PROFILE_PAIRS,
    PROFILE_PAIR_PASSES,
    require_constant_weight_admission,
    require_profile_admission,
    require_word_distance_output_bound,
    source_wire_upper_bound,
)
from jacobian.math.code_nonlinear._models import (
    BinaryCodeDistanceWitness,
    BinaryCodeRequest,
    ConstantWeightProfileRequest,
    ConstantWeightProfileResult,
    ConstantWeightRequest,
    ConstantWeightResult,
    ExplicitProfileRequest,
    ExplicitProfileResult,
    ToSetSystemRequest,
    ToSetSystemResult,
    WordDistanceRequest,
    WordDistanceResult,
)
from jacobian.math.code_nonlinear._operations import (
    compute_constant_weight,
    compute_constant_weight_profile,
    compute_distance_profile,
    compute_explicit_profile,
    compute_to_set_system,
    compute_word_distance,
)
from jacobian.math.code_nonlinear.values import (
    MAX_EXPLICIT_CODE_BITS,
    MAX_EXPLICIT_CODE_LENGTH,
)

ROOT = Path(__file__).parents[3]
SOURCE_CODEWORDS = (
    ROOT / "benchmarks/datasets/conjecture-probes-v1/"
    "constant-weight-code-a23-6-10-2992/solution/codewords.txt"
)


def _code(*words: tuple[int, ...], length: int | None = None) -> ExplicitBinaryCode:
    if length is None:
        if not words:
            raise ValueError("empty test codes must declare a length")
        length = len(words[0])
    return ExplicitBinaryCode(length=length, codewords=words)


def _binary_words(length: int, count: int) -> tuple[tuple[int, ...], ...]:
    return tuple(
        tuple((value >> coordinate) & 1 for coordinate in reversed(range(length)))
        for value in range(count)
    )


def _witness(
    source: ExplicitBinaryCode, left_index: int, right_index: int
) -> BinaryCodeDistanceWitness:
    left = source.codewords[left_index]
    right = source.codewords[right_index]
    differing = tuple(
        i for i, pair in enumerate(zip(left, right, strict=True)) if pair[0] != pair[1]
    )
    return BinaryCodeDistanceWitness(
        left_index=left_index,
        right_index=right_index,
        left_word=left,
        right_word=right,
        left_support=tuple(i for i, bit in enumerate(left) if bit),
        right_support=tuple(i for i, bit in enumerate(right) if bit),
        differing_coordinates=differing,
        left_weight=sum(left),
        right_weight=sum(right),
        support_intersection=sum(
            left_bit == right_bit == 1
            for left_bit, right_bit in zip(left, right, strict=True)
        ),
        distance=len(differing),
    )


def _source_scale_code() -> ExplicitBinaryCode:
    words = tuple(
        tuple(int(bit) for bit in f"{int(line, 16):023b}")
        for raw_line in SOURCE_CODEWORDS.read_text().splitlines()
        if (line := raw_line.strip()) and not line.startswith("$")
    )
    return ExplicitBinaryCode(length=23, codewords=words)


class TestCanonicalExplicitBinaryCode:
    def test_normalizes_order_and_retains_empty_axis(self) -> None:
        code = _code((1, 1, 0), (0, 0, 0), (0, 1, 1))
        assert code.codewords == ((0, 0, 0), (0, 1, 1), (1, 1, 0))
        assert ExplicitBinaryCode(length=7, codewords=()).model_dump() == {
            "length": 7,
            "codewords": (),
        }

    def test_distinguishes_zero_axis_empty_code_from_sole_empty_word(self) -> None:
        empty_code = ExplicitBinaryCode(length=0, codewords=())
        empty_word_code = ExplicitBinaryCode(length=0, codewords=((),))
        assert empty_code != empty_word_code
        assert empty_code.codewords == ()
        assert empty_word_code.codewords == ((),)

        with pytest.raises(ValidationError, match="sole empty word"):
            ExplicitBinaryCode(length=0, codewords=((), ()))
        with pytest.raises(ValidationError, match="declared length"):
            ExplicitBinaryCode(length=0, codewords=((0,),))

    @pytest.mark.parametrize(
        "payload",
        [
            {"length": 3, "codewords": [[0, 0, 0], [0, 0, 0]]},
            {"length": 3, "codewords": [[0, 0]]},
            {"length": 3, "codewords": [[0, 2, 0]]},
            {"length": 3, "codewords": [[False, 0, 0]]},
        ],
    )
    def test_rejects_duplicate_malformed_and_non_strict_words(
        self, payload: dict[str, object]
    ) -> None:
        with pytest.raises(ValidationError):
            ExplicitBinaryCode.model_validate(payload)

    def test_materialized_bit_boundary_is_immediate(self) -> None:
        accepted = ExplicitBinaryCode(
            length=512,
            codewords=_binary_words(512, MAX_EXPLICIT_CODE_BITS // 512),
        )
        assert accepted.length * len(accepted.codewords) == MAX_EXPLICIT_CODE_BITS
        with pytest.raises(ValidationError, match="bit source bound"):
            ExplicitBinaryCode(
                length=512,
                codewords=_binary_words(512, MAX_EXPLICIT_CODE_BITS // 512 + 1),
            )

    def test_exact_aggregate_boundary_passes_prevalidation(self) -> None:
        source = ExplicitBinaryCode.model_validate(
            {
                "length": 32,
                "codewords": [list(word) for word in _binary_words(32, 16_384)],
            }
        )
        assert source.length * len(source.codewords) == MAX_EXPLICIT_CODE_BITS

    def test_millions_of_empty_words_rejected_before_nested_conversion(self) -> None:
        codewords: list[list[int]] = [[]] * MAX_EXPLICIT_CODE_BITS
        codewords.append([2])
        with pytest.raises(ValidationError, match="word container bound"):
            ExplicitBinaryCode.model_validate({"length": 0, "codewords": codewords})

    def test_aggregate_bit_overflow_rejected_before_nested_conversion(self) -> None:
        codewords: list[list[int]] = [[0] * 1_000] * 2_000
        codewords[-1][-1] = 2
        with pytest.raises(ValidationError, match="bit source bound"):
            ExplicitBinaryCode.model_validate({"length": 1_000, "codewords": codewords})

    def test_enormous_single_word_rejected_before_nested_conversion(self) -> None:
        codewords: list[list[str]] = [["x"] * (MAX_EXPLICIT_CODE_LENGTH + 1)]
        with pytest.raises(ValidationError, match="bit source bound"):
            ExplicitBinaryCode.model_validate({"length": 3, "codewords": codewords})


class TestWordDistance:
    def test_exact_relation(self) -> None:
        result = compute_word_distance(
            WordDistanceRequest(word1=(1, 1, 0, 0), word2=(1, 0, 1, 0))
        )
        assert result.distance == 2
        assert result.differing_coordinates == (1, 2)
        assert result.weight1 == result.weight2 == 2
        assert result.support_intersection == 1

    def test_result_rejects_conclusion_mutation(self) -> None:
        result = compute_word_distance(
            WordDistanceRequest(word1=(1, 0, 1), word2=(0, 1, 1))
        )
        payload = result.model_dump(mode="json")
        payload["distance"] = 1
        with pytest.raises(ValidationError, match="replay"):
            type(result).model_validate(payload)

    def test_contract_version_tracks_the_wire_shape_change(self) -> None:
        from jacobian.math.code_nonlinear._tools import TOOLS

        operation = next(
            tool
            for tool in TOOLS
            if tool.operation_id == "code.binary.word_distance.compute"
        )
        assert operation.version == "2"

    def test_identical_maximal_words_admit_result_sensitive_bound(self) -> None:
        word = [0] * MAX_EXPLICIT_CODE_LENGTH
        result = compute_word_distance(WordDistanceRequest(word1=word, word2=word))
        assert result.distance == 0
        assert result.differing_coordinates == ()
        assert result.weight1 == result.weight2 == 0
        assert result.support_intersection == 0
        replayed = WordDistanceResult.model_validate(result.model_dump(mode="json"))
        assert replayed == result

    def test_word_distance_output_bound_covers_the_canonical_result(self) -> None:
        word = [0] * MAX_EXPLICIT_CODE_LENGTH
        result = compute_word_distance(WordDistanceRequest(word1=word, word2=word))
        bound = require_word_distance_output_bound(word, word)
        assert len(encode_strict_json(result.model_dump(mode="json"))) <= bound
        assert bound <= MAX_CODE_RESULT_BYTES

    def test_all_different_maximal_words_still_exceed_the_result_bound(self) -> None:
        left = [0] * MAX_EXPLICIT_CODE_LENGTH
        right = [1] * MAX_EXPLICIT_CODE_LENGTH
        with pytest.raises(ValidationError, match="5657085"):
            WordDistanceRequest(word1=left, word2=right)
        payload = {
            "word1": left,
            "word2": right,
            "distance": MAX_EXPLICIT_CODE_LENGTH,
            "differing_coordinates": list(range(MAX_EXPLICIT_CODE_LENGTH)),
            "weight1": 0,
            "weight2": MAX_EXPLICIT_CODE_LENGTH,
            "support_intersection": 0,
        }
        with pytest.raises(ValidationError, match="result bound"):
            WordDistanceResult.model_validate(payload)

    def test_differing_coordinate_wire_size_tracks_actual_difference_positions(
        self,
    ) -> None:
        left = [0] * MAX_EXPLICIT_CODE_LENGTH
        low = [0] * MAX_EXPLICIT_CODE_LENGTH
        for index in range(200_000):
            low[index] = 1
        admitted = WordDistanceRequest(word1=left, word2=low)
        assert (
            sum(a != b for a, b in zip(admitted.word1, admitted.word2, strict=True))
            == 200_000
        )
        high = [0] * MAX_EXPLICIT_CODE_LENGTH
        for index in range(
            MAX_EXPLICIT_CODE_LENGTH - 300_000, MAX_EXPLICIT_CODE_LENGTH
        ):
            high[index] = 1
        with pytest.raises(ValidationError, match="result bound"):
            WordDistanceRequest(word1=left, word2=high)


class TestExplicitProfile:
    def test_known_answer_and_defining_accounting(self) -> None:
        source = _code((0, 0, 0, 0), (0, 0, 1, 1), (1, 1, 1, 1))
        result = compute_explicit_profile(ExplicitProfileRequest(code=source))
        assert result.source is source
        assert result.length == 4
        assert result.cardinality == 3
        assert result.pair_count == 3
        assert result.weight_distribution == (1, 0, 1, 0, 1)
        assert result.distance_histogram == (0, 0, 2, 0, 1)
        assert result.minimum_distance == 2
        assert result.maximum_distance == 4
        assert sum(result.weight_distribution) == result.cardinality
        assert sum(result.distance_histogram) == result.pair_count
        assert result.minimum_distance_witness == _witness(source, 0, 1)
        assert result.maximum_distance_witness == _witness(source, 0, 2)

    def test_empty_and_singleton_codes_retain_context_without_fake_extrema(
        self,
    ) -> None:
        empty = compute_explicit_profile(
            ExplicitProfileRequest(code=ExplicitBinaryCode(length=5, codewords=()))
        )
        assert empty.length == 5
        assert empty.cardinality == empty.pair_count == 0
        assert empty.minimum_distance is empty.maximum_distance is None
        assert empty.minimum_distance_witness is None
        assert empty.maximum_distance_witness is None
        assert empty.weight_distribution == (0, 0, 0, 0, 0, 0)

        singleton = compute_explicit_profile(
            ExplicitProfileRequest(code=_code((1, 0, 1)))
        )
        assert singleton.pair_count == 0
        assert singleton.minimum_distance is singleton.maximum_distance is None
        assert singleton.weight_distribution == (0, 0, 1, 0)

        zero_empty = compute_explicit_profile(
            ExplicitProfileRequest(code=ExplicitBinaryCode(length=0, codewords=()))
        )
        zero_singleton = compute_explicit_profile(
            ExplicitProfileRequest(code=ExplicitBinaryCode(length=0, codewords=((),)))
        )
        assert zero_empty.weight_distribution == (0,)
        assert zero_singleton.weight_distribution == (1,)
        for result in (zero_empty, zero_singleton):
            assert result.length == 0
            assert result.pair_count == 0
            assert result.distance_histogram == (0,)
            assert result.minimum_distance is result.maximum_distance is None
            assert result.minimum_distance_witness is None
            assert result.maximum_distance_witness is None

    def test_equivalent_tied_extremal_witness_revalidates(self) -> None:
        source = _code((0, 0, 0), (0, 1, 1), (1, 0, 1), (1, 1, 0))
        result = compute_explicit_profile(ExplicitProfileRequest(code=source))
        assert result.minimum_distance == result.maximum_distance == 2
        payload = result.model_dump(mode="json")
        payload["minimum_distance_witness"] = _witness(source, 1, 2).model_dump(
            mode="json"
        )
        replayed = ExplicitProfileResult.model_validate(payload)
        assert replayed.minimum_distance_witness == _witness(source, 1, 2)

    @pytest.mark.parametrize(
        ("path", "replacement"),
        [
            (("length",), 5),
            (("cardinality",), 4),
            (("pair_count",), 2),
            (("distance_histogram", 2), 1),
            (("minimum_distance_witness", "right_index"), 2),
            (("minimum_distance_witness", "distance"), 4),
        ],
    )
    def test_rejects_metadata_histogram_and_witness_mutations(
        self, path: tuple[str | int, ...], replacement: object
    ) -> None:
        result = compute_explicit_profile(
            ExplicitProfileRequest(code=_code((0, 0, 0, 0), (0, 0, 1, 1), (1, 1, 1, 1)))
        )
        payload = result.model_dump(mode="json")
        target: object = payload
        for component in path[:-1]:
            target = target[component]  # type: ignore[index]
        target[path[-1]] = replacement  # type: ignore[index]
        with pytest.raises(ValidationError):
            ExplicitProfileResult.model_validate(payload)

    def test_rejects_one_bit_source_mutation(self) -> None:
        result = compute_explicit_profile(
            ExplicitProfileRequest(code=_code((0, 0, 0, 0), (0, 0, 1, 1), (1, 1, 1, 1)))
        )
        payload = result.model_dump(mode="json")
        payload["source"]["codewords"][1][0] = 1
        with pytest.raises(ValidationError):
            ExplicitProfileResult.model_validate(payload)

    def test_standard_a_23_6_10_2992_profile_replays_completely(self) -> None:
        """The source has length 23, minimum distance 6, and constant weight 10."""
        source = _source_scale_code()
        plan = require_profile_admission(source)
        assert source.length == 23
        assert len(source.codewords) == 2_992
        assert all(sum(word) == 10 for word in source.codewords)
        assert plan.pair_count == 4_474_536
        assert plan.pair_passes == PROFILE_PAIR_PASSES == 2
        assert plan.bitset_chunks == 1
        assert plan.bitset_chunk_work == 8_949_072
        assert plan.bitset_chunk_work <= MAX_PROFILE_BITSET_CHUNK_WORK
        assert plan.result_wire_upper_bound <= MAX_CODE_RESULT_BYTES

        result = compute_explicit_profile(ExplicitProfileRequest(code=source))
        assert (
            len(encode_strict_json(result.model_dump(mode="json")))
            <= plan.result_wire_upper_bound
        )
        assert result.minimum_distance == 6
        assert result.maximum_distance == 20
        assert result.weight_distribution[10] == 2_992
        assert result.distance_histogram == (
            0,
            0,
            0,
            0,
            0,
            0,
            184_837,
            0,
            551_360,
            0,
            1_299_434,
            0,
            1_426_181,
            0,
            749_495,
            0,
            241_176,
            0,
            20_512,
            0,
            1_541,
            0,
            0,
            0,
        )
        assert sum(result.distance_histogram) == 4_474_536


class TestConstantWeightProfile:
    def test_distance_and_intersection_defining_identity(self) -> None:
        source = _code((0, 0, 1, 1), (0, 1, 0, 1), (1, 0, 1, 0))
        result = compute_constant_weight_profile(
            ConstantWeightProfileRequest(code=source)
        )
        assert result.source is source
        assert result.length == 4
        assert result.weight == 2
        assert result.cardinality == 3
        assert result.pair_count == 3
        assert result.distance_histogram == (0, 0, 2, 0, 1)
        assert result.intersection_histogram == (1, 2, 0, 0, 0)
        assert result.minimum_distance == 2
        assert result.maximum_distance == 4
        for left_index, left in enumerate(source.codewords):
            for right in source.codewords[left_index + 1 :]:
                distance = sum(a != b for a, b in zip(left, right, strict=True))
                intersection = sum(
                    a == b == 1 for a, b in zip(left, right, strict=True)
                )
                assert distance == 2 * (result.weight - intersection)

    def test_singleton_has_no_pairwise_extremum(self) -> None:
        result = compute_constant_weight_profile(
            ConstantWeightProfileRequest(code=_code((1, 0, 1)))
        )
        assert result.pair_count == 0
        assert result.minimum_distance is result.maximum_distance is None
        assert result.minimum_distance_witness is None
        assert result.maximum_distance_witness is None

    def test_zero_coordinate_singleton_has_weight_zero_profile(self) -> None:
        result = compute_constant_weight_profile(
            ConstantWeightProfileRequest(
                code=ExplicitBinaryCode(length=0, codewords=((),))
            )
        )
        assert result.length == result.weight == result.pair_count == 0
        assert result.cardinality == 1
        assert result.distance_histogram == (0,)
        assert result.intersection_histogram == (0,)
        assert result.minimum_distance is result.maximum_distance is None
        assert result.minimum_distance_witness is None
        assert result.maximum_distance_witness is None

    def test_rejects_mixed_weight_source(self) -> None:
        with pytest.raises(ValidationError, match="same Hamming weight"):
            ConstantWeightProfileRequest(code=_code((0, 0, 1, 1), (1, 1, 1, 0)))

    @pytest.mark.parametrize(
        ("field", "replacement"),
        [
            ("weight", 1),
            ("pair_count", 2),
            ("intersection_histogram", [0, 3, 0, 0, 0]),
        ],
    )
    def test_rejects_constant_metadata_and_conclusion_mutations(
        self, field: str, replacement: object
    ) -> None:
        result = compute_constant_weight_profile(
            ConstantWeightProfileRequest(
                code=_code((0, 0, 1, 1), (0, 1, 0, 1), (1, 0, 1, 0))
            )
        )
        payload = result.model_dump(mode="json")
        payload[field] = replacement
        with pytest.raises(ValidationError):
            ConstantWeightProfileResult.model_validate(payload)


class TestSetSystemConversion:
    def test_exact_source_bound_bijection(self) -> None:
        source = _code((1, 0, 1, 0), (0, 1, 0, 1))
        result = compute_to_set_system(ToSetSystemRequest(code=source))
        assert result.source is source
        assert result.length == 4
        assert result.cardinality == 2
        assert result.coordinate_axis == (0, 1, 2, 3)
        # The canonical code order, not caller list order, owns the block axis.
        assert result.supports == ((1, 3), (0, 2))
        assert to_set_system(source) == result

    def test_empty_code_retains_coordinate_axis(self) -> None:
        result = to_set_system(ExplicitBinaryCode(length=4, codewords=()))
        assert result.coordinate_axis == (0, 1, 2, 3)
        assert result.supports == ()
        assert result.cardinality == 0

    def test_native_conversion_does_not_apply_mcp_output_bound(self) -> None:
        length = 300_000
        source = ExplicitBinaryCode(length=length, codewords=((1,) * length,))
        result = to_set_system(source)
        assert result.source is source
        assert result.length == length
        assert result.supports == (tuple(range(length)),)

    @pytest.mark.parametrize(
        ("source", "expected_supports"),
        [
            (ExplicitBinaryCode(length=0, codewords=()), ()),
            (ExplicitBinaryCode(length=0, codewords=((),)), ((),)),
        ],
    )
    def test_zero_coordinate_support_conversion(
        self,
        source: ExplicitBinaryCode,
        expected_supports: tuple[tuple[int, ...], ...],
    ) -> None:
        result = to_set_system(source)
        assert result.length == 0
        assert result.coordinate_axis == ()
        assert result.supports == expected_supports
        assert result.cardinality == len(source.codewords)

    @pytest.mark.parametrize("mutation", ["bit", "axis", "support"])
    def test_rejects_source_axis_and_support_mutations(self, mutation: str) -> None:
        result = to_set_system(_code((0, 1, 0, 1), (1, 0, 1, 0)))
        payload = result.model_dump(mode="json")
        if mutation == "bit":
            payload["source"]["codewords"][0][0] = 1
        elif mutation == "axis":
            payload["coordinate_axis"][2] = 3
        else:
            payload["supports"][0][0] = 0
        with pytest.raises(ValidationError):
            ToSetSystemResult.model_validate(payload)


class TestProducerConsumerClosure:
    def test_generated_code_serializes_unchanged_into_all_consumers(self) -> None:
        generated = compute_constant_weight(ConstantWeightRequest(length=4, weight=2))
        serialized_code = generated.model_dump(mode="json")["code"]

        explicit_request = ExplicitProfileRequest.model_validate(
            {"code": deepcopy(serialized_code)}
        )
        constant_request = ConstantWeightProfileRequest.model_validate(
            {"code": deepcopy(serialized_code)}
        )
        support_request = ToSetSystemRequest.model_validate(
            {"code": deepcopy(serialized_code)}
        )
        assert explicit_request.code == generated.code
        assert constant_request.code == generated.code
        assert support_request.code == generated.code
        assert compute_explicit_profile(explicit_request).pair_count == 15
        assert compute_constant_weight_profile(constant_request).weight == 2
        assert compute_to_set_system(support_request).cardinality == 6

    def test_each_source_bound_result_serializes_into_every_consumer(self) -> None:
        source = _code((0, 0, 1, 1), (0, 1, 0, 1), (1, 0, 1, 0))
        results = (
            compute_explicit_profile(ExplicitProfileRequest(code=source)),
            compute_constant_weight_profile(ConstantWeightProfileRequest(code=source)),
            compute_to_set_system(ToSetSystemRequest(code=source)),
        )
        for result in results:
            serialized_source = result.model_dump(mode="json")["source"]
            assert (
                ExplicitProfileRequest.model_validate(
                    {"code": deepcopy(serialized_source)}
                ).code
                == source
            )
            assert (
                ConstantWeightProfileRequest.model_validate(
                    {"code": deepcopy(serialized_source)}
                ).code
                == source
            )
            assert (
                ToSetSystemRequest.model_validate(
                    {"code": deepcopy(serialized_source)}
                ).code
                == source
            )


class TestDerivedAdmissionBoundaries:
    def test_pair_count_exact_and_immediate_over_bound(self) -> None:
        accepted = ExplicitProfileRequest(
            code=ExplicitBinaryCode(length=12, codewords=_binary_words(12, 3_162))
        )
        assert require_profile_admission(accepted.code).pair_count == 4_997_541
        assert require_profile_admission(accepted.code).pair_count <= MAX_PROFILE_PAIRS
        with pytest.raises(ValidationError, match="unordered pairs"):
            ExplicitProfileRequest(
                code=ExplicitBinaryCode(length=12, codewords=_binary_words(12, 3_163))
            )

    def test_bitset_chunk_work_immediate_boundary(self) -> None:
        accepted = ExplicitProfileRequest(
            code=ExplicitBinaryCode(length=90, codewords=_binary_words(90, 1_826))
        )
        plan = require_profile_admission(accepted.code)
        assert plan.pair_count == 1_666_225
        assert plan.bitset_chunks == 3
        assert plan.bitset_chunk_work == 9_997_350
        with pytest.raises(ValidationError, match="pair-by-bitset-chunk"):
            ExplicitProfileRequest(
                code=ExplicitBinaryCode(length=90, codewords=_binary_words(90, 1_827))
            )

    def test_profile_output_size_immediate_boundary(self) -> None:
        accepted_length = 100_770
        accepted = ExplicitProfileRequest(
            code=_code(
                (0,) * accepted_length,
                (1,) * accepted_length,
                length=accepted_length,
            )
        )
        assert (
            require_profile_admission(accepted.code).result_wire_upper_bound
            <= MAX_CODE_RESULT_BYTES
        )
        with pytest.raises(ValidationError, match="result bound"):
            ExplicitProfileRequest(
                code=_code(
                    (0,) * (accepted_length + 1),
                    (1,) * (accepted_length + 1),
                    length=accepted_length + 1,
                )
            )

    @pytest.mark.parametrize("length", [200_000, MAX_EXPLICIT_CODE_LENGTH])
    def test_empty_profile_charges_no_witnesses_and_fits_output_bound(
        self, length: int
    ) -> None:
        source = ExplicitBinaryCode(length=length, codewords=())
        plan = require_profile_admission(source)
        histogram_bytes = 1 + (length + 1) * 2
        assert plan.pair_count == 0
        assert plan.result_wire_upper_bound == (
            source_wire_upper_bound(source) + 3 * histogram_bytes + 2_048
        )

        result = compute_explicit_profile(ExplicitProfileRequest(code=source))
        actual_wire_bytes = len(encode_strict_json(result.model_dump(mode="json")))
        assert actual_wire_bytes <= plan.result_wire_upper_bound
        assert plan.result_wire_upper_bound <= MAX_CODE_RESULT_BYTES
        assert result.minimum_distance_witness is None
        assert result.maximum_distance_witness is None

    def test_singleton_profile_charges_no_witnesses(self) -> None:
        length = 200_000
        source = ExplicitBinaryCode(length=length, codewords=((0,) * length,))
        plan = require_profile_admission(source)
        histogram_bytes = 1 + (length + 1) * 2
        assert plan.pair_count == 0
        assert plan.result_wire_upper_bound == (
            source_wire_upper_bound(source) + 3 * histogram_bytes + 2_048
        )

        result = compute_explicit_profile(ExplicitProfileRequest(code=source))
        actual_wire_bytes = len(encode_strict_json(result.model_dump(mode="json")))
        assert actual_wire_bytes <= plan.result_wire_upper_bound
        assert result.minimum_distance_witness is None
        assert result.maximum_distance_witness is None

    def test_set_system_output_size_immediate_boundary(self) -> None:
        accepted_length = 275_963
        ToSetSystemRequest(code=_code((1,) * accepted_length, length=accepted_length))
        with pytest.raises(ValidationError, match="result bound"):
            ToSetSystemRequest(
                code=_code((1,) * (accepted_length + 1), length=accepted_length + 1)
            )

    @pytest.mark.parametrize(
        ("length", "weight", "expected"),
        [
            (0, 0, 1),
            (17, 8, 24_310),
            (101, 2, 5_050),
            (101, 99, 5_050),
        ],
    )
    def test_constant_weight_admission_is_exact_within_entry_bound(
        self, length: int, weight: int, expected: int
    ) -> None:
        assert expected == comb(length, weight)
        cardinality = require_constant_weight_admission(length, weight)
        assert cardinality == comb(length, weight)
        assert length * cardinality <= MAX_GENERATED_CODE_ENTRIES

    def test_central_weight_rejection_stops_before_the_full_binomial(self) -> None:
        weight = MAX_EXPLICIT_CODE_LENGTH // 2
        with pytest.raises(ValueError, match="entry bound") as excinfo:
            require_constant_weight_admission(MAX_EXPLICIT_CODE_LENGTH, weight)
        message = str(excinfo.value)
        # The thresholded crossing reports small counts instead of computing
        # the ~524k-bit central coefficient and hitting the integer-to-decimal
        # digit limit.
        assert "Exceeds the limit" not in message
        assert f"{MAX_EXPLICIT_CODE_LENGTH} coordinates * {weight + 1} words" in message

    def test_constant_weight_request_rejects_at_the_immediate_entry_boundary(
        self,
    ) -> None:
        ConstantWeightRequest.model_validate({"length": 101, "weight": 2})
        with pytest.raises(ValidationError, match="entry bound"):
            ConstantWeightRequest.model_validate({"length": 102, "weight": 2})
        with pytest.raises(ValidationError, match="entry bound"):
            ConstantWeightRequest.model_validate(
                {
                    "length": MAX_EXPLICIT_CODE_LENGTH,
                    "weight": MAX_EXPLICIT_CODE_LENGTH // 2,
                }
            )


class TestLegacyCanonicalConsumers:
    def test_distance_profile_uses_the_same_canonical_source(self) -> None:
        source = _code((1, 1, 0), (0, 0, 0), (0, 1, 1))
        result = compute_distance_profile(BinaryCodeRequest(code=source))
        assert result.source is source
        assert result.minimum_distance == 2
        assert result.weight_profile == (0, 2, 2)

    def test_constant_weight_generator_is_source_bound(self) -> None:
        result = compute_constant_weight(ConstantWeightRequest(length=4, weight=2))
        assert result.length == 4
        assert result.weight == 2
        assert result.count == 6
        assert all(sum(word) == 2 for word in result.code.codewords)

    def test_constant_weight_admission_derives_the_work_from_length_and_weight(
        self,
    ) -> None:
        zero = compute_constant_weight(ConstantWeightRequest(length=64, weight=0))
        assert zero.count == 1
        assert zero.code.codewords == ((0,) * 64,)
        unit = compute_constant_weight(ConstantWeightRequest(length=64, weight=1))
        assert unit.count == 64
        assert all(sum(word) == 1 for word in unit.code.codewords)
        wide = compute_constant_weight(ConstantWeightRequest(length=17, weight=8))
        assert wide.count == 24310
        assert all(sum(word) == 8 for word in wide.code.codewords)
        serialized = wide.model_dump(mode="json")
        replayed = ConstantWeightResult.model_validate(serialized)
        assert replayed.code == wide.code
        axis = compute_constant_weight(
            ConstantWeightRequest(length=MAX_EXPLICIT_CODE_LENGTH, weight=0)
        )
        assert axis.count == 1
        assert axis.code.codewords == ((0,) * MAX_EXPLICIT_CODE_LENGTH,)

    def test_constant_weight_admission_rejects_central_binomial_work(self) -> None:
        with pytest.raises(ValidationError, match="entry bound"):
            ConstantWeightRequest.model_validate({"length": 64, "weight": 32})
        properties = ConstantWeightRequest.model_json_schema()["properties"]
        assert properties["length"]["maximum"] == MAX_EXPLICIT_CODE_LENGTH

    def test_zero_coordinate_generator_returns_the_sole_empty_word(self) -> None:
        result = compute_constant_weight(ConstantWeightRequest(length=0, weight=0))
        assert result.length == result.weight == 0
        assert result.count == 1
        assert result.code == ExplicitBinaryCode(length=0, codewords=((),))

        serialized_code = result.model_dump(mode="json")["code"]
        explicit = compute_explicit_profile(
            ExplicitProfileRequest.model_validate({"code": deepcopy(serialized_code)})
        )
        constant = compute_constant_weight_profile(
            ConstantWeightProfileRequest.model_validate(
                {"code": deepcopy(serialized_code)}
            )
        )
        supports = compute_to_set_system(
            ToSetSystemRequest.model_validate({"code": deepcopy(serialized_code)})
        )
        assert explicit.source == constant.source == supports.source == result.code
        assert explicit.weight_distribution == (1,)
        assert constant.intersection_histogram == (0,)
        assert supports.supports == ((),)
