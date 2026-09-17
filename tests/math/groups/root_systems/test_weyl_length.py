"""Tests for the Weyl-group word-length operation."""

import pytest

from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.groups.root_systems._models import (
    MAX_WEYL_WORD_LENGTH,
    WeylElementLengthResult,
    WeylElementRequest,
)
from jacobian.math.groups.root_systems._models import (
    CartanMatrix as CartanMatrixValue,
)
from jacobian.math.groups.root_systems.operations import (
    cartan_matrix_from_type,
    weyl_element_length,
)

Matrix = tuple[tuple[int, ...], ...]

A2: Matrix = ((2, -1), (-1, 2))
A3: Matrix = ((2, -1, 0), (-1, 2, -1), (0, -1, 2))
B2: Matrix = ((2, -2), (-1, 2))
G2: Matrix = ((2, -3), (-1, 2))
A2_AFFINE: Matrix = ((2, -1, -1), (-1, 2, -1), (-1, -1, 2))

A2_POSITIVES: Matrix = ((1, 0), (0, 1), (1, 1))


def _cartan(rows: Matrix) -> CartanMatrixValue:
    return CartanMatrixValue.model_validate(rows)


def _reflect(root: tuple[int, ...], index: int, rows: Matrix) -> tuple[int, ...]:
    """Independent simple-reflection replay for inversion checks."""
    coefficient = sum(root[j] * rows[index][j] for j in range(len(rows)))
    image = list(root)
    image[index] -= coefficient
    return tuple(image)


def _apply(
    root: tuple[int, ...], word: tuple[int, ...], rows: Matrix
) -> tuple[int, ...]:
    image = root
    for index in word:
        image = _reflect(image, index, rows)
    return image


class TestWeylLengthKnownAnswers:
    @pytest.mark.parametrize(
        ("matrix", "word", "length", "reduced", "inversions"),
        (
            (A2, (), 0, True, ()),
            (A2, (0,), 1, True, ((1, 0),)),
            (A2, (0, 0), 0, False, ()),
            (A2, (0, 1, 0), 3, True, ((0, 1), (1, 0), (1, 1))),
            (A2, (0, 1, 1, 0), 0, False, ()),
            (A3, (0, 1, 2, 0, 1, 0), 6, True, None),
            (B2, (0,), 1, True, ((1, 0),)),
            (B2, (0, 1, 0, 1), 4, True, None),
            (G2, (0, 1, 0, 1, 0, 1), 6, True, None),
        ),
    )
    def test_length_reducedness_and_inversions(
        self,
        matrix: Matrix,
        word: tuple[int, ...],
        length: int,
        reduced: bool,
        inversions: Matrix | None,
    ) -> None:
        result = weyl_element_length(matrix, word)
        assert result.length == length
        assert result.is_reduced is reduced
        assert len(result.inversions) == length
        if inversions is not None:
            assert result.inversions == inversions

    def test_a3_staircase_is_the_longest_element(self) -> None:
        result = weyl_element_length(A3, (0, 1, 2, 0, 1, 0))
        assert result.length == 6 == 3 * 4 // 2

    def test_braid_words_share_length_and_inversions(self) -> None:
        """s_0 s_1 s_0 and s_1 s_0 s_1 are the same A2 element."""
        first = weyl_element_length(A2, (0, 1, 0))
        second = weyl_element_length(A2, (1, 0, 1))
        assert first.length == second.length == 3
        assert first.inversions == second.inversions

    def test_list_word_matches_tuple_word(self) -> None:
        assert weyl_element_length(A2, [0, 1, 0]) == weyl_element_length(A2, (0, 1, 0))


class TestWeylLengthDefiningInvariant:
    @pytest.mark.parametrize(
        ("matrix", "positives", "word"),
        (
            (A2, A2_POSITIVES, (0, 1, 0)),
            (A2, A2_POSITIVES, (0, 1)),
            (B2, ((1, 0), (0, 1), (1, 1), (2, 1)), (0, 1, 0)),
        ),
    )
    def test_inversions_are_exactly_the_roots_sent_negative(
        self, matrix: Matrix, positives: Matrix, word: tuple[int, ...]
    ) -> None:
        """Replay the word with an independent reflection loop: claimed
        inversions land strictly negative and every other positive root
        stays positive."""
        result = weyl_element_length(matrix, word)
        assert result.length == len(result.inversions)
        for root in positives:
            image = _apply(root, word, matrix)
            assert any(image)
            if root in result.inversions:
                assert all(coordinate <= 0 for coordinate in image)
            else:
                assert all(coordinate >= 0 for coordinate in image)

    def test_length_never_exceeds_positive_root_count(self) -> None:
        for word in ((), (0,), (1, 0), (0, 1, 0), (1, 0, 1, 0)):
            assert weyl_element_length(A2, word).length <= 3


class TestWeylLengthRejections:
    @pytest.mark.parametrize(
        "word",
        (
            (2,),
            (0, 2),
            (-1,),
            (True,),
            (0.0,),
            ("0",),
            ((0,),),
        ),
    )
    def test_invalid_indices_rejected(self, word: object) -> None:
        with pytest.raises(OperationDomainValidationError) as exc_info:
            weyl_element_length(A2, word)  # type: ignore[arg-type]
        assert exc_info.value.errors()[0]["type"] == "root_system.invalid_weyl_word"

    def test_overlong_word_rejected(self) -> None:
        with pytest.raises(OperationDomainValidationError) as exc_info:
            weyl_element_length(A2, (0,) * (MAX_WEYL_WORD_LENGTH + 1))
        assert exc_info.value.errors()[0]["type"] == "root_system.invalid_weyl_word"

    def test_non_tuple_word_rejected(self) -> None:
        with pytest.raises(OperationDomainValidationError):
            weyl_element_length(A2, "010")  # type: ignore[arg-type]

    def test_affine_matrix_rejected(self) -> None:
        with pytest.raises(OperationDomainValidationError) as exc_info:
            weyl_element_length(A2_AFFINE, (0,))
        assert exc_info.value.errors()[0]["type"] == "root_system.finite_type"

    def test_request_model_rejects_negative_and_overlong_words(self) -> None:
        from pydantic import ValidationError as PydanticValidationError

        with pytest.raises(PydanticValidationError):
            WeylElementRequest(matrix=_cartan(A2), word=(-1,))
        with pytest.raises(PydanticValidationError):
            WeylElementRequest(
                matrix=_cartan(A2), word=(0,) * (MAX_WEYL_WORD_LENGTH + 1)
            )


class TestWeylLengthComposition:
    def test_request_model_and_native_paths_agree(self) -> None:
        request = WeylElementRequest(matrix=_cartan(A2), word=(0, 1, 0))
        assert weyl_element_length(request.matrix, request.word) == weyl_element_length(
            A2, (0, 1, 0)
        )

    def test_constructor_result_feeds_length(self) -> None:
        built = cartan_matrix_from_type("A", 2)
        result = weyl_element_length(built.matrix, (0, 1, 0))
        assert (result.length, result.is_reduced) == (3, True)

    def test_serialized_result_round_trips(self) -> None:
        result = weyl_element_length(A2, (0, 1, 0))
        revived = WeylElementLengthResult.model_validate_json(result.model_dump_json())
        assert revived == result

    def test_e8_simple_reflection_has_length_one(self) -> None:
        built = cartan_matrix_from_type("E", 8)
        result = weyl_element_length(built.matrix, (0,))
        assert (result.length, result.is_reduced) == (1, True)

    def test_catalog_declares_the_operation_with_a_valid_example(self) -> None:
        from jacobian.canonical import encode_strict_json
        from jacobian.math.groups.root_systems._tools import TOOLS

        tools = {
            tool.operation_id: tool
            for tool in TOOLS
            if tool.operation_id == "weyl_group.element.length.compute"
        }
        assert set(tools) == {"weyl_group.element.length.compute"}
        tool = tools["weyl_group.element.length.compute"]
        assert tool.examples
        payload = tool.request_type.model_validate_json(
            encode_strict_json(tool.examples[0].input), strict=True
        )
        assert tool.run(payload) == weyl_element_length(payload.matrix, payload.word)
