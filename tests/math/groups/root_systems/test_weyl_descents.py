"""Tests for the Weyl-group descent-sets operation."""

import pytest

from jacobian.canonical import encode_strict_json
from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.groups.root_systems._models import (
    CartanMatrix as CartanMatrixValue,
)
from jacobian.math.groups.root_systems._models import (
    WeylDescentsResult,
    WeylElementRequest,
)
from jacobian.math.groups.root_systems.operations import (
    weyl_element_descents,
    weyl_element_length,
    weyl_longest_element,
)

Matrix = tuple[tuple[int, ...], ...]

A2: Matrix = ((2, -1), (-1, 2))
B2: Matrix = ((2, -2), (-1, 2))
A2_AFFINE: Matrix = ((2, -1, -1), (-1, 2, -1), (-1, -1, 2))


def _cartan(rows: Matrix) -> CartanMatrixValue:
    return CartanMatrixValue.model_validate(rows)


class TestDescentsKnownAnswers:
    @pytest.mark.parametrize(
        ("matrix", "word", "left", "right"),
        (
            (A2, (), (), ()),
            (A2, (0,), (0,), (0,)),
            (A2, (1,), (1,), (1,)),
            (A2, (0, 0), (), ()),
            (A2, (0, 1), (1,), (0,)),
            (A2, (1, 0), (0,), (1,)),
            (A2, (0, 1, 0), (0, 1), (0, 1)),
            (B2, (0, 1), (1,), (0,)),
            (B2, (0, 1, 0, 1), (0, 1), (0, 1)),
        ),
    )
    def test_descent_sets(
        self,
        matrix: Matrix,
        word: tuple[int, ...],
        left: tuple[int, ...],
        right: tuple[int, ...],
    ) -> None:
        result = weyl_element_descents(matrix, word)
        assert result.left_descents == left
        assert result.right_descents == right


class TestDescentsDefiningInvariant:
    @pytest.mark.parametrize(
        "word", [(), (0,), (1,), (0, 1), (1, 0), (0, 0), (0, 1, 0), (1, 0, 1, 0)]
    )
    def test_descents_lower_length(self, word: tuple[int, ...]) -> None:
        """Prepending a right descent (appending a left descent) drops
        the length by exactly one; any other simple reflection raises
        it by exactly one. Lengths come from the length operation."""
        result = weyl_element_descents(A2, word)
        base = weyl_element_length(A2, word).length
        for index in (0, 1):
            prepended = weyl_element_length(A2, (index, *word)).length
            if index in result.right_descents:
                assert prepended == base - 1
            else:
                assert prepended == base + 1
            appended = weyl_element_length(A2, (*word, index)).length
            if index in result.left_descents:
                assert appended == base - 1
            else:
                assert appended == base + 1

    def test_longest_word_descends_everywhere(self) -> None:
        longest = weyl_longest_element(A2)
        result = weyl_element_descents(A2, longest.word)
        assert result.left_descents == result.right_descents == (0, 1)


class TestDescentsRejections:
    @pytest.mark.parametrize("word", ((2,), (-1,), (True,), (0.0,)))
    def test_invalid_indices_rejected(self, word: object) -> None:
        with pytest.raises(OperationDomainValidationError) as exc_info:
            weyl_element_descents(A2, word)  # type: ignore[arg-type]
        assert exc_info.value.errors()[0]["type"] == "root_system.invalid_weyl_word"

    def test_affine_matrix_rejected(self) -> None:
        with pytest.raises(OperationDomainValidationError) as exc_info:
            weyl_element_descents(A2_AFFINE, (0,))
        assert exc_info.value.errors()[0]["type"] == "root_system.finite_type"


class TestDescentsComposition:
    def test_request_model_and_native_paths_agree(self) -> None:
        request = WeylElementRequest(matrix=_cartan(A2), word=(0, 1))
        assert weyl_element_descents(
            request.matrix, request.word
        ) == weyl_element_descents(A2, (0, 1))

    def test_serialized_result_round_trips(self) -> None:
        result = weyl_element_descents(A2, (0, 1))
        revived = WeylDescentsResult.model_validate_json(result.model_dump_json())
        assert revived == result

    def test_catalog_declares_the_operation_with_a_valid_example(self) -> None:
        from jacobian.math.groups.root_systems._tools import TOOLS

        tools = {
            tool.operation_id: tool
            for tool in TOOLS
            if tool.operation_id == "weyl_group.element.descents.compute"
        }
        assert set(tools) == {"weyl_group.element.descents.compute"}
        tool = tools["weyl_group.element.descents.compute"]
        assert tool.examples
        payload = tool.request_type.model_validate_json(
            encode_strict_json(tool.examples[0].input), strict=True
        )
        assert tool.run(payload) == weyl_element_descents(payload.matrix, payload.word)
