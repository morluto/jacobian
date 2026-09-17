"""Tests for the longest Weyl-group element operation."""

from collections.abc import Callable

import pytest

from jacobian.canonical import encode_strict_json
from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.groups.root_systems._models import (
    CartanMatrix as CartanMatrixValue,
)
from jacobian.math.groups.root_systems._models import (
    CartanMatrixRequest,
    WeylLongestElementResult,
)
from jacobian.math.groups.root_systems.operations import (
    cartan_matrix_from_type,
    root_system_data,
    weyl_element_length,
    weyl_longest_element,
)

Matrix = tuple[tuple[int, ...], ...]

A1: Matrix = ((2,),)
A2: Matrix = ((2, -1), (-1, 2))
A2_AFFINE: Matrix = ((2, -1, -1), (-1, 2, -1), (-1, -1, 2))

EXPECTED_LENGTHS: dict[str, Callable[[int], int]] = {
    "A": lambda rank: rank * (rank + 1) // 2,
    "B": lambda rank: rank * rank,
    "C": lambda rank: rank * rank,
    "D": lambda rank: rank * (rank - 1),
    "E": lambda rank: {6: 36, 7: 63, 8: 120}[rank],
    "F": lambda rank: 24,
    "G": lambda rank: 6,
}

VALID_RANKS = {
    "A": (1, 2, 3, 4, 5, 6, 7, 8),
    "B": (2, 3, 4, 5, 6, 7, 8),
    "C": (2, 3, 4, 5, 6, 7, 8),
    "D": (4, 5, 6, 7, 8),
    "E": (6, 7, 8),
    "F": (4,),
    "G": (2,),
}


def _cartan(rows: Matrix) -> CartanMatrixValue:
    return CartanMatrixValue.model_validate(rows)


def _reflect(root: tuple[int, ...], index: int, rows: Matrix) -> tuple[int, ...]:
    coefficient = sum(root[j] * rows[index][j] for j in range(len(rows)))
    image = list(root)
    image[index] -= coefficient
    return tuple(image)


class TestLongestElementClassification:
    @pytest.mark.parametrize("cartan_type", ["A", "B", "C", "D", "E", "F", "G"])
    def test_length_matches_positive_root_count(self, cartan_type: str) -> None:
        for rank in VALID_RANKS[cartan_type]:
            built = cartan_matrix_from_type(cartan_type, rank)
            result = weyl_longest_element(built.matrix)
            expected = EXPECTED_LENGTHS[cartan_type](rank)
            assert result.length == expected
            assert result.num_positive_roots == expected
            assert len(result.word) == expected

    def test_a1_longest_word(self) -> None:
        result = weyl_longest_element(A1)
        assert result.word == (0,)
        assert (result.length, result.num_positive_roots) == (1, 1)

    def test_a2_longest_word_is_reduced(self) -> None:
        result = weyl_longest_element(A2)
        assert result.length == 3
        assert len(result.word) == 3
        assert all(index in (0, 1) for index in result.word)


class TestLongestElementMaximality:
    def test_a2_word_sends_every_positive_root_negative(self) -> None:
        result = weyl_longest_element(A2)
        for root in ((1, 0), (0, 1), (1, 1)):
            image: tuple[int, ...] = root
            for index in result.word:
                image = _reflect(image, index, A2)
            assert all(coordinate <= 0 for coordinate in image)
            assert any(image)

    def test_word_agrees_with_length_operation(self) -> None:
        """The longest word fed to the length operation is reduced with
        the same length: producer-consumer composition."""
        for matrix in (A1, A2):
            longest = weyl_longest_element(matrix)
            measured = weyl_element_length(matrix, longest.word)
            assert measured.length == longest.length
            assert measured.is_reduced is True

    def test_length_matches_root_system_data(self) -> None:
        built = cartan_matrix_from_type("D", 4)
        longest = weyl_longest_element(built.matrix)
        data = root_system_data(built.matrix)
        assert longest.length == data.num_positive_roots == 12


class TestLongestElementRejections:
    def test_affine_matrix_rejected(self) -> None:
        with pytest.raises(OperationDomainValidationError) as exc_info:
            weyl_longest_element(A2_AFFINE)
        assert exc_info.value.errors()[0]["type"] == "root_system.finite_type"

    def test_native_accepts_canonical_matrix_values(self) -> None:
        assert weyl_longest_element(A2).length == 3


class TestLongestElementComposition:
    def test_request_model_and_native_paths_agree(self) -> None:
        request = CartanMatrixRequest(matrix=_cartan(A2))
        assert weyl_longest_element(request.matrix) == weyl_longest_element(A2)

    def test_serialized_result_round_trips(self) -> None:
        result = weyl_longest_element(A2)
        revived = WeylLongestElementResult.model_validate_json(result.model_dump_json())
        assert revived == result

    def test_catalog_declares_the_operation_with_a_valid_example(self) -> None:
        from jacobian.math.groups.root_systems._tools import TOOLS

        tools = {
            tool.operation_id: tool
            for tool in TOOLS
            if tool.operation_id == "weyl_group.longest_element.compute"
        }
        assert set(tools) == {"weyl_group.longest_element.compute"}
        tool = tools["weyl_group.longest_element.compute"]
        assert tool.examples
        payload = tool.request_type.model_validate_json(
            encode_strict_json(tool.examples[0].input), strict=True
        )
        assert tool.run(payload) == weyl_longest_element(payload.matrix)
