"""Tests for the Cartan-matrix named constructor operation."""

from collections.abc import Callable
from fractions import Fraction

import pytest

from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.groups.root_systems._cartan import VALID_CARTAN_TYPE_RANKS
from jacobian.math.groups.root_systems._models import (
    CartanMatrix as CartanMatrixValue,
)
from jacobian.math.groups.root_systems._models import (
    CartanTypeRequest,
    CartanTypeResult,
)
from jacobian.math.groups.root_systems.operations import (
    cartan_datum,
    cartan_matrix_from_type,
    root_system_data,
    weyl_group_order,
)

Matrix = tuple[tuple[int, ...], ...]

A1: Matrix = ((2,),)
A2: Matrix = ((2, -1), (-1, 2))
A3: Matrix = ((2, -1, 0), (-1, 2, -1), (0, -1, 2))
B2: Matrix = ((2, -1), (-2, 2))
C2: Matrix = ((2, -2), (-1, 2))
G2: Matrix = ((2, -3), (-1, 2))
D4: Matrix = (
    (2, -1, 0, 0),
    (-1, 2, -1, -1),
    (0, -1, 2, 0),
    (0, -1, 0, 2),
)
F4: Matrix = (
    (2, -1, 0, 0),
    (-1, 2, -2, 0),
    (0, -1, 2, -1),
    (0, 0, -1, 2),
)
E8: Matrix = (
    (2, -1, 0, 0, 0, 0, 0, 0),
    (-1, 2, -1, 0, 0, 0, 0, 0),
    (0, -1, 2, -1, 0, 0, 0, -1),
    (0, 0, -1, 2, -1, 0, 0, 0),
    (0, 0, 0, -1, 2, -1, 0, 0),
    (0, 0, 0, 0, -1, 2, -1, 0),
    (0, 0, 0, 0, 0, -1, 2, 0),
    (0, 0, -1, 0, 0, 0, 0, 2),
)

EXPECTED_RANKS = {
    "A": (1, 2, 3, 4, 5, 6, 7, 8),
    "B": (2, 3, 4, 5, 6, 7, 8),
    "C": (2, 3, 4, 5, 6, 7, 8),
    "D": (4, 5, 6, 7, 8),
    "E": (6, 7, 8),
    "F": (4,),
    "G": (2,),
}

EXPECTED_DETERMINANTS: dict[str, Callable[[int], int]] = {
    "A": lambda rank: rank + 1,
    "B": lambda rank: 2,
    "C": lambda rank: 2,
    "D": lambda rank: 4,
    "E": lambda rank: {6: 3, 7: 2, 8: 1}[rank],
    "F": lambda rank: 1,
    "G": lambda rank: 1,
}

EXPECTED_POSITIVE_ROOTS: dict[str, Callable[[int], int]] = {
    "A": lambda rank: rank * (rank + 1) // 2,
    "B": lambda rank: rank * rank,
    "C": lambda rank: rank * rank,
    "D": lambda rank: rank * (rank - 1),
    "E": lambda rank: {6: 36, 7: 63, 8: 120}[rank],
    "F": lambda rank: 24,
    "G": lambda rank: 6,
}


def _bareiss_determinant(matrix: Matrix) -> int:
    """Exact integer determinant by the fraction-free Bareiss algorithm."""
    size = len(matrix)
    table = [list(row) for row in matrix]
    previous = 1
    for step in range(size - 1):
        if table[step][step] == 0:
            return 0
        for row in range(step + 1, size):
            for column in range(step + 1, size):
                table[row][column] = (
                    table[row][column] * table[step][step]
                    - table[row][step] * table[step][column]
                ) // previous
        previous = table[step][step]
    return table[size - 1][size - 1]


class TestCartanTypeAdmissionTable:
    def test_valid_rank_table_matches_classification(self) -> None:
        assert dict(VALID_CARTAN_TYPE_RANKS) == {
            key: tuple(value) for key, value in EXPECTED_RANKS.items()
        }

    def test_all_valid_pairs_build(self) -> None:
        for cartan_type, ranks in EXPECTED_RANKS.items():
            for rank in ranks:
                result = cartan_matrix_from_type(cartan_type, rank)
                assert result.cartan_type == cartan_type
                assert result.rank == rank
                assert len(result.matrix) == rank


class TestCartanTypeKnownAnswers:
    @pytest.mark.parametrize(
        ("cartan_type", "rank", "expected"),
        (
            ("A", 1, A1),
            ("A", 2, A2),
            ("A", 3, A3),
            ("B", 2, B2),
            ("C", 2, C2),
            ("G", 2, G2),
            ("D", 4, D4),
            ("F", 4, F4),
            ("E", 8, E8),
        ),
    )
    def test_exact_matrices(
        self, cartan_type: str, rank: int, expected: Matrix
    ) -> None:
        result = cartan_matrix_from_type(cartan_type, rank)
        assert result.matrix.entries == expected
        assert result.matrix == CartanMatrixValue.model_validate(expected)

    def test_b_and_c_differ_only_in_bond_direction(self) -> None:
        short_end = cartan_matrix_from_type("B", 3).matrix.entries
        long_end = cartan_matrix_from_type("C", 3).matrix.entries
        assert short_end[1][2] == -1 and short_end[2][1] == -2
        assert long_end[1][2] == -2 and long_end[2][1] == -1
        assert cartan_datum(short_end).symmetrizer[-1].as_fraction() == Fraction(1, 2)
        assert cartan_datum(long_end).symmetrizer[-1].as_fraction() == 2

    def test_e_family_shares_the_branch_node(self) -> None:
        for rank in (6, 7, 8):
            entries = cartan_matrix_from_type("E", rank).matrix.entries
            assert entries[2][rank - 1] == -1
            assert entries[rank - 1][2] == -1


class TestCartanTypeDefiningInvariants:
    @pytest.mark.parametrize("cartan_type", ["A", "B", "C", "D", "E", "F", "G"])
    def test_determinant_table(self, cartan_type: str) -> None:
        for rank in EXPECTED_RANKS[cartan_type]:
            entries = cartan_matrix_from_type(cartan_type, rank).matrix.entries
            assert _bareiss_determinant(entries) == EXPECTED_DETERMINANTS[cartan_type](
                rank
            )

    @pytest.mark.parametrize("cartan_type", ["A", "B", "C", "D", "E", "F", "G"])
    def test_positive_root_counts(self, cartan_type: str) -> None:
        for rank in EXPECTED_RANKS[cartan_type]:
            built = cartan_matrix_from_type(cartan_type, rank)
            data = root_system_data(built.matrix)
            assert data.num_positive_roots == EXPECTED_POSITIVE_ROOTS[cartan_type](rank)

    @pytest.mark.parametrize(
        ("cartan_type", "rank", "expected"),
        (
            ("A", 1, 2),
            ("A", 2, 6),
            ("A", 3, 24),
            ("B", 2, 8),
            ("C", 3, 48),
            ("D", 4, 192),
            ("F", 4, 1152),
            ("G", 2, 12),
        ),
    )
    def test_weyl_group_orders(
        self, cartan_type: str, rank: int, expected: int
    ) -> None:
        built = cartan_matrix_from_type(cartan_type, rank)
        assert weyl_group_order(built.matrix).group_order == expected


class TestCartanTypeRejections:
    @pytest.mark.parametrize("cartan_type", ["H", "a", "", "X", "AE"])
    def test_unknown_type(self, cartan_type: str) -> None:
        with pytest.raises(OperationDomainValidationError) as exc_info:
            cartan_matrix_from_type(cartan_type, 2)
        assert exc_info.value.errors()[0]["type"] == "root_system.unknown_cartan_type"

    @pytest.mark.parametrize(
        ("cartan_type", "rank"),
        (
            ("A", 0),
            ("A", 9),
            ("B", 1),
            ("C", 1),
            ("D", 3),
            ("D", 9),
            ("E", 5),
            ("E", 9),
            ("F", 3),
            ("F", 5),
            ("G", 1),
            ("G", 3),
        ),
    )
    def test_invalid_rank_for_type(self, cartan_type: str, rank: int) -> None:
        with pytest.raises(OperationDomainValidationError) as exc_info:
            cartan_matrix_from_type(cartan_type, rank)
        assert (
            exc_info.value.errors()[0]["type"] == "root_system.invalid_cartan_type_rank"
        )

    @pytest.mark.parametrize("bad_type", [None, 123, b"A", ("A",), True])
    def test_non_string_type_rejected(self, bad_type: object) -> None:
        with pytest.raises(OperationDomainValidationError) as exc_info:
            cartan_matrix_from_type(bad_type, 2)  # type: ignore[arg-type]
        assert exc_info.value.errors()[0]["type"] == "root_system.unknown_cartan_type"

    @pytest.mark.parametrize("bad_rank", ["2", 2.0, None, True, False, (2,)])
    def test_non_integer_rank_rejected(self, bad_rank: object) -> None:
        with pytest.raises(OperationDomainValidationError) as exc_info:
            cartan_matrix_from_type("A", bad_rank)  # type: ignore[arg-type]
        assert (
            exc_info.value.errors()[0]["type"] == "root_system.invalid_cartan_type_rank"
        )


class TestCartanTypeComposition:
    def test_request_model_and_native_paths_agree(self) -> None:
        request = CartanTypeRequest(cartan_type="B", rank=3)
        assert cartan_matrix_from_type(
            request.cartan_type, request.rank
        ) == cartan_matrix_from_type("B", 3)

    def test_serialized_result_feeds_root_system_data(self) -> None:
        built = cartan_matrix_from_type("D", 4)
        revived = CartanTypeResult.model_validate_json(built.model_dump_json())
        assert revived == built
        assert root_system_data(revived.matrix).num_positive_roots == 12

    def test_built_e8_matches_shipped_kernel_fixture(self) -> None:
        assert cartan_matrix_from_type(
            "E", 8
        ).matrix == CartanMatrixValue.model_validate(E8)

    def test_catalog_declares_the_operation_with_a_valid_example(self) -> None:
        from jacobian.math.groups.root_systems._tools import TOOLS

        tools = {
            tool.operation_id: tool
            for tool in TOOLS
            if tool.operation_id == "root_system.cartan_matrix.from_type.compute"
        }
        assert set(tools) == {"root_system.cartan_matrix.from_type.compute"}
        tool = tools["root_system.cartan_matrix.from_type.compute"]
        assert tool.examples
        payload = CartanTypeRequest.model_validate(tool.examples[0].input)
        assert tool.run(payload) == cartan_matrix_from_type(
            payload.cartan_type, payload.rank
        )
