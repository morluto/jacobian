"""Tests for root system operations."""

import pytest
from pydantic import ValidationError

from jacobian.math.root_systems._models import CartanMatrixRequest
from jacobian.math.root_systems._operations import compute_root_system_data

A2 = [[2, -1], [-1, 2]]
A3 = [[2, -1, 0], [-1, 2, -1], [0, -1, 2]]
G2 = [[2, -3], [-1, 2]]
B2 = [[2, -2], [-1, 2]]
A1_X_A2 = [[2, 0, 0], [0, 2, -1], [0, -1, 2]]
D4 = [[2, -1, 0, 0], [-1, 2, -1, -1], [0, -1, 2, 0], [0, -1, 0, 2]]
E8 = [
    [2, -1, 0, 0, 0, 0, 0, 0],
    [-1, 2, -1, 0, 0, 0, 0, 0],
    [0, -1, 2, -1, 0, 0, 0, -1],
    [0, 0, -1, 2, -1, 0, 0, 0],
    [0, 0, 0, -1, 2, -1, 0, 0],
    [0, 0, 0, 0, -1, 2, -1, 0],
    [0, 0, 0, 0, 0, -1, 2, 0],
    [0, 0, -1, 0, 0, 0, 0, 2],
]


class TestCartanMatrix:
    def test_valid_a2(self) -> None:
        CartanMatrixRequest(matrix=A2)

    def test_valid_g2(self) -> None:
        CartanMatrixRequest(matrix=G2)

    def test_invalid_non_symmetric(self) -> None:
        with pytest.raises(ValidationError) as exc_info:
            CartanMatrixRequest(matrix=[[2, -4], [-1, 2]])
        assert exc_info.value.errors()[0]["type"] == "root_system.off_diagonal_product"

    def test_invalid_diagonal(self) -> None:
        with pytest.raises(ValidationError) as exc_info:
            CartanMatrixRequest(matrix=[[3, -1], [-1, 2]])
        assert exc_info.value.errors()[0]["type"] == "root_system.diagonal_entry"

    def test_invalid_positive_offdiag(self) -> None:
        with pytest.raises(ValidationError) as exc_info:
            CartanMatrixRequest(matrix=[[2, 1], [-1, 2]])
        assert exc_info.value.errors()[0]["type"] == "root_system.positive_off_diagonal"


class TestRootSystemData:
    def test_a2_positive_roots(self) -> None:
        result = compute_root_system_data(CartanMatrixRequest(matrix=A2))
        assert result.rank == 2
        assert result.num_positive_roots == 3
        assert result.components[0].coxeter_number == 3
        assert result.components[0].highest_root == (1, 1)

    def test_a3_positive_roots(self) -> None:
        result = compute_root_system_data(CartanMatrixRequest(matrix=A3))
        assert result.rank == 3
        assert result.num_positive_roots == 6
        assert result.components[0].coxeter_number == 4

    def test_g2_positive_roots(self) -> None:
        result = compute_root_system_data(CartanMatrixRequest(matrix=G2))
        assert result.rank == 2
        assert result.num_positive_roots == 6
        assert result.components[0].coxeter_number == 6

    def test_negative_roots(self) -> None:
        result = compute_root_system_data(CartanMatrixRequest(matrix=A2))
        for pos, neg in zip(result.positive_roots, result.negative_roots, strict=True):
            assert all(a + b == 0 for a, b in zip(pos, neg, strict=True))

    def test_simple_roots(self) -> None:
        result = compute_root_system_data(CartanMatrixRequest(matrix=A2))
        assert result.simple_roots == ((1, 0), (0, 1))


class TestSimpleReflection:
    """Tests for simple reflection operation."""

    def test_reflect_onto_itself(self) -> None:
        """s_i(alpha_i) = -alpha_i."""
        from jacobian.math.root_systems._models import SimpleReflectionRequest
        from jacobian.math.root_systems._operations import compute_simple_reflection

        result = compute_simple_reflection(
            SimpleReflectionRequest(matrix=A2, vector=(1, 0), simple_index=0)
        )
        assert result.reflected_vector == (-1, 0)

    def test_reflect_other_simple_root(self) -> None:
        """s_0(alpha_1) = alpha_1 - A[0][1]*alpha_0 = alpha_1 + alpha_0."""
        from jacobian.math.root_systems._models import SimpleReflectionRequest
        from jacobian.math.root_systems._operations import compute_simple_reflection

        result = compute_simple_reflection(
            SimpleReflectionRequest(matrix=A2, vector=(0, 1), simple_index=0)
        )
        assert result.reflected_vector == (1, 1)

    def test_reflect_in_a3(self) -> None:
        """s_1(alpha_0) in A3."""
        from jacobian.math.root_systems._models import SimpleReflectionRequest
        from jacobian.math.root_systems._operations import compute_simple_reflection

        # s_1(alpha_0) = alpha_0 - A[1][0]*alpha_1 = alpha_0 + alpha_1
        result = compute_simple_reflection(
            SimpleReflectionRequest(matrix=A3, vector=(1, 0, 0), simple_index=1)
        )
        assert result.reflected_vector == (1, 1, 0)


class TestWeylGroupOrder:
    """Tests for exact Weyl-group order through the signed-root action."""

    @pytest.mark.parametrize(
        ("matrix", "expected"),
        (
            ([[2]], 2),
            (A2, 6),
            (B2, 8),
            (G2, 12),
            (A1_X_A2, 12),
            (D4, 192),
        ),
    )
    def test_known_orders(self, matrix: list[list[int]], expected: int) -> None:
        from jacobian.math.root_systems._operations import compute_weyl_group_order

        result = compute_weyl_group_order(CartanMatrixRequest(matrix=matrix))

        assert result.group_order == expected
        assert result.matrix == tuple(tuple(row) for row in matrix)
        assert result.method == "SYMPY_SCHREIER_SIMS_SIGNED_ROOT_ACTION"

    def test_e8_order_does_not_materialize_weyl_group_elements(self) -> None:
        from jacobian.math.root_systems._operations import compute_weyl_group_order

        result = compute_weyl_group_order(CartanMatrixRequest(matrix=E8))

        assert result.group_order == 696_729_600

    def test_result_replays_signed_root_action_order(self) -> None:
        from jacobian.math.root_systems._models import WeylGroupOrderResult
        from jacobian.math.root_systems._operations import (
            verify_weyl_group_order_result,
        )

        claimed = WeylGroupOrderResult(matrix=A2, group_order=5)
        assert not verify_weyl_group_order_result(claimed)

    def test_kernel_results_are_verified_by_explicit_owner_paths(self) -> None:
        from jacobian.math.root_systems._models import SimpleReflectionRequest
        from jacobian.math.root_systems._operations import (
            compute_simple_reflection,
            compute_weyl_group_order,
            verify_root_system_data_result,
            verify_simple_reflection_result,
            verify_weyl_group_order_result,
        )

        root_data = compute_root_system_data(CartanMatrixRequest(matrix=A2))
        reflection = compute_simple_reflection(
            SimpleReflectionRequest(matrix=A2, vector=(1, 0), simple_index=0)
        )
        order = compute_weyl_group_order(CartanMatrixRequest(matrix=A2))

        assert verify_root_system_data_result(root_data)
        assert verify_simple_reflection_result(reflection)
        assert verify_weyl_group_order_result(order)

    def test_catalog_replaces_the_invalid_mixed_weyl_data_contract(self) -> None:
        from jacobian.math.root_systems._tools import TOOLS

        operation_ids = {tool.operation_id for tool in TOOLS}
        assert "root_system.weyl_group_order.compute" in operation_ids
        assert "root_system.weyl_group_data.compute" not in operation_ids
