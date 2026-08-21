"""Tests for root system operations."""

import pytest
from pydantic import ValidationError

from jacobian.math.root_systems._models import CartanMatrixRequest
from jacobian.math.root_systems._operations import compute_root_system_data

A2 = [[2, -1], [-1, 2]]
A3 = [[2, -1, 0], [-1, 2, -1], [0, -1, 2]]
G2 = [[2, -3], [-1, 2]]


class TestCartanMatrix:
    def test_valid_a2(self) -> None:
        CartanMatrixRequest(matrix=A2)

    def test_valid_g2(self) -> None:
        CartanMatrixRequest(matrix=G2)

    def test_invalid_non_symmetric(self) -> None:
        with pytest.raises(ValidationError, match="product"):
            CartanMatrixRequest(matrix=[[2, -4], [-1, 2]])

    def test_invalid_diagonal(self) -> None:
        with pytest.raises(ValidationError, match="diagonal"):
            CartanMatrixRequest(matrix=[[3, -1], [-1, 2]])

    def test_invalid_positive_offdiag(self) -> None:
        with pytest.raises(ValidationError, match="non-positive"):
            CartanMatrixRequest(matrix=[[2, 1], [-1, 2]])


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


class TestWeylGroupData:
    """Tests for Weyl group data computation."""

    def test_a2_weyl_group(self) -> None:
        """|W(A2)| = 6, h = 3."""
        from jacobian.math.root_systems._models import WeylGroupDataRequest
        from jacobian.math.root_systems._operations import compute_weyl_group_data

        result = compute_weyl_group_data(WeylGroupDataRequest(matrix=A2))
        assert result.group_order == 6
        assert result.coxeter_number == 3

    def test_a3_weyl_group(self) -> None:
        """|W(A3)| = 24, h = 4."""
        from jacobian.math.root_systems._models import WeylGroupDataRequest
        from jacobian.math.root_systems._operations import compute_weyl_group_data

        result = compute_weyl_group_data(WeylGroupDataRequest(matrix=A3))
        assert result.group_order == 24
        assert result.coxeter_number == 4

    def test_g2_weyl_group(self) -> None:
        """|W(G2)| = 12, h = 6."""
        from jacobian.math.root_systems._models import WeylGroupDataRequest
        from jacobian.math.root_systems._operations import compute_weyl_group_data

        result = compute_weyl_group_data(WeylGroupDataRequest(matrix=G2))
        assert result.group_order == 12
        assert result.coxeter_number == 6
