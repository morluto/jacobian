"""Tests for root system operations."""

import pytest

from jacobian.canonical import canonicalize_json
from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.groups.root_systems._models import (
    CartanMatrix as CartanMatrixValue,
)
from jacobian.math.groups.root_systems._models import (
    CartanMatrixRequest,
    RootSystemDataResult,
    SimpleReflectionRequest,
    SimpleReflectionResult,
    WeylGroupOrderResult,
)
from jacobian.math.groups.root_systems.operations import (
    MAX_ROOT_PROFILE_OUTPUT_BYTES,
    positive_root_profile,
    positive_roots,
    root_system_data,
    simple_reflection,
    weyl_group_order,
)

CartanMatrix = tuple[tuple[int, ...], ...]


def _cartan(rows: CartanMatrix) -> CartanMatrixValue:
    return CartanMatrixValue.model_validate(rows)


def compute_root_system_data(request: CartanMatrixRequest) -> RootSystemDataResult:
    return root_system_data(request.matrix)


def compute_simple_reflection(
    request: SimpleReflectionRequest,
) -> SimpleReflectionResult:
    return simple_reflection(request.matrix, request.vector, request.simple_index)


def compute_weyl_group_order(request: CartanMatrixRequest) -> WeylGroupOrderResult:
    return weyl_group_order(request.matrix)


A2: CartanMatrix = ((2, -1), (-1, 2))
A3: CartanMatrix = ((2, -1, 0), (-1, 2, -1), (0, -1, 2))
G2: CartanMatrix = ((2, -3), (-1, 2))
B2: CartanMatrix = ((2, -2), (-1, 2))
A1_X_A2: CartanMatrix = ((2, 0, 0), (0, 2, -1), (0, -1, 2))
A2_AFFINE: CartanMatrix = ((2, -1, -1), (-1, 2, -1), (-1, -1, 2))
D4: CartanMatrix = (
    (2, -1, 0, 0),
    (-1, 2, -1, -1),
    (0, -1, 2, 0),
    (0, -1, 0, 2),
)
E8: CartanMatrix = (
    (2, -1, 0, 0, 0, 0, 0, 0),
    (-1, 2, -1, 0, 0, 0, 0, 0),
    (0, -1, 2, -1, 0, 0, 0, -1),
    (0, 0, -1, 2, -1, 0, 0, 0),
    (0, 0, 0, -1, 2, -1, 0, 0),
    (0, 0, 0, 0, -1, 2, -1, 0),
    (0, 0, 0, 0, 0, -1, 2, 0),
    (0, 0, -1, 0, 0, 0, 0, 2),
)


class TestCartanMatrix:
    def test_valid_a2(self) -> None:
        CartanMatrixRequest(matrix=_cartan(A2))

    def test_valid_g2(self) -> None:
        CartanMatrixRequest(matrix=_cartan(G2))

    def test_invalid_non_symmetric(self) -> None:
        request = CartanMatrixRequest.model_validate({"matrix": [[2, -4], [-1, 2]]})
        with pytest.raises(OperationDomainValidationError) as exc_info:
            compute_root_system_data(request)
        assert exc_info.value.errors()[0]["type"] == "root_system.off_diagonal_product"

    def test_invalid_diagonal(self) -> None:
        request = CartanMatrixRequest.model_validate({"matrix": [[3, -1], [-1, 2]]})
        with pytest.raises(OperationDomainValidationError) as exc_info:
            compute_root_system_data(request)
        assert exc_info.value.errors()[0]["type"] == "root_system.diagonal_entry"

    def test_invalid_positive_offdiag(self) -> None:
        request = CartanMatrixRequest.model_validate({"matrix": [[2, 1], [-1, 2]]})
        with pytest.raises(OperationDomainValidationError) as exc_info:
            compute_root_system_data(request)
        assert exc_info.value.errors()[0]["type"] == "root_system.positive_off_diagonal"

    def test_finite_type_is_admitted_by_the_owner_operation(self) -> None:
        request = CartanMatrixRequest(matrix=_cartan(A2_AFFINE))
        with pytest.raises(OperationDomainValidationError) as exc_info:
            compute_root_system_data(request)
        assert exc_info.value.errors()[0]["type"] == "root_system.finite_type"

    def test_native_surface_accepts_canonical_matrix_values(self) -> None:
        assert root_system_data(A2).num_positive_roots == 3
        assert len(positive_roots(A2).positive_roots) == 3
        assert simple_reflection(A2, (1, 0), 0).reflected_vector == (-1, 0)
        assert weyl_group_order(A2).group_order == 6


class TestRootSystemData:
    def test_a2_positive_roots(self) -> None:
        result = compute_root_system_data(CartanMatrixRequest(matrix=_cartan(A2)))
        assert result.rank == 2
        assert result.num_positive_roots == 3
        assert result.components[0].coxeter_number == 3
        assert result.components[0].highest_root == (1, 1)

    def test_a3_positive_roots(self) -> None:
        result = compute_root_system_data(CartanMatrixRequest(matrix=_cartan(A3)))
        assert result.rank == 3
        assert result.num_positive_roots == 6
        assert result.components[0].coxeter_number == 4

    def test_g2_positive_roots(self) -> None:
        result = compute_root_system_data(CartanMatrixRequest(matrix=_cartan(G2)))
        assert result.rank == 2
        assert result.num_positive_roots == 6
        assert result.components[0].coxeter_number == 6

    def test_negative_roots(self) -> None:
        result = compute_root_system_data(CartanMatrixRequest(matrix=_cartan(A2)))
        for pos, neg in zip(result.positive_roots, result.negative_roots, strict=True):
            assert all(a + b == 0 for a, b in zip(pos, neg, strict=True))

    def test_simple_roots(self) -> None:
        result = compute_root_system_data(CartanMatrixRequest(matrix=_cartan(A2)))
        assert result.simple_roots == ((1, 0), (0, 1))


class TestPositiveRootProfile:
    @pytest.mark.parametrize(
        ("matrix", "roots", "heights", "supports", "highest"),
        (
            (
                ((2,),),
                ((1,),),
                (1,),
                ((0,),),
                ((0, (1,)),),
            ),
            (
                A2,
                ((0, 1), (1, 0), (1, 1)),
                (1, 1, 2),
                ((1,), (0,), (0, 1)),
                ((2, (1, 1)),),
            ),
            (
                B2,
                ((0, 1), (1, 0), (1, 1), (2, 1)),
                (1, 1, 2, 3),
                ((1,), (0,), (0, 1), (0, 1)),
                ((3, (2, 1)),),
            ),
            (
                G2,
                ((0, 1), (1, 0), (1, 1), (2, 1), (3, 1), (3, 2)),
                (1, 1, 2, 3, 4, 5),
                ((1,), (0,), (0, 1), (0, 1), (0, 1), (0, 1)),
                ((5, (3, 2)),),
            ),
        ),
    )
    def test_independent_root_profiles(
        self,
        matrix: CartanMatrix,
        roots: tuple[tuple[int, ...], ...],
        heights: tuple[int, ...],
        supports: tuple[tuple[int, ...], ...],
        highest: tuple[tuple[int, tuple[int, ...]], ...],
    ) -> None:
        result = positive_root_profile(matrix)
        assert result.datum.cartan_matrix.entries == matrix
        assert (
            tuple(entry.root_coefficients for entry in result.positive_roots) == roots
        )
        assert tuple(entry.height for entry in result.positive_roots) == heights
        assert (
            tuple(entry.support_simple_root_indices for entry in result.positive_roots)
            == supports
        )
        assert (
            tuple(
                (
                    component.highest_root_index,
                    result.positive_roots[
                        component.highest_root_index
                    ].root_coefficients,
                )
                for component in result.components
            )
            == highest
        )

    def test_disconnected_datum_preserves_factor_and_root_axes(self) -> None:
        result = positive_root_profile(A1_X_A2)
        assert tuple(
            component.simple_root_indices for component in result.components
        ) == (
            (0,),
            (1, 2),
        )
        assert tuple(
            component.positive_root_indices for component in result.components
        ) == (
            (3,),
            (0, 1, 2),
        )
        assert tuple(
            component.highest_root_index for component in result.components
        ) == (
            3,
            2,
        )

    def test_e8_full_root_family_fits_the_admitted_profile_output_bound(self) -> None:
        result = positive_root_profile(E8)
        encoded = canonicalize_json(result.model_dump(mode="json"))
        assert len(result.positive_roots) == 120
        assert len(encoded) <= MAX_ROOT_PROFILE_OUTPUT_BYTES
        highest = result.positive_roots[result.components[0].highest_root_index]
        assert highest.height == 29

    def test_serialized_profile_roundtrips_with_parent_and_axis(self) -> None:
        result = positive_root_profile(G2)
        revived = type(result).model_validate_json(result.model_dump_json())
        assert revived == result
        assert revived.datum.cartan_matrix.entries == G2
        assert (
            revived.positive_roots[revived.components[0].highest_root_index].height == 5
        )


class TestSimpleReflection:
    """Tests for simple reflection operation."""

    def test_reflect_onto_itself(self) -> None:
        """s_i(alpha_i) = -alpha_i."""
        from jacobian.math.groups.root_systems._models import SimpleReflectionRequest

        result = compute_simple_reflection(
            SimpleReflectionRequest(matrix=_cartan(A2), vector=(1, 0), simple_index=0)
        )
        assert result.reflected_vector == (-1, 0)

    def test_reflect_other_simple_root(self) -> None:
        """s_0(alpha_1) = alpha_1 - A[0][1]*alpha_0 = alpha_1 + alpha_0."""
        from jacobian.math.groups.root_systems._models import SimpleReflectionRequest

        result = compute_simple_reflection(
            SimpleReflectionRequest(matrix=_cartan(A2), vector=(0, 1), simple_index=0)
        )
        assert result.reflected_vector == (1, 1)

    def test_reflect_in_a3(self) -> None:
        """s_1(alpha_0) in A3."""
        from jacobian.math.groups.root_systems._models import SimpleReflectionRequest

        # s_1(alpha_0) = alpha_0 - A[1][0]*alpha_1 = alpha_0 + alpha_1
        result = compute_simple_reflection(
            SimpleReflectionRequest(
                matrix=_cartan(A3), vector=(1, 0, 0), simple_index=1
            )
        )
        assert result.reflected_vector == (1, 1, 0)


class TestWeylGroupOrder:
    """Tests for exact Weyl-group order through the signed-root action."""

    @pytest.mark.parametrize(
        ("matrix", "expected"),
        (
            (((2,),), 2),
            (A2, 6),
            (B2, 8),
            (G2, 12),
            (A1_X_A2, 12),
            (D4, 192),
        ),
    )
    def test_known_orders(self, matrix: CartanMatrix, expected: int) -> None:
        result = compute_weyl_group_order(CartanMatrixRequest(matrix=_cartan(matrix)))

        assert result.group_order == expected
        assert result.matrix == matrix

    def test_e8_order_does_not_materialize_weyl_group_elements(self) -> None:
        result = compute_weyl_group_order(CartanMatrixRequest(matrix=_cartan(E8)))

        assert result.group_order == 696_729_600

    def test_catalog_replaces_the_invalid_mixed_weyl_data_contract(self) -> None:
        from jacobian.math.groups.root_systems._tools import TOOLS

        operation_ids = {tool.operation_id for tool in TOOLS}
        assert "root_system.weyl_group_order.compute" in operation_ids
        assert "root_system.weyl_group_data.compute" not in operation_ids
