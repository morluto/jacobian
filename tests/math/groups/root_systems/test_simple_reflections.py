"""Tests for the Weyl simple-reflections operation."""

from typing import cast

import pytest

from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.groups.root_systems._models import (
    CartanMatrix as CartanMatrixValue,
)
from jacobian.math.groups.root_systems._models import (
    CartanMatrixRequest,
    SimpleReflectionsResult,
)
from jacobian.math.groups.root_systems.operations import simple_reflections

CartanMatrix = tuple[tuple[int, ...], ...]

A1: CartanMatrix = ((2,),)
A2: CartanMatrix = ((2, -1), (-1, 2))
B2: CartanMatrix = ((2, -2), (-1, 2))
G2: CartanMatrix = ((2, -3), (-1, 2))
A2_AFFINE: CartanMatrix = ((2, -1, -1), (-1, 2, -1), (-1, -1, 2))


def _cartan(rows: CartanMatrix) -> CartanMatrixValue:
    return CartanMatrixValue.model_validate(rows)


def _entries(result: SimpleReflectionsResult, lattice: str, index: int) -> CartanMatrix:
    return cast("CartanMatrix", getattr(result, f"{lattice}_matrices")[index].entries)


def _matmul(left: CartanMatrix, right: CartanMatrix) -> CartanMatrix:
    rank = len(left)
    return tuple(
        tuple(
            sum(left[row][k] * right[k][column] for k in range(rank))
            for column in range(rank)
        )
        for row in range(rank)
    )


def _matpow(matrix: CartanMatrix, exponent: int) -> CartanMatrix:
    rank = len(matrix)
    result: CartanMatrix = tuple(
        tuple(int(row == column) for column in range(rank)) for row in range(rank)
    )
    for _ in range(exponent):
        result = _matmul(matrix, result)
    return result


def _coxeter_product(result: SimpleReflectionsResult, lattice: str) -> CartanMatrix:
    first = _entries(result, lattice, 0)
    second = _entries(result, lattice, 1)
    return _matmul(first, second)


class TestSimpleReflectionsKnownAnswers:
    def test_a1_reflection_is_negation_on_every_lattice(self) -> None:
        result = simple_reflections(A1)
        assert result.rank == 1
        for lattice in ("root", "coroot", "weight", "coweight"):
            assert _entries(result, lattice, 0) == ((-1,),)
        assert result.involution_verified is True

    def test_a2_root_matrices(self) -> None:
        result = simple_reflections(A2)
        assert _entries(result, "root", 0) == ((-1, 1), (0, 1))
        assert _entries(result, "root", 1) == ((1, 0), (1, -1))

    def test_a2_weight_matrices(self) -> None:
        result = simple_reflections(A2)
        assert _entries(result, "weight", 0) == ((-1, 0), (1, 1))
        assert _entries(result, "weight", 1) == ((1, 1), (0, -1))

    def test_b2_root_and_weight_matrices(self) -> None:
        result = simple_reflections(B2)
        assert _entries(result, "root", 0) == ((-1, 2), (0, 1))
        assert _entries(result, "weight", 0) == ((-1, 0), (1, 1))
        assert _entries(result, "coroot", 0) == ((-1, 1), (0, 1))

    def test_g2_long_root_reflection(self) -> None:
        result = simple_reflections(G2)
        assert _entries(result, "root", 0) == ((-1, 3), (0, 1))
        assert _entries(result, "weight", 0) == ((-1, 0), (1, 1))

    def test_symmetric_cartan_gives_matching_root_and_coroot(self) -> None:
        result = simple_reflections(A2)
        assert result.root_matrices == result.coroot_matrices
        assert result.weight_matrices == result.coweight_matrices

    def test_nonsymmetric_cartan_separates_root_and_coroot(self) -> None:
        result = simple_reflections(B2)
        assert result.root_matrices != result.coroot_matrices
        assert result.weight_matrices != result.coweight_matrices


class TestReflectionIdentities:
    @pytest.mark.parametrize("matrix", (A1, A2, B2, G2))
    def test_every_reflection_squares_to_identity(self, matrix: CartanMatrix) -> None:
        result = simple_reflections(matrix)
        rank = len(matrix)
        identity: CartanMatrix = tuple(
            tuple(int(row == column) for column in range(rank)) for row in range(rank)
        )
        for lattice in ("root", "coroot", "weight", "coweight"):
            for index in range(rank):
                assert _matpow(_entries(result, lattice, index), 2) == identity

    @pytest.mark.parametrize(
        ("matrix", "braid_order"),
        ((A2, 3), (B2, 4), (G2, 6)),
    )
    def test_braid_relations_on_root_matrices(
        self, matrix: CartanMatrix, braid_order: int
    ) -> None:
        result = simple_reflections(matrix)
        assert _matpow(_coxeter_product(result, "root"), braid_order) == (
            (1, 0),
            (0, 1),
        )

    @pytest.mark.parametrize(
        ("matrix", "braid_order"),
        ((A2, 3), (B2, 4), (G2, 6)),
    )
    def test_braid_relations_on_weight_matrices(
        self, matrix: CartanMatrix, braid_order: int
    ) -> None:
        result = simple_reflections(matrix)
        assert _matpow(_coxeter_product(result, "weight"), braid_order) == (
            (1, 0),
            (0, 1),
        )

    def test_reflection_matches_single_vector_kernel(self) -> None:
        from jacobian.math.groups.root_systems.operations import simple_reflection

        result = simple_reflections(A2)
        matrix = _entries(result, "root", 0)
        vector = (3, -2)
        applied = tuple(
            sum(matrix[row][column] * vector[column] for column in range(2))
            for row in range(2)
        )
        assert simple_reflection(A2, vector, 0).reflected_vector == applied


class TestReflectionAdmission:
    def test_affine_cartan_rejected(self) -> None:
        with pytest.raises(OperationDomainValidationError) as exc_info:
            simple_reflections(A2_AFFINE)
        assert exc_info.value.errors()[0]["type"] == "root_system.finite_type"

    def test_non_cartan_product_rejected(self) -> None:
        with pytest.raises(OperationDomainValidationError) as exc_info:
            simple_reflections(((2, -4), (-1, 2)))
        assert exc_info.value.errors()[0]["type"] == "root_system.off_diagonal_product"

    def test_native_and_catalog_paths_agree(self) -> None:
        from jacobian.math.groups.root_systems._tools import TOOLS

        tool = next(
            tool
            for tool in TOOLS
            if tool.operation_id == "weyl_group.simple_reflections.compute"
        )
        request = CartanMatrixRequest(matrix=_cartan(B2))
        assert tool.run(request) == simple_reflections(request.matrix)

    def test_published_example_validates(self) -> None:
        from jacobian.math.groups.root_systems._tools import TOOLS

        tool = next(
            tool
            for tool in TOOLS
            if tool.operation_id == "weyl_group.simple_reflections.compute"
        )
        from jacobian.canonical import encode_strict_json

        example = tool.examples[0]
        request = tool.request_type.model_validate_json(
            encode_strict_json(example.input), strict=True
        )
        result = tool.run(request)
        assert result.rank == 2
        assert result.involution_verified is True
