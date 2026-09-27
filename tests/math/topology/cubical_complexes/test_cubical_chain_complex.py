"""Tests for the oriented cubical chain-complex operation."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.topology.chain_complexes.operations import (
    differential_squares_to_zero,
    homology_groups,
)
from jacobian.math.topology.chain_complexes.values import (
    CoefficientRing,
    IntegralHomologyGroupValue,
)
from jacobian.math.topology.cubical_complexes._models import (
    CubicalCell,
    CubicalCellBasis,
    CubicalChainCoefficient,
    CubicalChainComplexRequest,
    CubicalChainComplexResult,
)
from jacobian.math.topology.cubical_complexes._tools import TOOLS
from jacobian.math.topology.cubical_complexes.operations import chain_complex

_EDGE = (CubicalCell(intervals=((0, 1),)),)
_SQUARE = (CubicalCell(intervals=((0, 1), (0, 1))),)
_CUBE = (CubicalCell(intervals=((0, 1), (0, 1), (0, 1))),)


def _tool_result(
    request: CubicalChainComplexRequest,
) -> CubicalChainComplexResult:
    tool = next(
        tool
        for tool in TOOLS
        if tool.operation_id == "topology.cubical_complex.chain_complex.compute"
    )
    return tool.run(request)


class TestKnownAnswers:
    def test_single_edge(self) -> None:
        result = chain_complex(_EDGE)
        assert result.value.basis_sizes == (2, 1)
        assert result.value.differential_matrices == (((-1,), (1,)),)
        assert [basis.dimension for basis in result.cell_bases] == [0, 1]
        assert result.differential_squared_zero[0].product_rows == 0
        assert result.differential_squared_zero[0].product_columns == 1

    def test_single_square_boundary(self) -> None:
        result = chain_complex(_SQUARE)
        assert result.value.basis_sizes == (4, 4, 1)
        assert result.value.differential_matrices[0] == (
            (-1, -1, 0, 0),
            (1, 0, -1, 0),
            (0, 1, 0, -1),
            (0, 0, 1, 1),
        )
        assert result.value.differential_matrices[1] == (
            (-1,),
            (1,),
            (-1,),
            (1,),
        )

    def test_single_cube_f_vector(self) -> None:
        result = chain_complex(_CUBE)
        assert result.value.basis_sizes == (8, 12, 6, 1)
        assert tuple(len(basis.cells) for basis in result.cell_bases) == (8, 12, 6, 1)
        assert tuple(
            entry.upper_dimension for entry in result.differential_squared_zero
        ) == (1, 2, 3)

    def test_prime_field_reduces_residues(self) -> None:
        result = chain_complex(_SQUARE, CubicalChainCoefficient.PRIME_FIELD, 3)
        assert result.value.coefficient_ring is CoefficientRing.PRIME_FIELD
        assert result.value.prime == 3
        assert result.value.differential_matrices[1] == (
            (2,),
            (1,),
            (2,),
            (1,),
        )

    def test_degenerate_vertex_retains_ambient_axis(self) -> None:
        cell = CubicalCell(intervals=((4, 4), (7, 7)))
        result = chain_complex((cell,))
        assert result.value.basis_sizes == (1,)
        assert result.value.differential_matrices == ()
        assert result.differential_squared_zero == ()
        assert result.cell_bases[0].cells == (cell,)


class TestDefiningInvariants:
    @pytest.mark.parametrize(
        "cells",
        [
            _EDGE,
            _SQUARE,
            _CUBE,
            (
                CubicalCell(intervals=((0, 1), (0, 0))),
                CubicalCell(intervals=((1, 1), (0, 1))),
                CubicalCell(intervals=((1, 2), (0, 1))),
            ),
        ],
    )
    def test_d_squared_zero_replay(self, cells: tuple[CubicalCell, ...]) -> None:
        result = chain_complex(cells)
        assert differential_squares_to_zero(result.value).is_valid

    @pytest.mark.parametrize("prime", [2, 3, 5, 7])
    def test_d_squared_zero_over_prime_field(self, prime: int) -> None:
        result = chain_complex(_CUBE, CubicalChainCoefficient.PRIME_FIELD, prime)
        assert differential_squares_to_zero(result.value).is_valid

    def test_two_adjacent_squares(self) -> None:
        left = CubicalCell(intervals=((0, 1), (0, 1)))
        right = CubicalCell(intervals=((1, 2), (0, 1)))
        result = chain_complex((left, right))
        assert result.value.basis_sizes == (6, 7, 2)
        assert differential_squares_to_zero(result.value).is_valid

    def test_boundary_of_square_has_circle_homology(self) -> None:
        edges = (
            CubicalCell(intervals=((0, 1), (0, 0))),
            CubicalCell(intervals=((0, 0), (0, 1))),
            CubicalCell(intervals=((0, 1), (1, 1))),
            CubicalCell(intervals=((1, 1), (0, 1))),
        )
        result = chain_complex(edges)
        assert result.value.basis_sizes == (4, 4)
        groups = homology_groups(result.value).homology_groups
        assert all(isinstance(group, IntegralHomologyGroupValue) for group in groups)
        assert [group.free_rank for group in groups] == [1, 1]

    def test_serialization_round_trip(self) -> None:
        result = chain_complex(_SQUARE)
        restored = CubicalChainComplexResult.model_validate_json(
            result.model_dump_json()
        )
        assert restored == result
        assert chain_complex(_SQUARE) == result


class TestAdversarial:
    def test_integer_ring_rejects_prime(self) -> None:
        with pytest.raises(OperationDomainValidationError) as error:
            chain_complex(_SQUARE, CubicalChainCoefficient.INTEGER, 7)
        assert (
            error.value.errors()[0]["type"]
            == "cubical_complex.chain_coefficient_invalid"
        )

    def test_prime_field_requires_prime(self) -> None:
        with pytest.raises(OperationDomainValidationError):
            chain_complex(_SQUARE, CubicalChainCoefficient.PRIME_FIELD, None)

    def test_prime_field_rejects_composite_modulus(self) -> None:
        with pytest.raises(OperationDomainValidationError):
            chain_complex(_SQUARE, CubicalChainCoefficient.PRIME_FIELD, 4)

    def test_inconsistent_ambient_axis_rejected(self) -> None:
        bad = (
            CubicalCell(intervals=((0, 1),)),
            CubicalCell(intervals=((0, 1), (0, 1))),
        )
        with pytest.raises(OperationDomainValidationError):
            chain_complex(bad)

    def test_tampered_result_is_rejected(self) -> None:
        result = chain_complex(_SQUARE)
        with pytest.raises(ValidationError):
            CubicalChainComplexResult.model_validate(
                result.model_dump() | {"coefficient_ring": "GF_p", "prime": 2}
            )
        with pytest.raises(ValidationError):
            CubicalChainComplexResult.model_validate(
                result.model_dump()
                | {
                    "cell_bases": [
                        CubicalCellBasis(dimension=1, cells=_EDGE).model_dump()
                    ]
                }
            )


class TestResourceAdmission:
    def test_full_ten_cube_exceeds_chain_group_budget(self) -> None:
        with pytest.raises(OperationResourceAdmissionError) as error:
            chain_complex((CubicalCell(intervals=((0, 1),) * 10),))
        assert error.value.errors()[0]["type"] == "cubical_complex.chain_group_budget"

    def test_group_boundary_is_admitted(self) -> None:
        result = chain_complex(_CUBE, CubicalChainCoefficient.PRIME_FIELD, 2)
        assert all(len(basis.cells) <= 64 for basis in result.cell_bases)


class TestCatalogParity:
    def test_native_matches_catalog_entry(self) -> None:
        request = CubicalChainComplexRequest(
            cells=_SQUARE,
            coefficient_ring=CubicalChainCoefficient.PRIME_FIELD,
            prime=5,
        )
        assert _tool_result(request) == chain_complex(
            request.cells, request.coefficient_ring, request.prime
        )

    def test_examples_execute(self) -> None:
        tool = next(
            tool
            for tool in TOOLS
            if tool.operation_id == "topology.cubical_complex.chain_complex.compute"
        )
        for example in tool.examples:
            request = tool.request_type.model_validate(example.input)
            assert tool.run(request).value.basis_sizes
