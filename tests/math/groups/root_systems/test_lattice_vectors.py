"""Exact typed vectors in the four lattices of finite Cartan data."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.groups.root_systems._models import (
    CartanMatrix,
    CorootLatticeVector,
    CoweightLatticeVector,
    FiniteCartanDatum,
    RootLatticeVector,
    RootToWeightLatticeRequest,
    WeightLatticeVector,
)
from jacobian.math.groups.root_systems.operations import (
    coroot_lattice_vector,
    coroot_to_coweight_lattice,
    coweight_lattice_vector,
    root_lattice_vector,
    root_to_weight_lattice,
    weight_lattice_vector,
)
from jacobian.math.matrices.values import IntegerMatrix


def _cartan(rows: tuple[tuple[int, ...], ...]) -> CartanMatrix:
    return CartanMatrix.model_validate(rows)


def test_a2_vectors_and_canonical_inclusion_q_into_p() -> None:
    matrix = _cartan(((2, -1), (-1, 2)))
    roots = root_lattice_vector(matrix, (3, -2))
    coroots = coroot_lattice_vector(matrix, (-1, 4))

    assert roots.coordinates == (3, -2)
    assert coroots.coordinates == (-1, 4)
    assert isinstance(roots, RootLatticeVector)
    assert isinstance(coroots, CorootLatticeVector)

    weights = root_to_weight_lattice(roots)
    coweights = coroot_to_coweight_lattice(coroots)
    assert isinstance(weights, WeightLatticeVector)
    assert isinstance(coweights, CoweightLatticeVector)
    assert weights.coordinates == (8, -7)
    assert coweights.coordinates == (-6, 9)
    assert weights.datum == roots.datum
    assert coweights.datum == coroots.datum


@pytest.mark.parametrize(
    ("rows", "root_image", "coroot_image"),
    [
        (((2, -2), (-1, 2)), (8, -5), (7, -8)),  # B2 orientation
        (((2, -1), (-2, 2)), (7, -8), (8, -5)),  # C2 orientation
        (((2, -3), (-1, 2)), (9, -5), (7, -11)),  # G2 orientation
    ],
)
def test_nonsymmetric_root_and_coroot_inclusions_keep_orientation(
    rows: tuple[tuple[int, ...], ...],
    root_image: tuple[int, int],
    coroot_image: tuple[int, int],
) -> None:
    matrix = _cartan(rows)
    root = root_lattice_vector(matrix, (3, -1))
    coroot = coroot_lattice_vector(matrix, (3, -1))

    assert root_to_weight_lattice(root).coordinates == root_image
    assert coroot_to_coweight_lattice(coroot).coordinates == coroot_image


def test_reducible_datum_maps_each_component_without_reordering() -> None:
    # A1 + A2, with the direct-sum ordering retained by the datum.
    matrix = _cartan(
        (
            (2, 0, 0),
            (0, 2, -1),
            (0, -1, 2),
        )
    )
    root = root_lattice_vector(matrix, (4, 3, -2))
    coroot = coroot_lattice_vector(matrix, (-2, 1, 5))
    assert root_to_weight_lattice(root).coordinates == (8, 8, -7)
    assert coroot_to_coweight_lattice(coroot).coordinates == (-4, -3, 9)


@pytest.mark.parametrize(
    ("constructor", "carrier"),
    [
        (root_lattice_vector, RootLatticeVector),
        (coroot_lattice_vector, CorootLatticeVector),
        (weight_lattice_vector, WeightLatticeVector),
        (coweight_lattice_vector, CoweightLatticeVector),
    ],
)
def test_lattice_vectors_roundtrip_with_their_datum_parent(
    constructor, carrier
) -> None:
    vector = constructor(_cartan(((2, -1), (-1, 2))), (2, -3))
    restored = carrier.model_validate_json(vector.model_dump_json())
    assert restored == vector
    assert restored.datum.cartan_matrix.entries == ((2, -1), (-1, 2))
    assert restored.coordinates == (2, -3)


def test_coordinate_axis_must_match_rank_and_coordinates_are_bounded() -> None:
    matrix = _cartan(((2, -1), (-1, 2)))
    with pytest.raises((ValidationError, OperationDomainValidationError)):
        root_lattice_vector(matrix, (1,))
    with pytest.raises((ValidationError, OperationDomainValidationError)):
        root_lattice_vector(matrix, (1, 2, 3))
    with pytest.raises((ValidationError, OperationDomainValidationError)):
        root_lattice_vector(matrix, (1 << 128, 0))


def test_rank_above_admitted_cartan_bound_is_rejected() -> None:
    rank = 9
    rows = tuple(tuple(2 if i == j else 0 for j in range(rank)) for i in range(rank))
    with pytest.raises((ValidationError, OperationDomainValidationError)):
        root_lattice_vector(
            CartanMatrix.model_validate(rows),
            (0,) * rank,
        )


def test_map_rejects_forged_datum_and_malformed_nested_cartan() -> None:
    matrix = _cartan(((2, -1), (-1, 2)))
    honest = root_lattice_vector(matrix, (1, 0))
    malformed_matrix = IntegerMatrix.model_construct(
        row_count=2,
        column_count=2,
        entries=((2,),),
    )
    malformed_cartan = CartanMatrix.model_construct(
        matrix=malformed_matrix,
        simple_root_axis=(0,),
    )
    malformed_datum = FiniteCartanDatum.model_construct(
        cartan_matrix=malformed_cartan,
        symmetrizer=honest.datum.symmetrizer,
        root_to_weight=honest.datum.root_to_weight,
        coroot_to_coweight=honest.datum.coroot_to_coweight,
    )
    forged = RootLatticeVector.model_construct(
        datum=malformed_datum,
        coordinates=(1, 0),
    )
    with pytest.raises(OperationDomainValidationError):
        root_to_weight_lattice(forged)

    forged_datum = FiniteCartanDatum.model_construct(
        cartan_matrix=matrix,
        symmetrizer=honest.datum.symmetrizer,
        root_to_weight=IntegerMatrix.model_validate(
            {
                "row_count": 2,
                "column_count": 2,
                "entries": ((1, 0), (0, 1)),
            }
        ),
        coroot_to_coweight=honest.datum.coroot_to_coweight,
    )
    forged = RootLatticeVector.model_construct(
        datum=forged_datum,
        coordinates=(1, 0),
    )
    with pytest.raises(OperationDomainValidationError):
        root_to_weight_lattice(forged)


def test_map_request_rejects_a_vector_from_a_different_lattice() -> None:
    matrix = _cartan(((2, -1), (-1, 2)))
    coroot = coroot_lattice_vector(matrix, (1, 0))
    with pytest.raises(OperationDomainValidationError):
        root_to_weight_lattice(coroot)  # type: ignore[arg-type]
    with pytest.raises(ValidationError):
        RootToWeightLatticeRequest(vector=coroot)  # type: ignore[arg-type]
