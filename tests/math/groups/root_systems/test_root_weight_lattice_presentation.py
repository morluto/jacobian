from __future__ import annotations

import json

import pytest

from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.groups.root_systems._models import CartanMatrix, FiniteCartanDatum
from jacobian.math.groups.root_systems.lattice_presentations import (
    RootWeightLatticePresentation,
    root_weight_lattice_presentation,
)
from jacobian.math.groups.root_systems.operations import cartan_datum
from jacobian.math.lattices.operations import compute_sublattice_index
from jacobian.math.matrices.values import IntegerMatrix


@pytest.mark.parametrize(
    ("matrix", "expected_index", "expected_factors"),
    (
        (((2,),), 2, (2,)),
        (((2, -1), (-1, 2)), 3, (3,)),
        (((2, -2), (-1, 2)), 2, (2,)),
        (((2, -3), (-1, 2)), 1, ()),
        (
            ((2, 0, 0), (0, 2, -1), (0, -1, 2)),
            6,
            (6,),
        ),
    ),
)
def test_root_weight_lattices_compose_with_exact_sublattice_quotient(
    matrix: tuple[tuple[int, ...], ...],
    expected_index: int,
    expected_factors: tuple[int, ...],
) -> None:
    datum = cartan_datum(CartanMatrix.model_validate(matrix))
    restored_datum = FiniteCartanDatum.model_validate_json(datum.model_dump_json())

    presentation = root_weight_lattice_presentation(restored_datum)
    quotient = compute_sublattice_index(
        presentation.root_lattice,
        presentation.weight_lattice,
        presentation.root_to_weight_embedding,
    )

    rank = len(matrix)
    identity = tuple(
        tuple(int(row == column) for column in range(rank)) for row in range(rank)
    )
    expected_embedding = tuple(
        tuple(matrix[column][row] for column in range(rank)) for row in range(rank)
    )
    assert presentation.weight_lattice.basis.entries == identity
    assert presentation.root_lattice.basis.entries == expected_embedding
    assert presentation.root_to_weight_embedding.entries == expected_embedding
    assert quotient.index == expected_index
    assert quotient.invariant_factors == expected_factors
    assert quotient.free_rank == 0

    restored = RootWeightLatticePresentation.model_validate_json(
        presentation.model_dump_json()
    )
    assert restored == presentation
    assert (
        compute_sublattice_index(
            restored.root_lattice,
            restored.weight_lattice,
            restored.root_to_weight_embedding,
        )
        == quotient
    )


def test_b2_rows_encode_cartan_columns_in_fundamental_weight_basis() -> None:
    presentation = root_weight_lattice_presentation(
        cartan_datum(CartanMatrix.model_validate(((2, -2), (-1, 2))))
    )

    assert presentation.root_to_weight_embedding.entries == ((2, -1), (-2, 2))


def test_rejects_noncanonical_datum_maps_at_operation_boundary() -> None:
    datum = cartan_datum(CartanMatrix.model_validate(((2, -1), (-1, 2))))
    forged = datum.model_copy(
        update={
            "root_to_weight": IntegerMatrix(
                row_count=2,
                column_count=2,
                entries=((1, 0), (0, 1)),
            )
        }
    )

    with pytest.raises(OperationDomainValidationError) as error:
        root_weight_lattice_presentation(forged)
    assert error.value.errors()[0]["type"] == (
        "root_system.lattice_presentation.datum_mismatch"
    )


def test_result_validator_rejects_a_tampered_embedding() -> None:
    presentation = root_weight_lattice_presentation(
        cartan_datum(CartanMatrix.model_validate(((2, -1), (-1, 2))))
    )
    payload = presentation.model_dump(mode="json")
    payload["root_to_weight_embedding"]["entries"] = [["1", "0"], ["0", "1"]]

    with pytest.raises(ValueError):
        RootWeightLatticePresentation.model_validate(payload)


def test_result_validator_revalidates_the_canonical_datum() -> None:
    presentation = root_weight_lattice_presentation(
        cartan_datum(CartanMatrix.model_validate(((2, -1), (-1, 2))))
    )
    payload = presentation.model_dump(mode="json")
    identity = [["1", "0"], ["0", "1"]]
    payload["datum"]["root_to_weight"]["entries"] = identity
    payload["datum"]["coroot_to_coweight"]["entries"] = identity
    payload["root_lattice"]["basis"]["entries"] = identity
    payload["root_to_weight_embedding"]["entries"] = identity

    with pytest.raises(ValueError) as error:
        RootWeightLatticePresentation.model_validate_json(json.dumps(payload))
    assert error.value.errors()[0]["type"] == (
        "root_system.lattice_presentation_datum"
    )
