"""Finite Cartan root/coroot/weight datum contract."""

from __future__ import annotations

from fractions import Fraction

import pytest

from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.groups.root_systems._models import CartanMatrix
from jacobian.math.groups.root_systems.operations import (
    cartan_datum,
    simple_reflections,
)
from jacobian.math.matrices.values import IntegerMatrix


def test_nonsymmetric_g2_basis_transport_and_symmetrizer() -> None:
    datum = cartan_datum(CartanMatrix.model_validate(((2, -3), (-1, 2))))
    assert tuple(value.as_fraction() for value in datum.symmetrizer) == (
        Fraction(1),
        Fraction(3),
    )
    assert datum.root_to_weight.entries == ((2, -3), (-1, 2))
    assert datum.coroot_to_coweight.entries == ((2, -1), (-3, 2))


def test_datum_reflection_maps_are_involutions_for_b2_and_c2() -> None:
    for rows in (((2, -2), (-1, 2)), ((2, -1), (-2, 2))):
        reflections = simple_reflections(CartanMatrix.model_validate(rows))
        for family_name in (
            "root_matrices",
            "coroot_matrices",
            "weight_matrices",
            "coweight_matrices",
        ):
            for matrix in getattr(reflections, family_name):
                product = tuple(
                    tuple(
                        sum(
                            matrix.entries[i][k] * matrix.entries[k][j]
                            for k in range(2)
                        )
                        for j in range(2)
                    )
                    for i in range(2)
                )
                assert product == ((1, 0), (0, 1))


def test_native_malformed_cartan_is_an_owner_error() -> None:
    with pytest.raises(OperationDomainValidationError) as raised:
        cartan_datum(CartanMatrix.model_construct(matrix=None, simple_root_axis=()))
    assert raised.value.errors()[0]["type"] == "root_system.invalid_cartan_carrier"


def test_all_cartan_consumers_reject_inconsistent_nested_matrix_shape() -> None:
    nested = IntegerMatrix.model_construct(row_count=2, column_count=2, entries=((2,),))
    forged = CartanMatrix.model_construct(matrix=nested, simple_root_axis=(0,))
    for operation in (cartan_datum, simple_reflections):
        with pytest.raises(OperationDomainValidationError) as raised:
            operation(forged)
        assert raised.value.errors()[0]["type"] == (
            "root_system.invalid_cartan_carrier"
        )


def test_datum_serializes_without_reconstructing_cartan_parent() -> None:
    datum = cartan_datum(CartanMatrix.model_validate(((2, -3), (-1, 2))))
    restored = type(datum).model_validate_json(datum.model_dump_json())
    assert restored == datum
    assert restored.cartan_matrix.entries == ((2, -3), (-1, 2))
