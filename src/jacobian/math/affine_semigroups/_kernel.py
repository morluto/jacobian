"""Bounded exact integer-kernel reconstruction for generator configurations.

The kernel is ``ker_Z(A) = V ker_Z(D)`` for a Smith decomposition ``D = U A V``
of the configuration ``A``. Because ``D`` is a positive divisibility diagonal,
``ker_Z(D)`` is spanned by the last ``nullity`` standard basis vectors, so the
corresponding columns of the unimodular ``V`` form a primitive integer basis.
A row-Hermite normal form then makes that basis canonical.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from jacobian.math.lattices._lattice_ops import saturate_lattice
from jacobian.math.lattices._models import IntegerLattice
from jacobian.math.lattices.operations import hermite_normal_form
from jacobian.math.matrices.certified_snf.operations import smith_reduce
from jacobian.math.matrices.values import IntegerMatrix


def _rows_from_flint(value: Any) -> list[list[int]]:
    return [
        [int(value[row, column]) for column in range(value.ncols())]
        for row in range(value.nrows())
    ]


def _rows_from_matrix(value: IntegerMatrix) -> list[list[int]]:
    return [[int(entry) for entry in row] for row in value.entries]


def _integer_matrix(rows: list[list[int]], *, column_count: int) -> IntegerMatrix:
    return IntegerMatrix(
        row_count=len(rows),
        column_count=column_count,
        entries=tuple(tuple(int(value) for value in row) for row in rows),
    )


def _require_zero_product(
    configuration: IntegerMatrix, basis_rows: list[list[int]]
) -> None:
    """Replay ``A B^T = 0`` for the returned relation basis ``B``."""

    for row in configuration.entries:
        for generator in basis_rows:
            product = sum(
                int(value) * coefficient
                for value, coefficient in zip(row, generator, strict=True)
            )
            if product != 0:
                raise ArithmeticError(
                    "Smith-derived relation basis does not lie in ker_Z(A)"
                )


@dataclass(frozen=True, slots=True)
class RelationLatticeData:
    """Every exact value produced by the admitted integer-kernel kernel."""

    relation_basis: IntegerMatrix
    relation_lattice: IntegerLattice
    hnf_transformation: IntegerMatrix
    rank: int
    nullity: int
    smith_invariant_factors: tuple[int, ...]
    smith_rank: int
    saturated_basis: IntegerMatrix
    saturation_inclusion_transform: IntegerMatrix
    saturation_index: int


def compute_relation_lattice_data(
    configuration: IntegerMatrix,
) -> RelationLatticeData:
    """Compute the complete canonical integer kernel lattice of ``A``."""

    rows = _rows_from_matrix(configuration)
    row_count = configuration.row_count
    column_count = configuration.column_count
    reduction = smith_reduce(
        rows, row_count=row_count, column_count=column_count
    )
    rank = reduction.rank
    nullity = column_count - rank

    # D = U A V with V unimodular. The columns of V indexed by the zero
    # diagonal positions of D form a primitive basis of ker_Z(A).
    right = reduction.right
    generators = [
        [right[row][column] for row in range(column_count)]
        for column in range(rank, column_count)
    ]
    _require_zero_product(configuration, generators)

    if generators:
        normal_form, transformation = hermite_normal_form(generators)
        basis_rows = _rows_from_flint(normal_form)
        transform_rows = _rows_from_flint(transformation)
    else:
        basis_rows = []
        transform_rows = []

    relation_basis = _integer_matrix(basis_rows, column_count=column_count)
    hnf_transformation = _integer_matrix(
        transform_rows, column_count=len(transform_rows)
    )
    relation_lattice = IntegerLattice(
        ambient_dimension=column_count,
        basis=relation_basis,
    )

    if basis_rows:
        saturated_rows, inclusion_rows, saturation_index = saturate_lattice(
            basis_rows
        )
    else:
        saturated_rows, inclusion_rows, saturation_index = [], [], 1
    saturated_basis = _integer_matrix(saturated_rows, column_count=column_count)
    saturation_inclusion_transform = _integer_matrix(
        inclusion_rows, column_count=len(inclusion_rows)
    )

    return RelationLatticeData(
        relation_basis=relation_basis,
        relation_lattice=relation_lattice,
        hnf_transformation=hnf_transformation,
        rank=rank,
        nullity=nullity,
        smith_invariant_factors=tuple(int(value) for value in reduction.invariant_factors),
        smith_rank=rank,
        saturated_basis=saturated_basis,
        saturation_inclusion_transform=saturation_inclusion_transform,
        saturation_index=int(saturation_index),
    )


__all__ = ["RelationLatticeData", "compute_relation_lattice_data"]
