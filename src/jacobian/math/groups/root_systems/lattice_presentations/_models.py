"""Typed root and weight lattice presentations on a finite Cartan datum."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import model_validator

from jacobian._models import StrictModel
from jacobian.math.groups.root_systems._models import (
    FiniteCartanDatum,
    _validation_error,
)
from jacobian.math.lattices._models import IntegerLattice
from jacobian.math.matrices.values import IntegerMatrix


def _identity_matrix(rank: int) -> tuple[tuple[int, ...], ...]:
    return tuple(
        tuple(int(row == column) for column in range(rank)) for row in range(rank)
    )


def _transpose(matrix: tuple[tuple[int, ...], ...]) -> tuple[tuple[int, ...], ...]:
    return tuple(tuple(row[column] for row in matrix) for column in range(len(matrix)))


class RootWeightLatticePresentation(StrictModel):
    """The simple-root lattice embedded in the fundamental-weight lattice.

    Both lattices use the fundamental-weight coordinate space as their common
    ambient integer lattice. The rows of ``root_lattice.basis`` are the simple
    roots in those coordinates, and ``root_to_weight_embedding`` expresses
    them in the standard basis of ``weight_lattice``.
    """

    datum: FiniteCartanDatum
    root_lattice: IntegerLattice
    weight_lattice: IntegerLattice
    root_to_weight_embedding: IntegerMatrix
    relation: Literal["ROOT_LATTICE_IS_SUBLATTICE_OF_WEIGHT_LATTICE"] = (
        "ROOT_LATTICE_IS_SUBLATTICE_OF_WEIGHT_LATTICE"
    )

    @model_validator(mode="after")
    def require_canonical_embedding(self) -> Self:
        try:
            from jacobian.math.groups.root_systems.operations import cartan_datum

            canonical_datum = cartan_datum(self.datum.cartan_matrix)
            datum_matches = (
                self.datum.cartan_matrix.simple_root_axis
                == canonical_datum.cartan_matrix.simple_root_axis
                and self.datum.symmetrizer == canonical_datum.symmetrizer
                and self.datum.root_to_weight.entries
                == canonical_datum.root_to_weight.entries
                and self.datum.coroot_to_coweight.entries
                == canonical_datum.coroot_to_coweight.entries
            )
        except (AttributeError, TypeError, ValueError) as error:
            raise _validation_error(
                "lattice_presentation_datum",
                "presentation datum must be a canonical finite Cartan datum",
            ) from error
        if not datum_matches:
            raise _validation_error(
                "lattice_presentation_datum",
                "presentation datum must retain the canonical Cartan basis maps",
            )
        rank = len(canonical_datum.cartan_matrix)
        identity = _identity_matrix(rank)
        expected_embedding = _transpose(canonical_datum.root_to_weight.entries)
        if (
            self.weight_lattice.ambient_dimension != rank
            or self.weight_lattice.basis.entries != identity
            or self.root_lattice.ambient_dimension != rank
            or self.root_lattice.basis.entries != expected_embedding
            or self.root_to_weight_embedding.row_count != rank
            or self.root_to_weight_embedding.column_count != rank
            or self.root_to_weight_embedding.entries != expected_embedding
        ):
            raise _validation_error(
                "lattice_presentation_shape",
                "root and weight lattices must use the canonical Cartan inclusion",
            )
        reconstructed = tuple(
            tuple(
                sum(
                    self.root_to_weight_embedding.entries[row][inner]
                    * self.weight_lattice.basis.entries[inner][column]
                    for inner in range(rank)
                )
                for column in range(rank)
            )
            for row in range(rank)
        )
        if reconstructed != self.root_lattice.basis.entries:
            raise _validation_error(
                "lattice_presentation_relation",
                "root lattice basis must equal embedding times parent basis",
            )
        return self
