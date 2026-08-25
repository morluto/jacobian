"""Exact rational Hermitian matrices with explicit subsystem coordinates."""

from __future__ import annotations

from fractions import Fraction
from itertools import product
from math import isqrt, prod
from typing import Literal, Self

from pydantic import Field, model_validator
from pydantic_core import PydanticCustomError

from jacobian._models import StrictModel
from jacobian.canonical import format_canonical_integer
from jacobian.math.matrices.values import RationalMatrix

MAX_SUBSYSTEM_FACTORS = 4
MAX_SUBSYSTEM_DIMENSION = 16
MAX_SUBSYSTEM_LABEL_LENGTH = 64


class MatrixSubsystem(StrictModel):
    """One named finite subsystem in an ordered product basis."""

    label: str = Field(
        min_length=1,
        max_length=MAX_SUBSYSTEM_LABEL_LENGTH,
        description="Unique identifier of this subsystem within one ordered product.",
    )
    dimension: int = Field(
        ge=1,
        le=MAX_SUBSYSTEM_DIMENSION,
        description="Dimension of this finite subsystem basis.",
    )


class FactorizedHermitianMatrix(StrictModel):
    """A rational Hermitian matrix on an explicitly ordered subsystem product.

    ``factors`` describes both rows and columns.  Product coordinates use the
    lexicographic convention in which the final factor varies fastest.  Over
    ``QQ`` Hermitian means symmetric, so no complex-conjugation convention is
    implicit in this value.
    """

    matrix: RationalMatrix = Field(
        description="Square symmetric rational coordinates in the declared product basis."
    )
    factors: tuple[MatrixSubsystem, ...] = Field(
        max_length=MAX_SUBSYSTEM_FACTORS,
        description=(
            "Unique ordered subsystem factors whose dimensions multiply to the "
            "matrix order; an empty tuple denotes the scalar 1x1 context."
        ),
    )
    basis_linearization: Literal["LEXICOGRAPHIC_LAST_FACTOR_FASTEST"] = Field(
        default="LEXICOGRAPHIC_LAST_FACTOR_FASTEST",
        description=(
            "Product-basis order: lexicographic coordinates with the final "
            "factor varying fastest."
        ),
    )

    @model_validator(mode="after")
    def require_factorized_symmetric_square_matrix(self) -> Self:
        if len({factor.label for factor in self.factors}) != len(self.factors):
            raise _validation_error("invalid", "subsystem factor labels must be unique")
        dimension = prod((factor.dimension for factor in self.factors), start=1)
        if dimension > MAX_SUBSYSTEM_DIMENSION:
            raise _validation_error(
                "invalid",
                "subsystem product dimension exceeds the "
                f"{MAX_SUBSYSTEM_DIMENSION} bound",
            )
        if len(self.matrix.entries) != dimension or any(
            len(row) != dimension for row in self.matrix.entries
        ):
            raise _validation_error(
                "invalid",
                "matrix shape must equal the ordered subsystem product dimension",
            )
        if any(
            self.matrix.entries[row][column] != self.matrix.entries[column][row]
            for row in range(dimension)
            for column in range(row)
        ):
            raise _validation_error(
                "invalid", "a rational Hermitian matrix must be symmetric"
            )
        return self


def partial_trace_source_index_groups(
    matrix: FactorizedHermitianMatrix,
    traced_factor_labels: tuple[str, ...],
) -> tuple[tuple[tuple[int, int], ...], ...]:
    """Return, per reduced output cell, the folded source (row, column) indices.

    Cells follow the reduced matrix's row-major order, and each cell lists its
    terms in the kernel's fold order, so consumers of this walk share one
    contraction layout with :func:`partial_trace_entries`.
    """

    positions = {
        factor.label: position for position, factor in enumerate(matrix.factors)
    }
    traced_positions = tuple(positions[label] for label in traced_factor_labels)
    kept_positions = tuple(
        position
        for position in range(len(matrix.factors))
        if position not in traced_positions
    )
    dimensions = tuple(factor.dimension for factor in matrix.factors)
    traced_dimensions = tuple(dimensions[position] for position in traced_positions)
    kept_dimensions = tuple(dimensions[position] for position in kept_positions)

    def flat_index(coordinates: tuple[int, ...]) -> int:
        index = 0
        for coordinate, dimension in zip(coordinates, dimensions, strict=True):
            index = index * dimension + coordinate
        return index

    kept_coordinates = tuple(
        product(*(range(dimension) for dimension in kept_dimensions))
    )
    traced_coordinates = tuple(
        product(*(range(dimension) for dimension in traced_dimensions))
    )
    cells: list[tuple[tuple[int, int], ...]] = []
    for row_coordinates in kept_coordinates:
        for column_coordinates in kept_coordinates:
            cell: list[tuple[int, int]] = []
            for trace_coordinates in traced_coordinates:
                source_row = [0] * len(dimensions)
                source_column = [0] * len(dimensions)
                for position, coordinate in zip(
                    kept_positions, row_coordinates, strict=True
                ):
                    source_row[position] = coordinate
                for position, coordinate in zip(
                    kept_positions, column_coordinates, strict=True
                ):
                    source_column[position] = coordinate
                for position, coordinate in zip(
                    traced_positions, trace_coordinates, strict=True
                ):
                    source_row[position] = coordinate
                    source_column[position] = coordinate
                cell.append(
                    (
                        flat_index(tuple(source_row)),
                        flat_index(tuple(source_column)),
                    )
                )
            cells.append(tuple(cell))
    return tuple(cells)


def partial_trace_measured_entries(
    matrix: FactorizedHermitianMatrix,
    traced_factor_labels: tuple[str, ...],
) -> tuple[tuple[tuple[Fraction, ...], ...], int]:
    """Return the exact trace over named factors and its widest intermediate.

    Each reduced output cell sums its source terms with one
    cancellation-aware exact strategy: numerators are first accumulated per
    distinct denominator -- integer sums that cannot widen past one
    denominator -- and only the groups whose accumulated numerator survives
    are folded together in ascending-denominator order.  ``Fraction`` reduces
    between additions, so the recorded peak is the exact widest reduced
    numerator or denominator this walk observes, whatever cancellation delays
    or prevents.
    """

    source = tuple(
        tuple(entry.as_fraction() for entry in row) for row in matrix.matrix.entries
    )
    groups = partial_trace_source_index_groups(matrix, traced_factor_labels)
    kept_order = isqrt(len(groups))
    peak_component_digits = 1

    def charge(value: Fraction) -> None:
        nonlocal peak_component_digits
        peak_component_digits = max(
            peak_component_digits,
            len(format_canonical_integer(value.numerator)),
            len(format_canonical_integer(value.denominator)),
        )

    rows: list[tuple[Fraction, ...]] = []
    for row in range(len(groups) // kept_order):
        cell_row: list[Fraction] = []
        for group in groups[row * kept_order : (row + 1) * kept_order]:
            grouped: dict[int, int] = {}
            for row_index, column_index in group:
                term = source[row_index][column_index]
                grouped[term.denominator] = (
                    grouped.get(term.denominator, 0) + term.numerator
                )
                charge(Fraction(grouped[term.denominator], term.denominator))
            value = Fraction(0)
            for denominator in sorted(grouped):
                if grouped[denominator] == 0:
                    continue
                value += Fraction(grouped[denominator], denominator)
                charge(value)
            cell_row.append(value)
        rows.append(tuple(cell_row))
    return tuple(rows), peak_component_digits


def partial_trace_entries(
    matrix: FactorizedHermitianMatrix,
    traced_factor_labels: tuple[str, ...],
) -> tuple[tuple[Fraction, ...], ...]:
    """Return the exact trace over named factors in the declared product basis."""

    return partial_trace_measured_entries(matrix, traced_factor_labels)[0]


__all__ = [
    "MAX_SUBSYSTEM_DIMENSION",
    "MAX_SUBSYSTEM_FACTORS",
    "FactorizedHermitianMatrix",
    "MatrixSubsystem",
]


def _validation_error(reason: str, message: str) -> PydanticCustomError:
    return PydanticCustomError(f"matrix.{reason}", message)
