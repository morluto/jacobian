"""Typed wire contracts for cubical complex operations."""

from __future__ import annotations

from enum import StrEnum
from typing import Any, Literal, Self

from pydantic import Field, model_validator
from pydantic_core import PydanticCustomError

from jacobian._models import StrictModel
from jacobian.math.topology.chain_complexes.values import ChainComplexValue

MAX_DIM = 10
MAX_CELLS = 5000
MAX_FACE_CELLS = 3**MAX_DIM
"""Enough distinct faces for a full cube at every supported ambient dimension."""

MAX_CUBICAL_CHAIN_GROUP = 64
MAX_CUBICAL_CHAIN_CELLS = 16384
MAX_CUBICAL_PRIME = 1000003
MAX_TRIANGULATION_POINTS = 4096
MAX_TRIANGULATION_SIMPLICES = 16384
MAX_TRIANGULATION_CELL_SIMPLICES = 720


def _validation_error(reason: str, message: str) -> PydanticCustomError:
    """Build a stable validation error owned by cubical-complex contracts."""

    return PydanticCustomError(f"cubical_complex.{reason}", message)


class CubicalCell(StrictModel):
    """An elementary cube: a tuple of intervals [a_i, b_i] on integer lattice."""

    intervals: tuple[tuple[int, int], ...] = Field(min_length=1, max_length=MAX_DIM)

    @model_validator(mode="after")
    def require_valid_intervals(self) -> Self:
        for a, b in self.intervals:
            if a > b:
                raise _validation_error(
                    "interval_order",
                    "each interval must have a <= b (interval is [a, b])",
                )
            if b - a > 1:
                raise _validation_error(
                    "interval_length",
                    "each interval must have length 0 or 1 (b <= a + 1)",
                )
        return self

    @property
    def dimension(self) -> int:
        return sum(1 for a, b in self.intervals if b > a)


class CubicalComplexRequest(StrictModel):
    """A finite cubical complex: a set of elementary cubes."""

    cells: tuple[CubicalCell, ...] = Field(min_length=1, max_length=MAX_CELLS)


class CubicalComplex(StrictModel):
    """Canonical cubical complex with an explicit ambient coordinate axis.

    ``cells`` is a sorted family of distinct cells.  Operations establish that
    it is face closed before constructing this value; decoding checks only the
    bounded cell and axis representation.
    """

    ambient_dimension: int = Field(ge=1, le=MAX_DIM)
    cells: tuple[CubicalCell, ...] = Field(min_length=1, max_length=MAX_FACE_CELLS)

    @model_validator(mode="after")
    def require_structural_cells(self) -> Self:
        if any(len(cell.intervals) != self.ambient_dimension for cell in self.cells):
            raise _validation_error(
                "ambient_dimension_mismatch",
                "every cell must use the declared ambient coordinate axis",
            )
        if tuple(sorted(self.cells, key=lambda cell: cell.intervals)) != self.cells:
            raise _validation_error(
                "cells_not_canonical", "cells must be sorted canonically"
            )
        if len(set(self.cells)) != len(self.cells):
            raise _validation_error("duplicate_cells", "cells must be distinct")
        return self


class FVector(StrictModel):
    """An f-vector whose entries are indexed by explicit cell dimension."""

    dimension_axis: tuple[int, ...] = Field(min_length=1, max_length=MAX_DIM + 1)
    counts: tuple[int, ...] = Field(min_length=1, max_length=MAX_DIM + 1)

    @model_validator(mode="after")
    def require_structural_axis(self) -> Self:
        if self.dimension_axis != tuple(range(len(self.dimension_axis))):
            raise _validation_error(
                "dimension_axis_not_canonical",
                "dimension axis must enumerate dimensions from zero",
            )
        if len(self.counts) != len(self.dimension_axis) or any(
            count < 0 for count in self.counts
        ):
            raise _validation_error(
                "f_vector_shape", "f-vector counts must match its dimension axis"
            )
        return self


class FVectorResult(StrictModel):
    """The f-vector and Euler characteristic bound to a cubical complex."""

    complex: CubicalComplex
    source_cells: tuple[CubicalCell, ...] = Field(min_length=1, max_length=MAX_CELLS)
    f_vector: FVector
    euler_characteristic: int


class FaceClosureRequest(StrictModel):
    """Compute the full face closure of a set of cells."""

    cells: tuple[CubicalCell, ...] = Field(min_length=1, max_length=MAX_CELLS)


class FaceClosureResult(StrictModel):
    """A face-closed complex and its dimensional cell-count summary."""

    complex: CubicalComplex
    source_cells: tuple[CubicalCell, ...] = Field(min_length=1, max_length=MAX_CELLS)
    original_cells: int
    total_cells: int
    cells_by_dimension: FVector


class CubicalChainCoefficient(StrEnum):
    """Exact coefficient rings supported by cubical chain complexes."""

    INTEGER = "ZZ"
    PRIME_FIELD = "GF_p"


class CubicalChainComplexRequest(StrictModel):
    """A finite elementary cubical complex with exact chain coefficients.

    The kernel closes the supplied cells under faces once during admission;
    the request itself need only use one ambient coordinate axis.
    """

    cells: tuple[CubicalCell, ...] = Field(min_length=1, max_length=MAX_CELLS)
    coefficient_ring: CubicalChainCoefficient = CubicalChainCoefficient.INTEGER
    prime: int | None = Field(default=None, ge=2, le=MAX_CUBICAL_PRIME)


class CubicalCellBasis(StrictModel):
    """The canonically ordered cells spanning one cubical chain group."""

    dimension: int = Field(ge=0, le=MAX_DIM)
    cells: tuple[CubicalCell, ...] = Field(
        min_length=1,
        max_length=MAX_CUBICAL_CHAIN_GROUP,
        description=(
            "Canonically ordered cells of one dimension; the order is the "
            "implicit basis of the dense boundary matrices."
        ),
    )


class CubicalSquareLedgerEntry(StrictModel):
    """One replayed d^2 = 0 product between adjacent cubical degrees."""

    upper_dimension: int = Field(ge=1, le=MAX_DIM)
    product_rows: int = Field(ge=0)
    product_columns: int = Field(ge=0)
    nonzero_entries: Literal[0] = 0


class CubicalChainComplexResult(StrictModel):
    """The based cubical chain complex with its replayed square-zero ledger."""

    complex: CubicalComplex
    coefficient_ring: CubicalChainCoefficient
    prime: int | None = Field(default=None, ge=2, le=MAX_CUBICAL_PRIME)
    cell_bases: tuple[CubicalCellBasis, ...] = Field(min_length=1)
    value: ChainComplexValue
    differential_squared_zero: tuple[CubicalSquareLedgerEntry, ...] = ()

    @model_validator(mode="after")
    def require_structural_chain_contract(self) -> Self:
        dimensions = tuple(basis.dimension for basis in self.cell_bases)
        if dimensions != tuple(range(len(self.cell_bases))):
            raise _validation_error(
                "cell_basis_coverage_invalid",
                "cell bases must cover contiguous dimensions from zero",
            )
        expected_sizes = tuple(len(basis.cells) for basis in self.cell_bases)
        if self.value.basis_sizes != expected_sizes:
            raise _validation_error(
                "chain_basis_binding_invalid",
                "canonical chain basis sizes must match the cell bases",
            )
        if self.value.degree_min != 0:
            raise _validation_error(
                "chain_degree_binding_invalid",
                "cubical chain complexes are concentrated in degrees 0..d",
            )
        expected_ring = (
            "ZZ" if self.coefficient_ring is CubicalChainCoefficient.INTEGER else "GF_p"
        )
        if (
            self.value.coefficient_ring.value != expected_ring
            or self.value.prime != self.prime
        ):
            raise _validation_error(
                "chain_coefficient_binding_invalid",
                "canonical chain coefficients must match the requested ring",
            )
        if tuple(entry.upper_dimension for entry in self.differential_squared_zero) != (
            tuple(range(1, len(self.cell_bases)))
        ):
            raise _validation_error(
                "square_ledger_incomplete",
                "the square-zero ledger must cover every adjacent degree pair",
            )
        return self

    @classmethod
    def _from_kernel(cls, **values: Any) -> Self:
        return cls.model_construct(**values)


__all__ = [
    "MAX_CELLS",
    "MAX_CUBICAL_CHAIN_CELLS",
    "MAX_CUBICAL_CHAIN_GROUP",
    "MAX_CUBICAL_PRIME",
    "MAX_DIM",
    "MAX_TRIANGULATION_CELL_SIMPLICES",
    "MAX_TRIANGULATION_POINTS",
    "MAX_TRIANGULATION_SIMPLICES",
    "CubicalCell",
    "CubicalCellBasis",
    "CubicalChainCoefficient",
    "CubicalChainComplexRequest",
    "CubicalChainComplexResult",
    "CubicalComplex",
    "CubicalComplexRequest",
    "CubicalSquareLedgerEntry",
    "FVector",
    "FVectorResult",
    "FaceClosureRequest",
    "FaceClosureResult",
]
