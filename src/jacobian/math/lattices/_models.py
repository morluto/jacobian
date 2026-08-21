"""Bounded contracts for exact lattice operations."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import Field, model_validator

from jacobian._models import StrictModel
from jacobian.math.matrices.values import (
    MAX_MATRIX_DIMENSION,
    IntegerMatrix,
    RationalMatrix,
    require_matrix_scalar_digits,
)

_MAX_LATTICE_INPUT_SCALAR_DIGITS = 256


class HermiteNormalFormRequest(StrictModel):
    """One bounded integer matrix for row Hermite normal form."""

    matrix: IntegerMatrix

    @model_validator(mode="after")
    def require_hnf_input_budget(self) -> Self:
        require_matrix_scalar_digits(
            self.matrix.entries,
            maximum=_MAX_LATTICE_INPUT_SCALAR_DIGITS,
            label="Hermite normal form input",
        )
        return self


class HermiteNormalFormResult(StrictModel):
    """Exact row HNF and its left unimodular transformation."""

    normal_form: IntegerMatrix
    transformation: IntegerMatrix
    relation: Literal["NORMAL_FORM_EQUALS_TRANSFORMATION_TIMES_MATRIX"] = (
        "NORMAL_FORM_EQUALS_TRANSFORMATION_TIMES_MATRIX"
    )

    @model_validator(mode="after")
    def require_compatible_shapes(self) -> Self:
        rows = len(self.normal_form.entries)
        if len(self.transformation.entries) != rows:
            raise ValueError("HNF transformation must have one row per source row")
        if any(len(row) != rows for row in self.transformation.entries):
            raise ValueError("HNF transformation must be square")
        return self


class LatticeReductionRequest(StrictModel):
    """One bounded integer row basis for exact LLL reduction."""

    basis: IntegerMatrix

    @model_validator(mode="after")
    def require_lattice_input_budget(self) -> Self:
        require_matrix_scalar_digits(
            self.basis.entries,
            maximum=_MAX_LATTICE_INPUT_SCALAR_DIGITS,
            label="basis input",
        )
        return self


class LatticeReductionResult(StrictModel):
    """An exact reduced basis and its left transformation."""

    reduced_basis: IntegerMatrix
    transformation: IntegerMatrix
    rank: int = Field(ge=0, le=MAX_MATRIX_DIMENSION)
    relation: Literal["REDUCED_BASIS_EQUALS_TRANSFORMATION_TIMES_BASIS"] = (
        "REDUCED_BASIS_EQUALS_TRANSFORMATION_TIMES_BASIS"
    )
    representation: Literal["INTEGER_ROW_BASIS"] = "INTEGER_ROW_BASIS"
    gram_mode: Literal["EXACT"] = "EXACT"
    delta: Literal["0.99"] = "0.99"
    eta: Literal["0.51"] = "0.51"

    @model_validator(mode="after")
    def require_transformation_shape(self) -> Self:
        rows = len(self.reduced_basis.entries)
        if len(self.transformation.entries) != rows:
            raise ValueError("LLL transformation must have one row per basis row")
        if len(self.transformation.entries[0]) != rows:
            raise ValueError("LLL transformation must be square by basis row count")
        return self


# ---------------------------------------------------------------------------
# Issue #1739: integer-lattice structural operations
#
# An ``IntegerLattice`` represents a rank-``r`` lattice in ``ZZ^n`` by a
# full-row-rank integer basis matrix whose rows are basis vectors under the
# standard bilinear form.  The operations below are exact, deterministic, and
# bounded by ``MAX_MATRIX_DIMENSION`` and the per-scalar digit budget enforced
# by the request validators.
# ---------------------------------------------------------------------------


class IntegerLattice(StrictModel):
    """A rank-``r`` lattice in ``ZZ^n`` given by full-row-rank integer rows."""

    ambient_dimension: int = Field(ge=1, le=MAX_MATRIX_DIMENSION)
    basis: IntegerMatrix

    @model_validator(mode="after")
    def require_full_row_rank(self) -> Self:
        rows = len(self.basis.entries)
        columns = len(self.basis.entries[0]) if rows else 0
        if rows == 0:
            raise ValueError("lattice basis must contain at least one row")
        if columns != self.ambient_dimension:
            raise ValueError("basis columns must equal ambient_dimension")
        if rows > self.ambient_dimension:
            raise ValueError("lattice rank cannot exceed the ambient dimension")
        require_matrix_scalar_digits(
            self.basis.entries,
            maximum=_MAX_LATTICE_INPUT_SCALAR_DIGITS,
            label="lattice basis",
        )
        _require_full_rank_qq(self.basis.entries, rows, columns)
        return self


def _require_full_rank_qq(
    entries: tuple[tuple[str, ...], ...],
    rows: int,
    columns: int,
) -> None:
    """Raise when the integer rows are not full row rank over ``QQ``."""

    from fractions import Fraction

    matrix: list[list[Fraction]] = [
        [Fraction(entry) for entry in row] for row in entries
    ]
    rank = 0
    col = 0
    row = 0
    work = [row[:] for row in matrix]
    while row < rows and col < columns:
        pivot = None
        for r in range(row, rows):
            if work[r][col] != 0:
                pivot = r
                break
        if pivot is None:
            col += 1
            continue
        work[row], work[pivot] = work[pivot], work[row]
        for r in range(row + 1, rows):
            if work[r][col] != 0:
                factor = work[r][col] / work[row][col]
                for c in range(col, columns):
                    work[r][c] -= factor * work[row][c]
        row += 1
        col += 1
        rank += 1
    if rank != rows:
        raise ValueError("lattice basis must have full row rank over QQ")


class RankGramRequest(StrictModel):
    """One integer lattice for rank, Gram matrix, and covolume."""

    lattice: IntegerLattice

    @model_validator(mode="after")
    def require_budget(self) -> Self:
        require_matrix_scalar_digits(
            self.lattice.basis.entries,
            maximum=_MAX_LATTICE_INPUT_SCALAR_DIGITS,
            label="rank-gram input",
        )
        return self


class RankGramResult(StrictModel):
    """Exact rank, labelled Gram matrix ``G = B B^T``, and squared covolume."""

    rank: int = Field(ge=0, le=MAX_MATRIX_DIMENSION)
    ambient_dimension: int = Field(ge=1, le=MAX_MATRIX_DIMENSION)
    gram_matrix: IntegerMatrix
    squared_covolume: str
    covolume_rational: bool
    relation: Literal["GRAM_EQUALS_BASIS_TIMES_BASIS_TRANSPOSE"] = (
        "GRAM_EQUALS_BASIS_TIMES_BASIS_TRANSPOSE"
    )
    gram_mode: Literal["EXACT"] = "EXACT"


class CanonicalBasisResult(StrictModel):
    """Canonical HNF basis of a lattice and its unimodular transformation."""

    canonical_basis: IntegerMatrix
    transformation: IntegerMatrix
    rank: int = Field(ge=0, le=MAX_MATRIX_DIMENSION)
    relation: Literal["CANONICAL_BASIS_EQUALS_TRANSFORMATION_TIMES_BASIS"] = (
        "CANONICAL_BASIS_EQUALS_TRANSFORMATION_TIMES_BASIS"
    )


class DualRequest(StrictModel):
    """One integer lattice for dual-basis computation."""

    lattice: IntegerLattice

    @model_validator(mode="after")
    def require_budget(self) -> Self:
        require_matrix_scalar_digits(
            self.lattice.basis.entries,
            maximum=_MAX_LATTICE_INPUT_SCALAR_DIGITS,
            label="dual input",
        )
        return self


class DualResult(StrictModel):
    """Exact rational dual basis ``L^* = {x in span_Q(L) : <x,L> subset ZZ}``."""

    dual_basis: RationalMatrix
    dual_gram: RationalMatrix
    relation: Literal["DUAL_BASIS_BASIS_PAIRING_IS_INTEGER"] = (
        "DUAL_BASIS_BASIS_PAIRING_IS_INTEGER"
    )


class SaturationResult(StrictModel):
    """Primitive closure ``sat(L) = span_Q(L) cap ZZ^n`` and its index."""

    saturated_basis: IntegerMatrix
    inclusion_transform: IntegerMatrix
    saturation_index: int = Field(ge=1)
    relation: Literal["SATURATED_BASIS_SPANS_PRIMITIVE_CLOSURE"] = (
        "SATURATED_BASIS_SPANS_PRIMITIVE_CLOSURE"
    )


class SublatticeIndexRequest(StrictModel):
    """An inclusion of a sublattice into a parent lattice.

    ``sublattice`` and ``parent`` are each full-row-rank integer bases, and
    ``embedding`` is the integer matrix ``E`` expressing every sublattice basis
    vector as an integer linear combination of the parent basis rows, i.e.
    ``sublattice = E @ parent``.
    """

    sublattice: IntegerLattice
    parent: IntegerLattice
    embedding: IntegerMatrix

    @model_validator(mode="after")
    def require_compatible_inclusion(self) -> Self:
        if self.sublattice.ambient_dimension != self.parent.ambient_dimension:
            raise ValueError("sublattice and parent ambient dimensions must match")
        if self.embedding.entries and len(self.embedding.entries[0]) != len(
            self.parent.basis.entries
        ):
            raise ValueError("embedding columns must match parent basis rows")
        if len(self.embedding.entries) != len(self.sublattice.basis.entries):
            raise ValueError("embedding rows must match sublattice basis rows")
        require_matrix_scalar_digits(
            self.embedding.entries,
            maximum=_MAX_LATTICE_INPUT_SCALAR_DIGITS,
            label="sublattice embedding",
        )
        return self


class SublatticeIndexResult(StrictModel):
    """Finite quotient invariant factors and the sublattice index."""

    index: int = Field(ge=1)
    invariant_factors: tuple[str, ...]
    free_rank: int = Field(ge=0, le=MAX_MATRIX_DIMENSION)
    relation: Literal["QUOTIENT_IS_DIRECT_SUM_OF_CYCLIC_GROUPS"] = (
        "QUOTIENT_IS_DIRECT_SUM_OF_CYCLIC_GROUPS"
    )


class DiscriminantGroupRequest(StrictModel):
    """One nondegenerate integer lattice for discriminant-group computation."""

    lattice: IntegerLattice

    @model_validator(mode="after")
    def require_budget(self) -> Self:
        require_matrix_scalar_digits(
            self.lattice.basis.entries,
            maximum=_MAX_LATTICE_INPUT_SCALAR_DIGITS,
            label="discriminant-group input",
        )
        return self


class DiscriminantGroupResult(StrictModel):
    """Finite abelian group ``L^*/L`` and the discriminant order ``|det G|``."""

    discriminant_order: int = Field(ge=1)
    invariant_factors: tuple[str, ...]
    relation: Literal["DISCRIMINANT_GROUP_EQUALS_DUAL_MOD_LATTICE"] = (
        "DISCRIMINANT_GROUP_EQUALS_DUAL_MOD_LATTICE"
    )


class OrthogonalComplementRequest(StrictModel):
    """One integer lattice whose orthogonal complement in ``QQ^n`` is sought."""

    lattice: IntegerLattice

    @model_validator(mode="after")
    def require_budget(self) -> Self:
        require_matrix_scalar_digits(
            self.lattice.basis.entries,
            maximum=_MAX_LATTICE_INPUT_SCALAR_DIGITS,
            label="orthogonal-complement input",
        )
        return self


class OrthogonalComplementResult(StrictModel):
    """A canonical rational basis for the orthogonal complement."""

    complement_basis: RationalMatrix
    complement_rank: int = Field(ge=0, le=MAX_MATRIX_DIMENSION)
    relation: Literal["COMPLEMENT_BASIS_SPANS_ORTHOGONAL_COMPLEMENT"] = (
        "COMPLEMENT_BASIS_SPANS_ORTHOGONAL_COMPLEMENT"
    )


class DirectSumRequest(StrictModel):
    """Two integer lattices to direct-sum."""

    first: IntegerLattice
    second: IntegerLattice

    @model_validator(mode="after")
    def require_budget(self) -> Self:
        require_matrix_scalar_digits(
            self.first.basis.entries,
            maximum=_MAX_LATTICE_INPUT_SCALAR_DIGITS,
            label="direct-sum first operand",
        )
        require_matrix_scalar_digits(
            self.second.basis.entries,
            maximum=_MAX_LATTICE_INPUT_SCALAR_DIGITS,
            label="direct-sum second operand",
        )
        return self


class OrthogonalSumRequest(StrictModel):
    """Two integer lattices to orthogonally sum."""

    first: IntegerLattice
    second: IntegerLattice

    @model_validator(mode="after")
    def require_budget(self) -> Self:
        require_matrix_scalar_digits(
            self.first.basis.entries,
            maximum=_MAX_LATTICE_INPUT_SCALAR_DIGITS,
            label="orthogonal-sum first operand",
        )
        require_matrix_scalar_digits(
            self.second.basis.entries,
            maximum=_MAX_LATTICE_INPUT_SCALAR_DIGITS,
            label="orthogonal-sum second operand",
        )
        return self


class DirectSumResult(StrictModel):
    """Block-coordinate direct sum of two lattices."""

    direct_sum_basis: IntegerMatrix
    ambient_dimension: int = Field(ge=1, le=MAX_MATRIX_DIMENSION)
    relation: Literal["DIRECT_SUM_IS_BLOCK_DIAGONAL_EMBEDDING"] = (
        "DIRECT_SUM_IS_BLOCK_DIAGONAL_EMBEDDING"
    )


class OrthogonalSumResult(StrictModel):
    """Block-diagonal orthogonal sum of two lattices under the standard form."""

    orthogonal_sum_basis: IntegerMatrix
    ambient_dimension: int = Field(ge=1, le=MAX_MATRIX_DIMENSION)
    relation: Literal["ORTHOGONAL_SUM_IS_BLOCK_DIAGONAL_GRAM"] = (
        "ORTHOGONAL_SUM_IS_BLOCK_DIAGONAL_GRAM"
    )


__all__ = [
    "CanonicalBasisResult",
    "DirectSumRequest",
    "DirectSumResult",
    "DiscriminantGroupRequest",
    "DiscriminantGroupResult",
    "DualRequest",
    "DualResult",
    "HermiteNormalFormRequest",
    "HermiteNormalFormResult",
    "IntegerLattice",
    "LatticeReductionRequest",
    "LatticeReductionResult",
    "OrthogonalComplementRequest",
    "OrthogonalComplementResult",
    "OrthogonalSumRequest",
    "OrthogonalSumResult",
    "RankGramRequest",
    "RankGramResult",
    "SaturationResult",
    "SublatticeIndexRequest",
    "SublatticeIndexResult",
]
