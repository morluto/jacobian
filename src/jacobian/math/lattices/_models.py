"""Bounded contracts for exact lattice operations."""

from __future__ import annotations

from collections.abc import Callable
from typing import Annotated, Literal, Self

from pydantic import Field, WithJsonSchema, model_validator
from pydantic_core import PydanticCustomError

from jacobian._models import StrictModel
from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.matrices.values import (
    MAX_INTEGER_MATRIX_ORDER,
    MAX_MATRIX_DIMENSION,
    IntegerMatrix,
    RationalMatrix,
    RationalVectorSpaceBasis,
    integer_matrix_axis_schema,
    require_matrix_scalar_digits,
)

_MAX_LATTICE_INPUT_SCALAR_DIGITS = 256


def _validation_error(reason: str, message: str) -> PydanticCustomError:
    """Build a stable validation error owned by lattice models."""

    return PydanticCustomError(f"lattice.{reason}", message)


def _require_lattice_matrix_envelope(matrix: IntegerMatrix, *, label: str) -> None:
    """Admit one integer matrix into the lattice 32-axis computation envelope."""

    rows = len(matrix.entries)
    columns = matrix.column_count
    if (
        not rows
        or not columns
        or rows > MAX_MATRIX_DIMENSION
        or columns > MAX_MATRIX_DIMENSION
    ):
        raise _validation_error(
            "budget_exceeded",
            f"{label} dimensions are limited to {MAX_MATRIX_DIMENSION} rows and columns",
        )
    require_matrix_scalar_digits(
        matrix.entries,
        maximum=_MAX_LATTICE_INPUT_SCALAR_DIGITS,
        label=label,
    )


def _run_admission(admission: Callable[[], None], *, location: tuple[str, ...]) -> None:
    """Expose lattice envelope rejection through the domain API."""

    try:
        admission()
    except PydanticCustomError as exc:
        raise OperationDomainValidationError(
            location=location, code=exc.type, message=exc.message()
        ) from exc


class HermiteNormalFormRequest(StrictModel):
    """An integer matrix for row HNF, with H = U A and unimodular U.

    HNF-specific admission bounds the full augmented matrix [A | I],
    including coefficient growth and the square transformation.
    """

    matrix: IntegerMatrix = Field(
        description=(
            f"Nonempty integer matrix with axes at most {MAX_INTEGER_MATRIX_ORDER} "
            "and at most 256 digits per scalar. HNF admission bounds the "
            "fraction-free elimination, modular reduction and the square "
            "transformation using row-norm minor bounds. At most 250,000,000 "
            "scalar-operation units, 262,144 bits per intermediate and "
            "12,000,000,000 retained coefficient bits are admitted."
        ),
    )

    @model_validator(mode="after")
    def require_scalar_envelope(self) -> Self:
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
            raise _validation_error(
                "hnf_transformation_rows",
                "HNF transformation must have one row per source row",
            )
        if any(len(row) != rows for row in self.transformation.entries):
            raise _validation_error(
                "hnf_transformation_square", "HNF transformation must be square"
            )
        return self


class LatticeReductionRequest(StrictModel):
    """One bounded integer row basis for exact LLL reduction.

    Row and column counts are at most ``MAX_MATRIX_DIMENSION``.
    """

    basis: Annotated[
        IntegerMatrix,
        WithJsonSchema(integer_matrix_axis_schema(MAX_MATRIX_DIMENSION)),
    ] = Field(
        description=(
            "Integer row basis whose row and column counts are at most "
            f"{MAX_MATRIX_DIMENSION}."
        ),
    )

    @model_validator(mode="after")
    def require_admitted_envelope(self) -> Self:
        _require_lattice_matrix_envelope(self.basis, label="basis input")
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
            raise _validation_error(
                "lll_transformation_rows",
                "LLL transformation must have one row per basis row",
            )
        if self.transformation.column_count != rows:
            raise _validation_error(
                "lll_transformation_square",
                "LLL transformation must be square by basis row count",
            )
        return self


# ---------------------------------------------------------------------------
# Integer-lattice structural operations
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
        columns = self.basis.column_count
        if rows == 0:
            raise _validation_error(
                "basis_empty", "lattice basis must contain at least one row"
            )
        if columns != self.ambient_dimension:
            raise _validation_error(
                "basis_columns_mismatch", "basis columns must equal ambient_dimension"
            )
        if rows > self.ambient_dimension:
            raise _validation_error(
                "rank_exceeds_ambient",
                "lattice rank cannot exceed the ambient dimension",
            )
        require_matrix_scalar_digits(
            self.basis.entries,
            maximum=_MAX_LATTICE_INPUT_SCALAR_DIGITS,
            label="lattice basis",
        )
        return self


class RankGramRequest(StrictModel):
    """One integer lattice for rank, Gram matrix, and covolume."""

    lattice: IntegerLattice


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

    @model_validator(mode="after")
    def require_inclusion_shape(self) -> Self:
        rank = len(self.saturated_basis.entries)
        if len(self.inclusion_transform.entries) != rank or any(
            len(row) != rank for row in self.inclusion_transform.entries
        ):
            raise _validation_error(
                "saturation_inclusion_shape",
                "saturation inclusion must be square by lattice rank",
            )
        return self


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
            raise _validation_error(
                "ambient_dimensions_mismatch",
                "sublattice and parent ambient dimensions must match",
            )
        if self.embedding.entries and len(self.embedding.entries[0]) != len(
            self.parent.basis.entries
        ):
            raise _validation_error(
                "embedding_columns_mismatch",
                "embedding columns must match parent basis rows",
            )
        if len(self.embedding.entries) != len(self.sublattice.basis.entries):
            raise _validation_error(
                "embedding_rows_mismatch",
                "embedding rows must match sublattice basis rows",
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


class OrthogonalComplementResult(StrictModel):
    """A canonical rational basis for the orthogonal complement."""

    complement_basis: RationalVectorSpaceBasis
    complement_rank: int = Field(ge=0, le=MAX_MATRIX_DIMENSION)
    relation: Literal["COMPLEMENT_BASIS_SPANS_ORTHOGONAL_COMPLEMENT"] = (
        "COMPLEMENT_BASIS_SPANS_ORTHOGONAL_COMPLEMENT"
    )

    @model_validator(mode="after")
    def require_rank_shape(self) -> Self:
        if self.complement_rank != len(self.complement_basis.vectors):
            raise _validation_error(
                "orthogonal_complement_rank",
                "complement rank must equal the number of basis vectors",
            )
        return self


class DirectSumRequest(StrictModel):
    """Two integer lattices to direct-sum."""

    first: IntegerLattice
    second: IntegerLattice


class OrthogonalSumRequest(StrictModel):
    """Two integer lattices to orthogonally sum."""

    first: IntegerLattice
    second: IntegerLattice


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
