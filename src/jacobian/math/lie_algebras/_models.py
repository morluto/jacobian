"""Typed wire contracts for finite-dimensional Lie algebras over QQ."""

from __future__ import annotations

from typing import Annotated, Self

from pydantic import Field, StringConstraints, model_validator
from pydantic_core import PydanticCustomError

from jacobian._exact import CanonicalRational
from jacobian._models import StrictModel

LieBasisLabel = Annotated[
    str,
    StringConstraints(pattern=r"^[A-Za-z][A-Za-z0-9_]{0,31}$", strict=True),
]

MAX_LIE_DIMENSION = 8
MAX_STRUCTURE_NONZEROS = 256
MAX_STRUCTURE_COEFFICIENT_DIGITS = 64
MAX_ELEMENT_COEFFICIENT_DIGITS = 64
MAX_BRACKET_LEDGER_ROWS = MAX_LIE_DIMENSION * (MAX_LIE_DIMENSION - 1) // 2


def _validation_error(reason: str, message: str) -> PydanticCustomError:
    """Build a stable error owned by Lie-algebra contracts."""

    return PydanticCustomError(f"lie_algebra.{reason}", message)


class StructureConstant(StrictModel):
    """One exact bracket coefficient [b_i, b_j] = sum_k c_ij^k b_k, stored for i < j."""

    i: int = Field(ge=0, le=MAX_LIE_DIMENSION - 1)
    j: int = Field(ge=0, le=MAX_LIE_DIMENSION - 1)
    k: int = Field(ge=0, le=MAX_LIE_DIMENSION - 1)
    coefficient: CanonicalRational

    @model_validator(mode="after")
    def require_ordered_nonzero_constant(self) -> Self:
        if not self.i < self.j:
            raise _validation_error(
                "constant_order",
                "structure constants must be stored with i < j",
            )
        if self.coefficient.as_fraction() == 0:
            raise _validation_error(
                "zero_constant", "zero structure constants must be omitted"
            )
        return self


class FiniteDimensionalLieAlgebra(StrictModel):
    """One finite-dimensional Lie algebra over QQ by ordered structure constants."""

    basis: tuple[LieBasisLabel, ...] = Field(
        min_length=1,
        max_length=MAX_LIE_DIMENSION,
        description="Ordered basis axis; row order is a transport convention.",
    )
    structure_constants: tuple[StructureConstant, ...] = Field(
        min_length=0,
        max_length=MAX_STRUCTURE_NONZEROS,
        description=(
            "Sparse nonzero bracket coefficients with i < j in lexicographic "
            "(i, j, k) order; antisymmetry is canonical and Jacobi is "
            "established by consuming-operation admission."
        ),
    )

    @model_validator(mode="after")
    def require_canonical_constant_table(self) -> Self:
        if len(set(self.basis)) != len(self.basis):
            raise _validation_error(
                "duplicate_basis", "Lie-algebra basis labels must be unique"
            )
        dimension = len(self.basis)
        keys = tuple(
            (constant.i, constant.j, constant.k)
            for constant in self.structure_constants
        )
        if tuple(sorted(keys)) != keys or len(set(keys)) != len(keys):
            raise _validation_error(
                "constant_order",
                "structure constants must use unique lexicographic (i, j, k) order",
            )
        if any(
            constant.i >= dimension
            or constant.j >= dimension
            or constant.k >= dimension
            for constant in self.structure_constants
        ):
            raise _validation_error(
                "constant_axis",
                "structure-constant indices must lie on the basis axis",
            )
        return self


class LieAlgebraElement(StrictModel):
    """One exact vector with its source basis axis."""

    basis: tuple[LieBasisLabel, ...] = Field(min_length=1, max_length=MAX_LIE_DIMENSION)
    coordinates: tuple[CanonicalRational, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def require_basis_coordinates_shape(self) -> Self:
        if len(set(self.basis)) != len(self.basis):
            raise _validation_error(
                "duplicate_basis", "element basis labels must be unique"
            )
        if len(self.coordinates) != len(self.basis):
            raise _validation_error(
                "coordinate_shape",
                "element coordinates must match the basis axis",
            )
        return self


class LieBracketRequest(StrictModel):
    """Two elements of one Lie algebra whose bracket is requested."""

    algebra: FiniteDimensionalLieAlgebra
    left: LieAlgebraElement
    right: LieAlgebraElement

    @model_validator(mode="after")
    def require_shared_basis(self) -> Self:
        if (
            self.left.basis != self.algebra.basis
            or self.right.basis != self.algebra.basis
        ):
            raise _validation_error(
                "element_basis",
                "bracket elements must use the algebra's ordered basis",
            )
        return self


class BracketPairContribution(StrictModel):
    """One exact basis-pair term of the bracket expansion."""

    i: int = Field(ge=0, le=MAX_LIE_DIMENSION - 1)
    j: int = Field(ge=0, le=MAX_LIE_DIMENSION - 1)
    pair_coefficient: CanonicalRational = Field(
        description="Exact scalar x_i * y_j - x_j * y_i for the basis pair."
    )
    terms: tuple[StructureConstant, ...] = Field(
        description="Scaled structure constants (i, j, k) contributing to the bracket."
    )

    @model_validator(mode="after")
    def require_pair_shape(self) -> Self:
        if not self.i < self.j:
            raise _validation_error("pair_order", "ledger pairs must satisfy i < j")
        if self.pair_coefficient.as_fraction() == 0:
            raise _validation_error(
                "zero_pair", "ledger rows must have nonzero pair coefficients"
            )
        if (
            any(term.i != self.i or term.j != self.j for term in self.terms)
            or not self.terms
        ):
            raise _validation_error(
                "pair_binding",
                "ledger terms must bind exactly the row's basis pair",
            )
        return self


class LieBracketResult(StrictModel):
    """The exact bracket coordinates with their basis-pair ledger."""

    algebra: FiniteDimensionalLieAlgebra
    left: LieAlgebraElement
    right: LieAlgebraElement
    bracket: LieAlgebraElement
    ledger: tuple[BracketPairContribution, ...] = Field(
        max_length=MAX_BRACKET_LEDGER_ROWS,
        description=(
            "One row per nonzero basis pair in lexicographic (i, j) order; "
            "rows sum exactly to the bracket coordinates."
        ),
    )

    @model_validator(mode="after")
    def require_bracket_shape(self) -> Self:
        if self.bracket.basis != self.algebra.basis:
            raise _validation_error(
                "bracket_basis",
                "the bracket must use the algebra's ordered basis",
            )
        pairs = tuple((row.i, row.j) for row in self.ledger)
        if tuple(sorted(pairs)) != pairs or len(set(pairs)) != len(pairs):
            raise _validation_error(
                "ledger_order",
                "ledger rows must use unique lexicographic (i, j) order",
            )
        return self

    @classmethod
    def _from_kernel(
        cls,
        algebra: FiniteDimensionalLieAlgebra,
        left: LieAlgebraElement,
        right: LieAlgebraElement,
        *,
        bracket: LieAlgebraElement,
        ledger: tuple[BracketPairContribution, ...],
    ) -> Self:
        return cls.model_construct(
            algebra=algebra,
            left=left,
            right=right,
            bracket=bracket,
            ledger=ledger,
        )
