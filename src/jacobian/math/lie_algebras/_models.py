"""Typed wire contracts for finite-dimensional Lie algebras over QQ."""

from __future__ import annotations

from itertools import pairwise
from typing import Annotated, Self

from pydantic import Field, StringConstraints, model_validator
from pydantic_core import PydanticCustomError

from jacobian._exact import CanonicalRational
from jacobian._models import StrictModel
from jacobian.math.matrices.values import RationalMatrix

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


class LieSubspace(StrictModel):
    """One subspace of a Lie algebra's underlying vector space.

    ``generators`` is the canonical reduced row echelon basis: every row
    carries a leading one in a strictly increasing pivot column, pivot
    columns are otherwise zero, and zero rows are absent. RREF is unique,
    so equal subspaces have byte-identical generators, including the
    zero subspace as a zero-row matrix retaining the ambient axis.
    """

    basis: tuple[LieBasisLabel, ...] = Field(
        min_length=1,
        max_length=MAX_LIE_DIMENSION,
        description="The ambient ordered basis axis the rows are coordinatized in.",
    )
    generators: RationalMatrix = Field(
        description=(
            "RREF basis rows spanning the subspace; row count is the "
            "dimension, column count matches the ambient basis axis."
        )
    )

    @model_validator(mode="after")
    def require_rref_generators(self) -> Self:
        dimension = len(self.basis)
        rows = self.generators.entries
        if self.generators.column_count != dimension or any(
            len(row) != dimension for row in rows
        ):
            raise _validation_error(
                "subspace_axis",
                "subspace generator rows must match the ambient basis axis",
            )
        if self.generators.row_count != len(rows) or len(rows) > dimension:
            raise _validation_error(
                "subspace_dimension",
                "subspace generator count must not exceed the ambient dimension",
            )
        pivots: list[int] = []
        for row in rows:
            pivot = next(
                (
                    column
                    for column, value in enumerate(row)
                    if value.as_fraction() != 0
                ),
                None,
            )
            if pivot is None:
                raise _validation_error(
                    "subspace_zero_row", "subspace generators omit zero rows"
                )
            pivots.append(pivot)
        if pivots != sorted(pivots) or len(set(pivots)) != len(pivots):
            raise _validation_error(
                "subspace_pivot_order", "RREF pivot columns strictly increase"
            )
        for rank, pivot in enumerate(pivots):
            if rows[rank][pivot].as_fraction() != 1:
                raise _validation_error(
                    "subspace_leading_one", "RREF pivots carry leading ones"
                )
            if any(
                rows[other][pivot].as_fraction() != 0
                for other in range(len(rows))
                if other != rank
            ):
                raise _validation_error(
                    "subspace_pivot_column",
                    "RREF pivot columns are zero outside their row",
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


class LieAlgebraRequest(StrictModel):
    """One Lie algebra whose invariant is requested."""

    algebra: FiniteDimensionalLieAlgebra = Field(
        description=(
            "Finite-dimensional Lie algebra over QQ of dimension at most "
            f"{MAX_LIE_DIMENSION} by ordered basis labels and sparse "
            "structure constants; antisymmetry is canonical and every "
            "basis-triple Jacobi identity is established by operation "
            "admission before computation."
        )
    )


class LieKillingResult(StrictModel):
    """The exact Killing-form Gram matrix with its source algebra.

    ``killing_form[i][j] = tr(ad_{b_i} ad_{b_j})`` in the algebra's
    ordered basis. The form is symmetric and ad-invariant; Cartan's
    criterion reads semisimplicity off its nondegeneracy, which
    consuming operations decide.
    """

    algebra: FiniteDimensionalLieAlgebra
    killing_form: RationalMatrix

    @model_validator(mode="after")
    def require_killing_shape(self) -> Self:
        dimension = len(self.algebra.basis)
        entries = self.killing_form.entries
        if (
            self.killing_form.row_count != dimension
            or self.killing_form.column_count != dimension
            or len(entries) != dimension
            or any(len(row) != dimension for row in entries)
        ):
            raise _validation_error(
                "killing_shape",
                "the Killing matrix must be square on the algebra basis axis",
            )
        if any(
            entries[row][column].as_fraction() != entries[column][row].as_fraction()
            for row in range(dimension)
            for column in range(dimension)
        ):
            raise _validation_error(
                "killing_symmetry",
                "the Killing form is symmetric in its basis arguments",
            )
        return self

    @classmethod
    def _from_kernel(
        cls,
        algebra: FiniteDimensionalLieAlgebra,
        killing_form: RationalMatrix,
    ) -> Self:
        return cls.model_construct(
            algebra=algebra,
            killing_form=killing_form,
        )


class LieCenterResult(StrictModel):
    """The exact center with its source algebra.

    The center retains the canonical RREF subspace of vectors bracketing
    to zero against every basis element. Consuming series, quotient, and
    nilpotency operations accept the subspace unchanged.
    """

    algebra: FiniteDimensionalLieAlgebra
    center: LieSubspace

    @model_validator(mode="after")
    def require_center_binding(self) -> Self:
        if self.center.basis != self.algebra.basis:
            raise _validation_error(
                "center_basis",
                "the center subspace must use the algebra's ordered basis",
            )
        return self

    @classmethod
    def _from_kernel(
        cls,
        algebra: FiniteDimensionalLieAlgebra,
        center: LieSubspace,
    ) -> Self:
        return cls.model_construct(
            algebra=algebra,
            center=center,
        )


class LieIdealRequest(StrictModel):
    """One algebra with a candidate ideal subspace."""

    algebra: FiniteDimensionalLieAlgebra
    candidate: LieSubspace = Field(
        description=(
            "Candidate RREF subspace on the algebra's ordered basis; "
            "admission decides whether every algebra bracket with it "
            "stays inside."
        )
    )

    @model_validator(mode="after")
    def require_candidate_binding(self) -> Self:
        if self.candidate.basis != self.algebra.basis:
            raise _validation_error(
                "candidate_basis",
                "the candidate subspace must use the algebra's ordered basis",
            )
        return self


class IdealViolationWitness(StrictModel):
    """The first basis bracket escaping the candidate subspace."""

    basis_index: int = Field(ge=0, le=MAX_LIE_DIMENSION - 1)
    subspace_row: int = Field(
        ge=0,
        description="The generator row whose bracket with the basis element escapes.",
    )
    bracket: LieAlgebraElement = Field(
        description="The exact escaping bracket coordinates on the algebra basis."
    )


class LieIdealCheckResult(StrictModel):
    """Whether a subspace absorbs every algebra bracket.

    ``IDEAL`` retains no witness; ``NOT_IDEAL`` retains the first
    escaping bracket in increasing (basis index, generator row) order.
    """

    algebra: FiniteDimensionalLieAlgebra
    candidate: LieSubspace
    is_ideal: bool
    witness: IdealViolationWitness | None = None

    @model_validator(mode="after")
    def require_ideal_binding(self) -> Self:
        if self.candidate.basis != self.algebra.basis:
            raise _validation_error(
                "ideal_binding",
                "the candidate subspace must use the algebra's ordered basis",
            )
        if (self.witness is None) == (not self.is_ideal):
            raise _validation_error(
                "ideal_witness_binding",
                "a non-ideal owns its first escaping bracket and an ideal "
                "owns no witness",
            )
        if self.witness is not None and (
            self.witness.bracket.basis != self.algebra.basis
            or self.witness.basis_index >= len(self.algebra.basis)
            or self.witness.subspace_row >= self.candidate.generators.row_count
        ):
            raise _validation_error(
                "witness_axis",
                "the witness must address the algebra basis and a generator row",
            )
        return self

    @classmethod
    def _from_kernel(
        cls,
        algebra: FiniteDimensionalLieAlgebra,
        candidate: LieSubspace,
        is_ideal: bool,
        witness: IdealViolationWitness | None,
    ) -> Self:
        return cls.model_construct(
            algebra=algebra,
            candidate=candidate,
            is_ideal=is_ideal,
            witness=witness,
        )


class LieQuotientRequest(StrictModel):
    """One algebra with an ideal subspace and quotient basis labels."""

    algebra: FiniteDimensionalLieAlgebra
    ideal: LieSubspace = Field(
        description=(
            "RREF ideal subspace on the algebra's ordered basis; operation "
            "admission verifies ideal absorption before forming cosets."
        )
    )
    quotient_basis: tuple[LieBasisLabel, ...] = Field(
        min_length=1,
        max_length=MAX_LIE_DIMENSION,
        description=(
            "Fresh ordered labels for the quotient basis, one per free "
            "column of the ideal RREF in increasing order; the count must "
            "equal the algebra dimension minus the ideal dimension."
        ),
    )

    @model_validator(mode="after")
    def require_quotient_binding(self) -> Self:
        if self.ideal.basis != self.algebra.basis:
            raise _validation_error(
                "quotient_binding",
                "the ideal subspace must use the algebra's ordered basis",
            )
        if len(set(self.quotient_basis)) != len(self.quotient_basis):
            raise _validation_error(
                "quotient_labels",
                "quotient basis labels must be unique",
            )
        if len(self.quotient_basis) != len(self.algebra.basis) - (
            self.ideal.generators.row_count
        ):
            raise _validation_error(
                "quotient_dimension",
                "quotient labels must number dimension minus ideal dimension",
            )
        return self


class LieQuotientResult(StrictModel):
    """A Lie algebra quotient by an ideal with its source data."""

    algebra: FiniteDimensionalLieAlgebra
    ideal: LieSubspace
    quotient: FiniteDimensionalLieAlgebra

    @model_validator(mode="after")
    def require_quotient_shape(self) -> Self:
        if (
            self.ideal.basis != self.algebra.basis
            or len(self.quotient.basis)
            != len(self.algebra.basis) - self.ideal.generators.row_count
        ):
            raise _validation_error(
                "quotient_shape",
                "the quotient dimension is algebra dimension minus ideal "
                "dimension on the bound bases",
            )
        return self

    @classmethod
    def _from_kernel(
        cls,
        algebra: FiniteDimensionalLieAlgebra,
        ideal: LieSubspace,
        quotient: FiniteDimensionalLieAlgebra,
    ) -> Self:
        return cls.model_construct(
            algebra=algebra,
            ideal=ideal,
            quotient=quotient,
        )


def _require_series_shape(
    terms: tuple[LieSubspace, ...],
    basis: tuple[LieBasisLabel, ...],
    terminates: bool,
    *,
    kind: str,
) -> None:
    """Share the descending-series structural contract between both series."""

    dimension = len(basis)
    if not terms:
        raise _validation_error(
            "series_coverage", f"a {kind} series retains at least its head term"
        )
    identity = tuple(
        tuple(1 if column == row else 0 for column in range(dimension))
        for row in range(dimension)
    )
    head = terms[0].generators.entries
    if (
        terms[0].basis != basis
        or tuple(tuple(value.as_fraction() for value in row) for row in head)
        != identity
    ):
        raise _validation_error(
            "series_head",
            f"a {kind} series starts from the whole algebra",
        )
    dimensions = tuple(term.generators.row_count for term in terms)
    if any(term.basis != basis for term in terms) or any(
        first <= second for first, second in pairwise(dimensions)
    ):
        raise _validation_error(
            "series_descent",
            f"a {kind} series strictly descends along bound subspaces",
        )
    if terminates != (dimensions[-1] == 0):
        raise _validation_error(
            "series_decision",
            f"a {kind} series terminates exactly at the zero subspace",
        )


class LieDerivedSeriesResult(StrictModel):
    """The derived series with its solvability decision.

    ``terms[0]`` is the algebra, ``terms[k+1] = [terms[k], terms[k]]``,
    stopping at the zero subspace or the first fixed term. The algebra
    is solvable exactly when the series reaches zero. Dimensions
    strictly decrease, so the series holds at most ``dim + 1`` terms.
    """

    algebra: FiniteDimensionalLieAlgebra
    terms: tuple[LieSubspace, ...]
    solvable: bool

    @model_validator(mode="after")
    def require_derived_shape(self) -> Self:
        _require_series_shape(
            terms=self.terms,
            basis=self.algebra.basis,
            terminates=self.solvable,
            kind="derived",
        )
        return self

    @classmethod
    def _from_kernel(
        cls,
        algebra: FiniteDimensionalLieAlgebra,
        terms: tuple[LieSubspace, ...],
        solvable: bool,
    ) -> Self:
        return cls.model_construct(
            algebra=algebra,
            terms=terms,
            solvable=solvable,
        )


class LieLowerCentralSeriesResult(StrictModel):
    """The lower central series with its nilpotency decision.

    ``terms[0]`` is the algebra, ``terms[k+1] = [algebra, terms[k]]``,
    stopping at the zero subspace or the first fixed term. The algebra
    is nilpotent exactly when the series reaches zero. Dimensions
    strictly decrease, so the series holds at most ``dim + 1`` terms.
    """

    algebra: FiniteDimensionalLieAlgebra
    terms: tuple[LieSubspace, ...]
    nilpotent: bool

    @model_validator(mode="after")
    def require_central_shape(self) -> Self:
        _require_series_shape(
            terms=self.terms,
            basis=self.algebra.basis,
            terminates=self.nilpotent,
            kind="lower central",
        )
        return self

    @classmethod
    def _from_kernel(
        cls,
        algebra: FiniteDimensionalLieAlgebra,
        terms: tuple[LieSubspace, ...],
        nilpotent: bool,
    ) -> Self:
        return cls.model_construct(
            algebra=algebra,
            terms=terms,
            nilpotent=nilpotent,
        )


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
