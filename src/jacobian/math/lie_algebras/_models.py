"""Typed wire contracts for finite-dimensional Lie algebras over QQ."""

from __future__ import annotations

from collections.abc import Mapping
from fractions import Fraction
from itertools import pairwise
from typing import Annotated, Any, Self
from weakref import ReferenceType, ref

from pydantic import (
    Field,
    StrictBool,
    StringConstraints,
    TypeAdapter,
    model_validator,
)
from pydantic_core import PydanticCustomError

from jacobian._exact import MAX_CANONICAL_RATIONAL_DIGITS, CanonicalRational
from jacobian._models import StrictModel
from jacobian.canonical import decimal_digit_width
from jacobian.math.matrices.values import RationalMatrix

LieBasisLabel = Annotated[
    str,
    StringConstraints(pattern=r"^[A-Za-z][A-Za-z0-9_]{0,31}$", strict=True),
]

MAX_LIE_DIMENSION = 8
MAX_STRUCTURE_NONZEROS = 256
MAX_STRUCTURE_COEFFICIENT_DIGITS = 64
# Jacobi is alternating, so it suffices to check increasing basis triples.
# For each of three cyclic terms there are at most n choices for the inner
# bracket output and n for the outer bracket output.
MAX_LIE_JACOBI_WORK = (
    3
    * (MAX_LIE_DIMENSION * (MAX_LIE_DIMENSION - 1) * (MAX_LIE_DIMENSION - 2) // 6)
    * MAX_LIE_DIMENSION**2
)
MAX_LIE_JACOBI_TERMS_PER_TRIPLE = 3 * MAX_LIE_DIMENSION**2
MAX_LIE_JACOBI_INTERMEDIATE_DIGITS = (
    2 * MAX_STRUCTURE_COEFFICIENT_DIGITS * MAX_LIE_JACOBI_TERMS_PER_TRIPLE
    + len(str(MAX_LIE_JACOBI_TERMS_PER_TRIPLE))
)
MAX_ELEMENT_COEFFICIENT_DIGITS = 64
MAX_BRACKET_LEDGER_ROWS = MAX_LIE_DIMENSION * (MAX_LIE_DIMENSION - 1) // 2
# A bracket pair performs two coordinate products and every retained ledger
# term performs one scaling and one accumulation.  Keep this derived envelope
# explicit so the native operation admits its complete ledger before doing
# exact result construction.
MAX_BRACKET_WORK = 2 * MAX_BRACKET_LEDGER_ROWS + 2 * MAX_STRUCTURE_NONZEROS
MAX_BRACKET_RESULT_COEFFICIENT_DIGITS = MAX_CANONICAL_RATIONAL_DIGITS
MAX_UPPER_CENTRAL_WORK = (MAX_LIE_DIMENSION + 1) * MAX_LIE_DIMENSION**4
MAX_UPPER_CENTRAL_RESULT_COEFFICIENTS = (MAX_LIE_DIMENSION + 1) * MAX_LIE_DIMENSION**2
MAX_SUBALGEBRA_CHECK_WORK = (
    MAX_LIE_DIMENSION**2 * MAX_STRUCTURE_NONZEROS + MAX_LIE_DIMENSION**4
)
MAX_CENTRALIZER_WORK = (
    3 * MAX_LIE_DIMENSION**4 + MAX_LIE_DIMENSION**2 * MAX_STRUCTURE_NONZEROS
)
# Cardinality-and-digits budget for one intermediate canonical matrix built
# during generated-subalgebra or generated-ideal closure: the entry count
# times the admitted per-entry decimal digit width must fit this envelope.
MAX_GENERATED_MATRIX_DECIMAL_DIGITS = 5_000_000

# Keep proof of construction outside the frozen model. Pydantic private
# attributes remain assignable through normal attribute syntax, so an
# instance-local boolean can be forged on a `model_construct` value.
_JACOBI_ADMITTED: dict[int, ReferenceType[Any]] = {}


def _register_jacobi_admitted(value: FiniteDimensionalLieAlgebra) -> None:
    identity = id(value)

    def discard(reference: ReferenceType[Any]) -> None:
        if _JACOBI_ADMITTED.get(identity) is reference:
            _JACOBI_ADMITTED.pop(identity, None)

    _JACOBI_ADMITTED[identity] = ref(value, discard)


def has_jacobi_admission(value: FiniteDimensionalLieAlgebra) -> bool:
    """Whether this exact immutable instance crossed a Jacobi proof boundary."""
    reference = _JACOBI_ADMITTED.get(id(value))
    return reference is not None and reference() is value


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


class LieAlgebraStructureConstant(StructureConstant):
    """A structure coefficient at the finite-dimensional algebra boundary."""

    coefficient: CanonicalRational = Field(
        description=(
            "Exact rational with at most 64 decimal digits in each reduced "
            "numerator and denominator."
        ),
        json_schema_extra={
            "properties": {
                "num": {
                    "maxLength": MAX_STRUCTURE_COEFFICIENT_DIGITS + 1,
                    "pattern": rf"^(?:0|-?[1-9][0-9]{{0,{MAX_STRUCTURE_COEFFICIENT_DIGITS - 1}}})(?![\s\S])",
                },
                "den": {
                    "maxLength": MAX_STRUCTURE_COEFFICIENT_DIGITS,
                    "pattern": rf"^[1-9][0-9]{{0,{MAX_STRUCTURE_COEFFICIENT_DIGITS - 1}}}(?![\s\S])",
                },
            }
        },
    )

    @model_validator(mode="after")
    def require_bounded_coefficient(self) -> Self:
        if (
            decimal_digit_width(self.coefficient.num) > MAX_STRUCTURE_COEFFICIENT_DIGITS
            or decimal_digit_width(self.coefficient.den)
            > MAX_STRUCTURE_COEFFICIENT_DIGITS
        ):
            raise _validation_error(
                "structure_coefficient_bound",
                "structure constants must use at most "
                f"{MAX_STRUCTURE_COEFFICIENT_DIGITS} decimal digits",
            )
        return self


class FiniteDimensionalLieAlgebra(StrictModel):
    """One finite-dimensional Lie algebra over QQ by ordered structure constants."""

    @classmethod
    def _from_jacobi_proved_kernel(
        cls,
        *,
        basis: tuple[LieBasisLabel, ...],
        structure_constants: tuple[StructureConstant, ...],
    ) -> Self:
        """Build an internal result after its operation proves Jacobi exactly.

        Public and serialized construction always runs full validation. This
        path is only for kernels whose defining computation already proves
        Jacobi, avoiding a second expansion while retaining the private
        admission fact needed by subsequent consumers.
        """
        basis = TypeAdapter(tuple[LieBasisLabel, ...]).validate_python(
            basis, strict=True
        )
        if not 1 <= len(basis) <= MAX_LIE_DIMENSION:
            raise _validation_error(
                "dimension_bound", "Lie-algebra dimension is outside 1..8"
            )
        if len(set(basis)) != len(basis):
            raise _validation_error(
                "duplicate_basis", "Lie-algebra basis labels must be unique"
            )
        if len(structure_constants) > MAX_STRUCTURE_NONZEROS:
            raise _validation_error(
                "structure_constant_bound", "too many Lie-algebra structure constants"
            )
        canonical_constants = tuple(
            LieAlgebraStructureConstant.model_validate(
                constant.model_dump(mode="python")
            )
            for constant in structure_constants
        )
        keys = tuple(
            (constant.i, constant.j, constant.k) for constant in canonical_constants
        )
        if tuple(sorted(keys)) != keys or len(set(keys)) != len(keys):
            raise _validation_error(
                "constant_order", "structure constants must be unique and ordered"
            )
        if any(
            constant.i >= len(basis)
            or constant.j >= len(basis)
            or constant.k >= len(basis)
            for constant in canonical_constants
        ):
            raise _validation_error(
                "constant_axis", "structure-constant indices must lie on the basis axis"
            )
        value = cls.model_construct(
            basis=basis, structure_constants=canonical_constants
        )
        _register_jacobi_admitted(value)
        return value

    basis: tuple[LieBasisLabel, ...] = Field(
        min_length=1,
        max_length=MAX_LIE_DIMENSION,
        description="Ordered basis axis; row order is a transport convention",
    )
    structure_constants: tuple[LieAlgebraStructureConstant, ...] = Field(
        min_length=0,
        max_length=MAX_STRUCTURE_NONZEROS,
        description=(
            "Sparse nonzero bracket coefficients with i < j in lexicographic "
            "(i, j, k) order; antisymmetry is canonical and construction "
            "checks Jacobi on the complete basis."
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
        # The Jacobiator is alternating for an antisymmetric bracket, so
        # increasing triples prove the identity on every basis triple. The
        # fixed dimension, nonzero, coefficient-height, work, and rational
        # intermediate-height ceilings bound this exact check.
        table: dict[tuple[int, int], dict[int, Fraction]] = {}
        for constant in self.structure_constants:
            value = constant.coefficient.as_fraction()
            table.setdefault((constant.i, constant.j), {})[constant.k] = value
            table.setdefault((constant.j, constant.i), {})[constant.k] = -value
        jacobi_work = 0
        triples = tuple(
            (first, second, third)
            for first in range(dimension)
            for second in range(first + 1, dimension)
            for third in range(second + 1, dimension)
        )
        for first, second, third in triples:
            for outer_first, outer_second, inner in (
                (first, second, third),
                (second, third, first),
                (third, first, second),
            ):
                outer_terms = table.get((outer_first, outer_second), {})
                jacobi_work += sum(
                    len(table.get((middle, inner), {})) for middle in outer_terms
                )
        if jacobi_work > MAX_LIE_JACOBI_WORK:
            raise _validation_error(
                "jacobi_work_bound", "Jacobi validation exceeds its fixed work bound"
            )
        if MAX_LIE_JACOBI_INTERMEDIATE_DIGITS > MAX_CANONICAL_RATIONAL_DIGITS:
            raise _validation_error(
                "jacobi_height_bound",
                "Jacobi validation exceeds its exact intermediate-height bound",
            )
        for first, second, third in triples:
            accumulator: dict[int, Fraction] = {}
            for outer_first, outer_second, inner in (
                (first, second, third),
                (second, third, first),
                (third, first, second),
            ):
                for middle, outer_value in table.get(
                    (outer_first, outer_second), {}
                ).items():
                    inner_terms = table.get((middle, inner), {})
                    for target, inner_value in inner_terms.items():
                        accumulator[target] = (
                            accumulator.get(target, 0) + outer_value * inner_value
                        )
            if any(value != 0 for value in accumulator.values()):
                raise _validation_error(
                    "jacobi_identity",
                    "structure constants must satisfy the Jacobi identity",
                )
        _register_jacobi_admitted(self)
        return self

    def model_copy(
        self,
        *,
        update: Mapping[str, Any] | None = None,
        deep: bool = False,
    ) -> Self:
        """Keep the internal Jacobi admission fact bound to unchanged fields."""
        if not update:
            return super().model_copy(deep=deep)
        payload = self.model_dump(mode="python")
        payload.update(update)
        return type(self).model_validate(payload)

    def __eq__(self, other: object) -> bool:
        if type(other) is not type(self):
            return NotImplemented
        assert isinstance(other, FiniteDimensionalLieAlgebra)
        return (
            self.basis == other.basis
            and self.structure_constants == other.structure_constants
        )

    def __hash__(self) -> int:
        return hash((type(self), self.basis, self.structure_constants))


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


class LieSubalgebra(LieSubspace):
    """An exact Lie subalgebra represented in its source algebra's basis.

    Closure is established by the operation that produces this value. If a
    caller supplies the value to another operation, that operation admits and
    checks the claimed closure before relying on it.
    """

    algebra: FiniteDimensionalLieAlgebra

    @model_validator(mode="after")
    def require_source_axis(self) -> Self:
        if self.basis != self.algebra.basis:
            raise _validation_error(
                "subalgebra_source_axis",
                "the subalgebra rows must use the source algebra's ordered basis",
            )
        return self


class LieIdeal(LieSubspace):
    """An ambient-bound ideal subspace produced by exact closure.

    The type retains its ambient algebra and RREF coordinate axis. Its ideal
    property is established by the producer; a consuming operation that
    relies on a caller-supplied claim must check ideal absorption itself.
    """

    algebra: FiniteDimensionalLieAlgebra

    @model_validator(mode="after")
    def require_source_axis(self) -> Self:
        if self.basis != self.algebra.basis:
            raise _validation_error(
                "ideal_source_axis",
                "the ideal rows must use the source algebra's ordered basis",
            )
        return self


class LieCentralizerRequest(StrictModel):
    """A finite family whose common centralizer is requested."""

    algebra: FiniteDimensionalLieAlgebra
    elements: tuple[LieAlgebraElement, ...] = Field(
        max_length=MAX_LIE_DIMENSION,
        description=(
            "Up to the ambient dimension many source-bound vectors; the empty "
            "family has centralizer equal to the whole algebra."
        ),
    )

    @model_validator(mode="after")
    def require_element_axes(self) -> Self:
        if len(self.elements) > len(self.algebra.basis):
            raise _validation_error(
                "centralizer_family_bound",
                "the centralizer family cannot exceed the ambient dimension",
            )
        if any(element.basis != self.algebra.basis for element in self.elements):
            raise _validation_error(
                "centralizer_element_basis",
                "centralizer elements must use the algebra's ordered basis",
            )
        return self


class LieGeneratedSubalgebraRequest(StrictModel):
    """An ambient Lie algebra and a bounded family of generators."""

    algebra: FiniteDimensionalLieAlgebra
    generators: tuple[LieAlgebraElement, ...] = Field(
        max_length=MAX_LIE_DIMENSION,
        description="Vectors on the exact ambient ordered basis; the empty family generates zero.",
    )

    @model_validator(mode="after")
    def require_generator_axes(self) -> Self:
        if any(item.basis != self.algebra.basis for item in self.generators):
            raise _validation_error(
                "generated_subalgebra_generator_basis",
                "generators must use the algebra's ordered basis",
            )
        return self


class LieGeneratedIdealRequest(StrictModel):
    """An ambient Lie algebra and a bounded family of ideal generators."""

    algebra: FiniteDimensionalLieAlgebra
    generators: tuple[LieAlgebraElement, ...] = Field(
        max_length=MAX_LIE_DIMENSION,
        description=(
            "Vectors on the exact ambient ordered basis; the empty family "
            "generates the zero ideal."
        ),
    )

    @model_validator(mode="after")
    def require_generator_axes(self) -> Self:
        if any(item.basis != self.algebra.basis for item in self.generators):
            raise _validation_error(
                "generated_ideal_generator_basis",
                "ideal generators must use the algebra's ordered basis",
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


class LieAdjointRequest(StrictModel):
    """One element whose adjoint endomorphism is requested."""

    algebra: FiniteDimensionalLieAlgebra
    element: LieAlgebraElement

    @model_validator(mode="after")
    def require_element_axis(self) -> Self:
        if self.element.basis != self.algebra.basis:
            raise _validation_error(
                "adjoint_element_basis",
                "the adjoint element must use the algebra's ordered basis",
            )
        return self


class LieAlgebraRequest(StrictModel):
    """One Lie algebra whose invariant is requested."""

    algebra: FiniteDimensionalLieAlgebra = Field(
        description=(
            "Finite-dimensional Lie algebra over QQ of dimension at most "
            f"{MAX_LIE_DIMENSION} by ordered basis labels and sparse "
            "structure constants with bounded rational entries; "
            "antisymmetry is canonical and construction checks every "
            "basis-triple Jacobi identity."
        )
    )


class LieDirectSumRequest(StrictModel):
    """Two admitted Lie algebras and the ordered basis of their direct sum."""

    left: FiniteDimensionalLieAlgebra
    right: FiniteDimensionalLieAlgebra
    basis: tuple[LieBasisLabel, ...] = Field(
        min_length=2,
        max_length=MAX_LIE_DIMENSION,
        description=(
            "Unique output labels in left-block then right-block order; their count "
            "must equal the sum of the two source dimensions."
        ),
    )

    @model_validator(mode="after")
    def require_output_axis(self) -> Self:
        expected = len(self.left.basis) + len(self.right.basis)
        if len(self.basis) != expected or len(set(self.basis)) != expected:
            raise _validation_error(
                "direct_sum_basis",
                "direct-sum labels must be unique across both dimensions",
            )
        return self


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


class LieKillingRadicalResult(StrictModel):
    """Nullspace of the Killing bilinear form, bound to its source algebra.

    This is the radical of the bilinear form. It is not asserted to be the
    solvable radical of the Lie algebra.
    """

    killing_result: LieKillingResult
    radical: LieSubspace

    @model_validator(mode="after")
    def require_source_axis(self) -> Self:
        if self.radical.basis != self.killing_result.algebra.basis:
            raise _validation_error(
                "killing_radical_source_axis",
                "the Killing-form radical must use the source algebra basis",
            )
        return self

    @classmethod
    def _from_kernel(
        cls, killing_result: LieKillingResult, radical: LieSubspace
    ) -> Self:
        return cls.model_construct(killing_result=killing_result, radical=radical)


class LieSemisimplicityResult(StrictModel):
    """The boolean decision from Cartan's semisimplicity criterion over QQ."""

    is_semisimple: StrictBool


class LieAdjointRepresentationResult(StrictModel):
    """Adjoint matrices in the exact order of the retained source basis."""

    algebra: FiniteDimensionalLieAlgebra
    matrices: tuple[RationalMatrix, ...] = Field(
        min_length=1,
        max_length=MAX_LIE_DIMENSION,
        description="One exact adjoint matrix per source basis element, in order.",
    )

    @model_validator(mode="after")
    def require_adjoint_matrix_axes(self) -> Self:
        dimension = len(self.algebra.basis)
        if len(self.matrices) != dimension or any(
            matrix.row_count != dimension
            or matrix.column_count != dimension
            or len(matrix.entries) != dimension
            or any(len(row) != dimension for row in matrix.entries)
            for matrix in self.matrices
        ):
            raise _validation_error(
                "adjoint_representation_shape",
                "there must be one square adjoint matrix per algebra basis element",
            )
        return self

    @classmethod
    def _from_kernel(
        cls,
        algebra: FiniteDimensionalLieAlgebra,
        matrices: tuple[RationalMatrix, ...],
    ) -> Self:
        return cls.model_construct(algebra=algebra, matrices=matrices)


class LieAdjointResult(StrictModel):
    """The exact matrix of ad_x on its retained source algebra."""

    algebra: FiniteDimensionalLieAlgebra
    element: LieAlgebraElement
    matrix: RationalMatrix

    @model_validator(mode="after")
    def require_adjoint_axes(self) -> Self:
        dimension = len(self.algebra.basis)
        if self.element.basis != self.algebra.basis:
            raise _validation_error(
                "adjoint_element_basis",
                "the adjoint element must use the algebra's ordered basis",
            )
        if (
            self.matrix.row_count != dimension
            or self.matrix.column_count != dimension
            or len(self.matrix.entries) != dimension
            or any(len(row) != dimension for row in self.matrix.entries)
        ):
            raise _validation_error(
                "adjoint_shape",
                "the adjoint matrix must be square on the algebra basis axis",
            )
        return self

    @classmethod
    def _from_kernel(
        cls,
        algebra: FiniteDimensionalLieAlgebra,
        element: LieAlgebraElement,
        matrix: RationalMatrix,
    ) -> Self:
        return cls.model_construct(algebra=algebra, element=element, matrix=matrix)


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


class LieCentralizerResult(StrictModel):
    """The common centralizer, as a subalgebra in its exact source algebra."""

    algebra: FiniteDimensionalLieAlgebra
    centralizer: LieSubalgebra

    @model_validator(mode="after")
    def require_source_binding(self) -> Self:
        if self.centralizer.algebra != self.algebra:
            raise _validation_error(
                "centralizer_source",
                "the centralizer must retain the exact source algebra",
            )
        return self

    @classmethod
    def _from_kernel(
        cls, algebra: FiniteDimensionalLieAlgebra, centralizer: LieSubalgebra
    ) -> Self:
        return cls.model_construct(algebra=algebra, centralizer=centralizer)


class LieIdealRequest(StrictModel):
    """One algebra with a candidate ideal subspace."""

    algebra: FiniteDimensionalLieAlgebra
    candidate: LieIdeal | LieSubalgebra | LieSubspace = Field(
        description=(
            "Candidate RREF subspace on the algebra's ordered basis; "
            "admission decides whether every algebra bracket with it "
            "stays inside."
        )
    )

    @model_validator(mode="after")
    def require_candidate_binding(self) -> Self:
        if self.candidate.basis != self.algebra.basis or (
            isinstance(self.candidate, (LieIdeal, LieSubalgebra))
            and self.candidate.algebra != self.algebra
        ):
            raise _validation_error(
                "candidate_basis",
                "the candidate subspace must use the algebra's ordered basis",
            )
        return self


class LieSubalgebraRequest(StrictModel):
    """One algebra with a candidate Lie subalgebra subspace."""

    algebra: FiniteDimensionalLieAlgebra
    candidate: LieIdeal | LieSubalgebra | LieSubspace = Field(
        description="Candidate RREF subspace on the algebra's ordered basis."
    )

    @model_validator(mode="after")
    def require_candidate_binding(self) -> Self:
        if self.candidate.basis != self.algebra.basis or (
            isinstance(self.candidate, (LieIdeal, LieSubalgebra))
            and self.candidate.algebra != self.algebra
        ):
            raise _validation_error(
                "candidate_basis",
                "the candidate subspace must use the algebra's ordered basis",
            )
        return self


class LieSubalgebraConstructionRequest(StrictModel):
    """One source algebra, closed subspace, and ordered induced basis labels."""

    algebra: FiniteDimensionalLieAlgebra
    candidate: LieIdeal | LieSubalgebra | LieSubspace
    subalgebra_basis: tuple[LieBasisLabel, ...] = Field(
        max_length=MAX_LIE_DIMENSION,
        description="Labels for candidate RREF rows in their existing order.",
    )

    @model_validator(mode="after")
    def require_candidate_binding_and_dimension(self) -> Self:
        if self.candidate.basis != self.algebra.basis or (
            isinstance(self.candidate, (LieIdeal, LieSubalgebra))
            and self.candidate.algebra != self.algebra
        ):
            raise _validation_error(
                "subalgebra_binding",
                "the candidate must use the source algebra's ordered basis",
            )
        if len(self.subalgebra_basis) != self.candidate.generators.row_count:
            raise _validation_error(
                "subalgebra_dimension",
                "the induced basis labels must match the candidate dimension",
            )
        if len(set(self.subalgebra_basis)) != len(self.subalgebra_basis):
            raise _validation_error(
                "subalgebra_labels", "induced basis labels must be unique"
            )
        return self


class LieSubalgebraResult(StrictModel):
    """An induced structure-constant algebra and its inclusion coordinates."""

    algebra: FiniteDimensionalLieAlgebra
    subspace: LieSubalgebra
    induced: FiniteDimensionalLieAlgebra
    inclusion: RationalMatrix

    @model_validator(mode="after")
    def require_induced_shape(self) -> Self:
        dimension = self.subspace.generators.row_count
        if (
            self.subspace.algebra != self.algebra
            or len(self.induced.basis) != dimension
            or self.inclusion.row_count != dimension
            or self.inclusion.column_count != len(self.algebra.basis)
            or self.inclusion.entries != self.subspace.generators.entries
        ):
            raise _validation_error(
                "subalgebra_result_shape",
                "the induced algebra and inclusion must retain the exact source subspace basis",
            )
        return self

    @classmethod
    def _from_kernel(
        cls,
        algebra: FiniteDimensionalLieAlgebra,
        subspace: LieSubalgebra,
        induced: FiniteDimensionalLieAlgebra,
        inclusion: RationalMatrix,
    ) -> Self:
        return cls.model_construct(
            algebra=algebra,
            subspace=subspace,
            induced=induced,
            inclusion=inclusion,
        )


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


class LieSubalgebraViolationWitness(StrictModel):
    """The first generator-pair bracket outside a candidate subalgebra."""

    left_row: int = Field(ge=0, le=MAX_LIE_DIMENSION - 1)
    right_row: int = Field(ge=0, le=MAX_LIE_DIMENSION - 1)
    bracket: LieAlgebraElement


class LieSubalgebraCheckResult(StrictModel):
    """Whether a candidate subspace is closed under its own Lie bracket."""

    algebra: FiniteDimensionalLieAlgebra
    candidate: LieSubspace
    is_subalgebra: StrictBool
    witness: LieSubalgebraViolationWitness | None = None

    @model_validator(mode="after")
    def require_subalgebra_binding(self) -> Self:
        if self.candidate.basis != self.algebra.basis:
            raise _validation_error(
                "subalgebra_binding",
                "the candidate subspace must use the algebra's ordered basis",
            )
        if (self.witness is None) == (not self.is_subalgebra):
            raise _validation_error(
                "subalgebra_witness_binding",
                "a non-subalgebra owns its first escaping bracket and a subalgebra owns no witness",
            )
        if self.witness is not None and (
            self.witness.bracket.basis != self.algebra.basis
            or self.witness.left_row >= self.candidate.generators.row_count
            or self.witness.right_row >= self.candidate.generators.row_count
        ):
            raise _validation_error(
                "subalgebra_witness_axis",
                "the witness must address generator rows on the algebra basis",
            )
        return self

    @classmethod
    def _from_kernel(
        cls,
        algebra: FiniteDimensionalLieAlgebra,
        candidate: LieSubspace,
        is_subalgebra: bool,
        witness: LieSubalgebraViolationWitness | None,
    ) -> Self:
        return cls.model_construct(
            algebra=algebra,
            candidate=candidate,
            is_subalgebra=is_subalgebra,
            witness=witness,
        )


class LieQuotientRequest(StrictModel):
    """One algebra with an ideal subspace and quotient basis labels."""

    algebra: FiniteDimensionalLieAlgebra
    ideal: LieIdeal | LieSubalgebra | LieSubspace = Field(
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
        if self.ideal.basis != self.algebra.basis or (
            isinstance(self.ideal, (LieIdeal, LieSubalgebra))
            and self.ideal.algebra != self.algebra
        ):
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


class LieUpperCentralSeriesResult(StrictModel):
    """The upper central series, starting at zero and ending at L if nilpotent.

    ``terms[0]`` is zero and ``terms[k+1] / terms[k]`` is the center of
    ``L / terms[k]``. A non-nilpotent algebra stops at the first stable proper
    term; a nilpotent algebra reaches the whole algebra.
    """

    algebra: FiniteDimensionalLieAlgebra
    terms: tuple[LieSubspace, ...]
    nilpotent: bool

    @model_validator(mode="after")
    def require_upper_central_shape(self) -> Self:
        dimension = len(self.algebra.basis)
        if not self.terms:
            raise _validation_error(
                "series_coverage", "an upper central series contains its zero term"
            )
        if any(term.basis != self.algebra.basis for term in self.terms):
            raise _validation_error(
                "series_basis", "upper central terms use the algebra basis"
            )
        dimensions = tuple(term.generators.row_count for term in self.terms)
        if dimensions[0] != 0 or any(a >= b for a, b in pairwise(dimensions)):
            raise _validation_error(
                "series_ascent", "upper central terms strictly ascend from zero"
            )
        reaches_algebra = dimensions[-1] == dimension
        if self.nilpotent != reaches_algebra or len(self.terms) > dimension + 1:
            raise _validation_error(
                "series_decision",
                "the upper central series is nilpotent exactly when it reaches the algebra",
            )
        return self

    @classmethod
    def _from_kernel(
        cls,
        algebra: FiniteDimensionalLieAlgebra,
        terms: tuple[LieSubspace, ...],
        nilpotent: bool,
    ) -> Self:
        return cls.model_construct(algebra=algebra, terms=terms, nilpotent=nilpotent)


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
