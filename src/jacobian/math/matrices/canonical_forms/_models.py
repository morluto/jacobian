"""Typed wire contracts for exact canonical-form operations over QQ."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import Field, model_validator
from pydantic_core import PydanticCustomError

from jacobian._models import StrictModel
from jacobian.math.matrices.values import (
    MAX_MATRIX_DIMENSION,
    RationalMatrix,
)
from jacobian.math.polynomials.values import (
    MAX_POLYNOMIAL_EXPONENT,
    MonicPolynomial,
    RationalPolynomial,
)

MAX_CANONICAL_FORM_DIMENSION = 16
MAX_CANONICAL_FORM_SCALAR_DIGITS = 256
# The centralizer system has n^2 rows and columns.  This owner-local budget
# admits the dense FLINT kernel through the published n=16 matrix boundary
# while keeping its cubic exact-arithmetic work bounded independently of the
# generic linear-algebra axis limit.
MAX_CENTRALIZER_RREF_WORK = 2_000_000_000
MAX_CENTRALIZER_OUTPUT_DIGIT_WORK = 20_000_000

# A Horner pass performs ``degree`` dense n-by-n products. The retained
# conservative envelope charges the producer pass and its source-bound
# accounting without widening this established public admission bound.
MAX_MATRIX_POLYNOMIAL_SCALAR_PRODUCTS = 4_000_000
MATRIX_POLYNOMIAL_EVALUATION_PASSES = 2
# Couple exact operation count to the largest admitted rational component.
# Reference cases, including source retention: a dense 32x32 degree-61 request
# costs about 33 billion proxy units and 1.5 seconds; a 1x1 degree-327 request
# with a 32,701-digit denominator costs about 700 billion units and one second.
MAX_MATRIX_POLYNOMIAL_DIGIT_WORK = 1_000_000_000_000
# Polynomial division can repeatedly multiply by every non-leading modulus
# coefficient.  Charge the source degree and the resulting component width
# before materializing the quotient, rather than discovering growth midway.
MAX_MATRIX_POLYNOMIAL_REMAINDER_DIGIT_WORK = 1_000_000_000_000


def _mathematical_polynomial_degree(
    polynomial: RationalPolynomial,
) -> int | None:
    if not polynomial.polynomial.terms:
        return None
    return polynomial.polynomial.terms[0].exponents[0]


class MatrixPolynomialEvaluationRequest(StrictModel):
    """Evaluate one exact univariate rational polynomial at a square matrix."""

    matrix: RationalMatrix = Field(
        description=(
            "Nonempty square matrix over QQ through order "
            f"{MAX_MATRIX_DIMENSION}. Matrix and polynomial coefficients "
            "share the exact rational field."
        )
    )
    polynomial: RationalPolynomial = Field(
        description=(
            "Sparse polynomial over QQ in exactly one declared variable; terms "
            "use the canonical descending exponent order of RationalPolynomial. "
            "Exact admission couples the ordinary degree to the matrix order: "
            f"{MATRIX_POLYNOMIAL_EVALUATION_PASSES} * degree * order^3 must "
            f"stay within {MAX_MATRIX_POLYNOMIAL_SCALAR_PRODUCTS:,} scalar "
            "products across both Horner passes, so the largest admitted "
            "ordinary degree at matrix order 32 is "
            f"{MAX_MATRIX_POLYNOMIAL_SCALAR_PRODUCTS // (MATRIX_POLYNOMIAL_EVALUATION_PASSES * MAX_MATRIX_DIMENSION**3)}. "
            "Exact admission additionally multiplies the total "
            "(2 * degree * order^3) scalar products by the square of the "
            "largest decimal-digit component among the matrix entries, the "
            "polynomial coefficients, the predicted exact-result "
            "components, and the predicted shifted Horner intermediate "
            "components, whose denominators include coprime coefficient "
            "denominators whenever their matrix-power supports overlap in "
            "one shared entry during evaluation; that digit-work product "
            "must stay within "
            f"{MAX_MATRIX_POLYNOMIAL_DIGIT_WORK:,} units."
        )
    )


class MatrixPolynomialEvaluationResult(StrictModel):
    """Exact value of one polynomial at one rational matrix.

    Kernel-produced instances use :meth:`_from_kernel`. This transport model
    checks source/value shape and declared Horner accounting only;
    """

    source_matrix: RationalMatrix
    polynomial: RationalPolynomial
    value: RationalMatrix
    polynomial_degree: int | None = Field(
        default=None,
        ge=0,
        le=MAX_POLYNOMIAL_EXPONENT,
        description="The ordinary degree for a nonzero polynomial; null for zero.",
    )
    matrix_multiplications: int = Field(ge=0, le=MAX_POLYNOMIAL_EXPONENT)
    scalar_product_terms: int = Field(
        ge=0,
        le=MAX_MATRIX_POLYNOMIAL_SCALAR_PRODUCTS // MATRIX_POLYNOMIAL_EVALUATION_PASSES,
    )

    @model_validator(mode="after")
    def require_structural_accounting(self) -> Self:
        expected_degree = _mathematical_polynomial_degree(self.polynomial)
        if self.polynomial_degree != expected_degree:
            raise _validation_error(
                "budget_exceeded",
                "matrix polynomial result degree does not match its source",
            )
        expected_multiplications = expected_degree or 0
        if self.matrix_multiplications != expected_multiplications:
            raise _validation_error(
                "budget_exceeded",
                "Horner matrix multiplication count must equal the polynomial degree",
            )
        dimension = len(self.source_matrix.entries)
        if len(self.value.entries) != dimension or any(
            len(row) != dimension for row in self.value.entries
        ):
            raise _validation_error(
                "shape_mismatch",
                "matrix polynomial value must have the source matrix shape",
            )
        if self.scalar_product_terms != expected_multiplications * dimension**3:
            raise _validation_error(
                "shape_mismatch",
                "Horner scalar-product count must equal degree times matrix order cubed",
            )
        return self

    @classmethod
    def _from_kernel(
        cls,
        *,
        matrix: RationalMatrix,
        polynomial: RationalPolynomial,
        value: RationalMatrix,
    ) -> Self:
        polynomial_degree = _mathematical_polynomial_degree(polynomial)
        return cls.model_construct(
            source_matrix=matrix,
            polynomial=polynomial,
            value=value,
            polynomial_degree=polynomial_degree,
            matrix_multiplications=polynomial_degree or 0,
            scalar_product_terms=(polynomial_degree or 0) * len(matrix.entries) ** 3,
        )


class MatrixPolynomialRemainderRequest(StrictModel):
    """Reduce one exact univariate polynomial modulo a matrix's minimal polynomial."""

    matrix: RationalMatrix = Field(
        description="Nonempty square rational matrix whose minimal polynomial supplies the modulus."
    )
    polynomial: RationalPolynomial = Field(
        description="Sparse univariate rational polynomial; its declared variable is retained in every result polynomial."
    )


class MatrixPolynomialRemainderResult(StrictModel):
    """Exact Euclidean division by the matrix minimal polynomial.

    The quotient and remainder establish the defining relation
    ``polynomial = quotient * minimal_polynomial + remainder`` and the
    remainder has degree strictly below the minimal-polynomial degree.
    """

    source_matrix: RationalMatrix
    polynomial: RationalPolynomial
    minimal_polynomial: MonicPolynomial
    quotient: RationalPolynomial
    remainder: RationalPolynomial

    @model_validator(mode="after")
    def require_division_shape(self) -> Self:
        if self.source_matrix.row_count == 0:
            raise _validation_error("shape_mismatch", "source matrix must be nonempty")
        if self.source_matrix.row_count != self.source_matrix.column_count:
            raise _validation_error("shape_mismatch", "source matrix must be square")
        variable = self.polynomial.variables
        if self.minimal_polynomial.variables != variable:
            raise _validation_error(
                "invariant_mismatch", "minimal polynomial must use the source variable"
            )
        if self.quotient.variables != variable or self.remainder.variables != variable:
            raise _validation_error(
                "invariant_mismatch", "division outputs must use the source variable"
            )
        minimal_degree = self.minimal_polynomial.polynomial.terms[0].exponents[0]
        remainder_degree = max(
            (term.exponents[0] for term in self.remainder.polynomial.terms),
            default=-1,
        )
        if remainder_degree >= minimal_degree:
            raise _validation_error(
                "invariant_mismatch",
                "remainder degree must be smaller than the minimal-polynomial degree",
            )
        return self

    @classmethod
    def _from_kernel(
        cls,
        *,
        matrix: RationalMatrix,
        polynomial: RationalPolynomial,
        minimal_polynomial: MonicPolynomial,
        quotient: RationalPolynomial,
        remainder: RationalPolynomial,
    ) -> Self:
        return cls.model_construct(
            source_matrix=matrix,
            polynomial=polynomial,
            minimal_polynomial=minimal_polynomial,
            quotient=quotient,
            remainder=remainder,
        )


class SquareMatrixRequest(StrictModel):
    """One square rational matrix bounded for canonical-form computation."""

    matrix: RationalMatrix


class MinimalPolynomialResult(StrictModel):
    """Exact minimal polynomial of a square rational matrix.

    Retains source and structural polynomial metadata. Kernel output uses
    :meth:`_from_kernel`.
    """

    matrix: RationalMatrix
    minimal_polynomial: MonicPolynomial
    characteristic_polynomial: MonicPolynomial
    degree: int = Field(ge=1, le=MAX_CANONICAL_FORM_DIMENSION)

    @model_validator(mode="after")
    def require_structural_metadata(self) -> Self:
        if self.degree != len(self.minimal_polynomial.coefficients) - 1:
            raise _validation_error(
                "invariant_mismatch", "degree must equal the minimal-polynomial degree"
            )
        if len(self.characteristic_polynomial.coefficients) - 1 != len(
            self.matrix.entries
        ):
            raise _validation_error(
                "shape_mismatch",
                "characteristic-polynomial degree must equal matrix order",
            )
        return self

    @classmethod
    def _from_kernel(
        cls,
        *,
        matrix: RationalMatrix,
        minimal_polynomial: MonicPolynomial,
        characteristic_polynomial: MonicPolynomial,
    ) -> Self:
        return cls.model_construct(
            matrix=matrix,
            minimal_polynomial=minimal_polynomial,
            characteristic_polynomial=characteristic_polynomial,
            degree=len(minimal_polynomial.coefficients) - 1,
        )


class InvariantFactorEntry(StrictModel):
    """One monic invariant factor from the rational canonical form."""

    factor: MonicPolynomial
    block_size: int = Field(ge=1, le=MAX_CANONICAL_FORM_DIMENSION)


class RationalCanonicalFormResult(StrictModel):
    """Exact rational (Frobenius) canonical form of a square rational matrix.

    Retains source and structural metadata. Kernel output uses
    :meth:`_from_kernel`.
    """

    matrix: RationalMatrix
    invariant_factors: tuple[InvariantFactorEntry, ...] = Field(min_length=1)
    characteristic_polynomial: MonicPolynomial
    minimal_polynomial: MonicPolynomial
    total_block_size: int = Field(ge=1, le=MAX_CANONICAL_FORM_DIMENSION)

    @model_validator(mode="after")
    def require_structural_metadata(self) -> Self:
        dimension = len(self.matrix.entries)
        if self.total_block_size != sum(
            entry.block_size for entry in self.invariant_factors
        ):
            raise _validation_error(
                "shape_mismatch", "total block size must equal the summed block sizes"
            )
        if self.total_block_size != dimension:
            raise _validation_error(
                "shape_mismatch", "block sizes must total the matrix dimension"
            )
        for entry in self.invariant_factors:
            if entry.block_size != len(entry.factor.coefficients) - 1:
                raise _validation_error(
                    "invariant_mismatch", "each block size must equal its factor degree"
                )
        if self.invariant_factors[-1].factor != self.minimal_polynomial:
            raise _validation_error(
                "invariant_mismatch",
                "the final invariant factor is the minimal polynomial",
            )
        return self

    @classmethod
    def _from_kernel(
        cls,
        *,
        matrix: RationalMatrix,
        invariant_factors: tuple[InvariantFactorEntry, ...],
        characteristic_polynomial: MonicPolynomial,
        minimal_polynomial: MonicPolynomial,
    ) -> Self:
        return cls.model_construct(
            matrix=matrix,
            invariant_factors=invariant_factors,
            characteristic_polynomial=characteristic_polynomial,
            minimal_polynomial=minimal_polynomial,
            total_block_size=sum(entry.block_size for entry in invariant_factors),
        )


class PrimaryDecompositionResult(StrictModel):
    """Primary decomposition of the minimal polynomial into irreducible-power components.

    Retains source and component values. Kernel output uses :meth:`_from_kernel`.
    """

    matrix: RationalMatrix
    components: tuple[MonicPolynomial, ...] = Field(min_length=1)
    minimal_polynomial: MonicPolynomial

    @model_validator(mode="after")
    def require_structural_metadata(self) -> Self:
        if len(self.minimal_polynomial.coefficients) - 1 > len(self.matrix.entries):
            raise _validation_error(
                "shape_mismatch",
                "minimal-polynomial degree cannot exceed matrix order",
            )
        return self

    @classmethod
    def _from_kernel(
        cls,
        *,
        matrix: RationalMatrix,
        components: tuple[MonicPolynomial, ...],
        minimal_polynomial: MonicPolynomial,
    ) -> Self:
        return cls.model_construct(
            matrix=matrix,
            components=components,
            minimal_polynomial=minimal_polynomial,
        )


class InvariantFactorProfileResult(StrictModel):
    """Complete monic invariant-factor profile with field-bound relations.

    The kernel establishes the divisibility chain, the product relation
    against the characteristic polynomial, and the terminal minimal-polynomial
    relation; structural decoding checks only monic shape and axis coverage.
    """

    matrix: RationalMatrix
    invariant_factors: tuple[InvariantFactorEntry, ...] = Field(min_length=1)
    characteristic_polynomial: MonicPolynomial
    minimal_polynomial: MonicPolynomial
    scope: Literal["COMPLETE_FIELD_BOUND_PROFILE"] = "COMPLETE_FIELD_BOUND_PROFILE"

    @model_validator(mode="after")
    def require_structural_profile(self) -> Self:
        dimension = len(self.matrix.entries)
        if sum(entry.block_size for entry in self.invariant_factors) != dimension:
            raise _validation_error(
                "shape_mismatch", "block sizes must total the matrix dimension"
            )
        for entry in self.invariant_factors:
            if entry.block_size != len(entry.factor.coefficients) - 1:
                raise _validation_error(
                    "invariant_mismatch", "each block size must equal its factor degree"
                )
        if self.invariant_factors[-1].factor != self.minimal_polynomial:
            raise _validation_error(
                "invariant_mismatch",
                "the final invariant factor is the minimal polynomial",
            )
        return self

    @classmethod
    def _from_kernel(
        cls,
        *,
        matrix: RationalMatrix,
        invariant_factors: tuple[InvariantFactorEntry, ...],
        characteristic_polynomial: MonicPolynomial,
        minimal_polynomial: MonicPolynomial,
    ) -> Self:
        return cls.model_construct(
            matrix=matrix,
            invariant_factors=invariant_factors,
            characteristic_polynomial=characteristic_polynomial,
            minimal_polynomial=minimal_polynomial,
        )


class SimilarityRequest(StrictModel):
    left: RationalMatrix
    right: RationalMatrix


class SimilarityResult(StrictModel):
    """Similarity decision from complete invariant-factor data.

    Two matrices are similar exactly when their canonical representatives
    agree over the same field and dimension. No constructive basis is returned;
    the shared profile is the proof.
    """

    left: RationalMatrix
    right: RationalMatrix
    similar: bool
    left_factors: tuple[InvariantFactorEntry, ...]
    right_factors: tuple[InvariantFactorEntry, ...]

    @classmethod
    def _from_kernel(
        cls,
        *,
        left: RationalMatrix,
        right: RationalMatrix,
        similar: bool,
        left_factors: tuple[InvariantFactorEntry, ...],
        right_factors: tuple[InvariantFactorEntry, ...],
    ) -> Self:
        return cls.model_construct(
            left=left,
            right=right,
            similar=similar,
            left_factors=left_factors,
            right_factors=right_factors,
        )


class CentralizerResult(StrictModel):
    """Complete exact centralizer as a nullspace basis of n-by-n matrices."""

    matrix: RationalMatrix
    dimension: int = Field(ge=1)
    basis: tuple[RationalMatrix, ...]

    @classmethod
    def _from_kernel(
        cls,
        *,
        matrix: RationalMatrix,
        dimension: int,
        basis: tuple[RationalMatrix, ...],
    ) -> Self:
        return cls.model_construct(matrix=matrix, dimension=dimension, basis=basis)


__all__ = [
    "MATRIX_POLYNOMIAL_EVALUATION_PASSES",
    "MAX_CANONICAL_FORM_DIMENSION",
    "MAX_CANONICAL_FORM_SCALAR_DIGITS",
    "MAX_MATRIX_POLYNOMIAL_DIGIT_WORK",
    "MAX_MATRIX_POLYNOMIAL_REMAINDER_DIGIT_WORK",
    "MAX_MATRIX_POLYNOMIAL_SCALAR_PRODUCTS",
    "CentralizerResult",
    "InvariantFactorEntry",
    "InvariantFactorProfileResult",
    "MatrixPolynomialEvaluationRequest",
    "MatrixPolynomialEvaluationResult",
    "MatrixPolynomialRemainderRequest",
    "MatrixPolynomialRemainderResult",
    "MinimalPolynomialResult",
    "MonicPolynomial",
    "PrimaryDecompositionResult",
    "RationalCanonicalFormResult",
    "SimilarityRequest",
    "SimilarityResult",
    "SquareMatrixRequest",
]


def _validation_error(reason: str, message: str) -> PydanticCustomError:
    return PydanticCustomError(f"matrix.{reason}", message)
