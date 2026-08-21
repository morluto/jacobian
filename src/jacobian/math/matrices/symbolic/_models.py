"""Typed wire contracts for symbolic matrix operations over QQ(t_1, ..., t_n)."""

from __future__ import annotations

from itertools import combinations, permutations
from typing import Literal, Self

from pydantic import Field, model_validator

from jacobian._models import StrictModel
from jacobian.math.polynomials.values import PolynomialVariable, RationalFunction

MAX_SYMBOLIC_MATRIX_DIMENSION = 8
MAX_SYMBOLIC_VARIABLES = 8
MAX_SYMBOLIC_MATRIX_TERMS = 512
MAX_SYMBOLIC_RESULT_TERMS = 256
MAX_SYMBOLIC_RESULT_EXPONENT = 64
MAX_SYMBOLIC_RESULT_COEFFICIENT_DIGITS = 128


def _is_polynomial_entry(value: RationalFunction) -> bool:
    terms = value.denominator.terms
    return (
        len(terms) == 1
        and terms[0].coefficient.num == "1"
        and terms[0].coefficient.den == "1"
        and all(exponent == 0 for exponent in terms[0].exponents)
    )


def _principal_minor_term_bounds(
    entries: tuple[tuple[RationalFunction, ...], ...],
) -> tuple[int, ...]:
    """Bound raw terms in each characteristic coefficient by Leibniz expansion."""

    dimension = len(entries)
    bounds = [1]
    for size in range(1, dimension + 1):
        coefficient_terms = 0
        for axes in combinations(range(dimension), size):
            for columns in permutations(axes):
                product_terms = 1
                for row, column in zip(axes, columns, strict=True):
                    product_terms *= len(entries[row][column].numerator.terms)
                coefficient_terms += product_terms
        bounds.append(coefficient_terms)
    return tuple(bounds)


def _require_determinant_family_result_budget(
    matrix: SymbolicMatrix,
    *,
    characteristic_polynomial: bool,
) -> None:
    dimension = len(matrix.entries)
    if dimension == 1:
        return
    values = tuple(value for row in matrix.entries for value in row)
    if any(not _is_polynomial_entry(value) for value in values):
        raise ValueError(
            "multi-dimensional determinant-family requests require polynomial entries"
        )
    term_bounds = _principal_minor_term_bounds(matrix.entries)
    relevant_bounds = term_bounds[1:] if characteristic_polynomial else term_bounds[-1:]
    if any(bound > MAX_SYMBOLIC_RESULT_TERMS for bound in relevant_bounds):
        raise ValueError("determinant-family expansion exceeds the result term budget")
    maximum_exponent = max(
        (
            exponent
            for value in values
            for term in value.numerator.terms
            for exponent in term.exponents
        ),
        default=0,
    )
    if dimension * maximum_exponent > MAX_SYMBOLIC_RESULT_EXPONENT:
        raise ValueError(
            "determinant-family expansion exceeds the result exponent budget"
        )
    coefficient_digits = max(
        (
            len(component.lstrip("-"))
            for value in values
            for term in value.numerator.terms
            for component in (term.coefficient.num, term.coefficient.den)
        ),
        default=1,
    )
    if any(
        bound * dimension * coefficient_digits + len(str(max(bound, 1)))
        > MAX_SYMBOLIC_RESULT_COEFFICIENT_DIGITS
        for bound in relevant_bounds
    ):
        raise ValueError(
            "determinant-family expansion exceeds the result coefficient budget"
        )


class SymbolicMatrix(StrictModel):
    """One nonempty rectangular matrix over a multivariate rational-function field.

    Every entry is a canonical reduced numerator/denominator value over the
    declared ordered variables. For example, the former expression ``a*c`` is
    represented by one numerator term with exponents ``(1, 0, 1, ...)`` and a
    unit denominator; ``f/e`` is represented by numerator ``f`` and denominator
    ``e``. This preserves every element of ``QQ(t_1, ..., t_n)`` without parsing
    caller text with SymPy.
    """

    matrix_schema_version: Literal["1"] = "1"
    variables: tuple[PolynomialVariable, ...] = Field(
        min_length=0,
        max_length=MAX_SYMBOLIC_VARIABLES,
    )
    entries: tuple[tuple[RationalFunction, ...], ...] = Field(
        min_length=1,
        max_length=MAX_SYMBOLIC_MATRIX_DIMENSION,
    )

    @model_validator(mode="after")
    def require_rectangular_nonempty_rows(self) -> Self:
        column_count = len(self.entries[0])
        if column_count == 0 or column_count > MAX_SYMBOLIC_MATRIX_DIMENSION:
            raise ValueError(
                "matrix rows must contain between 1 and "
                f"{MAX_SYMBOLIC_MATRIX_DIMENSION} entries"
            )
        if any(len(row) != column_count for row in self.entries):
            raise ValueError("matrix rows must all have the same length")
        if len(set(self.variables)) != len(self.variables):
            raise ValueError("symbolic matrix variables must be unique")
        values = tuple(value for row in self.entries for value in row)
        if any(value.variables != self.variables for value in values):
            raise ValueError(
                "every symbolic matrix entry must use the declared ordered field"
            )
        term_count = sum(
            len(value.numerator.terms) + len(value.denominator.terms)
            for value in values
        )
        if term_count > MAX_SYMBOLIC_MATRIX_TERMS:
            raise ValueError("symbolic matrix exceeds the 512-term operation budget")
        return self


class SymbolicMatrixRequest(StrictModel):
    """A symbolic matrix over a declared variable list."""

    matrix: SymbolicMatrix

    @model_validator(mode="after")
    def require_request_consistency(self) -> Self:
        return self


class SquareSymbolicMatrixRequest(SymbolicMatrixRequest):
    """A square symbolic matrix for operations requiring square input.

    Operations like determinant, characteristic polynomial, and eigenvalues
    are only defined for square matrices.  This request type enforces
    squareness at the request boundary rather than relying on a backend
    ValueError.
    """

    @model_validator(mode="after")
    def require_square(self) -> Self:
        rows = len(self.matrix.entries)
        cols = len(self.matrix.entries[0])
        if rows != cols:
            raise ValueError("operation requires a square symbolic matrix")
        return self


class SymbolicDeterminantRequest(SquareSymbolicMatrixRequest):
    """A square matrix whose exact determinant fits the public result type."""

    matrix: SymbolicMatrix = Field(
        description=(
            "A square symbolic matrix. One-dimensional matrices may contain any "
            "accepted rational function; larger matrices require polynomial "
            "entries whose derived determinant expansion has at most 256 terms, "
            "exponent 64, and 128-digit coefficient components."
        )
    )

    @model_validator(mode="after")
    def require_representable_determinant(self) -> Self:
        _require_determinant_family_result_budget(
            self.matrix,
            characteristic_polynomial=False,
        )
        return self


class SymbolicCharacteristicPolynomialRequest(SquareSymbolicMatrixRequest):
    """A square matrix whose characteristic polynomial fits the result type."""

    matrix: SymbolicMatrix = Field(
        description=(
            "A square symbolic matrix. One-dimensional matrices may contain any "
            "accepted rational function; larger matrices require polynomial "
            "entries whose derived principal-minor expansions each have at most "
            "256 terms, exponent 64, and 128-digit coefficient components."
        )
    )

    @model_validator(mode="after")
    def require_representable_characteristic_polynomial(self) -> Self:
        _require_determinant_family_result_budget(
            self.matrix,
            characteristic_polynomial=True,
        )
        return self


class SymbolicDeterminantResult(StrictModel):
    """The exact determinant in the matrix's rational-function field."""

    determinant: RationalFunction
    method: Literal["SYMPY_BAREISS"] = "SYMPY_BAREISS"


class SymbolicRankResult(StrictModel):
    """The exact symbolic rank and the canonical pivot columns."""

    rank: int = Field(ge=0, le=MAX_SYMBOLIC_MATRIX_DIMENSION)
    pivot_columns: tuple[int, ...] = Field(max_length=MAX_SYMBOLIC_MATRIX_DIMENSION)
    method: Literal["EXACT_SYMBOLIC_ROW_REDUCTION"] = "EXACT_SYMBOLIC_ROW_REDUCTION"


class SymbolicCharacteristicPolynomialResult(StrictModel):
    """The dense monic characteristic polynomial coefficients (descending)."""

    variable: Literal["lambda"] = "lambda"
    degree: int = Field(ge=1, le=MAX_SYMBOLIC_MATRIX_DIMENSION)
    coefficients_descending: tuple[RationalFunction, ...] = Field(
        min_length=2,
        max_length=MAX_SYMBOLIC_MATRIX_DIMENSION + 1,
    )
    convention: Literal["DET_LAMBDA_I_MINUS_A"] = "DET_LAMBDA_I_MINUS_A"


class SymbolicEigenvaluesResult(StrictModel):
    """The exact eigenvalues with algebraic multiplicities.

    The representation discriminates between:
    - EXPLICIT_ROOTS: individual eigenvalue expressions are returned
    - ROOTS_BY_POLYNOMIAL: eigenvalues are the roots of the returned
      characteristic polynomial over QQ(t_1, ..., t_n); individual root
      expressions are not materialized because the backend cannot
      represent them in radicals.
    """

    representation: Literal["EXPLICIT_ROOTS", "ROOTS_BY_POLYNOMIAL"] = "EXPLICIT_ROOTS"
    eigenvalues: tuple[str, ...] | None = Field(
        default=None,
        min_length=1,
        max_length=MAX_SYMBOLIC_MATRIX_DIMENSION,
    )
    multiplicities: tuple[int, ...] | None = Field(
        default=None,
        min_length=1,
        max_length=MAX_SYMBOLIC_MATRIX_DIMENSION,
    )
    characteristic_polynomial: tuple[RationalFunction, ...] | None = Field(
        default=None,
        min_length=2,
        max_length=MAX_SYMBOLIC_MATRIX_DIMENSION + 1,
    )
    degree: int | None = Field(default=None, ge=1, le=MAX_SYMBOLIC_MATRIX_DIMENSION)
    convention: Literal["SYMPY_EIGENVALS"] = "SYMPY_EIGENVALS"

    @model_validator(mode="after")
    def require_representation_consistency(self) -> Self:
        if self.representation == "EXPLICIT_ROOTS":
            if self.eigenvalues is None or self.multiplicities is None:
                raise ValueError(
                    "EXPLICIT_ROOTS must populate eigenvalues and multiplicities"
                )
            if len(self.eigenvalues) != len(self.multiplicities):
                raise ValueError(
                    "eigenvalues and multiplicities must have the same length"
                )
            if self.characteristic_polynomial is not None or self.degree is not None:
                raise ValueError(
                    "EXPLICIT_ROOTS must not populate characteristic_polynomial or degree"
                )
        else:  # ROOTS_BY_POLYNOMIAL
            if self.eigenvalues is not None or self.multiplicities is not None:
                raise ValueError(
                    "ROOTS_BY_POLYNOMIAL must not populate eigenvalues or multiplicities"
                )
            if self.characteristic_polynomial is None or self.degree is None:
                raise ValueError(
                    "ROOTS_BY_POLYNOMIAL must populate characteristic_polynomial and degree"
                )
            if len(self.characteristic_polynomial) != self.degree + 1:
                raise ValueError(
                    "characteristic polynomial coefficients must equal degree plus one"
                )
        return self
