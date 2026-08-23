"""Typed wire contracts for symbolic matrix operations over QQ(t_1, ..., t_n)."""

from __future__ import annotations

from itertools import combinations, permutations
from typing import Any, Literal, Self

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


# ---------------------------------------------------------------------------
# Symbolic linear system over QQ(t_1, ..., t_n)
# ---------------------------------------------------------------------------


def _entry_growth_factor(value: RationalFunction) -> int:
    """Term-count factor one entry contributes to an unreduced product.

    Polynomial entries contribute their numerator term count; rational
    entries contribute squared factors because products of fractions
    accumulate numerator and denominator terms on both sides.
    """
    numerator_terms = len(value.numerator.terms)
    denominator_terms = len(value.denominator.terms)
    unit_denominator = denominator_terms == 1 and all(
        term.coefficient.num == "1" and all(e == 0 for e in term.exponents)
        for term in value.denominator.terms
    )
    if unit_denominator:
        return numerator_terms
    return max(numerator_terms, denominator_terms) ** 2


_EXPANSION_ENUMERATION_NODE_BUDGET = 200_000


class _ExpansionBudgetExhaustedError(Exception):
    """Exact Leibniz enumeration exceeded its admission node budget."""


def _augmented_growth_support(
    entries: tuple[tuple[RationalFunction, ...], ...],
    rhs: tuple[RationalFunction, ...],
) -> tuple[tuple[tuple[int, int], ...], ...]:
    """Nonzero ``(column, growth factor)`` cells per row of ``[A | b]``."""
    columns = len(entries[0])
    support: list[tuple[tuple[int, int], ...]] = []
    for row_index, row in enumerate(entries):
        cells = [
            (column, _entry_growth_factor(value)) for column, value in enumerate(row)
        ]
        if row_index < len(rhs):
            cells.append((columns, _entry_growth_factor(rhs[row_index])))
        support.append(tuple(cell for cell in cells if cell[1] > 0))
    return tuple(support)


def _injection_count_bound(
    support: tuple[tuple[tuple[int, int], ...], ...],
    columns_count: int,
    size: int,
) -> int:
    """Closed-form expansion bound over every size-k minor of ``[A | b]``.

    A Leibniz term survives only through an injection of its rows into
    distinct nonzero-growth columns, so the number of surviving terms is
    bounded by the product of the smaller of each largest row and column
    degree, times the largest single-cell growth factor to the size.
    """
    row_degrees = sorted((len(row) for row in support), reverse=True)
    column_degrees = [0] * (columns_count + 1)
    for row in support:
        for column, _ in row:
            column_degrees[column] += 1
    column_degrees.sort(reverse=True)
    growth_factors = [factor for row in support for _, factor in row]
    maximum_growth: int = max(growth_factors, default=0)
    degree_product = 1
    for index in range(size):
        degree_product *= min(row_degrees[index], column_degrees[index])
    expansion_bound: int = degree_product * maximum_growth**size
    return expansion_bound


def _exact_size_expansion(
    support: tuple[tuple[tuple[int, int], ...], ...],
    rows_count: int,
    columns_count: int,
    size: int,
    visited: list[int],
) -> int:
    """Exact maximum Leibniz expansion over every size-k minor.

    The structural walk visits only nonzero-growth cells and charges each
    visit to ``visited[0]``, aborting once the shared admission budget is
    exhausted.
    """

    def walk(
        row_position: int,
        row_indices: tuple[int, ...],
        available: int,
        product: int,
    ) -> int:
        if row_position == len(row_indices):
            return product
        total = 0
        for column, factor in support[row_indices[row_position]]:
            bit = 1 << column
            if available & bit:
                visited[0] += 1
                if visited[0] > _EXPANSION_ENUMERATION_NODE_BUDGET:
                    raise _ExpansionBudgetExhaustedError()
                total += walk(
                    row_position + 1,
                    row_indices,
                    available ^ bit,
                    product * factor,
                )
        return total

    maximum = 0
    for row_indices in combinations(range(rows_count), size):
        for column_indices in combinations(range(columns_count + 1), size):
            column_mask = sum(1 << column for column in column_indices)
            maximum = max(maximum, walk(0, row_indices, column_mask, 1))
    return maximum


def _expansion_bounds_by_size(
    entries: tuple[tuple[RationalFunction, ...], ...],
    rhs: tuple[RationalFunction, ...],
) -> list[int]:
    """Per-size maximum minor expansion, exact within a node budget.

    Sparse systems complete the exact structural enumeration far below
    the budget; once it is exceeded, remaining sizes fall back to
    ``_injection_count_bound``, so request validation never performs
    factorial permutation work.
    """
    rows_count = len(entries)
    columns_count = len(entries[0])
    work = min(rows_count, columns_count)
    support = _augmented_growth_support(entries, rhs)
    bounds = [0] * (work + 1)
    visited = [0]
    exhausted = False
    for size in range(1, work + 1):
        if exhausted:
            bounds[size] = _injection_count_bound(support, columns_count, size)
            continue
        try:
            bounds[size] = _exact_size_expansion(
                support, rows_count, columns_count, size, visited
            )
        except _ExpansionBudgetExhaustedError:
            exhausted = True
            bounds[size] = _injection_count_bound(support, columns_count, size)
    return bounds


def _solution_component_growth_bound(
    entries: tuple[tuple[RationalFunction, ...], ...],
    rhs: tuple[RationalFunction, ...],
) -> tuple[int, int, int]:
    """Conservative (terms, exponent, coefficient digits) for solved components.

    Every solution, particular-solution, and nullspace component is an
    exact ratio of minors of the augmented system ``[A | b]`` over minors
    of ``A`` of every size up to ``work = min(rows, columns)`` (Cramer/RREF
    structure; rank-deficient systems are decided by their largest
    nonvanishing minors). Both sides of such a ratio multiply up to
    ``2 * size`` entry factors, and each unreduced minor numerator expands
    over the Leibniz sum of per-entry term-count products.
    """
    rows = len(entries)
    columns = len(entries[0])
    work = min(rows, columns)

    # Every k-size minor of [A | b] with 1 <= k <= work bounds the
    # expansion work behind some solution component; A's own minors are a
    # subset of these. Lower-rank minors matter when all work-size minors
    # are structurally zero.
    maximum_expansion_by_size = _expansion_bounds_by_size(entries, rhs)

    values = tuple(value for row in entries for value in row) + tuple(rhs)
    maximum_exponent = max(
        (
            exponent
            for value in values
            for polynomial in (value.numerator, value.denominator)
            for term in polynomial.terms
            for exponent in term.exponents
        ),
        default=0,
    )
    coefficient_digits = max(
        (
            max(len(term.coefficient.num.lstrip("-")), len(term.coefficient.den))
            for value in values
            for polynomial in (value.numerator, value.denominator)
            for term in polynomial.terms
        ),
        default=1,
    )
    terms_bound = max(expansion**2 for expansion in maximum_expansion_by_size)
    exponent_bound = 2 * work * maximum_exponent
    digits_bound = max(
        (
            expansion * 2 * size * coefficient_digits + len(str(max(expansion, 1)))
            for size, expansion in enumerate(maximum_expansion_by_size)
            if size >= 1
        ),
        default=1,
    )
    return terms_bound, exponent_bound, digits_bound


def _require_linear_system_solution_budget(
    request: SymbolicLinearSystemRequest,
) -> None:
    """Admit only systems whose derived solutions fit the result type.

    Runs at request admission so no accepted request can fail inside the
    backend conversion with a host exception instead of returning its
    declared typed result.
    """
    _require_linear_system_growth_admission(request.matrix.entries, request.rhs)


def _require_linear_system_growth_admission(
    entries: tuple[tuple[RationalFunction, ...], ...],
    rhs: tuple[RationalFunction, ...],
) -> None:
    """Admit only systems whose derived solutions fit the result type.

    Shared by the wire request validator and the native solve entry point so
    direct callers cannot bypass the derived-solution bounds.
    """
    growth = _solution_component_growth_bound(entries, rhs)
    if growth[0] > MAX_SYMBOLIC_RESULT_TERMS:
        raise ValueError(
            "linear-system solution exceeds the derived result term budget; "
            "reduce entry term counts or dimension"
        )
    if growth[1] > MAX_SYMBOLIC_RESULT_EXPONENT:
        raise ValueError(
            "linear-system solution exceeds the derived result exponent "
            "budget; reduce entry exponents or dimension"
        )
    if growth[2] > MAX_SYMBOLIC_RESULT_COEFFICIENT_DIGITS:
        raise ValueError(
            "linear-system solution exceeds the derived result coefficient "
            "budget; reduce coefficient sizes or dimension"
        )


class SymbolicLinearSystemRequest(StrictModel):
    """Solve one bounded system ``A x = b`` over ``QQ(t_1, ..., t_n)``.

    The declared parameters are algebraically independent: the result is the
    generic solution over the rational-function field, not a case split over
    parameter specializations.  The coefficient matrix ``A`` and right-hand
    side ``b`` must use the same declared ordered variable list.
    """

    matrix: SymbolicMatrix
    rhs: tuple[RationalFunction, ...] = Field(
        min_length=1,
        max_length=MAX_SYMBOLIC_MATRIX_DIMENSION,
    )

    @model_validator(mode="after")
    def require_consistent_system(self) -> Self:
        rows = len(self.matrix.entries)
        if len(self.rhs) != rows:
            raise ValueError(
                "the right-hand side length must equal the coefficient row count"
            )
        for value in self.rhs:
            if value.variables != self.matrix.variables:
                raise ValueError(
                    "the right-hand side must use the declared ordered field"
                )
        # Derived-solution admission: bound exponent, term, and coefficient
        # growth before the backend runs so every accepted system returns
        # its declared typed result instead of failing inside conversion.
        _require_linear_system_solution_budget(self)
        return self


def _raw_system_column_bound(system: Any) -> int:
    """Best-effort column count of a not-yet-validated raw system payload."""
    if isinstance(system, dict):
        matrix = system.get("matrix")
        if isinstance(matrix, dict):
            entries = matrix.get("entries")
            if (
                isinstance(entries, (list, tuple))
                and entries
                and isinstance(entries[0], (list, tuple))
            ):
                return len(entries[0])
    return MAX_SYMBOLIC_MATRIX_DIMENSION


class SymbolicLinearSystemResult(StrictModel):
    """Classification and solution data for one symbolic linear system.

    The source system is retained so the classification and every solution
    vector remain verifiable against it after serialization.
    """

    system: SymbolicLinearSystemRequest
    classification: Literal["UNIQUE", "NON_UNIQUE", "INCONSISTENT"]
    solution: tuple[RationalFunction, ...] | None = None
    particular_solution: tuple[RationalFunction, ...] | None = None
    nullspace_basis: tuple[tuple[RationalFunction, ...], ...] | None = None
    consistency: Literal["EXACT_RATIONAL_FUNCTION"] = "EXACT_RATIONAL_FUNCTION"
    field_semantics: Literal["GENERIC_OVER_QQ_FIELD"] = "GENERIC_OVER_QQ_FIELD"

    def _replayed_solution(self) -> tuple[str, object, object, object]:
        from jacobian.math.matrices.symbolic.operations import (
            symbolic_linear_system_solve,
        )

        return symbolic_linear_system_solve(
            self.system.matrix.entries,
            self.system.rhs,
            self.system.matrix.variables,
        )

    @model_validator(mode="before")
    @classmethod
    def require_bounded_payload_shapes(cls, data: Any) -> Any:
        # Cap relayed solution payloads against the retained source's column
        # count BEFORE nested RationalFunction parsing; an unbounded tuple of
        # individually valid values would otherwise be fully parsed before
        # any later check rejects it.
        if not isinstance(data, dict):
            return data
        limit = _raw_system_column_bound(data.get("system"))
        for key in ("solution", "particular_solution"):
            value = data.get(key)
            if isinstance(value, (list, tuple)) and len(value) > limit:
                raise ValueError(
                    f"{key} length {len(value)} exceeds the retained system's "
                    f"column count {limit}"
                )
        basis = data.get("nullspace_basis")
        if isinstance(basis, (list, tuple)):
            if len(basis) > limit:
                raise ValueError(
                    f"nullspace_basis length {len(basis)} exceeds the "
                    f"retained system's column count {limit}"
                )
            for vector in basis:
                if isinstance(vector, (list, tuple)) and len(vector) > limit:
                    raise ValueError(
                        "a nullspace basis vector exceeds the retained "
                        f"system's column count {limit}"
                    )
        return data

    def _require_mathematical_witnesses(self) -> None:
        """Verify NON_UNIQUE witnesses by their defining equations.

        A particular solution is valid iff ``A p = b`` exactly, and a basis
        list is complete iff its vectors lie in the kernel, are linearly
        independent, and number ``n - rank(A)``. The public contract fixes no
        canonical free-variable normalization, so backend-identity comparison
        would reject mathematically equivalent witnesses.
        """
        import sympy

        from jacobian.math.polynomials._conversions import (
            rational_function_to_sympy,
        )

        entries = self.system.matrix.entries
        n_cols = len(entries[0])
        declared_variables = self.system.matrix.variables
        particular = self.particular_solution
        assert particular is not None
        # Every witness must live on the retained system's declared ordered
        # field and match its exact column count. Both are checked before
        # any SymPy arithmetic so a malformed relayed payload fails as a
        # contract violation instead of a backend host exception.
        if len(particular) != n_cols:
            raise ValueError(
                "particular_solution must have exactly the retained "
                "system's column count"
            )
        for value in particular:
            if value.variables != declared_variables:
                raise ValueError(
                    "witness vectors must use the retained system's "
                    "declared ordered field"
                )
        basis = self.nullspace_basis or ()
        for vector in basis:
            if len(vector) != n_cols:
                raise ValueError(
                    "every nullspace basis vector must have exactly the "
                    "retained system's column count"
                )
            for value in vector:
                if value.variables != declared_variables:
                    raise ValueError(
                        "witness vectors must use the retained system's "
                        "declared ordered field"
                    )
        coefficient = sympy.Matrix(
            [[rational_function_to_sympy(e) for e in row] for row in entries]
        )
        rhs_vec = sympy.Matrix(
            [[rational_function_to_sympy(v)] for v in self.system.rhs]
        )
        p_vec = sympy.Matrix([[rational_function_to_sympy(v)] for v in particular])
        residual = coefficient * p_vec - rhs_vec
        if any(sympy.cancel(entry) != 0 for entry in residual):
            raise ValueError(
                "particular_solution must satisfy the retained system exactly"
            )
        kernel_columns = []
        for vector in basis:
            v_vec = sympy.Matrix([[rational_function_to_sympy(v)] for v in vector])
            image = coefficient * v_vec
            if any(sympy.cancel(entry) != 0 for entry in image):
                raise ValueError("every nullspace basis vector must satisfy A v = 0")
            kernel_columns.append(v_vec)
        rank_coefficient = coefficient.rank()
        nullity = n_cols - rank_coefficient
        if len(kernel_columns) != nullity:
            raise ValueError(
                "nullspace_basis must carry exactly n - rank(A) independent "
                "vectors to span the kernel completely"
            )
        if nullity and (sympy.Matrix.hstack(*kernel_columns).rank() != nullity):
            raise ValueError("nullspace basis vectors must be linearly independent")

    def _require_classification_payload_shape(self) -> None:
        if self.classification == "UNIQUE":
            if self.solution is None:
                raise ValueError("UNIQUE must carry a solution vector")
            if self.particular_solution is not None or self.nullspace_basis is not None:
                raise ValueError(
                    "UNIQUE must not populate particular_solution or nullspace_basis"
                )
        elif self.classification == "NON_UNIQUE":
            if self.particular_solution is None:
                raise ValueError("NON_UNIQUE must carry a particular_solution")
            if self.solution is not None:
                raise ValueError("NON_UNIQUE must not populate the unique solution")
        elif (
            self.solution is not None
            or self.particular_solution is not None
            or self.nullspace_basis is not None
        ):
            raise ValueError("INCONSISTENT must not carry solution data")

    @model_validator(mode="after")
    def require_consistent_result(self) -> Self:
        # Solution growth is bounded at request admission
        # (_require_linear_system_solution_budget), before the backend runs:
        # a parsed canonical RationalFunction already caps each side at
        # MAX_SYMBOLIC_RESULT_TERMS, so per-component term checks here would
        # be ineffective anyway.
        self._require_classification_payload_shape()
        # Source-bound replay: the retained classification must be the exact
        # solve of this result's own coefficient matrix and right-hand side,
        # so a relayed or forged payload cannot validate as a solution of an
        # unrelated system. The unique solution is compared by identity (it
        # is mathematically unique); NON_UNIQUE witnesses are validated by
        # their defining equations instead of backend identity, since the
        # contract fixes no canonical free-variable normalization.
        (
            expected_classification,
            expected_solution,
            _expected_particular,
            _expected_nullspace,
        ) = self._replayed_solution()
        if self.classification != expected_classification:
            raise ValueError(
                "linear-system conclusion must be the exact solve of the "
                "retained source system"
            )
        if self.classification == "UNIQUE" and self.solution != expected_solution:
            raise ValueError(
                "the unique solution must be the exact solve of the "
                "retained source system"
            )
        if self.classification == "NON_UNIQUE":
            self._require_mathematical_witnesses()
        return self
