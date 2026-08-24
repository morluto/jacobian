"""Bounded exact contracts for graded Jacobian syzygies over ``QQ``."""

from __future__ import annotations

import hashlib
from fractions import Fraction
from math import comb
from typing import Literal, Self

from pydantic import Field, StrictInt, model_validator

from jacobian._digest import Sha256Digest
from jacobian._exact import (
    MAX_CANONICAL_RATIONAL_DIGITS,
    CanonicalRational,
    require_bounded_rational,
)
from jacobian._models import StrictModel
from jacobian.canonical import canonicalize_json
from jacobian.math.geometry.projective.values import RationalProjectiveLine
from jacobian.math.polynomials.values import (
    MAX_POLYNOMIAL_VARIABLES,
    PolynomialVariable,
    RationalPolynomial,
    require_polynomial_budget,
)

ExponentVector = tuple[StrictInt, ...]

MAX_SOURCE_DEGREE = 16
MAX_MULTIPLIER_DEGREE = 8
MAX_MAP_DIMENSION = 512
MAX_TOTAL_BASIS_MONOMIALS = 2_048
MAX_AGGREGATE_MATRIX_ENTRIES = 250_000
MAX_SPARSE_MATRIX_ENTRIES = 50_000
MAX_LINEAR_ALGEBRA_WORK = 15_000_000
MAX_SOURCE_COEFFICIENT_DIGITS = 32
MAX_LINEAR_FACTOR_COEFFICIENT_DIGITS = 12


def _basis_size(variable_count: int, degree: int) -> int:
    return comb(degree + variable_count - 1, variable_count - 1)


def _decimal_log_upper(value: int) -> int:
    """Return ``ceil(log10(value))`` without floating-point arithmetic."""

    return 0 if value <= 1 else len(str(value - 1))


def _homogeneous_basis(variable_count: int, degree: int) -> tuple[tuple[int, ...], ...]:
    if variable_count == 1:
        return ((degree,),)
    return tuple(
        (first, *tail)
        for first in range(degree, -1, -1)
        for tail in _homogeneous_basis(variable_count - 1, degree - first)
    )


def _compute_homogeneous_source_degree(
    polynomial: RationalPolynomial | None,
    linear_factors: tuple[RationalProjectiveLine, ...] | None,
) -> int:
    if polynomial is not None:
        terms = polynomial.polynomial.terms
        if not terms:
            raise ValueError("the source homogeneous polynomial must be nonzero")
        degrees = {sum(term.exponents) for term in terms}
        if len(degrees) != 1:
            raise ValueError("the source polynomial must be homogeneous")
        return next(iter(degrees))
    if linear_factors is None:
        raise ValueError("labelled linear factors are required")
    return len(linear_factors)


def _require_coefficient_map_budget(
    *,
    variable_count: int,
    source_degree: int,
    max_degree: int,
    coefficient_map_detail: Literal["CERTIFICATES", "SPARSE_ENTRIES"],
    entry_coefficient_digits: int,
) -> None:
    """Bound rows, columns, dense work, and the materialized result separately."""

    # Exact work is charged before SymPy runs. At each degree we perform a
    # producer RREF, an RREF to select independent minor rows, a determinant,
    # a possible nullspace RREF, then validator RREF and determinant replay.
    # With k=min(rows, columns), their conservative scalar-update bound is
    # 3*rows*columns*k + k*k*rows + 2*k**3. A fraction-free intermediate is a
    # ratio of k-minors; each numerator and denominator has at most
    # k*(entry_digits + ceil(log10(k))) digits. This covers both the rank
    # certificate and every exact value that can cross the result boundary.
    aggregate_entries = 0
    total_basis_monomials = 0
    total_linear_algebra_work = 0
    maximum_intermediate_digits = 0
    for degree in range(max_degree + 1):
        source_basis_count = _basis_size(variable_count, degree)
        target_basis_count = _basis_size(variable_count, source_degree - 1 + degree)
        column_count = variable_count * source_basis_count
        if (
            source_basis_count > MAX_MAP_DIMENSION
            or target_basis_count > MAX_MAP_DIMENSION
            or column_count > MAX_MAP_DIMENSION
        ):
            raise ValueError(
                "graded coefficient-map dimensions exceed the 512-monomial or "
                "512-column exact certificate budget"
            )
        aggregate_entries += target_basis_count * column_count
        total_basis_monomials += source_basis_count + target_basis_count
        rank_bound = min(target_basis_count, column_count)
        total_linear_algebra_work += (
            3 * target_basis_count * column_count * rank_bound
            + rank_bound * rank_bound * target_basis_count
            + 2 * rank_bound**3
        )
        maximum_intermediate_digits = max(
            maximum_intermediate_digits,
            rank_bound * (entry_coefficient_digits + _decimal_log_upper(rank_bound)),
        )
    if aggregate_entries > MAX_AGGREGATE_MATRIX_ENTRIES:
        raise ValueError(
            "graded coefficient maps exceed the 250000-entry exact rank budget"
        )
    if total_basis_monomials > MAX_TOTAL_BASIS_MONOMIALS:
        raise ValueError(
            "graded coefficient-map bases exceed the 2048-monomial result budget"
        )
    if total_linear_algebra_work > MAX_LINEAR_ALGEBRA_WORK:
        raise ValueError(
            "graded coefficient maps exceed the 15000000-update exact linear-algebra budget"
        )
    if maximum_intermediate_digits > MAX_CANONICAL_RATIONAL_DIGITS:
        raise ValueError(
            "graded coefficient maps exceed the exact rational intermediate-height budget"
        )
    if (
        coefficient_map_detail == "SPARSE_ENTRIES"
        and aggregate_entries > MAX_SPARSE_MATRIX_ENTRIES
    ):
        raise ValueError(
            "materialized graded coefficient maps exceed the 50000-entry result budget"
        )


def _polynomial_terms(
    polynomial: RationalPolynomial,
) -> dict[tuple[int, ...], Fraction]:
    return {
        tuple(term.exponents): term.coefficient.as_fraction()
        for term in polynomial.polynomial.terms
    }


def _partial_derivative_terms(
    terms: dict[tuple[int, ...], Fraction], variable: int
) -> dict[tuple[int, ...], Fraction]:
    derivative: dict[tuple[int, ...], Fraction] = {}
    for exponents, coefficient in terms.items():
        power = exponents[variable]
        if power:
            derived = list(exponents)
            derived[variable] -= 1
            derivative[tuple(derived)] = coefficient * power
    return derivative


def _multiply_terms(
    left: dict[tuple[int, ...], Fraction], right: dict[tuple[int, ...], Fraction]
) -> dict[tuple[int, ...], Fraction]:
    product: dict[tuple[int, ...], Fraction] = {}
    for left_exponents, left_coefficient in left.items():
        for right_exponents, right_coefficient in right.items():
            exponents = tuple(
                first + second
                for first, second in zip(left_exponents, right_exponents, strict=True)
            )
            product[exponents] = (
                product.get(exponents, Fraction(0))
                + left_coefficient * right_coefficient
            )
    return {
        exponents: coefficient
        for exponents, coefficient in product.items()
        if coefficient
    }


def _expanded_source_terms(
    polynomial: RationalPolynomial | None,
    linear_factors: tuple[RationalProjectiveLine, ...] | None,
) -> dict[tuple[int, ...], Fraction]:
    if polynomial is not None:
        return _polynomial_terms(polynomial)
    if linear_factors is None:
        raise ValueError("labelled linear factors are required")
    terms: dict[tuple[int, ...], Fraction] = {(0, 0, 0): Fraction(1)}
    for factor in linear_factors:
        terms = _multiply_terms(
            terms,
            {
                tuple(1 if axis == position else 0 for axis in range(3)): (
                    coefficient.as_fraction()
                )
                for position, coefficient in enumerate(factor.coefficients)
            },
        )
    return terms


def _degree_zero_kernel_is_forced(
    *, variable_count: int, source_terms: dict[tuple[int, ...], Fraction]
) -> bool:
    """Decide whether the degree-zero Jacobian map provably has nonzero nullity.

    Every degree-zero column is exactly one partial derivative's coefficient
    vector, so the map's exact rank follows from the admitted request alone.
    Rank below the variable count forces the first kernel at degree zero, and
    execution then breaks before constructing any later map.
    """

    partials = tuple(
        _partial_derivative_terms(source_terms, variable)
        for variable in range(variable_count)
    )
    pivot_rows: list[tuple[int, list[Fraction]]] = []
    for exponents in sorted({key for partial in partials for key in partial}):
        row = [partial.get(exponents, Fraction(0)) for partial in partials]
        for leading_column, pivot_row in pivot_rows:
            factor = row[leading_column]
            if not factor:
                continue
            row = [
                value - factor * pivot_value
                for value, pivot_value in zip(row, pivot_row, strict=True)
            ]
        new_leading_column = next(
            (column for column, value in enumerate(row) if value), None
        )
        if new_leading_column is None:
            continue
        scale = row[new_leading_column]
        pivot_rows.append((new_leading_column, [value / scale for value in row]))
        if len(pivot_rows) == variable_count:
            return False
    return True


class GradedJacobianSyzygyRequestBase(StrictModel):
    """Search the homogeneous Jacobian map through one explicit degree bound."""

    polynomial: RationalPolynomial | None = None
    linear_factors: tuple[RationalProjectiveLine, ...] | None = Field(
        default=None,
        min_length=1,
        max_length=MAX_SOURCE_DEGREE,
    )
    linear_factor_variables: (
        tuple[PolynomialVariable, PolynomialVariable, PolynomialVariable] | None
    ) = None
    max_degree: StrictInt = Field(default=6, ge=0, le=MAX_MULTIPLIER_DEGREE)
    coefficient_map_detail: Literal["CERTIFICATES", "SPARSE_ENTRIES"]

    @model_validator(mode="after")
    def require_bounded_homogeneous_input(self) -> Self:
        if self.polynomial is not None:
            if self.linear_factors is not None:
                raise ValueError(
                    "supply exactly one of polynomial or labelled linear_factors"
                )
            if self.linear_factor_variables is not None:
                raise ValueError(
                    "linear_factor_variables is only valid with linear_factors"
                )
            polynomial = self.polynomial
            variables = polynomial.variables
            require_polynomial_budget(
                polynomial,
                maximum_terms=4_096,
                maximum_exponent=MAX_SOURCE_DEGREE,
                maximum_coefficient_digits=MAX_SOURCE_COEFFICIENT_DIGITS,
                label="graded Jacobian source polynomial",
            )
        elif self.linear_factors is not None:
            if self.linear_factor_variables is None:
                raise ValueError("linear_factors require an exact three-variable order")
            labels = tuple(factor.label for factor in self.linear_factors)
            if len(labels) != len(set(labels)):
                raise ValueError("labelled linear-factor names must be unique")
            for factor in self.linear_factors:
                for coefficient in factor.coefficients:
                    require_bounded_rational(
                        coefficient,
                        max_digits=MAX_LINEAR_FACTOR_COEFFICIENT_DIGITS,
                        label="graded Jacobian linear-factor coefficient",
                    )
            variables = self.linear_factor_variables
        else:
            raise ValueError(
                "supply exactly one of polynomial or labelled linear_factors"
            )
        if not 1 <= len(variables) <= MAX_POLYNOMIAL_VARIABLES:
            raise ValueError("graded Jacobian syzygies require one to eight variables")
        if len(set(variables)) != len(variables):
            raise ValueError("graded Jacobian syzygy variables must be unique")
        source_degree = _compute_homogeneous_source_degree(
            self.polynomial, self.linear_factors
        )
        if not 1 <= source_degree <= MAX_SOURCE_DEGREE:
            raise ValueError("the source homogeneous degree must lie between 1 and 16")
        entry_coefficient_digits = (
            MAX_SOURCE_COEFFICIENT_DIGITS + _decimal_log_upper(source_degree)
            if self.polynomial is not None
            else source_degree * MAX_LINEAR_FACTOR_COEFFICIENT_DIGITS
            + _decimal_log_upper(_basis_size(3, source_degree))
            + _decimal_log_upper(source_degree)
        )
        # Execution stops at the first non-injective map, so admission charges
        # only degrees a run can actually reach. The degree-zero rank is exact
        # and cheap from the request alone; later degrees stay fully budgeted.
        budgeted_max_degree = (
            0
            if _degree_zero_kernel_is_forced(
                variable_count=len(variables),
                source_terms=_expanded_source_terms(
                    self.polynomial, self.linear_factors
                ),
            )
            else self.max_degree
        )
        _require_coefficient_map_budget(
            variable_count=len(variables),
            source_degree=source_degree,
            max_degree=budgeted_max_degree,
            coefficient_map_detail=self.coefficient_map_detail,
            entry_coefficient_digits=entry_coefficient_digits,
        )
        return self


class GradedJacobianSyzygyRequest(GradedJacobianSyzygyRequestBase):
    """Minimum-degree request, whose result deliberately omits sparse entries."""

    coefficient_map_detail: Literal["CERTIFICATES"] = "CERTIFICATES"


class GradedJacobianSyzygyCoefficientRequest(GradedJacobianSyzygyRequestBase):
    """Coefficient-ledger request, whose result always materializes sparse entries."""

    coefficient_map_detail: Literal["SPARSE_ENTRIES"] = "SPARSE_ENTRIES"


class GradedJacobianMapEntry(StrictModel):
    row: StrictInt = Field(ge=0, lt=MAX_MAP_DIMENSION)
    column: StrictInt = Field(ge=0, lt=MAX_MAP_DIMENSION)
    coefficient: CanonicalRational

    @model_validator(mode="after")
    def require_nonzero_coefficient(self) -> Self:
        if self.coefficient.as_fraction() == 0:
            raise ValueError("zero coefficient-map entries must be omitted")
        return self


class GradedJacobianRankMinor(StrictModel):
    row_indices: tuple[StrictInt, ...] = Field(max_length=MAX_MAP_DIMENSION)
    column_indices: tuple[StrictInt, ...] = Field(max_length=MAX_MAP_DIMENSION)
    determinant: CanonicalRational

    @model_validator(mode="after")
    def require_square_nonzero_minor(self) -> Self:
        if len(self.row_indices) != len(self.column_indices):
            raise ValueError("rank certificate minor must be square")
        if any(index < 0 for index in (*self.row_indices, *self.column_indices)):
            raise ValueError(
                "rank certificate indices must be nonnegative zero-based positions"
            )
        if (
            tuple(sorted(set(self.row_indices))) != self.row_indices
            or tuple(sorted(set(self.column_indices))) != self.column_indices
        ):
            raise ValueError("rank certificate indices must be unique and sorted")
        if self.determinant.as_fraction() == 0:
            raise ValueError("rank certificate determinant must be nonzero")
        return self


class GradedJacobianCoefficientMap(StrictModel):
    multiplier_degree: StrictInt = Field(ge=0, le=MAX_MULTIPLIER_DEGREE)
    source_monomial_basis: tuple[ExponentVector, ...] = Field(
        min_length=1, max_length=MAX_MAP_DIMENSION
    )
    target_monomial_basis: tuple[ExponentVector, ...] = Field(
        max_length=MAX_MAP_DIMENSION
    )
    row_count: StrictInt = Field(ge=1, le=MAX_MAP_DIMENSION)
    column_count: StrictInt = Field(ge=1, le=MAX_MAP_DIMENSION)
    matrix_digest: Sha256Digest
    sparse_entries: tuple[GradedJacobianMapEntry, ...] = Field(
        max_length=MAX_SPARSE_MATRIX_ENTRIES
    )
    rank: StrictInt = Field(ge=0, le=MAX_MAP_DIMENSION)
    nullity: StrictInt = Field(ge=0, le=MAX_MAP_DIMENSION)
    pivot_columns: tuple[StrictInt, ...] = Field(max_length=MAX_MAP_DIMENSION)
    rank_minor: GradedJacobianRankMinor | None = None
    injective: bool

    @model_validator(mode="after")
    def bind_dimensions_rank_and_optional_entries(self) -> Self:
        variable_count = len(self.source_monomial_basis[0])
        if not 1 <= variable_count <= MAX_POLYNOMIAL_VARIABLES:
            raise ValueError(
                "source basis exponent vectors must have one to eight axes"
            )
        if self.source_monomial_basis != _homogeneous_basis(
            variable_count, self.multiplier_degree
        ):
            raise ValueError("source basis must be the canonical homogeneous basis")
        if any(
            len(exponents) != variable_count
            or any(exponent < 0 for exponent in exponents)
            for exponents in self.target_monomial_basis
        ):
            raise ValueError("target basis exponent vectors must match variable_count")
        if len(self.source_monomial_basis) * variable_count != self.column_count:
            raise ValueError(
                "source basis must induce one multiplier block per variable"
            )
        if len(self.target_monomial_basis) != self.row_count:
            raise ValueError("target basis length must equal row_count")
        if self.rank + self.nullity != self.column_count:
            raise ValueError("rank plus nullity must equal column_count")
        if (
            len(self.pivot_columns) != self.rank
            or tuple(sorted(set(self.pivot_columns))) != self.pivot_columns
            or any(column >= self.column_count for column in self.pivot_columns)
        ):
            raise ValueError("pivot columns must canonically bind the reported rank")
        if self.injective != (self.nullity == 0):
            raise ValueError("injective must be equivalent to zero nullity")
        if self.rank == 0:
            if self.rank_minor is not None:
                raise ValueError("rank-zero map must not carry a nonzero minor")
        elif (
            self.rank_minor is None
            or len(self.rank_minor.row_indices) != self.rank
            or any(row >= self.row_count for row in self.rank_minor.row_indices)
            or any(
                column >= self.column_count for column in self.rank_minor.column_indices
            )
        ):
            raise ValueError("positive rank requires one bound full-rank minor")
        positions = tuple((entry.row, entry.column) for entry in self.sparse_entries)
        if positions != tuple(sorted(set(positions))) or any(
            row >= self.row_count or column >= self.column_count
            for row, column in positions
        ):
            raise ValueError("sparse coefficient-map entries must be unique and sorted")
        return self


class GradedJacobianKernelWitness(StrictModel):
    multiplier_degree: StrictInt = Field(ge=0, le=MAX_MULTIPLIER_DEGREE)
    coefficient_vector: tuple[CanonicalRational, ...] = Field(
        min_length=1,
        max_length=MAX_MAP_DIMENSION,
    )
    multipliers: tuple[RationalPolynomial, ...] = Field(
        min_length=1,
        max_length=MAX_POLYNOMIAL_VARIABLES,
    )

    @model_validator(mode="after")
    def require_nonzero_homogeneous_vector(self) -> Self:
        if all(value.as_fraction() == 0 for value in self.coefficient_vector):
            raise ValueError("kernel witness coefficient vector must be nonzero")
        if any(
            any(
                sum(term.exponents) != self.multiplier_degree
                for term in multiplier.polynomial.terms
            )
            for multiplier in self.multipliers
        ):
            raise ValueError("kernel witness multipliers must be homogeneous")
        return self


class GradedJacobianSyzygyResult(StrictModel):
    """Exact rank ledger and first kernel through the requested finite bound.

    ``result_schema_version='1'`` remains compatible with the prior
    three-variable result shape: variable count is still carried only by the
    canonical ``variables`` and exponent vectors. Operation versions, rather
    than this result value version, distinguish the specialized request modes.
    """

    result_schema_version: Literal["1"] = "1"
    variables: tuple[PolynomialVariable, ...] = Field(
        min_length=1, max_length=MAX_POLYNOMIAL_VARIABLES
    )
    source_kind: Literal["EXPANDED_POLYNOMIAL", "LABELLED_LINEAR_FACTOR_PRODUCT"]
    expanded_polynomial: RationalPolynomial
    homogeneous_degree: StrictInt = Field(ge=1, le=MAX_SOURCE_DEGREE)
    searched_through_degree: StrictInt = Field(ge=0, le=MAX_MULTIPLIER_DEGREE)
    coefficient_map_detail: Literal["CERTIFICATES", "SPARSE_ENTRIES"]
    partial_derivatives: tuple[RationalPolynomial, ...] = Field(
        min_length=1, max_length=MAX_POLYNOMIAL_VARIABLES
    )
    degree_maps: tuple[GradedJacobianCoefficientMap, ...] = Field(
        min_length=1,
        max_length=MAX_MULTIPLIER_DEGREE + 1,
    )
    status: Literal["FOUND", "NONE_THROUGH_BOUND"]
    first_syzygy_degree: StrictInt | None = Field(
        default=None, ge=0, le=MAX_MULTIPLIER_DEGREE
    )
    kernel_witness: GradedJacobianKernelWitness | None = None
    completion: Literal["COMPLETE_THROUGH_BOUND"] = "COMPLETE_THROUGH_BOUND"

    @model_validator(mode="after")
    def bind_first_kernel_and_finite_scope(self) -> Self:
        variable_count = _validate_result_source_and_partials(self)
        _validate_result_maps(self, variable_count)
        noninjective = tuple(
            item.multiplier_degree for item in self.degree_maps if not item.injective
        )
        if self.status == "FOUND":
            _validate_found_witness(self, variable_count, noninjective)
        elif (
            noninjective
            or self.first_syzygy_degree is not None
            or self.kernel_witness is not None
        ):
            raise ValueError(
                "NONE_THROUGH_BOUND may not claim or expose a kernel witness"
            )
        return self


def _validate_result_source_and_partials(result: GradedJacobianSyzygyResult) -> int:
    variable_count = len(result.variables)
    if len(set(result.variables)) != variable_count:
        raise ValueError("result variable order must be unique")
    if result.source_kind == "LABELLED_LINEAR_FACTOR_PRODUCT" and variable_count != 3:
        raise ValueError(
            "labelled linear-factor provenance requires exactly three variables"
        )
    if result.expanded_polynomial.variables != result.variables:
        raise ValueError("expanded source must use the declared variable order")
    source_terms = _polynomial_terms(result.expanded_polynomial)
    if not source_terms or any(
        sum(exponents) != result.homogeneous_degree for exponents in source_terms
    ):
        raise ValueError(
            "expanded source must be nonzero and homogeneous of the stated degree"
        )
    if len(result.partial_derivatives) != variable_count or any(
        partial.variables != result.variables for partial in result.partial_derivatives
    ):
        raise ValueError("partial derivatives must retain the source variable order")
    expected_partials = tuple(
        _partial_derivative_terms(source_terms, variable)
        for variable in range(variable_count)
    )
    if any(
        _polynomial_terms(partial) != expected
        for partial, expected in zip(
            result.partial_derivatives, expected_partials, strict=True
        )
    ):
        raise ValueError("partial derivatives must reconstruct from the source")
    return variable_count


def _validate_result_maps(
    result: GradedJacobianSyzygyResult, variable_count: int
) -> None:
    expected_degrees = tuple(range(result.searched_through_degree + 1))
    actual_degrees = tuple(item.multiplier_degree for item in result.degree_maps)
    if actual_degrees != expected_degrees:
        raise ValueError("degree maps must cover every degree from zero in order")
    for item in result.degree_maps:
        expected_target_basis = _homogeneous_basis(
            variable_count, result.homogeneous_degree - 1 + item.multiplier_degree
        )
        if item.target_monomial_basis != expected_target_basis:
            raise ValueError("coefficient maps must use the source ring's exact bases")
    if result.coefficient_map_detail == "CERTIFICATES" and any(
        item.sparse_entries for item in result.degree_maps
    ):
        raise ValueError("certificate detail must omit full sparse matrices")
    if result.coefficient_map_detail == "SPARSE_ENTRIES" and any(
        not item.sparse_entries and item.rank > 0 for item in result.degree_maps
    ):
        raise ValueError("sparse-entry detail must expose every nonzero map")
    for item in result.degree_maps:
        _replay_coefficient_map(result, item)


def _replayed_matrix_digest(
    *,
    multiplier_degree: int,
    source_basis: tuple[tuple[int, ...], ...],
    target_basis: tuple[tuple[int, ...], ...],
    entries: tuple[tuple[int, int, Fraction], ...],
) -> str:
    payload = {
        "protocol": "jacobian.graded-jacobian-map.v1",
        "multiplier_degree": multiplier_degree,
        "source_monomial_basis": [list(item) for item in source_basis],
        "target_monomial_basis": [list(item) for item in target_basis],
        "entries": [
            [row, column, f"{value.numerator}/{value.denominator}"]
            for row, column, value in entries
        ],
    }
    return f"sha256:{hashlib.sha256(canonicalize_json(payload)).hexdigest()}"


def _replay_coefficient_map(
    result: GradedJacobianSyzygyResult,
    item: GradedJacobianCoefficientMap,
) -> None:
    """Replay one admitted exact map so rank proves the reported minimum."""

    from sympy import Matrix, Rational

    matrix = Matrix.zeros(item.row_count, item.column_count)
    row_by_exponent = {
        exponents: row for row, exponents in enumerate(item.target_monomial_basis)
    }
    block_size = len(item.source_monomial_basis)
    for component, partial in enumerate(result.partial_derivatives):
        for basis_index, multiplier_exponents in enumerate(item.source_monomial_basis):
            column = component * block_size + basis_index
            for partial_exponents, coefficient in _polynomial_terms(partial).items():
                target_exponents = tuple(
                    left + right
                    for left, right in zip(
                        multiplier_exponents, partial_exponents, strict=True
                    )
                )
                matrix[row_by_exponent[target_exponents], column] += Rational(
                    coefficient.numerator, coefficient.denominator
                )
    entries = tuple(
        (row, column, Fraction(matrix[row, column]))
        for row in range(matrix.rows)
        for column in range(matrix.cols)
        if matrix[row, column] != 0
    )
    if item.matrix_digest != _replayed_matrix_digest(
        multiplier_degree=item.multiplier_degree,
        source_basis=item.source_monomial_basis,
        target_basis=item.target_monomial_basis,
        entries=entries,
    ):
        raise ValueError("coefficient-map digest must bind the reconstructed matrix")
    if (
        item.sparse_entries
        and tuple(
            (entry.row, entry.column, entry.coefficient.as_fraction())
            for entry in item.sparse_entries
        )
        != entries
    ):
        raise ValueError("sparse coefficient-map entries must reconstruct exactly")
    _, pivot_columns = matrix.rref()
    rank = len(pivot_columns)
    if (
        item.rank != rank
        or item.pivot_columns != tuple(int(column) for column in pivot_columns)
        or item.nullity != matrix.cols - rank
        or item.injective != (rank == matrix.cols)
    ):
        raise ValueError("coefficient-map rank certificate must replay exactly")
    if item.rank_minor is not None:
        minor = matrix.extract(
            item.rank_minor.row_indices, item.rank_minor.column_indices
        ).det()
        if Fraction(minor) != item.rank_minor.determinant.as_fraction():
            raise ValueError("rank-minor determinant must replay exactly")


def _validate_found_witness(
    result: GradedJacobianSyzygyResult,
    variable_count: int,
    noninjective: tuple[int, ...],
) -> None:
    witness = result.kernel_witness
    if (
        not noninjective
        or result.first_syzygy_degree != noninjective[0]
        or witness is None
        or witness.multiplier_degree != noninjective[0]
        or len(witness.multipliers) != variable_count
        or len(witness.coefficient_vector)
        != result.degree_maps[noninjective[0]].column_count
        or result.searched_through_degree != noninjective[0]
        or any(
            multiplier.variables != result.variables
            for multiplier in witness.multipliers
        )
    ):
        raise ValueError("FOUND must bind the first nonzero graded kernel")
    basis = result.degree_maps[noninjective[0]].source_monomial_basis
    expected_vector = tuple(
        _polynomial_terms(multiplier).get(exponents, Fraction(0))
        for multiplier in witness.multipliers
        for exponents in basis
    )
    if (
        tuple(value.as_fraction() for value in witness.coefficient_vector)
        != expected_vector
    ):
        raise ValueError("kernel vector must reconstruct its multipliers")
    reconstructed: dict[tuple[int, ...], Fraction] = {}
    for multiplier, partial in zip(
        witness.multipliers, result.partial_derivatives, strict=True
    ):
        for exponents, coefficient in _multiply_terms(
            _polynomial_terms(multiplier), _polynomial_terms(partial)
        ).items():
            reconstructed[exponents] = (
                reconstructed.get(exponents, Fraction(0)) + coefficient
            )
    if any(reconstructed.values()):
        raise ValueError("kernel witness must reconstruct a Jacobian syzygy")


__all__ = [
    "GradedJacobianCoefficientMap",
    "GradedJacobianKernelWitness",
    "GradedJacobianMapEntry",
    "GradedJacobianRankMinor",
    "GradedJacobianSyzygyCoefficientRequest",
    "GradedJacobianSyzygyRequest",
    "GradedJacobianSyzygyRequestBase",
    "GradedJacobianSyzygyResult",
]
