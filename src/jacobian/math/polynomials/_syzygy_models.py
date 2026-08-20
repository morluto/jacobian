"""Bounded exact contracts for graded Jacobian syzygies over QQ[x,y,z]."""

from __future__ import annotations

from math import comb
from typing import Literal, Self

from pydantic import Field, StrictInt, model_validator

from jacobian._digest import Sha256Digest
from jacobian._exact import CanonicalRational
from jacobian._models import StrictModel
from jacobian.math.geometry.projective.values import RationalProjectiveLine
from jacobian.math.polynomials.values import PolynomialVariable, RationalPolynomial

ExponentTriple = tuple[int, int, int]


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


def _require_coefficient_map_entry_budget(source_degree: int, max_degree: int) -> None:
    aggregate_entries = 0
    for degree in range(max_degree + 1):
        columns = 3 * comb(degree + 2, 2)
        rows = comb(source_degree + degree + 1, 2)
        aggregate_entries += rows * columns
    if aggregate_entries > 250_000:
        raise ValueError(
            "graded coefficient maps exceed the 250000-entry exact rank budget"
        )


class GradedJacobianSyzygyRequest(StrictModel):
    """Search the homogeneous Jacobian map through one explicit degree bound."""

    polynomial: RationalPolynomial | None = None
    linear_factors: tuple[RationalProjectiveLine, ...] | None = Field(
        default=None,
        min_length=1,
        max_length=16,
    )
    linear_factor_variables: (
        tuple[
            PolynomialVariable,
            PolynomialVariable,
            PolynomialVariable,
        ]
        | None
    ) = None
    max_degree: StrictInt = Field(default=6, ge=0, le=8)
    coefficient_map_detail: Literal["CERTIFICATES", "SPARSE_ENTRIES"] = "CERTIFICATES"

    @model_validator(mode="after")
    def require_bounded_homogeneous_three_variable_input(self) -> Self:
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
        elif self.linear_factors is not None:
            if self.linear_factor_variables is None:
                raise ValueError("linear_factors require an exact three-variable order")
            labels = tuple(factor.label for factor in self.linear_factors)
            if len(labels) != len(set(labels)):
                raise ValueError("labelled linear-factor names must be unique")
            variables = self.linear_factor_variables
        else:
            raise ValueError(
                "supply exactly one of polynomial or labelled linear_factors"
            )
        if len(variables) != 3:
            raise ValueError(
                "graded Jacobian syzygies currently require exactly three variables"
            )
        if len(set(variables)) != len(variables):
            raise ValueError("graded Jacobian syzygy variables must be unique")
        source_degree = _compute_homogeneous_source_degree(
            self.polynomial, self.linear_factors
        )
        if source_degree < 1 or source_degree > 16:
            raise ValueError("the source homogeneous degree must lie between 1 and 16")
        _require_coefficient_map_entry_budget(source_degree, self.max_degree)
        return self


class GradedJacobianMapEntry(StrictModel):
    row: StrictInt = Field(ge=0, le=1023)
    column: StrictInt = Field(ge=0, le=1023)
    coefficient: CanonicalRational

    @model_validator(mode="after")
    def require_nonzero_coefficient(self) -> Self:
        if self.coefficient.as_fraction() == 0:
            raise ValueError("zero coefficient-map entries must be omitted")
        return self


class GradedJacobianRankMinor(StrictModel):
    row_indices: tuple[StrictInt, ...] = Field(max_length=512)
    column_indices: tuple[StrictInt, ...] = Field(max_length=512)
    determinant: CanonicalRational

    @model_validator(mode="after")
    def require_square_nonzero_minor(self) -> Self:
        if len(self.row_indices) != len(self.column_indices):
            raise ValueError("rank certificate minor must be square")
        if (
            tuple(sorted(set(self.row_indices))) != self.row_indices
            or tuple(sorted(set(self.column_indices))) != self.column_indices
        ):
            raise ValueError("rank certificate indices must be unique and sorted")
        if self.determinant.as_fraction() == 0:
            raise ValueError("rank certificate determinant must be nonzero")
        return self


class GradedJacobianCoefficientMap(StrictModel):
    multiplier_degree: StrictInt = Field(ge=0, le=8)
    source_monomial_basis: tuple[ExponentTriple, ...] = Field(max_length=64)
    target_monomial_basis: tuple[ExponentTriple, ...] = Field(max_length=512)
    row_count: StrictInt = Field(ge=1, le=512)
    column_count: StrictInt = Field(ge=3, le=512)
    matrix_digest: Sha256Digest
    sparse_entries: tuple[GradedJacobianMapEntry, ...] = Field(max_length=50_000)
    rank: StrictInt = Field(ge=0, le=512)
    nullity: StrictInt = Field(ge=0, le=512)
    pivot_columns: tuple[StrictInt, ...] = Field(max_length=512)
    rank_minor: GradedJacobianRankMinor | None = None
    injective: bool

    @model_validator(mode="after")
    def bind_dimensions_rank_and_optional_entries(self) -> Self:
        if len(self.source_monomial_basis) * 3 != self.column_count:
            raise ValueError("source basis must induce three multiplier blocks")
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
    multiplier_degree: StrictInt = Field(ge=0, le=8)
    coefficient_vector: tuple[CanonicalRational, ...] = Field(
        min_length=3,
        max_length=512,
    )
    multipliers: tuple[
        RationalPolynomial,
        RationalPolynomial,
        RationalPolynomial,
    ]

    @model_validator(mode="after")
    def require_nonzero_vector(self) -> Self:
        if all(value.as_fraction() == 0 for value in self.coefficient_vector):
            raise ValueError("kernel witness coefficient vector must be nonzero")
        return self


class GradedJacobianSyzygyResult(StrictModel):
    """Exact rank ledger and first kernel through the requested finite bound."""

    result_schema_version: Literal["1"] = "1"
    variables: tuple[str, str, str]
    source_kind: Literal["EXPANDED_POLYNOMIAL", "LABELLED_LINEAR_FACTOR_PRODUCT"]
    expanded_polynomial: RationalPolynomial
    homogeneous_degree: StrictInt = Field(ge=1, le=16)
    searched_through_degree: StrictInt = Field(ge=0, le=8)
    coefficient_map_detail: Literal["CERTIFICATES", "SPARSE_ENTRIES"]
    partial_derivatives: tuple[
        RationalPolynomial,
        RationalPolynomial,
        RationalPolynomial,
    ]
    degree_maps: tuple[GradedJacobianCoefficientMap, ...] = Field(
        min_length=1,
        max_length=9,
    )
    status: Literal["FOUND", "NONE_THROUGH_BOUND"]
    first_syzygy_degree: StrictInt | None = Field(default=None, ge=0, le=8)
    kernel_witness: GradedJacobianKernelWitness | None = None
    completion: Literal["COMPLETE_THROUGH_BOUND"] = "COMPLETE_THROUGH_BOUND"

    @model_validator(mode="after")
    def bind_first_kernel_and_finite_scope(self) -> Self:
        if self.expanded_polynomial.variables != self.variables:
            raise ValueError("expanded source must use the declared variable order")
        if not self.expanded_polynomial.polynomial.terms or any(
            sum(term.exponents) != self.homogeneous_degree
            for term in self.expanded_polynomial.polynomial.terms
        ):
            raise ValueError(
                "expanded source must be nonzero and homogeneous of the stated degree"
            )
        expected_degrees = tuple(range(self.searched_through_degree + 1))
        actual_degrees = tuple(item.multiplier_degree for item in self.degree_maps)
        if actual_degrees != expected_degrees:
            raise ValueError("degree maps must cover every degree from zero in order")
        if self.coefficient_map_detail == "CERTIFICATES" and any(
            item.sparse_entries for item in self.degree_maps
        ):
            raise ValueError("certificate detail must omit full sparse matrices")
        if self.coefficient_map_detail == "SPARSE_ENTRIES" and any(
            not item.sparse_entries and item.rank > 0 for item in self.degree_maps
        ):
            raise ValueError("sparse-entry detail must expose every nonzero map")
        noninjective = tuple(
            item.multiplier_degree for item in self.degree_maps if not item.injective
        )
        if self.status == "FOUND":
            if (
                not noninjective
                or self.first_syzygy_degree != noninjective[0]
                or self.kernel_witness is None
                or self.kernel_witness.multiplier_degree != noninjective[0]
                or len(self.kernel_witness.coefficient_vector)
                != self.degree_maps[noninjective[0]].column_count
                or self.searched_through_degree != noninjective[0]
            ):
                raise ValueError("FOUND must bind the first nonzero graded kernel")
        elif (
            noninjective
            or self.first_syzygy_degree is not None
            or self.kernel_witness is not None
        ):
            raise ValueError(
                "NONE_THROUGH_BOUND may not claim or expose a kernel witness"
            )
        return self


__all__ = [
    "GradedJacobianCoefficientMap",
    "GradedJacobianKernelWitness",
    "GradedJacobianMapEntry",
    "GradedJacobianRankMinor",
    "GradedJacobianSyzygyRequest",
    "GradedJacobianSyzygyResult",
]
