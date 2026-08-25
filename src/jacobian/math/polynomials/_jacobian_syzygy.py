"""Exact graded Jacobian coefficient maps and first-kernel search."""

from __future__ import annotations

import hashlib
from fractions import Fraction
from typing import Any, Literal

from jacobian._exact import CanonicalRational
from jacobian.canonical import canonicalize_json, format_canonical_integer
from jacobian.catalog.models import (
    OperationExample,
)
from jacobian.math.arithmetic import primitive_integer_vector
from jacobian.math.polynomials._conversions import (
    rational_from_sympy,
    rational_polynomial_from_sympy,
    rational_polynomial_to_sympy,
    symbols_for_variables,
)
from jacobian.math.polynomials._support import polynomial_operation
from jacobian.math.polynomials._syzygy_models import (
    GradedJacobianCoefficientMap,
    GradedJacobianKernelWitness,
    GradedJacobianMapEntry,
    GradedJacobianRankMinor,
    GradedJacobianSyzygyCoefficientRequest,
    GradedJacobianSyzygyRequest,
    GradedJacobianSyzygyRequestBase,
    GradedJacobianSyzygyResult,
    _homogeneous_basis,
)
from jacobian.math.polynomials.values import (
    RationalPolynomial,
    RationalPolynomialTerm,
    SparseRationalPolynomial,
)


def _fraction_text(value: Any) -> str:
    fraction = Fraction(value)
    return (
        f"{format_canonical_integer(fraction.numerator)}/"
        f"{format_canonical_integer(fraction.denominator)}"
    )


def _matrix_digest(
    *,
    multiplier_degree: int,
    source_basis: tuple[tuple[int, ...], ...],
    target_basis: tuple[tuple[int, ...], ...],
    entries: tuple[tuple[int, int, Any], ...],
) -> str:
    payload = {
        "protocol": "jacobian.graded-jacobian-map.v1",
        "multiplier_degree": multiplier_degree,
        "source_monomial_basis": [list(item) for item in source_basis],
        "target_monomial_basis": [list(item) for item in target_basis],
        "entries": [
            [row, column, _fraction_text(value)] for row, column, value in entries
        ],
    }
    return f"sha256:{hashlib.sha256(canonicalize_json(payload)).hexdigest()}"


def _primitive_kernel(vector: Any) -> tuple[Fraction, ...]:
    fractions = tuple(Fraction(value) for value in vector)
    try:
        primitive = primitive_integer_vector(fractions)
    except ValueError as exc:
        raise RuntimeError("symbolic nullspace returned a zero basis vector") from exc
    return tuple(Fraction(value) for value in primitive)


def _multiplier_polynomial(
    *,
    variables: tuple[str, ...],
    basis: tuple[tuple[int, ...], ...],
    coefficients: tuple[Fraction, ...],
) -> RationalPolynomial:
    return RationalPolynomial(
        variables=variables,
        polynomial=SparseRationalPolynomial(
            terms=tuple(
                RationalPolynomialTerm(
                    coefficient=CanonicalRational(
                        num=format_canonical_integer(coefficient.numerator),
                        den=format_canonical_integer(coefficient.denominator),
                    ),
                    exponents=exponents,
                )
                for exponents, coefficient in zip(basis, coefficients, strict=True)
                if coefficient
            )
        ),
    )


def _coefficient_matrix(
    partials: tuple[Any, ...],
    multiplier_degree: int,
    homogeneous_degree: int,
) -> tuple[
    tuple[tuple[int, ...], ...],
    tuple[tuple[int, ...], ...],
    Any,
    tuple[tuple[int, int, Any], ...],
]:
    from sympy import Matrix

    variable_count = len(partials)
    source_basis = _homogeneous_basis(variable_count, multiplier_degree)
    target_degree = homogeneous_degree - 1 + multiplier_degree
    target_basis = _homogeneous_basis(variable_count, target_degree)
    row_by_exponent = {exponents: index for index, exponents in enumerate(target_basis)}
    matrix = Matrix.zeros(len(target_basis), variable_count * len(source_basis))
    for component, partial in enumerate(partials):
        for basis_index, multiplier_exponents in enumerate(source_basis):
            column = component * len(source_basis) + basis_index
            for partial_exponents, coefficient in partial.terms():
                if coefficient == 0:
                    continue
                target_exponents = tuple(
                    left + right
                    for left, right in zip(
                        multiplier_exponents,
                        partial_exponents,
                        strict=True,
                    )
                )
                row = row_by_exponent[target_exponents]
                matrix[row, column] += coefficient
    entries = tuple(
        (row, column, matrix[row, column])
        for row in range(matrix.rows)
        for column in range(matrix.cols)
        if matrix[row, column] != 0
    )
    return source_basis, target_basis, matrix, entries


def compute_graded_jacobian_syzygy(
    request: GradedJacobianSyzygyRequest,
) -> GradedJacobianSyzygyResult:
    return _compute_graded_jacobian_syzygy(request)


def compute_graded_jacobian_syzygy_coefficients(
    request: GradedJacobianSyzygyCoefficientRequest,
) -> GradedJacobianSyzygyResult:
    """Retain the full sparse coefficient maps as explicit evidence."""
    return _compute_graded_jacobian_syzygy(request)


def _compute_graded_jacobian_syzygy(
    request: GradedJacobianSyzygyRequestBase,
) -> GradedJacobianSyzygyResult:
    if request.polynomial is not None:
        variables = request.polynomial.variables
        source = rational_polynomial_to_sympy(request.polynomial)
        source_kind: Literal[
            "EXPANDED_POLYNOMIAL", "LABELLED_LINEAR_FACTOR_PRODUCT"
        ] = "EXPANDED_POLYNOMIAL"
    else:
        from sympy import Poly, Rational

        linear_factors = request.linear_factors
        factor_variables = request.linear_factor_variables
        if linear_factors is None or factor_variables is None:
            raise ValueError("linear-factor input is incomplete")
        variables = factor_variables
        generators = symbols_for_variables(variables)
        source = Poly(1, *generators, domain="QQ")
        for factor in linear_factors:
            source *= Poly(
                sum(
                    Rational(coefficient.as_fraction()) * generator
                    for coefficient, generator in zip(
                        factor.coefficients,
                        generators,
                        strict=True,
                    )
                ),
                *generators,
                domain="QQ",
            )
        source_kind = "LABELLED_LINEAR_FACTOR_PRODUCT"
    source_degree = int(source.total_degree())
    partials = tuple(source.diff(variable) for variable in source.gens)
    maps: list[GradedJacobianCoefficientMap] = []
    kernel_witness: GradedJacobianKernelWitness | None = None
    first_degree: int | None = None

    for multiplier_degree in range(request.max_degree + 1):
        source_basis, target_basis, matrix, entries = _coefficient_matrix(
            partials,
            multiplier_degree,
            source_degree,
        )
        _, pivot_columns = matrix.rref()
        rank = len(pivot_columns)
        rank_minor: GradedJacobianRankMinor | None = None
        if rank:
            independent_rows = matrix[:, list(pivot_columns)].T.rref()[1]
            row_indices = tuple(int(index) for index in independent_rows)
            column_indices = tuple(int(index) for index in pivot_columns)
            determinant = matrix.extract(row_indices, column_indices).det()
            if determinant == 0:
                raise RuntimeError("rank-minor extraction returned a zero determinant")
            rank_minor = GradedJacobianRankMinor(
                row_indices=row_indices,
                column_indices=column_indices,
                determinant=rational_from_sympy(determinant),
            )
        nullity = matrix.cols - rank
        maps.append(
            GradedJacobianCoefficientMap(
                multiplier_degree=multiplier_degree,
                source_monomial_basis=source_basis,
                target_monomial_basis=target_basis,
                row_count=matrix.rows,
                column_count=matrix.cols,
                matrix_digest=_matrix_digest(
                    multiplier_degree=multiplier_degree,
                    source_basis=source_basis,
                    target_basis=target_basis,
                    entries=entries,
                ),
                sparse_entries=(
                    tuple(
                        GradedJacobianMapEntry(
                            row=row,
                            column=column,
                            coefficient=rational_from_sympy(value),
                        )
                        for row, column, value in entries
                    )
                    if request.coefficient_map_detail == "SPARSE_ENTRIES"
                    else ()
                ),
                rank=rank,
                nullity=nullity,
                pivot_columns=tuple(int(index) for index in pivot_columns),
                rank_minor=rank_minor,
                injective=nullity == 0,
            )
        )
        if nullity:
            first_degree = multiplier_degree
            vector = _primitive_kernel(matrix.nullspace()[0])
            block_size = len(source_basis)
            multipliers = tuple(
                _multiplier_polynomial(
                    variables=variables,
                    basis=source_basis,
                    coefficients=vector[
                        component * block_size : (component + 1) * block_size
                    ],
                )
                for component in range(len(variables))
            )
            kernel_witness = GradedJacobianKernelWitness(
                multiplier_degree=multiplier_degree,
                coefficient_vector=tuple(
                    CanonicalRational(
                        num=format_canonical_integer(value.numerator),
                        den=format_canonical_integer(value.denominator),
                    )
                    for value in vector
                ),
                multipliers=multipliers,
            )
            break

    searched_through = first_degree if first_degree is not None else request.max_degree
    return GradedJacobianSyzygyResult(
        variables=variables,
        source_kind=source_kind,
        expanded_polynomial=rational_polynomial_from_sympy(source, variables),
        homogeneous_degree=source_degree,
        searched_through_degree=searched_through,
        coefficient_map_detail=request.coefficient_map_detail,
        partial_derivatives=tuple(
            rational_polynomial_from_sympy(partial, variables) for partial in partials
        ),
        degree_maps=tuple(maps),
        status="FOUND" if first_degree is not None else "NONE_THROUGH_BOUND",
        first_syzygy_degree=first_degree,
        kernel_witness=kernel_witness,
    )


GRADED_JACOBIAN_SYZYGY_OPERATION = polynomial_operation(
    "polynomial.jacobian_syzygy.minimum_degree.compute",
    "Compute the first graded Jacobian syzygy degree",
    (
        "For bounded homogeneous h in QQ[x_1,...,x_n], supplied sparsely or, "
        "for n=3, as labelled linear forms, construct every graded map "
        "(QQ[x_1,...,x_n]_q)^n -> QQ[x_1,...,x_n]_(q+deg(h)-1) from q=0 and "
        "stop at the first nonzero kernel or the finite degree bound. This "
        "operation returns rank certificates only; use the coefficient ledger "
        "for sparse map entries."
    ),
    GradedJacobianSyzygyRequest,
    GradedJacobianSyzygyResult,
    compute_graded_jacobian_syzygy,
    "polynomial",
    "jacobian",
    "syzygy",
    "homogeneous",
    "graded",
    "rank",
    "kernel",
    "exact",
    examples=(
        OperationExample(
            name="sparse-homogeneous-polynomial",
            description=(
                "Supply h=x^2+y^2+z^2 with unique nonzero terms in descending "
                "lexicographic exponent order."
            ),
            input={
                "polynomial": {
                    "variables": ["x", "y", "z"],
                    "polynomial": {
                        "terms": [
                            {
                                "coefficient": {"num": "1", "den": "1"},
                                "exponents": [2, 0, 0],
                            },
                            {
                                "coefficient": {"num": "1", "den": "1"},
                                "exponents": [0, 2, 0],
                            },
                            {
                                "coefficient": {"num": "1", "den": "1"},
                                "exponents": [0, 0, 2],
                            },
                        ]
                    },
                },
                "max_degree": 0,
            },
        ),
        OperationExample(
            name="labelled-linear-factor-product",
            description=(
                "Bind h=x*y*z to labelled factors; supply exactly one source form, "
                "with a matching three-variable order and unique labels."
            ),
            input={
                "linear_factors": [
                    {
                        "label": "Lx",
                        "coefficients": [
                            {"num": "1", "den": "1"},
                            {"num": "0", "den": "1"},
                            {"num": "0", "den": "1"},
                        ],
                    },
                    {
                        "label": "Ly",
                        "coefficients": [
                            {"num": "0", "den": "1"},
                            {"num": "1", "den": "1"},
                            {"num": "0", "den": "1"},
                        ],
                    },
                    {
                        "label": "Lz",
                        "coefficients": [
                            {"num": "0", "den": "1"},
                            {"num": "0", "den": "1"},
                            {"num": "1", "den": "1"},
                        ],
                    },
                ],
                "linear_factor_variables": ["x", "y", "z"],
                "max_degree": 1,
            },
        ),
    ),
)

JACOBIAN_SYZYGY_COEFFICIENT_LEDGER_OPERATION = polynomial_operation(
    "polynomial.jacobian_syzygy.coefficients.compute",
    "Compute graded Jacobian syzygy coefficient ledger",
    (
        "Compute every sparse entry in the bounded graded Jacobian coefficient "
        "maps, together with the syzygy summary and rank certificates."
    ),
    GradedJacobianSyzygyCoefficientRequest,
    GradedJacobianSyzygyResult,
    compute_graded_jacobian_syzygy_coefficients,
    "polynomial",
    "jacobian",
    "syzygy",
    "coefficient-ledger",
    "evidence",
    examples=(
        OperationExample(
            name="sparse-homogeneous-polynomial",
            description="Compute sparse coefficient maps for h=x²+y²+z².",
            input={
                "polynomial": {
                    "variables": ["x", "y", "z"],
                    "polynomial": {
                        "terms": [
                            {
                                "coefficient": {"num": "1", "den": "1"},
                                "exponents": [2, 0, 0],
                            },
                            {
                                "coefficient": {"num": "1", "den": "1"},
                                "exponents": [0, 2, 0],
                            },
                            {
                                "coefficient": {"num": "1", "den": "1"},
                                "exponents": [0, 0, 2],
                            },
                        ]
                    },
                },
                "max_degree": 0,
            },
        ),
    ),
)


__all__ = [
    "GRADED_JACOBIAN_SYZYGY_OPERATION",
    "JACOBIAN_SYZYGY_COEFFICIENT_LEDGER_OPERATION",
]
