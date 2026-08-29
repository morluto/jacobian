"""Regression tests binding rational RREF, rank, and nullspace to their source."""

from __future__ import annotations

from fractions import Fraction
from itertools import islice

import pytest
import sympy
from pydantic import ValidationError
from sympy import primerange

from jacobian._exact import CanonicalRational
from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math import matrices
from jacobian.math.matrices._operation_models import (
    MAX_DETERMINANT_MATRIX_DIMENSION,
    MAX_DETERMINANT_SCALAR_WORK,
    IntegerMatrixRequest,
    MatrixDeterminantRequest,
    MatrixRankRequest,
    MatrixRankResult,
    NullspaceResult,
    RationalMatrixProductRequest,
    RationalMatrixRequest,
    RrefResult,
)
from jacobian.math.matrices._tools import (
    compute_determinant,
    compute_nullspace,
    compute_product,
    compute_rank,
    compute_rref,
    compute_smith_normal_form,
)
from jacobian.math.matrices.operations import rank_result
from jacobian.math.matrices.values import (
    MAX_EXACT_LINEAR_MATRIX_AXIS,
    MAX_MATRIX_DIMENSION,
    MAX_RATIONAL_MATRIX_ORDER,
    IntegerMatrix,
    RationalMatrix,
    rational_matrix_from_fractions,
)


def _tall_relation_rows() -> tuple[tuple[int, ...], ...]:
    rows: list[tuple[int, ...]] = []
    for row in range(46):
        values = [0] * 21
        if row < 20:
            values[row] = 1
        else:
            values[row % 20] = 1
        values[20] = values[0] + values[1]
        rows.append(tuple(values))
    return tuple(rows)


def _sympy_from_fractions(entries: tuple[tuple[Fraction, ...], ...]) -> sympy.Matrix:
    return sympy.Matrix(
        [
            [sympy.Rational(value.numerator, value.denominator) for value in row]
            for row in entries
        ]
    )


def test_flint_exact_linear_operations_support_reported_46_by_21_shape() -> None:
    integer_rows = _tall_relation_rows()
    rational = rational_matrix_from_fractions(
        tuple(tuple(Fraction(value) for value in row) for row in integer_rows)
    )

    rank = compute_rank(MatrixRankRequest(matrix=rational))
    rref = compute_rref(RationalMatrixRequest(matrix=rational))
    nullspace = compute_nullspace(RationalMatrixRequest(matrix=rational))

    assert rank.rank == rref.rank == 20
    assert nullspace.rank == 20
    assert nullspace.nullity == 1
    vector = tuple(value.as_fraction() for value in nullspace.basis_vectors[0])
    assert all(
        sum(
            Fraction(entry) * coefficient
            for entry, coefficient in zip(row, vector, strict=True)
        )
        == 0
        for row in integer_rows
    )

    smith = compute_smith_normal_form(
        IntegerMatrixRequest(
            matrix=IntegerMatrix(
                entries=tuple(
                    tuple(str(value) for value in row) for row in integer_rows
                )
            )
        )
    )
    assert smith.rank == 20
    assert smith.invariant_factors == ("1",) * 20
    assert len(smith.normal_form.entries) == 46
    assert len(smith.normal_form.entries[0]) == 21


def test_native_exact_linear_operations_share_the_46_by_21_flint_path() -> None:
    integer_rows = _tall_relation_rows()
    source = sympy.Matrix([list(row) for row in integer_rows])
    rational = rational_matrix_from_fractions(
        tuple(tuple(Fraction(value) for value in row) for row in integer_rows)
    )
    integer = IntegerMatrix(
        entries=tuple(tuple(str(value) for value in row) for row in integer_rows)
    )

    native_rank, native_pivots = matrices.rank(source)
    wire_rank = compute_rank(MatrixRankRequest(matrix=rational))
    native_reduced, native_rref_pivots = matrices.rref(source)
    wire_rref = compute_rref(RationalMatrixRequest(matrix=rational))
    native_smith = matrices.smith_normal_form(source)
    wire_smith = compute_smith_normal_form(IntegerMatrixRequest(matrix=integer))

    assert native_rank == wire_rank.rank == 20
    assert native_pivots == wire_rank.pivot_columns
    assert native_rref_pivots == wire_rref.pivot_columns
    assert native_reduced == sympy.Matrix(
        [
            [sympy.Rational(value.as_fraction()) for value in row]
            for row in wire_rref.reduced_matrix.entries
        ]
    )
    assert native_smith == sympy.Matrix(
        [[int(value) for value in row] for row in wire_smith.normal_form.entries]
    )


def test_native_exact_linear_operations_reject_an_axis_above_the_flint_envelope() -> (
    None
):
    oversized = sympy.zeros(MAX_EXACT_LINEAR_MATRIX_AXIS + 1, 1)

    with pytest.raises(ValueError, match="between 1 and 64"):
        matrices.rank(oversized)
    with pytest.raises(ValueError, match="between 1 and 64"):
        matrices.rref(oversized)
    with pytest.raises(ValueError, match="between 1 and 64"):
        matrices.smith_normal_form(oversized)


def test_exact_linear_admission_rejects_unrepresentable_rref_height() -> None:
    denominators = tuple(islice(primerange(1_000_000_000, 2_000_000_000), 64))
    matrix = RationalMatrix(
        entries=tuple(
            tuple(
                CanonicalRational(num="1", den=str(denominator))
                for denominator in denominators
            )
            for _ in range(64)
        )
    )

    with pytest.raises(OperationDomainValidationError) as excinfo:
        compute_rref(RationalMatrixRequest(matrix=matrix))

    assert excinfo.value.errors()[0]["type"] == "matrix.budget_exceeded"
    assert "result-height" in excinfo.value.errors()[0]["msg"]


def _matrix(rows: list[list[str]]) -> RationalMatrix:
    return RationalMatrix.model_validate(
        {
            "domain": "QQ",
            "entries": [
                [
                    {
                        "num": value.split("/")[0],
                        "den": value.split("/")[1] if "/" in value else "1",
                    }
                    for value in row
                ]
                for row in rows
            ],
        }
    )


def test_rational_matrix_from_fractions_preserves_canonical_exact_entries() -> None:
    matrix = rational_matrix_from_fractions(
        ((Fraction(-6, 8), Fraction(5)), (Fraction(0), Fraction(7, 12)))
    )

    assert matrix.model_dump(mode="json") == {
        "domain": "QQ",
        "entries": [
            [{"num": "-3", "den": "4"}, {"num": "5", "den": "1"}],
            [{"num": "0", "den": "1"}, {"num": "7", "den": "12"}],
        ],
    }


def test_producer_results_replay_across_shapes() -> None:
    """Zero, rectangular, rank-deficient, and full-rank sources stay bound."""

    shapes = (
        _matrix([["0", "0"], ["0", "0"]]),
        _matrix([["1", "2", "3"], ["4", "5", "6"]]),
        _matrix([["1", "2"], ["2", "4"], ["3", "6"]]),
        _matrix([["1/2", "0"], ["0", "1/3"]]),
    )
    for matrix in shapes:
        request = RationalMatrixRequest(matrix=matrix)
        rref = compute_rref(request)
        assert rref.matrix == matrix

        rank_request = MatrixRankRequest(matrix=matrix)
        rank = compute_rank(rank_request)
        assert rank.matrix == matrix
        assert rank.rank == len(rank.pivot_columns)

        nullspace = compute_nullspace(request)
        assert nullspace.matrix == matrix
        assert nullspace.rank + nullspace.nullity == nullspace.ambient_dimension
        assert len(nullspace.basis_vectors) == nullspace.nullity


@pytest.mark.parametrize(
    "rows",
    (
        [["1", "2", "3"], ["4", "5", "6"]],
        [["0", "0", "0"], ["1", "0", "0"]],
        [["1/2", "1/3"], ["1/5", "1/7"], ["1", "1"]],
    ),
)
def test_serialized_results_round_trip(rows: list[list[str]]) -> None:
    matrix = _matrix(rows)
    rref = compute_rref(RationalMatrixRequest(matrix=matrix))
    assert RrefResult.model_validate(rref.model_dump()) == rref

    rank = compute_rank(MatrixRankRequest(matrix=matrix))
    assert MatrixRankResult.model_validate(rank.model_dump()) == rank

    nullspace = compute_nullspace(RationalMatrixRequest(matrix=matrix))
    assert NullspaceResult.model_validate(nullspace.model_dump()) == nullspace


def test_producer_to_serialized_interoperability() -> None:
    """Serialized RREF agrees with independently produced rank and nullspace."""

    matrix = _matrix([["1", "2", "3"], ["4", "5", "6"], ["7", "8", "9"]])
    rref = RrefResult.model_validate(
        compute_rref(RationalMatrixRequest(matrix=matrix)).model_dump()
    )
    rank = compute_rank(MatrixRankRequest(matrix=matrix))
    nullspace = compute_nullspace(RationalMatrixRequest(matrix=matrix))
    assert rref.rank == rank.rank == nullspace.rank == 2
    assert list(rref.pivot_columns) == list(rank.pivot_columns)
    assert list(rref.free_columns) == list(nullspace.free_columns)


def test_product_value_feeds_determinant_without_reencoding() -> None:
    """A matrix producer's canonical value is the determinant input value."""

    left = _matrix([["1", "2"], ["3", "4"]])
    right = _matrix([["0", "1"], ["1", "0"]])
    product = compute_product(RationalMatrixProductRequest(left=left, right=right))

    # Transport composition parses the producer's canonical payload directly,
    # and native composition retains its exact owner type unchanged.
    transported = MatrixDeterminantRequest.model_validate(
        {"matrix": product.product.model_dump(mode="json")}
    )
    assert transported.matrix == product.product

    request = MatrixDeterminantRequest(matrix=product.product)
    assert request.matrix is product.product
    assert compute_determinant(request).determinant == CanonicalRational(
        num="2", den="1"
    )


def test_flint_rectangular_product_exceeds_shared_matrix_axis() -> None:
    left_rows = 96
    right_columns = 80
    left = _matrix([[str(row + 1), "1", "-1"] for row in range(left_rows)])
    right = _matrix(
        [
            ["1" for _ in range(right_columns)],
            [str(column + 1) for column in range(right_columns)],
            ["2" for _ in range(right_columns)],
        ]
    )

    result = compute_product(RationalMatrixProductRequest(left=left, right=right))

    assert result.left_rows == left_rows
    assert result.inner_dimension == 3
    assert result.right_columns == right_columns
    assert tuple(
        tuple(value.as_fraction() for value in row) for row in result.product.entries
    ) == tuple(
        tuple(Fraction(row + column) for column in range(right_columns))
        for row in range(left_rows)
    )


def test_product_rejects_coefficient_growth_before_backend() -> None:
    inner_dimension = 128
    left = RationalMatrix(
        entries=(
            tuple(
                CanonicalRational(num="1", den=str(10**255 + index + 1))
                for index in range(inner_dimension)
            ),
        )
    )
    right = RationalMatrix(
        entries=tuple(
            (CanonicalRational(num="1", den="1"),) for _ in range(inner_dimension)
        )
    )

    with pytest.raises(OperationDomainValidationError, match="canonical digit budget"):
        compute_product(RationalMatrixProductRequest(left=left, right=right))


def _identity_entries(size: int) -> tuple[tuple[CanonicalRational, ...], ...]:
    one = CanonicalRational(num="1", den="1")
    zero = CanonicalRational(num="0", den="1")
    return tuple(
        tuple(one if index == column else zero for column in range(size))
        for index in range(size)
    )


def test_product_admits_dense_shared_denominator_order_32_times_identity() -> None:
    order = 32
    huge = CanonicalRational(num="1", den=str(10**255 + 1))
    left = RationalMatrix(
        entries=tuple(tuple(huge for _ in range(order)) for _ in range(order))
    )
    right = RationalMatrix(entries=_identity_entries(order))

    result = compute_product(RationalMatrixProductRequest(left=left, right=right))

    assert result.product == left


def test_product_admits_shared_denominator_dot_product() -> None:
    inner_dimension = 128
    denominator = 10**255 + 1
    value = CanonicalRational(num="1", den=str(denominator))
    left = RationalMatrix(entries=(tuple(value for _ in range(inner_dimension)),))
    right = RationalMatrix(entries=tuple((value,) for _ in range(inner_dimension)))

    result = compute_product(RationalMatrixProductRequest(left=left, right=right))

    assert result.product.entries == (
        (CanonicalRational.from_integer_ratio(inner_dimension, denominator**2),),
    )


def test_product_admits_cancelling_order_32_dot_product_terms() -> None:
    order = 32
    denominators = tuple(10**255 + 2 * index + 1 for index in range(order // 2))
    zero = CanonicalRational(num="0", den="1")
    one = CanonicalRational(num="1", den="1")
    row = tuple(
        CanonicalRational(num="1" if offset == 0 else "-1", den=str(denominator))
        for denominator in denominators
        for offset in range(2)
    )
    left = RationalMatrix(entries=tuple(row for _ in range(order)))
    right = RationalMatrix(
        entries=tuple(tuple(one for _ in range(order)) for _ in range(order))
    )

    result = compute_product(RationalMatrixProductRequest(left=left, right=right))

    assert result.product.entries == tuple(
        tuple(zero for _ in range(order)) for _ in range(order)
    )


def test_product_admits_cancelling_swapped_denominator_pairs() -> None:
    denominator_a = 10**255 + 1
    denominator_b = 10**255 + 3
    left = RationalMatrix(
        entries=(
            tuple(
                CanonicalRational(
                    num="1" if index % 2 == 0 else "-1",
                    den=str(denominator_a if index % 2 == 0 else denominator_b),
                )
                for index in range(128)
            ),
        )
    )
    right = RationalMatrix(
        entries=tuple(
            (
                CanonicalRational(
                    num="1",
                    den=str(denominator_b if index % 2 == 0 else denominator_a),
                ),
            )
            for index in range(128)
        )
    )

    result = compute_product(RationalMatrixProductRequest(left=left, right=right))

    assert result.product.entries == ((CanonicalRational(num="0", den="1"),),)


def test_product_admits_sparse_order_32_with_one_large_entry() -> None:
    order = 32
    denominator = str(10**255 + 1)
    one = CanonicalRational(num="1", den="1")
    zero = CanonicalRational(num="0", den="1")
    huge = CanonicalRational(num="1", den=denominator)
    left_entries = [
        [one if row == column else zero for column in range(order)]
        for row in range(order)
    ]
    left_entries[0][0] = huge
    left = RationalMatrix(entries=tuple(tuple(row) for row in left_entries))
    right = RationalMatrix(entries=_identity_entries(order))

    result = compute_product(RationalMatrixProductRequest(left=left, right=right))

    assert result.product.entries[0][0] == huge
    assert result.product.entries[1][1] == one


def test_native_multiply_shares_widened_flint_product_kernel() -> None:
    import sympy

    from jacobian.math import matrices

    left_rows = 40
    right_columns = 33
    left = sympy.Matrix([[row + 1, 1, -1] for row in range(left_rows)])
    right = sympy.Matrix(
        [
            [1 for _ in range(right_columns)],
            [column + 1 for column in range(right_columns)],
            [2 for _ in range(right_columns)],
        ]
    )

    native = matrices.multiply(left, right)
    wire = compute_product(
        RationalMatrixProductRequest(
            left=RationalMatrix(
                entries=tuple(
                    tuple(
                        CanonicalRational.from_integer_ratio(int(left[row, column]), 1)
                        for column in range(3)
                    )
                    for row in range(left_rows)
                )
            ),
            right=RationalMatrix(
                entries=tuple(
                    tuple(
                        CanonicalRational.from_integer_ratio(int(right[row, column]), 1)
                        for column in range(right_columns)
                    )
                    for row in range(3)
                )
            ),
        )
    )

    assert native == sympy.Matrix(
        [[value.as_fraction() for value in row] for row in wire.product.entries]
    )


def test_native_matrix_operations_keep_large_exact_scalar_fallbacks() -> None:
    import sympy

    from jacobian.math import matrices

    huge = 10**256
    source = sympy.Matrix([[huge]])

    assert matrices.multiply(source, sympy.ones(1)) == source
    assert matrices.characteristic_polynomial(source, "lambda").as_expr() == (
        sympy.Symbol("lambda") - huge
    )


def test_native_matrix_operations_admit_exact_256_digit_scalars() -> None:
    import sympy

    from jacobian.math import matrices

    boundary = 10**256 - 1
    source = sympy.diag(boundary, *([1] * 32))

    assert matrices.multiply(source, sympy.eye(33)) == source
    assert matrices.inverse(source)[0, 0] == sympy.Rational(1, boundary)
    polynomial = matrices.characteristic_polynomial(source, "lambda")
    assert polynomial.degree() == 33
    assert polynomial.LC() == 1


def test_request_admission_rejects_matrices_above_the_computation_dimension() -> None:
    oversized = RationalMatrix(
        entries=_identity_entries(MAX_EXACT_LINEAR_MATRIX_AXIS + 1)
    )
    rref_request = RationalMatrixRequest(matrix=oversized)
    rank_request = MatrixRankRequest(matrix=oversized)

    with pytest.raises(OperationDomainValidationError, match="64 rows and columns"):
        compute_rref(rref_request)
    with pytest.raises(OperationDomainValidationError, match="64 rows and columns"):
        compute_rank(rank_request)


def test_exact_linear_requests_admit_tall_matrices_above_the_square_dimension() -> None:
    tall = RationalMatrix(
        entries=tuple(
            tuple(
                CanonicalRational(num=str(int(row == column)), den="1")
                for column in range(21)
            )
            for row in range(46)
        )
    )
    assert RationalMatrixRequest(matrix=tall).matrix is tall
    assert MatrixRankRequest(matrix=tall).matrix is tall


def test_exact_linear_requests_reject_an_axis_above_the_operation_envelope() -> None:
    matrix = RationalMatrix(
        entries=tuple(
            tuple(
                CanonicalRational(num=str(column + 1), den="1") for column in range(2)
            )
            for _ in range(MAX_EXACT_LINEAR_MATRIX_AXIS + 1)
        )
    )

    with pytest.raises(OperationDomainValidationError) as excinfo:
        rank_result(matrix)

    assert excinfo.value.errors()[0]["type"] == "matrix.budget_exceeded"
    assert "64 rows and columns" in excinfo.value.errors()[0]["msg"]


def test_request_admission_keeps_the_boundary_computation_dimension() -> None:
    boundary = RationalMatrix(entries=_identity_entries(MAX_MATRIX_DIMENSION))
    request = MatrixRankRequest(matrix=boundary)
    rank = compute_rank(request)
    assert rank.rank == MAX_MATRIX_DIMENSION
    assert len(rank.pivot_columns) == MAX_MATRIX_DIMENSION
    assert rank.matrix == boundary


def test_determinant_accepts_its_operation_specific_matrix_boundary() -> None:
    assert MAX_RATIONAL_MATRIX_ORDER >= MAX_DETERMINANT_MATRIX_DIMENSION
    matrix = RationalMatrix(entries=_identity_entries(MAX_DETERMINANT_MATRIX_DIMENSION))
    assert MatrixDeterminantRequest(matrix=matrix).matrix is matrix


def test_flint_determinant_executes_above_the_previous_order_ceiling() -> None:
    order = 96
    entries = tuple(
        tuple(
            Fraction(-2, 3)
            if row == column == 0
            else Fraction(1)
            if row == column
            else Fraction(row + column + 1, 7)
            if column > row
            else Fraction(0)
            for column in range(order)
        )
        for row in range(order)
    )
    matrix = rational_matrix_from_fractions(entries)

    result = compute_determinant(MatrixDeterminantRequest(matrix=matrix))

    assert result.determinant == CanonicalRational(num="-2", den="3")
    native = matrices.determinant(_sympy_from_fractions(entries))
    assert native == sympy.Rational(-2, 3)


def test_determinant_preserves_row_swap_and_scaling_invariants() -> None:
    entries = (
        (Fraction(1, 2), Fraction(2), Fraction(-1)),
        (Fraction(3), Fraction(1, 3), Fraction(4)),
        (Fraction(-2), Fraction(5), Fraction(7, 11)),
    )
    source = rational_matrix_from_fractions(entries)
    transformed = rational_matrix_from_fractions(
        (tuple(-3 * value for value in entries[1]), entries[0], entries[2])
    )

    source_value = compute_determinant(
        MatrixDeterminantRequest(matrix=source)
    ).determinant.as_fraction()
    transformed_value = compute_determinant(
        MatrixDeterminantRequest(matrix=transformed)
    ).determinant.as_fraction()

    assert transformed_value == 3 * source_value


def test_determinant_rejects_requests_above_the_scalar_work_envelope() -> None:
    order = MAX_DETERMINANT_MATRIX_DIMENSION
    large = Fraction(int("9" * 256))
    entries = tuple(
        tuple(
            large if row == column == 0 else Fraction(int(row == column))
            for column in range(order)
        )
        for row in range(order)
    )
    matrix = rational_matrix_from_fractions(entries)
    assert order**3 * 256 > MAX_DETERMINANT_SCALAR_WORK

    with pytest.raises(OperationDomainValidationError, match="scalar-work budget"):
        compute_determinant(MatrixDeterminantRequest(matrix=matrix))
    with pytest.raises(OperationDomainValidationError, match="scalar-work budget"):
        matrices.determinant(_sympy_from_fractions(entries))


def test_determinant_rejects_an_unrepresentable_result_height_bound() -> None:
    denominators = tuple(primerange(2, 730))[:MAX_DETERMINANT_MATRIX_DIMENSION]
    row = tuple(Fraction(1, denominator) for denominator in denominators)
    entries = tuple(row for _ in range(MAX_DETERMINANT_MATRIX_DIMENSION))
    matrix = rational_matrix_from_fractions(entries)

    with pytest.raises(
        OperationDomainValidationError, match="rational result-height bound"
    ):
        compute_determinant(MatrixDeterminantRequest(matrix=matrix))
    with pytest.raises(
        OperationDomainValidationError, match="rational result-height bound"
    ):
        matrices.determinant(_sympy_from_fractions(entries))


def test_raw_preflight_keeps_exact_linear_and_determinant_axis_boundaries() -> None:
    def wire_identity(order: int) -> list[list[dict[str, str]]]:
        return [
            [{"num": str(int(row == column)), "den": "1"} for column in range(order)]
            for row in range(order)
        ]

    rank_boundary = {"matrix": {"entries": wire_identity(MAX_EXACT_LINEAR_MATRIX_AXIS)}}
    assert MatrixRankRequest.model_validate(
        rank_boundary
    ).matrix.entries == _identity_entries(MAX_EXACT_LINEAR_MATRIX_AXIS)
    rank_above_operation_axis = MatrixRankRequest.model_validate(
        {"matrix": {"entries": wire_identity(MAX_EXACT_LINEAR_MATRIX_AXIS + 1)}}
    )
    with pytest.raises(OperationDomainValidationError, match="64 rows and columns"):
        compute_rank(rank_above_operation_axis)

    determinant_boundary = {
        "matrix": {"entries": wire_identity(MAX_DETERMINANT_MATRIX_DIMENSION)}
    }
    assert MatrixDeterminantRequest.model_validate(
        determinant_boundary
    ).matrix.entries == (_identity_entries(MAX_DETERMINANT_MATRIX_DIMENSION))
    with pytest.raises(ValidationError):
        MatrixDeterminantRequest.model_validate(
            {"matrix": {"entries": wire_identity(MAX_DETERMINANT_MATRIX_DIMENSION + 1)}}
        )


def test_raw_preflight_rejects_a_257_digit_operation_scalar() -> None:
    with pytest.raises(ValidationError):
        MatrixRankRequest.model_validate(
            {
                "matrix": {
                    "entries": [
                        [{"num": "9" * 257, "den": "1"}],
                    ]
                }
            }
        )
