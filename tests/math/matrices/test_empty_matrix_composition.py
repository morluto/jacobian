"""Zero-dimensional matrices retain their field and axes through composition."""

from fractions import Fraction

import pytest
import sympy
from pydantic import ValidationError

from jacobian.math.matrices._conversions import (
    integer_matrix_from_sympy,
    integer_matrix_to_sympy,
    rational_matrix_from_sympy,
    rational_matrix_to_sympy,
)
from jacobian.math.matrices.operations import (
    characteristic_polynomial_result,
    determinant_result,
    nullspace_result,
    product_result,
    rank_result,
    rref_result,
)
from jacobian.math.matrices.values import (
    EmbeddedRealSimpleNumberFieldMatrix,
    RationalMatrix,
    RealQuadraticMatrix,
    dense_rational_matrix_from_sparse,
    rational_matrix_from_fractions,
    sparse_rational_matrix_from_dense,
)
from jacobian.math.number_theory.number_fields import (
    RealNumberFieldEmbedding,
    SimpleNumberFieldPresentation,
    embeddings,
)


@pytest.mark.parametrize(("rows", "columns"), [(0, 3), (3, 0), (0, 0)])
def test_empty_exact_matrix_backend_and_wire_round_trips(
    rows: int, columns: int
) -> None:
    backend = sympy.zeros(rows, columns)
    rational = rational_matrix_from_sympy(backend)
    parsed = RationalMatrix.model_validate_json(rational.model_dump_json())
    assert rational_matrix_to_sympy(parsed) == backend
    integer = integer_matrix_from_sympy(backend)
    assert (
        integer_matrix_to_sympy(
            type(integer).model_validate_json(integer.model_dump_json())
        )
        == backend
    )
    assert (
        dense_rational_matrix_from_sparse(sparse_rational_matrix_from_dense(rational))
        == rational
    )
    assert (
        rational_matrix_from_fractions([[] for _ in range(rows)], column_count=columns)
        == rational
    )


@pytest.mark.parametrize(("rows", "columns"), [(0, 3), (3, 0), (0, 0)])
def test_empty_rational_elimination_retains_source_and_kernel(
    rows: int, columns: int
) -> None:
    matrix = rational_matrix_from_sympy(sympy.zeros(rows, columns))
    rank = rank_result(matrix)
    reduced = rref_result(matrix)
    kernel = nullspace_result(matrix)
    for result in (rank, reduced, kernel):
        assert type(result).model_validate_json(result.model_dump_json()) == result
        assert result.matrix == matrix
        assert result.rank == 0
    assert reduced.reduced_matrix == matrix
    assert kernel.ambient_dimension == columns
    assert kernel.nullity == columns
    vectors = [tuple(x.as_fraction() for x in v) for v in kernel.basis_vectors]
    assert vectors == [
        tuple(Fraction(int(i == j)) for j in range(columns)) for i in range(columns)
    ]


@pytest.mark.parametrize(
    ("rows", "inner", "columns"), [(0, 2, 3), (2, 0, 3), (2, 3, 0), (0, 0, 0)]
)
def test_empty_matrix_product_preserves_composable_axes(
    rows: int, inner: int, columns: int
) -> None:
    left = rational_matrix_from_sympy(sympy.zeros(rows, inner))
    right = rational_matrix_from_sympy(sympy.zeros(inner, columns))
    result = product_result(left, right)
    parsed = type(result).model_validate_json(result.model_dump_json())
    assert rational_matrix_to_sympy(parsed.product) == sympy.zeros(rows, columns)
    assert (parsed.left_rows, parsed.inner_dimension, parsed.right_columns) == (
        rows,
        inner,
        columns,
    )
    assert rank_result(parsed.product).rank == 0


def test_empty_determinant_and_characteristic_polynomial_are_units() -> None:
    matrix = RationalMatrix(row_count=0, column_count=0, entries=())
    assert determinant_result(matrix).determinant.as_fraction() == 1
    result = characteristic_polynomial_result(matrix)
    assert type(result).model_validate_json(result.model_dump_json()) == result
    assert result.degree == 0
    assert tuple(c.as_fraction() for c in result.coefficients_descending) == (
        Fraction(1),
    )


@pytest.mark.parametrize(("rows", "columns"), [(0, 3), (3, 0), (0, 0)])
def test_empty_extension_field_matrices_retain_parent(rows: int, columns: int) -> None:
    from jacobian.math.matrices._number_field import (
        domain_matrix_from_embedded,
        embedded_matrix_from_domain,
        recognize_real_simple_number_field,
    )

    entries = tuple(() for _ in range(rows))
    quadratic = RealQuadraticMatrix(
        radicand=2, row_count=rows, column_count=columns, entries=entries
    )
    assert (
        RealQuadraticMatrix.model_validate_json(quadratic.model_dump_json())
        == quadratic
    )
    assert quadratic.radicand == 2
    presentation = SimpleNumberFieldPresentation(coefficients_descending=(1, 0, -2))
    embedding = embeddings(presentation).records[1].embedding
    assert isinstance(embedding, RealNumberFieldEmbedding)
    value = EmbeddedRealSimpleNumberFieldMatrix(
        embedding=embedding, row_count=rows, column_count=columns, entries=entries
    )
    assert (
        EmbeddedRealSimpleNumberFieldMatrix.model_validate_json(value.model_dump_json())
        == value
    )
    assert value.embedding == embedding
    recognized = recognize_real_simple_number_field(embedding)
    backend = domain_matrix_from_embedded(value, recognized)
    assert backend.shape == (rows, columns)
    assert embedded_matrix_from_domain(backend, recognized) == value


def test_empty_quadratic_matrix_requires_field_context() -> None:
    with pytest.raises(ValidationError, match="explicit radicand"):
        RealQuadraticMatrix(entries=())


def test_raw_matrix_admission_checks_explicit_empty_axes() -> None:
    from jacobian.math.matrices._operation_models import RationalMatrixRequest

    with pytest.raises(ValidationError, match="dimensions are limited"):
        RationalMatrixRequest.model_validate(
            {"matrix": {"row_count": 0, "column_count": 4096, "entries": []}}
        )


def test_empty_rank_wire_retains_large_axes() -> None:
    from jacobian.math.matrices._operation_models import MatrixRankRequest
    from jacobian.math.matrices.values import SparseRationalMatrix

    for matrix in (
        RationalMatrix(row_count=0, column_count=4096, entries=()),
        SparseRationalMatrix(row_count=0, column_count=4096, entries=()),
    ):
        request = MatrixRankRequest.model_validate_json(
            MatrixRankRequest(matrix=matrix).model_dump_json()
        )
        result = rank_result(request.matrix)
        assert result.rank == 0
        assert result.matrix.row_count == 0
        assert result.matrix.column_count == 4096
